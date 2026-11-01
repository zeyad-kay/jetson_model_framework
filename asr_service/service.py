import queue
import threading
import time
from pathlib import Path

from asr_service.config import Config
from common.service import WorkerService
from common.io import TimedTranscriptWriter, TranscriptWriter
from asr_service.log import get_logger
from asr_service.model import Nemotron
from asr_service.segments import SegmentTracker

log = get_logger(__name__)


class ASRService(WorkerService):
    """Transcribes each session's audio as it is recorded.

    Idles until start() names a session, then reads <session>/audio/audio.pcm,
    transcribes it and writes <session>/transcript/transcript.txt (plus timestamped
    transcript.jsonl) until the
    session ends (stop(), or start() of the next session). shutdown() ends the
    current session and makes run() return.
    """

    def __init__(self, config: Config):
        self.config = config
        self._sessions = queue.Queue()  # session ids waiting to be processed; None = shut down
        self._end_session = threading.Event()
        self._shutdown = threading.Event()
        self._model = None

    def start(self, session_id: str) -> None:
        """Queues a session; any session in progress is finished first.

        Raises ValueError if session_id isn't an existing session folder.
        Safe to call from any thread.
        """
        if Path(session_id).name != session_id or not self._session_dir(session_id).is_dir():
            raise ValueError(f"unknown session: {session_id!r}")
        # Order matters: run() clears the flag after taking an id off the queue.
        self._end_session.set()
        self._sessions.put(session_id)

    def stop(self) -> None:
        """Finishes the current session (processing its remaining audio). Safe from any thread."""
        self._end_session.set()

    def shutdown(self) -> None:
        """Finishes the current session and makes run() return. Safe from any thread or signal handler."""
        self._shutdown.set()
        self._end_session.set()
        self._sessions.put(None)

    def _session_dir(self, session_id: str) -> Path:
        return Path(self.config.sessions_dir) / session_id

    def run(self) -> None:
        cfg = self.config
        # audio.pcm is raw PCM, so there is no WAV header to skip.
        self._model = Nemotron(sampling_rate=cfg.sampling_rate, header_bytes=0)
        log.info(
            "model ready: %s (%d Hz, %s)",
            type(self._model).__name__,
            self._model.sampling_rate,
            self._model.channel,
        )
        log.info("waiting for a session")
        while True:
            session_id = self._sessions.get()
            if session_id is None or self._shutdown.is_set():
                return
            self._end_session.clear()
            self._run_session(session_id)
            log.info("waiting for a session")

    def _run_session(self, session_id: str) -> None:
        session_dir = self._session_dir(session_id)
        audio_path = session_dir / "audio" / "audio.pcm"
        log.info("session %s started", session_id)
        # Nemotron emits text fragments (tokens), so no separator between writes.
        transcript_dir = session_dir / "transcript"
        tracker = SegmentTracker(
            transcript_dir / "segments.json", self.config.segment_pause, self._model.chunk_seconds
        )
        with TranscriptWriter(transcript_dir / "transcript.txt", separator="") as writer, \
                TimedTranscriptWriter(transcript_dir / "transcript.jsonl") as timed_writer:
            log.info("transcribing %s, writing %s and %s", audio_path, writer.path, timed_writer.path)
            worker = threading.Thread(
                target=self._transcribe, args=(audio_path, writer, timed_writer, tracker), daemon=True
            )
            worker.start()
            self._end_session.wait()
            # Let the model drain the audio it has not decoded yet. finish() is a
            # no-op if transcribe() hasn't started yet, so repeat it until the
            # worker exits; fall back to stop() if draining takes too long.
            deadline = time.monotonic() + self.config.drain_timeout
            while worker.is_alive():
                if time.monotonic() < deadline:
                    self._model.finish()
                else:
                    log.warning("draining timed out, stopping transcription")
                    self._model.stop()
                worker.join(timeout=0.5)
        tracker.close()
        log.info("session %s finished, %d segments", session_id, len(tracker))

    def _transcribe(
        self,
        audio_path: Path,
        writer: TranscriptWriter,
        timed_writer: TimedTranscriptWriter,
        tracker: SegmentTracker,
    ) -> None:
        try:
            # The recorder creates the file shortly after the session starts.
            while not audio_path.exists():
                if self._end_session.wait(self.config.poll_interval):
                    return
            for text, seconds in self._model.transcribe(str(audio_path)):
                if text:
                    log.debug("transcribed at %.2fs: %r", seconds, text)
                    writer.write(text)
                    timed_writer.write(text, seconds)
                    tracker.add(text, seconds)
                tracker.tick(seconds)
        except Exception:
            log.exception("transcription failed for %s", audio_path)
