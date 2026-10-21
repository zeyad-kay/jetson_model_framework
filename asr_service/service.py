import queue
import threading
from pathlib import Path

from asr_service.config import Config
from asr_service.io import AudioReader, TranscriptWriter
from asr_service.log import get_logger
from asr_service.model import ASRModel

log = get_logger(__name__)


class ASRService:
    """Transcribes each session's audio as it is recorded.

    Idles until start() names a session, then reads <session>/audio/audio.pcm,
    transcribes it and writes <session>/transcript/transcript.txt until the
    session ends (stop(), or start() of the next session). shutdown() ends the
    current session and makes run() return.
    """

    def __init__(self, config: Config):
        self.config = config
        self._sessions = queue.Queue()  # session ids waiting to be processed; None = shut down
        self._end_session = threading.Event()
        self._shutdown = threading.Event()
        self._model = None
        self._writer = None

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
        self._model = ASRModel(sampling_rate=cfg.sampling_rate)
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
        cfg = self.config
        session_dir = self._session_dir(session_id)
        log.info("session %s started", session_id)
        with (
            AudioReader(
                session_dir / "audio" / "audio.pcm",
                sampling_rate=self._model.sampling_rate,
                chunk_seconds=cfg.chunk_seconds,
            ) as reader,
            TranscriptWriter(session_dir / "transcript" / "transcript.txt") as self._writer,
        ):
            log.info(
                "reading %s in %.1fs chunks (%d bytes), writing %s",
                reader.path,
                cfg.chunk_seconds,
                reader.chunk_size,
                self._writer.path,
            )
            self._read_loop(reader)
        log.info("session %s finished", session_id)

    def _read_loop(self, reader: AudioReader) -> None:
        while True:
            # Check the flag before reading so audio written before the end
            # signal is never missed.
            ending = self._end_session.is_set()
            chunk = reader.read_chunk()
            if chunk is not None:
                self._process(chunk)
                continue
            if ending:
                break
            self._end_session.wait(self.config.poll_interval)
        # Session ended and all full chunks are consumed: flush the tail.
        tail = reader.read_remaining()
        if tail:
            self._process(tail)

    def _process(self, audio: bytes) -> None:
        log.debug("processing %d bytes of audio", len(audio))
        # TODO: transcribe `audio` with self._model and write the text to self._writer
