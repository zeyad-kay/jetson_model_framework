from pathlib import Path

from asr_service.config import Config
from common.service import WorkerService
from common.io import TimedTranscriptWriter, TranscriptWriter
from common.log import get_logger
from asr_service.model import Nemotron
from asr_service.segments import SegmentTracker

log = get_logger(__name__)


class ASRService(WorkerService):
    """Transcribes each session's audio as it is recorded.

    Idles until queue_session() names a session, then reads <session>/audio/audio.pcm,
    transcribes it and writes <session>/transcript/transcript.txt (plus timestamped
    transcript.jsonl and segments.json, split at sentence ends and pauses; see SegmentTracker) until the
    session ends (end_session(), or queue_session() of the next session). shutdown() ends the
    current session and makes run() return.
    """

    def __init__(self, config: Config):
        super().__init__(config.sessions_dir, config.poll_interval, log)
        self.config = config

    def _load_model(self) -> Nemotron:
        cfg = self.config
        # audio.pcm is raw PCM, so there is no WAV header to skip.
        model = Nemotron(
            sampling_rate=cfg.sampling_rate,
            language=cfg.language,
            delay=cfg.delay,
            num_lookahead_tokens=cfg.num_lookahead_tokens,
            header_bytes=0,
        )
        log.info("model ready: %s (%d Hz, %s)", type(model).__name__, model.sampling_rate, model.channel)
        return model

    def _process_session(self, session_dir: Path) -> None:
        audio_path = session_dir / "audio" / "audio.pcm"
        transcript_dir = session_dir / "transcript"
        tracker = SegmentTracker(
            transcript_dir / "segments.json", self.config.segment_pause,
            self.config.segment_min_length, self._model.chunk_seconds,
        )
        try:
            # Nemotron emits text fragments (tokens), so no separator between writes.
            with TranscriptWriter(transcript_dir / "transcript.txt", separator="") as writer, \
                    TimedTranscriptWriter(transcript_dir / "transcript.jsonl") as timed_writer:
                log.info("transcribing %s, writing %s and %s", audio_path, writer.path, timed_writer.path)
                if not self._wait_for_file(audio_path):
                    return
                for text, seconds in self._model.transcribe(str(audio_path)):
                    if text:
                        log.debug("transcribed at %.2fs: %r", seconds, text)
                        writer.write(text)
                        timed_writer.write(text, seconds)
                        tracker.add(text, seconds)
                    tracker.tick(seconds)
        finally:
            # Writes the last segment, even on a crash.
            tracker.close()
            log.info("wrote %d segments to %s", len(tracker), tracker.path)
