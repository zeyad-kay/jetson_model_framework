from pathlib import Path
from typing import Any, Optional

from asr_service.reader import PcmReader
from asr_service.config import Config
from asr_service.model import Nemotron
from asr_service.segments import SegmentTracker
from common.io import TimedTranscriptWriter
from common.log import get_logger
from common.service import WorkerService

log = get_logger(__name__)


class ASRService(WorkerService):
    """Transcribes each session's audio as it is recorded.

    Works through the sessions the server creates (see WorkerService), in its steps:

    reader   a PcmReader: the next chunk the model decodes (Nemotron.chunk_span) of
             <session>/audio/audio.pcm, once recorded, which the server appends to as the
             session goes on (16-bit mono PCM): {"samples": float32 array, "final": whether
             they are the last, the rest of the audio, fewer than a chunk}.
    predict  Nemotron.predict(): the text fragments (tokens) decoded from it,
             {"fragments": [(text, seconds)], "seconds": the end of the audio decoded}.
    save     appends the fragments, timestamped, to <session>/transcript/transcript.jsonl
             (the transcript is their texts joined as they are), and adds them to
             segments.json, split at sentence ends and pauses (see SegmentTracker; only
             closed segments are written).
    cleanup  (the base's) stops the model's decoding, calls finalize_outputs, which closes
             transcript.jsonl and writes the last segment to segments.json, and then marks it
             complete: segments.json.done ({"status": "ok", "failed" or "abandoned"}), even if
             the session failed (its output marker).

    The session ends once the server marks audio.pcm complete (audio.pcm.done) and the rest
    of the audio is transcribed, or once nothing in the session has changed for
    abandon_timeout seconds without that marker ("abandoned").
    """

    def __init__(self, config: Config):
        super().__init__("asr_service", config.sessions_dir, config.poll_interval, log,
                         status_path=config.status_path, max_session_age=config.max_session_age,
                         abandon_timeout=config.abandon_timeout)
        self.config = config
        # The current session's outputs, set by setup().
        self._tracker: Optional[SegmentTracker] = None
        self._timed_writer: Optional[TimedTranscriptWriter] = None

    def load_model(self) -> Nemotron:
        cfg = self.config
        model = Nemotron(sampling_rate=cfg.sampling_rate, language=cfg.language,
                         num_lookahead_tokens=cfg.num_lookahead_tokens)
        log.info("model ready: %s (%d Hz, %s)", type(model).__name__, model.sampling_rate, model.language)
        return model

    def output_path(self, session_dir: Path) -> Path:
        return session_dir / "transcript" / "segments.json"

    def open_reader(self, session_dir: Path) -> PcmReader:
        return PcmReader(session_dir / "audio" / "audio.pcm", self._model.chunk_span)

    def setup(self, session_dir: Path) -> None:
        transcript_dir = session_dir / "transcript"
        self._tracker = SegmentTracker(transcript_dir / "segments.json", self.config.segment_pause,
                                       self.config.segment_min_length, self._model.chunk_seconds)
        self._timed_writer = TimedTranscriptWriter(transcript_dir / "transcript.jsonl")
        log.info("transcribing %s, writing %s and %s", session_dir / "audio" / "audio.pcm",
                 self._timed_writer.path, self._tracker.path)
        self._model.start()

    def save(self, inputs: dict, outputs: dict[str, Any]) -> None:
        for text, seconds in outputs["fragments"]:
            if text:
                log.debug("transcribed at %.2fs: %r", seconds, text)
                self._timed_writer.write(text, seconds)
                self._tracker.add(text, seconds)
        self._tracker.tick(outputs["seconds"])

    def finalize_outputs(self) -> None:
        if self._timed_writer is not None:
            self._timed_writer.close()
        tracker = self._tracker
        self._tracker = self._timed_writer = None
        if tracker is not None:
            # Writes the last segment, even after a failure. Before the output marker, so the
            # extraction service sees every segment before segments.json.done.
            tracker.close()
