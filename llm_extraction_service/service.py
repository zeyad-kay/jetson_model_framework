from pathlib import Path
from typing import Optional

from common.io import write_json
from common.log import get_logger
from common.service import WorkerService
from llm_extraction_service.config import Config
from llm_extraction_service.model import QwenExtractor
from llm_extraction_service.reader import SegmentReader

log = get_logger(__name__)


class LLMExtractionService(WorkerService):
    """Keeps each session's ePCR up to date as its transcript is recorded.

    Works through the sessions the server creates (see WorkerService), in its steps:

    reader   a SegmentReader: the next new segment of <session>/transcript/segments.json,
             which the ASR service rewrites (atomically, closed segments only) as the session goes on.
    predict  QwenExtractor.predict(segment): the segment can add a record to each
             category of the ePCR.
    save     rewrites <session>/extractions/<category>.json, one file per category
             (medications, interventions, ...) holding that category's records so far,
             [{"segment", "extractions": [...]}, ...], writes the current state of each
             medication administration to extractions/medication_administrations.json,
             and sets extractions/progress.json to {"segment", "start", "end"} of the segment.
    cleanup  (the base's) stops the model and writes extractions/extractions.done ({"status":
             "ok", "failed" or "abandoned"}), which marks the files final, even if the session
             failed (its output marker).

    The session ends once the ASR service marks segments.json complete
    (segments.json.done) and its last segment is extracted, or once nothing in the
    session has changed for abandon_timeout seconds without that marker ("abandoned").
    """

    def __init__(self, config: Config):
        super().__init__("llm_extraction_service", config.sessions_dir, config.poll_interval, log,
                         status_path=config.status_path, max_session_age=config.max_session_age,
                         abandon_timeout=config.abandon_timeout)
        self.config = config
        # The current session's state, set by setup().
        self._extractions_dir: Optional[Path] = None

    def load_model(self) -> QwenExtractor:
        cfg = self.config
        extractor = QwenExtractor(
            model_id=cfg.extract_model,
            prompt_path=str(Path(cfg.prompts_dir) / "separated"),
            link=cfg.link,
            index_dir=str(Path(cfg.llm_dir) / "terminology" / "index"),
        )
        log.info("model ready: %s (%s)", type(extractor).__name__, extractor.model_id)
        return extractor

    def output_path(self, session_dir: Path) -> Path:
        # Stands for the files in extractions/ together: its marker is extractions/extractions.done.
        return session_dir / "extractions" / "extractions"

    def open_reader(self, session_dir: Path) -> SegmentReader:
        return SegmentReader(session_dir / "transcript" / "segments.json")

    def setup(self, session_dir: Path) -> None:
        self._extractions_dir = session_dir / "extractions"
        log.info("following %s, writing %s/<category>.json",
                 session_dir / "transcript" / "segments.json", self._extractions_dir)
        self._model.start()

    def save(self, segment: dict, record: dict) -> None:
        out = self._extractions_dir
        for c in self._model.CATEGORIES:
            write_json(out / f"{c}.json", record[c])
        write_json(out / "medication_administrations.json", record["medication_administrations"])
        # Written last, so it never names a segment the category files don't hold yet.
        write_json(out / "progress.json", {k: record[k] for k in ("segment", "start", "end")})
        log.debug("extracted segment %s", record["segment"])
