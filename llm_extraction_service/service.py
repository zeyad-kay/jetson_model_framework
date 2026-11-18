from pathlib import Path

from common.io import write_json
from common.service import WorkerService
from llm_extraction_service.config import Config
from common.log import get_logger
from llm_extraction_service.model import QwenExtractor

log = get_logger(__name__)


class LLMExtractionService(WorkerService):
    """Keeps each session's ePCR up to date as its transcript is recorded.

    Idles until queue_session() names a session, then follows the segments the ASR
    service writes to <session>/transcript/segments.json. Each segment can add a
    record to each category of the ePCR (see QwenExtractor), and after each one the
    service rewrites <session>/structured/<category>.json, one file per category
    (medications, interventions, ...) holding that category's records so far,
    [{"segment", "extractions": [...]}, ...], writes the current state of each medication
    administration to structured/medication_administrations.json, and sets structured/progress.json to
    {"segment", "start", "end"} of the segment. Once the session ends (end_session(),
    or queue_session() of the next session) and the last segment is taken in, it
    creates structured/extractions.done, which marks the files final.
    shutdown() ends the current session and makes run() return.
    """

    def __init__(self, config: Config):
        super().__init__(config.sessions_dir, config.poll_interval, log)
        self.config = config

    def _load_model(self) -> QwenExtractor:
        cfg = self.config
        extractor = QwenExtractor(
            model_id=cfg.extract_model,
            prompt_path=str(Path(cfg.prompts_dir) / "separated"),
            link=cfg.link,
            index_dir=str(Path(cfg.llm_dir) / "terminology" / "index"),
            poll_interval=cfg.poll_interval,
            drain_idle=cfg.drain_idle,
        )
        log.info("model ready: %s (%s)", type(extractor).__name__, extractor.model_id)
        return extractor

    def _process_session(self, session_dir: Path) -> None:
        segments_path = session_dir / "transcript" / "segments.json"
        structured = session_dir / "structured"
        done_path = structured / "extractions.done"
        try:
            log.info("following %s, writing %s/<category>.json", segments_path, structured)
            if not self._wait_for_file(segments_path):
                return

            count = 0
            for record in self._model.extract(str(segments_path)):
                count += 1
                for c in self._model.CATEGORIES:
                    write_json(structured / f"{c}.json", record[c])
                write_json(structured / "medication_administrations.json", record["medication_administrations"])
                # Written last, so it never names a segment the category files don't hold yet.
                write_json(structured / "progress.json",
                           {k: record[k] for k in ("segment", "start", "end")})
                log.debug("extracted segment %s", record["segment"])
            log.info("extracted %d segments (%d unparseable replies, %d validation errors)",
                     count, self._model.parse_failures, self._model.validation_errors)
        finally:
            # Even when nothing was extracted or extraction failed, so the session reads as ended.
            done_path.parent.mkdir(parents=True, exist_ok=True)
            done_path.touch()
