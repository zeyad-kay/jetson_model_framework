from pathlib import Path

from common.io import write_json
from common.service import WorkerService
from llm_extraction_service.config import Config
from common.log import get_logger
from llm_extraction_service.model import QwenExtractor

log = get_logger(__name__)


class LLMExtractionService(WorkerService):
    """Extracts each session's ePCR entities from its transcript as it is recorded.

    Idles until queue_session() names a session, then follows the segments the ASR
    service writes to <session>/transcript/segments.json, extracting each
    segment as it appears into <session>/structured/extractions.json. Once the
    session ends (end_session(), or queue_session() of the next session) and the last
    segment is extracted, it creates structured/extractions.done, which tells the
    aggregation service (llm_aggregation_service/) that extractions.json is final.
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
        extractions_path = session_dir / "structured" / "extractions.json"
        done_path = session_dir / "structured" / "extractions.done"
        try:
            log.info("extracting %s, writing %s", segments_path, extractions_path)
            if not self._wait_for_file(segments_path):
                return

            records = []
            for record in self._model.process(str(segments_path)):
                records.append(record)
                write_json(extractions_path, records)
                log.debug("extracted segment %s", record["segment"])
            log.info("extracted %d segments (%d unparseable replies)", len(records), self._model.parse_failures)
        finally:
            # Even when nothing was extracted or extraction failed, so the aggregator stops waiting.
            done_path.parent.mkdir(parents=True, exist_ok=True)
            done_path.touch()
