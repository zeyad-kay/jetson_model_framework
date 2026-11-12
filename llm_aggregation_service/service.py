from pathlib import Path

from llm_aggregation_service.config import Config
from llm_aggregation_service.model import QwenAggregator
from common.io import write_json
from common.log import get_logger
from common.service import WorkerService

log = get_logger(__name__)


class LLMAggregationService(WorkerService):
    """Keeps each session's ePCR aggregate up to date as its segments are extracted.

    Idles until queue_session() names a session, then follows the extraction
    records the extraction service writes to <session>/structured/extractions.json.
    Each time records are added, all the records so far are aggregated into
    <session>/structured/aggregate.json, and a copy is kept as
    structured/aggregates/<segment>.json, named after the last segment it covers.
    Once the session ends (end_session(), or queue_session() of the next session),
    it waits for the extraction service to create structured/extractions.done, aggregates
    whatever is new, and finishes. shutdown() ends the current session and makes
    run() return.
    """

    def __init__(self, config: Config):
        super().__init__(config.sessions_dir, config.poll_interval, log)
        self.config = config

    def _load_model(self) -> QwenAggregator:
        cfg = self.config
        model = QwenAggregator(
            model_id=cfg.model,
            prompt_path=cfg.prompt_path,
            poll_interval=cfg.poll_interval,
            drain_timeout=cfg.drain_timeout,
        )
        log.info("model ready: %s (%s)", type(model).__name__, model.model_id)
        return model

    def _process_session(self, session_dir: Path) -> None:
        structured = session_dir / "structured"
        extractions_path = structured / "extractions.json"
        done_path = structured / "extractions.done"  # created by llm_extraction_service/service.py
        aggregate_path = structured / "aggregate.json"
        log.info("aggregating %s, writing %s", extractions_path, aggregate_path)

        count = 0
        for records, aggregate in self._model.process(str(extractions_path), str(done_path)):
            segment = records[-1].get("segment")
            write_json(structured / "aggregates" / f"{segment}.json", aggregate)
            write_json(aggregate_path, aggregate)
            count += 1
            log.info("aggregated %d records, up to segment %s", len(records), segment)
        if self._model.timed_out:
            log.warning("%s was never created; gave up after %gs without new records",
                        done_path, self.config.drain_timeout)
        log.info("wrote %d aggregates", count)
