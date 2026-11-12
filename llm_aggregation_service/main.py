from llm_aggregation_service.config import Config
from llm_aggregation_service.service import LLMAggregationService
from common.log import get_logger, setup_logging
from common.service import run_service

log = get_logger("llm_aggregation_service.main")


def main() -> None:
    config = Config.load()
    setup_logging("llm_aggregation_service", level=config.log_level, log_file=config.log_file, also_stderr=True)
    log.info("starting llm_aggregation_service")
    run_service(LLMAggregationService(config), config.bridge_path, log)
    log.info("llm_aggregation_service stopped")


if __name__ == "__main__":
    main()
