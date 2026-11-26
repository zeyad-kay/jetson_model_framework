from common.service import run_service
from llm_extraction_service.config import Config
from common.log import get_logger, setup_logging
from llm_extraction_service.service import LLMExtractionService

log = get_logger("llm_extraction_service.main")


def main() -> None:
    config = Config.load()
    setup_logging("llm_extraction_service", level=config.log_level, log_file=config.log_file, also_stderr=True)
    log.info("starting llm_extraction_service")
    run_service(LLMExtractionService(config))
    log.info("llm_extraction_service stopped")


if __name__ == "__main__":
    main()
