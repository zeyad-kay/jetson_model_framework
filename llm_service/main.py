from common.service import run_service
from llm_service.config import Config
from llm_service.log import get_logger, setup_logging
from llm_service.service import LLMService

log = get_logger(__name__)


def main() -> None:
    config = Config.load()
    setup_logging(level=config.log_level)
    log.info("starting llm_service")
    run_service(LLMService(config), config.bridge_path, log)
    log.info("llm_service stopped")


if __name__ == "__main__":
    main()
