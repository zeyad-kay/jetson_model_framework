from common.service import run_service
from llm_service.config import Config
from common.log import get_logger, setup_logging
from llm_service.service import LLMService

log = get_logger("llm_service.main")


def main() -> None:
    config = Config.load()
    setup_logging("llm_service", level=config.log_level, log_file=config.log_file, also_stderr=True)
    log.info("starting llm_service")
    run_service(LLMService(config), config.bridge_path, log)
    log.info("llm_service stopped")


if __name__ == "__main__":
    main()
