from asr_service.config import Config
from asr_service.log import get_logger, setup_logging
from asr_service.service import ASRService
from common.service import run_service

log = get_logger(__name__)


def main() -> None:
    config = Config.load()
    setup_logging(level=config.log_level, log_file=config.log_file)
    log.info("starting asr_service")
    run_service(ASRService(config), config.bridge_path, log)
    log.info("asr_service stopped")


if __name__ == "__main__":
    main()
