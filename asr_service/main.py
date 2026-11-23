from asr_service.config import Config
from common.log import get_logger, setup_logging
from asr_service.service import ASRService
from common.service import run_service

log = get_logger("asr_service.main")


def main() -> None:
    config = Config.load()
    setup_logging("asr_service", level=config.log_level, log_file=config.log_file, also_stderr=True)
    log.info("starting asr_service")
    run_service(ASRService(config))
    log.info("asr_service stopped")


if __name__ == "__main__":
    main()
