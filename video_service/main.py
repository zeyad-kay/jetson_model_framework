from common.service import run_service
from video_service.config import Config
from common.log import get_logger, setup_logging
from video_service.service import VideoService

log = get_logger("video_service.main")


def main() -> None:
    config = Config.load()
    setup_logging("video_service", level=config.log_level, log_file=config.log_file, also_stderr=True)
    log.info("starting video_service")
    run_service(VideoService(config), config.bridge_path, log)
    log.info("video_service stopped")


if __name__ == "__main__":
    main()
