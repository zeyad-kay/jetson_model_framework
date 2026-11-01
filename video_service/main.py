from common.service import run_service
from video_service.config import Config
from video_service.log import get_logger, setup_logging
from video_service.service import VideoService

log = get_logger(__name__)


def main() -> None:
    config = Config.load()
    setup_logging(level=config.log_level)
    log.info("starting video_service")
    run_service(VideoService(config), config.bridge_path, log)
    log.info("video_service stopped")


if __name__ == "__main__":
    main()
