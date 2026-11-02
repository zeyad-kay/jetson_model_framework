from pathlib import Path
from typing import Optional, Union

from common import log as _log

ROOT_NAME = "video_service"


def setup_logging(level: Union[int, str] = "INFO", log_file: Optional[Union[str, Path]] = None) -> None:
    """Configures the video_service logger. Call once at startup."""
    _log.setup_logging(ROOT_NAME, level, log_file, also_stderr=True)


def get_logger(name: str):
    """Returns a child logger, e.g. get_logger(__name__) -> video_service.<module>."""
    return _log.get_logger(ROOT_NAME, name)
