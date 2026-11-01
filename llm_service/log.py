from pathlib import Path
from typing import Optional, Union

from common import log as _log

ROOT_NAME = "llm_service"


def setup_logging(level: Union[int, str] = "INFO") -> None:
    """Configures the llm_service logger. Call once at startup."""
    _log.setup_logging(ROOT_NAME, level)


def get_logger(name: str):
    """Returns a child logger, e.g. get_logger(__name__) -> llm_service.<module>."""
    return _log.get_logger(ROOT_NAME, name)
