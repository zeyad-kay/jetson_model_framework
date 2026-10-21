import logging
import os
from pathlib import Path
from typing import Optional, Union

ROOT_NAME = "server"
FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

_configured = False


def setup_logging(
    level: Optional[Union[int, str]] = None,
    log_file: Optional[Union[str, Path]] = None,
) -> None:
    """Configures the server logger. Call once at startup.

    level defaults to the SERVER_LOG_LEVEL environment variable, then INFO.
    Logs go to stderr, and also to log_file if given. Repeated calls do
    nothing, so handlers are never duplicated.
    """
    global _configured
    if _configured:
        return
    if level is None:
        level = os.environ.get("SERVER_LOG_LEVEL", "INFO")

    root = logging.getLogger(ROOT_NAME)
    root.setLevel(level)
    root.propagate = False  # don't double-log through the global root logger

    formatter = logging.Formatter(FORMAT)
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Returns a child logger, e.g. get_logger(__name__) -> server.main."""
    if name != ROOT_NAME and not name.startswith(ROOT_NAME + "."):
        name = f"{ROOT_NAME}.{name}"
    return logging.getLogger(name)
