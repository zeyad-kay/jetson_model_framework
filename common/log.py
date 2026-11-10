import logging
from pathlib import Path
from typing import Optional, Union

FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

_configured = set()


def setup_logging(
    root_name: str,
    level: Union[int, str] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    also_stderr: bool = False,
) -> None:
    """Configures the logger named root_name (a service's top-level package, e.g. "asr_service"). Call once at startup.

    Logs go to log_file (overwritten at each start) if given, else to stderr.
    With also_stderr they go to stderr as well. Repeated calls for the same
    root_name do nothing, so handlers are never duplicated.
    """
    if root_name in _configured:
        return
    root = logging.getLogger(root_name)
    root.setLevel(level)
    root.propagate = False  # don't double-log through the global root logger

    handlers = []
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, mode="w", encoding="utf-8"))
    if log_file is None or also_stderr:
        handlers.append(logging.StreamHandler())
    formatter = logging.Formatter(FORMAT)
    for handler in handlers:
        handler.setFormatter(formatter)
        root.addHandler(handler)
    _configured.add(root_name)


def get_logger(name: str) -> logging.Logger:
    """Returns the logger for module `name`; pass __name__, e.g. get_logger(__name__) -> asr_service.service.

    A module's logger is a child of its top-level package's logger, which
    setup_logging() configures. A module run with `python -m` is named
    "__main__", so name it explicitly instead, e.g. get_logger("asr_service.main").
    """
    return logging.getLogger(name)
