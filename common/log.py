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
    """Configures the logger named root_name (a service's top-level package). Call once at startup.

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


def get_logger(root_name: str, name: str) -> logging.Logger:
    """Returns a child of the root_name logger, e.g. get_logger("server", __name__) -> server.main."""
    if name != root_name and not name.startswith(root_name + "."):
        name = f"{root_name}.{name}"
    return logging.getLogger(name)
