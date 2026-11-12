"""Logging for the test clients (integration/*/main.py, e2e/main.py): everything goes to a file.

Each run writes its own file, <log dir>/<name>-<YYYYmmdd-HHMMSS>.log, where the
log dir is --test-log-dir, else $TEST_LOG_DIR, else logs/tests at the repo root
(/logs/tests in the test containers, which mount the repo's logs/tests there).
The console gets one line naming the file; the exit code says whether the test passed.

Only the standard library is used.
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import NoReturn, TextIO

FORMAT = "%(asctime)s %(levelname)s %(message)s"
# The repo's logs/tests: tests/integration/testlog.py -> <repo>/logs/tests (/logs/tests in the containers).
DEFAULT_DIR = Path(__file__).resolve().parent.parent.parent / "logs" / "tests"

log = logging.getLogger("tests")
_stream: TextIO | None = None


def add_args(parser: argparse.ArgumentParser) -> None:
    default = os.environ.get("TEST_LOG_DIR") or str(DEFAULT_DIR)
    parser.add_argument("--test-log-dir", type=Path, default=Path(default),
                        help=f"folder for this run's log file (default: $TEST_LOG_DIR or {DEFAULT_DIR})")


def setup(name: str, log_dir: Path) -> Path:
    """Sends the "tests" logger, and uncaught exceptions, to a new file in log_dir; returns its path."""
    global _stream
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}.log"
    # Append mode, so a subprocess writing to the same file (stream()) never overwrites lines.
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(FORMAT))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    log.propagate = False
    _stream = handler.stream

    def excepthook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            log.error("interrupted")
        else:
            log.critical("uncaught exception", exc_info=(exc_type, exc, tb))
    sys.excepthook = excepthook

    print(f"{name}: logging to {path}", file=sys.stderr)
    log.info("%s", " ".join(sys.argv))
    return path


def stream() -> TextIO | None:
    """The open log file, for a subprocess's stderr (e.g. ffmpeg's warnings); None before setup()."""
    if _stream is not None:
        _stream.flush()
    return _stream


def fail(message: str) -> NoReturn:
    """Logs message as an error and exits 1."""
    log.error(message)
    sys.exit(1)
