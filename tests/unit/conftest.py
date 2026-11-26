"""Unit tests: the services' logic without models or GPUs.

Run from the repo root on any machine with pytest, numpy and Pillow (the video tests skip
without them, and test_frame_reader.py without a full ffmpeg; see its docstring):

    python3 -m pytest tests/unit
"""

import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))  # common/, llm_extraction_service/, ...


def make_session(sessions_dir, name, services=None, age=None, subdirs=("audio", "transcript", "video", "extractions")):
    """A session folder as the server creates it: its subfolders, then session.json naming `services`
    (None: no list, every service). age: seconds since it was created (session.json's mtime)."""
    import json
    import os

    path = Path(sessions_dir) / name
    for sub in subdirs:
        (path / sub).mkdir(parents=True, exist_ok=True)
    path.mkdir(parents=True, exist_ok=True)
    info = {"id": name, "created": "2026-11-20T00:00:00+00:00"}
    if services is not None:
        info["services"] = list(services)
    (path / "session.json").write_text(json.dumps(info))
    if age is not None:
        then = time.time() - age
        os.utime(path / "session.json", (then, then))
    return path


def wait_until(condition, timeout: float = 5.0, interval: float = 0.01) -> bool:
    """Polls condition() until it is true; False if timeout seconds pass first."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return condition()


class Runner:
    """Runs a service's run() on a background thread, and shuts it down at the end of a test."""

    def __init__(self, service):
        self.service = service
        self.thread = threading.Thread(target=service.run, daemon=True)

    def start(self) -> "Runner":
        self.thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        self.service.shutdown()
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "run() did not return after shutdown()"


@pytest.fixture
def run_service():
    """run_service(service) starts service.run() on a thread; it is shut down after the test."""
    runners = []

    def start(service) -> Runner:
        runner = Runner(service).start()
        runners.append(runner)
        return runner

    yield start
    for runner in runners:
        if runner.thread.is_alive():
            runner.stop()
