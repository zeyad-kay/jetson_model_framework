"""WorkerService (common/service.py): finding sessions on disk, the open_reader / setup / poll /
predict / save / cleanup steps, output markers, giving up, shutdown and the heartbeat."""

import json
import logging
import threading
import time

import pytest

from conftest import make_session, wait_until
from common.model import StreamingModel
from common.service import WorkerService

log = logging.getLogger("unit.worker_service")


class FakeReader:
    """Reads the service's inputs for one session: {"value": item}, or None while there are none.
    Raises EOFError once they are all read and <session>/input.done exists."""

    def __init__(self, service, session_dir):
        self.service = service
        self.session_dir = session_dir

    def read(self):
        service = self.service
        service.reads += 1
        complete = (self.session_dir / "input.done").exists()  # before reading, as real readers do
        with service.lock:
            inputs = service.inputs.setdefault(self.session_dir.name, [])
            if inputs:
                item = inputs.pop(0)
                service.calls.append(("read", item))
                return {"value": item}
        if complete:
            raise EOFError("input.done")
        return None

    def close(self):
        self.service._step("close")


class FakeService(WorkerService):
    """Takes each session's inputs from a list the test appends to; records every step it runs.

    Its output marker is <session>/fake.done. fail_in names a step that raises.
    """

    def __init__(self, sessions_dir, fail_in=None, **kwargs):
        kwargs.setdefault("scan_interval", 0.01)
        super().__init__("fake", sessions_dir, 0.01, log, **kwargs)
        self.fail_in = fail_in
        self.inputs = {}  # session id -> its inputs not read yet
        self.calls = []  # (step, argument)
        self.reads = 0
        self.lock = threading.Lock()

    def _step(self, name, arg=None):
        self.calls.append((name, arg))
        if self.fail_in == name:
            raise RuntimeError(f"{name} failed")

    def feed(self, session, *items, complete=False):
        with self.lock:
            self.inputs.setdefault(session.name, []).extend(items)
        if complete:
            (session / "input.done").touch()

    def load_model(self):
        self.calls.append(("load_model", None))
        return StreamingModel()

    def output_path(self, session_dir):
        return session_dir / "fake"

    def open_reader(self, session_dir):
        self.session_dir = session_dir
        self._step("open_reader", session_dir.name)
        return FakeReader(self, session_dir)

    def setup(self, session_dir):
        self._step("setup", session_dir.name)

    def predict(self, inputs):
        self._step("predict", inputs["value"])
        return inputs["value"] * 10

    def save(self, inputs, outputs):
        self._step("save", (inputs["value"], outputs))

    def cleanup(self):
        self.calls.append(("cleanup", None))
        if self.fail_in == "cleanup":
            raise RuntimeError("cleanup failed")
        super().cleanup()  # the base writes fake.done

    def steps(self):
        """The calls, without the reads."""
        return [c for c in self.calls if c[0] != "read"]

    def predicted(self):
        return [arg for name, arg in self.calls if name == "predict"]

    def sessions_run(self):
        return [arg for name, arg in self.calls if name == "open_reader"]


def marker(session):
    path = session / "fake.done"
    return json.loads(path.read_text()) if path.exists() else None


def finished(caplog, name):
    return f"session {name} finished" in caplog.text


@pytest.fixture(autouse=True)
def logs(caplog):
    caplog.set_level(logging.INFO, logger=log.name)


def test_runs_every_step_in_order_and_marks_the_session_done(tmp_path, run_service, caplog):
    service = FakeService(tmp_path)
    run_service(service)
    s1 = make_session(tmp_path, "s1")
    service.feed(s1, 1, 2)
    assert wait_until(lambda: ("save", (2, 20)) in service.calls)
    service.feed(s1, 3, complete=True)
    assert wait_until(lambda: finished(caplog, "s1"))
    assert service.steps() == [
        ("load_model", None),
        ("open_reader", "s1"),
        ("setup", "s1"),
        ("predict", 1), ("save", (1, 10)),
        ("predict", 2), ("save", (2, 20)),
        ("predict", 3), ("save", (3, 30)),
        ("close", None),
        ("cleanup", None),
    ]
    assert marker(s1) == {"status": "ok"}
    assert "input.done" in caplog.text


def test_everything_recorded_before_the_input_is_complete_is_processed(tmp_path, run_service, caplog):
    service = FakeService(tmp_path)
    s1 = make_session(tmp_path, "s1")
    service.feed(s1, *range(1, 6), complete=True)  # complete before the service even starts
    run_service(service)
    assert wait_until(lambda: finished(caplog, "s1"))
    assert service.predicted() == [1, 2, 3, 4, 5]


def test_picks_up_the_pending_sessions_for_it_oldest_first(tmp_path, run_service, caplog):
    service = FakeService(tmp_path, max_session_age=3600)
    sessions = {
        "newer": make_session(tmp_path, "newer", age=10),
        "older": make_session(tmp_path, "older", age=100, services=["fake", "other"]),
        "done": make_session(tmp_path, "done", age=50),  # already has its output marker
        "expired": make_session(tmp_path, "expired", age=7200),  # older than max_session_age
        "not_for_it": make_session(tmp_path, "not_for_it", age=20, services=["other"]),
    }
    (sessions["done"] / "fake.done").write_text("{}")
    (tmp_path / "no_session_json" / "audio").mkdir(parents=True)  # e.g. from before session.json
    (tmp_path / "stray_file").write_text("")
    for session in sessions.values():
        service.feed(session, complete=True)
    run_service(service)
    assert wait_until(lambda: finished(caplog, "newer"))
    time.sleep(0.1)
    assert service.sessions_run() == ["older", "newer"]


def test_a_session_created_while_it_works_is_picked_up_next(tmp_path, run_service, caplog):
    service = FakeService(tmp_path)
    run_service(service)
    s1 = make_session(tmp_path, "s1")
    service.feed(s1, 1)
    assert wait_until(lambda: service.predicted() == [1])
    s2 = make_session(tmp_path, "s2")
    service.feed(s2, 2, complete=True)
    time.sleep(0.1)
    assert service.sessions_run() == ["s1"]  # s1 isn't complete yet: s2 waits
    service.feed(s1, complete=True)
    assert wait_until(lambda: finished(caplog, "s2"))
    assert service.sessions_run() == ["s1", "s2"]
    assert marker(s1) == marker(s2) == {"status": "ok"}


@pytest.mark.parametrize("step", ["open_reader", "setup", "predict", "save", "close"])
def test_a_failing_step_ends_the_session_marked_failed(tmp_path, run_service, caplog, step):
    service = FakeService(tmp_path, fail_in=step)
    s1 = make_session(tmp_path, "s1", age=10)
    service.feed(s1, 1, 2, complete=True)
    run_service(service)
    assert wait_until(lambda: finished(caplog, "s1"))
    assert "session s1 failed" in caplog.text
    assert service.steps()[-1] == ("cleanup", None)
    if step != "open_reader":
        assert service.steps()[-2] == ("close", None)  # the reader is closed before cleanup
    if step != "close":
        assert ("predict", 2) not in service.calls  # nothing after the failure
        assert marker(s1) == {"status": "failed"}
    else:
        assert "session s1 failed to close its reader" in caplog.text
        assert marker(s1) == {"status": "ok"}  # the outputs were complete

    # Not retried; the service goes on with the next session.
    service.fail_in = None
    s2 = make_session(tmp_path, "s2")
    service.feed(s2, 7, complete=True)
    assert wait_until(lambda: finished(caplog, "s2"))
    assert service.sessions_run() == ["s1", "s2"]
    assert marker(s2) == {"status": "ok"}


def test_a_session_is_tried_once_per_run_even_if_it_is_left_unmarked(tmp_path, run_service, caplog):
    service = FakeService(tmp_path, fail_in="cleanup")
    s1 = make_session(tmp_path, "s1")
    service.feed(s1, complete=True)
    run_service(service)
    assert wait_until(lambda: "session s1 failed to clean up" in caplog.text)
    assert wait_until(lambda: finished(caplog, "s1"))
    time.sleep(0.1)
    assert service.sessions_run() == ["s1"]  # no marker, but not retried in a loop
    assert marker(s1) is None


def test_shutdown_leaves_an_unfinished_session_unmarked_and_it_is_redone(tmp_path, run_service, caplog):
    first = FakeService(tmp_path)
    runner = run_service(first)
    s1 = make_session(tmp_path, "s1")
    first.feed(s1, 1, 2)
    assert wait_until(lambda: first.predicted() == [1, 2])
    first.feed(s1, 3)  # recorded before the shutdown: still processed
    runner.stop()
    assert first.predicted() == [1, 2, 3]
    assert first.steps()[-1] == ("cleanup", None)
    assert marker(s1) is None
    assert "session s1 interrupted by shutdown" in caplog.text

    # The next run of the service redoes the session from the start.
    second = FakeService(tmp_path)
    second.feed(s1, 1, 2, 3, complete=True)
    run_service(second)
    assert wait_until(lambda: marker(s1) == {"status": "ok"})
    assert second.predicted() == [1, 2, 3]


def test_shutdown_after_the_input_is_complete_still_marks_the_session(tmp_path, run_service):
    service = FakeService(tmp_path)
    s1 = make_session(tmp_path, "s1")
    hold = threading.Event()
    save = service.save

    def slow_save(inputs, outputs):
        hold.wait(5)
        save(inputs, outputs)

    service.save = slow_save
    service.feed(s1, 1, complete=True)
    runner = run_service(service)
    assert wait_until(lambda: service.predicted() == [1])
    service.shutdown()
    hold.set()
    runner.stop()
    # The reader finds input.done once 1 is saved: the session ended with its input complete.
    assert marker(s1) == {"status": "ok"}


def test_a_session_whose_input_never_completes_is_given_up_on(tmp_path, run_service, caplog):
    service = FakeService(tmp_path, abandon_timeout=0.3)
    run_service(service)
    s1 = make_session(tmp_path, "s1")
    service.feed(s1, 1)  # and then nothing, and no input.done: its writer died
    assert wait_until(lambda: service.predicted() == [1])
    fed = time.monotonic()
    assert wait_until(lambda: finished(caplog, "s1"))
    assert time.monotonic() - fed >= 0.3
    assert "giving up on session s1" in caplog.text
    assert service.steps()[-2:] == [("close", None), ("cleanup", None)]
    assert marker(s1) == {"status": "abandoned"}

    # The service goes on with the next session.
    s2 = make_session(tmp_path, "s2")
    service.feed(s2, 2, complete=True)
    assert wait_until(lambda: marker(s2) == {"status": "ok"})


def test_new_inputs_or_changes_in_the_session_keep_it_going(tmp_path, run_service, caplog):
    service = FakeService(tmp_path, abandon_timeout=0.6)  # several times the 0.1 s between activities
    run_service(service)
    s1 = make_session(tmp_path, "s1")
    for value in range(10):  # inputs, 0.1 s apart
        service.feed(s1, value)
        time.sleep(0.1)
    recording = s1 / "audio" / "audio.pcm"
    for _ in range(10):  # no inputs, but its upstream's input still grows (e.g. a long silence)
        with recording.open("ab") as f:
            f.write(b"\0" * 100)
        time.sleep(0.1)
    assert not finished(caplog, "s1")
    assert wait_until(lambda: marker(s1) == {"status": "abandoned"})
    assert service.predicted() == list(range(10))


def test_shutdown_while_idle_returns_promptly(tmp_path, run_service):
    service = FakeService(tmp_path, scan_interval=30.0)
    runner = run_service(service)
    assert wait_until(lambda: ("load_model", None) in service.calls)
    started = time.monotonic()
    runner.stop()
    assert time.monotonic() - started < 5


def test_the_heartbeat_says_what_the_service_is_doing(tmp_path, run_service):
    status_path = tmp_path / "run" / "fake.json"
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    hold = threading.Event()
    service = FakeService(sessions, status_path=status_path, heartbeat_interval=0.05)
    load_model = service.load_model
    service.load_model = lambda: (hold.wait(5), load_model())[1]

    def status():
        try:
            return json.loads(status_path.read_text())
        except (OSError, ValueError):
            return None

    runner = run_service(service)
    assert wait_until(lambda: (status() or {}).get("state") == "loading")
    first = status()["updated"]
    assert wait_until(lambda: status()["updated"] > first)  # refreshed while loading
    assert status()["service"] == "fake" and status()["pid"] > 0
    hold.set()
    assert wait_until(lambda: status()["state"] == "idle" and status()["session"] is None)
    s1 = make_session(sessions, "s1")
    assert wait_until(lambda: status()["state"] == "working" and status()["session"] == "s1")
    service.feed(s1, complete=True)
    assert wait_until(lambda: status()["state"] == "idle")
    runner.stop()
    assert status()["state"] == "stopped"
