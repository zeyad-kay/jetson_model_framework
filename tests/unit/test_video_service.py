"""VideoService (video_service/service.py) with a fake model and a fake FrameReader: saving the
frames with a detection, draining at the end of the session, ending when the video stops."""

import json
import logging
import threading

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("PIL")  # write_jpeg

from conftest import make_session, wait_until  # noqa: E402
from video_service import service as video_service_module  # noqa: E402
from video_service.config import Config  # noqa: E402
from video_service.service import VideoService  # noqa: E402

FPS = 10.0


class FakeReader:
    """Stands in for FrameReader: hands out the frames the test feeds it, never blocking."""

    instances = []

    def __init__(self, path, probe_timeout=60.0):
        self.path = path
        self.probe_timeout = probe_timeout
        self.width, self.height, self.fps = 4, 2, FPS
        self.frames = []
        self.num_read = 0
        self.broken = False  # read() raises EOFError once the frames fed are read
        self.complete = False  # as if video.ts.done existed: likewise, EOFError after the frames
        self.closed = False
        self.lock = threading.Lock()
        FakeReader.instances.append(self)

    def feed(self, *values):
        """Frames filled with these values."""
        with self.lock:
            self.frames.extend(np.full((self.height, self.width, 3), v, np.uint8) for v in values)

    def read(self):
        with self.lock:
            if self.frames:
                seconds = self.num_read / self.fps
                self.num_read += 1
                return {"frame": self.frames.pop(0), "seconds": seconds}
        if self.broken:
            raise EOFError("the decoding stopped")
        if self.complete:
            raise EOFError("video.ts is complete")
        return None

    def close(self):
        self.closed = True


class FakeSam3:
    """Stands in for Sam3: detects one object on frames whose value is odd, scoring value/100."""

    prompt = "wound"

    def __init__(self, fail_on=None):
        self.fail_on = fail_on  # frame value whose predict() raises
        self.calls = []

    def start(self):
        self.calls.append("start")

    def predict(self, inputs):
        value = int(inputs["frame"][0, 0, 0])
        self.calls.append(("predict", value, inputs["seconds"]))
        if value == self.fail_on:
            raise RuntimeError("model failed")
        if value % 2:
            return {"object_ids": [1], "scores": [value / 100], "boxes": [[0.4, 1.0, 3.6, 2.0]]}
        return {"object_ids": [], "scores": [], "boxes": []}

    def stop(self):
        self.calls.append("stop")

    def predicted(self):
        return [c[1] for c in self.calls if isinstance(c, tuple)]


@pytest.fixture(autouse=True)
def fake_reader(monkeypatch):
    FakeReader.instances = []
    monkeypatch.setattr(video_service_module, "FrameReader", FakeReader)


@pytest.fixture
def session(tmp_path):
    return make_session(tmp_path, "s1", age=10)


def make_service(sessions_dir, model, min_score=0.5) -> VideoService:
    config = Config(sessions_dir=str(sessions_dir), poll_interval=0.01, status_path=None,
                    probe_timeout=12.5, min_score=min_score)
    service = VideoService(config)
    service.load_model = lambda: model
    service._scan_interval = 0.01
    return service


def start_session(service, run_service) -> FakeReader:
    """Runs the service and returns the reader of the session it picks up."""
    count = len(FakeReader.instances)
    run_service(service)
    assert wait_until(lambda: len(FakeReader.instances) > count)
    return FakeReader.instances[-1]


def marker(session_dir):
    path = session_dir / "video" / "wounds.done"
    return json.loads(path.read_text()) if path.exists() else None


@pytest.fixture(autouse=True)
def logs(caplog):
    caplog.set_level(logging.INFO, logger=video_service_module.log.name)


def test_saves_the_frames_with_a_detection_scoring_at_least_min_score(session, run_service, caplog):
    model = FakeSam3()
    service = make_service(session.parent, model, min_score=0.5)
    reader = start_session(service, run_service)
    assert reader.path == session / "video" / "video.ts"
    assert reader.probe_timeout == 12.5
    reader.feed(0, 51, 2, 49, 53)  # 51 and 53 score >= 0.5; 49 is below it
    reader.complete = True
    assert wait_until(lambda: "session s1 finished" in caplog.text)

    assert model.calls[0] == "start" and model.calls[-1] == "stop"
    assert [(v, round(s, 3)) for v, s in [c[1:] for c in model.calls[1:-1]]] == [
        (0, 0.0), (51, 0.1), (2, 0.2), (49, 0.3), (53, 0.4)]
    wounds = session / "video" / "wounds"
    assert sorted(p.name for p in wounds.iterdir()) == [
        "000000.100.box", "000000.100.jpg", "000000.400.box", "000000.400.jpg"]
    assert marker(session) == {"status": "ok"}
    assert (wounds / "000000.100.box").read_text() == "0 1 4 2\n"
    assert (wounds / "000000.100.jpg").read_bytes()[:2] == b"\xff\xd8"
    assert reader.closed
    assert "segmented 5 frames, saved 2 scoring >= 0.50; best score 0.530 at 0.400s" in caplog.text


def test_logs_the_best_score_even_when_no_frame_is_saved(session, run_service, caplog):
    service = make_service(session.parent, FakeSam3(), min_score=0.9)
    reader = start_session(service, run_service)
    reader.feed(0, 61, 2, 73)  # detections scoring 0.61 and 0.73, both below 0.9
    reader.complete = True
    assert wait_until(lambda: "session s1 finished" in caplog.text)

    assert not (session / "video" / "wounds").exists()
    assert "segmented 4 frames, saved 0 scoring >= 0.90; best score 0.730 at 0.300s" in caplog.text


def test_logs_when_nothing_is_detected(session, run_service, caplog):
    service = make_service(session.parent, FakeSam3())
    reader = start_session(service, run_service)
    reader.feed(0, 2)
    reader.complete = True
    assert wait_until(lambda: "session s1 finished" in caplog.text)

    assert "segmented 2 frames, saved 0 scoring >= 0.50; no object detected" in caplog.text


def test_keeps_following_the_video_until_it_is_complete(session, run_service, caplog):
    model = FakeSam3()
    service = make_service(session.parent, model)
    reader = start_session(service, run_service)
    reader.feed(*range(5))
    assert wait_until(lambda: model.predicted() == list(range(5)))
    assert not wait_until(lambda: "session s1 finished" in caplog.text, timeout=0.2)
    reader.feed(*range(5, 10))
    reader.complete = True
    assert wait_until(lambda: model.calls[-1:] == ["stop"])
    assert model.predicted() == list(range(10))


def test_a_broken_stream_ends_the_session(session, run_service, caplog):
    model = FakeSam3()
    service = make_service(session.parent, model)
    reader = start_session(service, run_service)
    reader.feed(1, 2)
    reader.broken = True
    assert wait_until(lambda: "session s1 finished" in caplog.text)
    assert model.predicted() == [1, 2]
    assert "the decoding stopped" in caplog.text
    assert marker(session) == {"status": "ok"}


def test_a_failing_model_still_stops_the_decoder_and_the_model(session, run_service, caplog):
    model = FakeSam3(fail_on=2)
    service = make_service(session.parent, model)
    reader = start_session(service, run_service)
    reader.feed(1, 2, 3)
    assert wait_until(lambda: "session s1 finished" in caplog.text)
    assert "session s1 failed" in caplog.text
    assert model.predicted() == [1, 2] and model.calls[-1] == "stop"
    assert reader.closed
    assert marker(session) == {"status": "failed"}


def test_shutdown_before_the_video_is_complete_leaves_the_session_to_redo(session, run_service, caplog):
    model = FakeSam3()
    service = make_service(session.parent, model)
    reader = start_session(service, run_service)
    reader.feed(1)
    assert wait_until(lambda: model.predicted() == [1])
    service.shutdown()
    assert wait_until(lambda: "session s1 finished" in caplog.text)
    assert model.calls[-1] == "stop" and reader.closed
    assert marker(session) is None


def test_each_session_starts_from_the_beginning_of_its_video(tmp_path, run_service, caplog):
    s1 = make_session(tmp_path, "s1", age=20)
    s2 = make_session(tmp_path, "s2", age=10)
    model = FakeSam3()
    service = make_service(tmp_path, model, min_score=0.0)
    first = start_session(service, run_service)
    assert first.path == s1 / "video" / "video.ts"
    first.feed(1, 3)
    first.complete = True
    assert wait_until(lambda: len(FakeReader.instances) == 2)
    second = FakeReader.instances[1]
    assert second.path == s2 / "video" / "video.ts"
    second.feed(5)
    second.complete = True
    assert wait_until(lambda: "session s2 finished" in caplog.text)
    assert model.calls.count("start") == 2 and model.calls.count("stop") == 2
    assert model.calls[-2] == ("predict", 5, 0.0)  # seconds restart at 0
    assert sorted(p.name for p in (s2 / "video" / "wounds").glob("*.jpg")) == ["000000.000.jpg"]
