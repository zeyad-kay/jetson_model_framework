"""The server's side of sessions: Session (server/sessions.py) writes session.json and marks its
recordings complete; WorkerServiceManager (server/services.py) reads the services' heartbeats."""

import json
import logging
import time

import pytest

from common.io import done_marker
from server import sessions as server_sessions
from server.services import WorkerServiceClient, WorkerServiceManager
from server.config import Config
from server.coordinator import Coordinator
from server.sessions import Session, abort_unfinished


def read(path):
    return json.loads(path.read_text())


# --- Session ----------------------------------------------------------------------------

events = []


class FakeListener:
    """Stands in for AudioListener / VideoListener: records when it stops."""

    def __init__(self, url, client, session_id, output_path, **_):
        self.output_path = output_path

    def start(self):
        pass

    def stop(self):
        # The recorder's last write: the marker must not exist yet.
        events.append(("stop", self.output_path.name, done_marker(self.output_path).exists()))


class FakeMedia:
    client = None

    def read_url(self, session_id):
        return f"rtsp://media/{session_id}"


@pytest.fixture(autouse=True)
def fake_listeners(monkeypatch):
    events.clear()
    monkeypatch.setattr(server_sessions, "AudioListener", FakeListener)
    monkeypatch.setattr(server_sessions, "VideoListener", FakeListener)


def test_a_new_session_has_session_json_naming_its_services(tmp_path):
    session = Session.create(tmp_path, FakeMedia(), 16000, services=["asr_service", "video_service"])
    info = read(session.path / "session.json")
    assert info["id"] == session.id and info["created"]
    assert info["services"] == ["asr_service", "video_service"]
    assert all((session.path / sub).is_dir() for sub in ("audio", "transcript", "video", "extractions"))
    assert "services" not in read(Session.create(tmp_path, FakeMedia(), 16000).path / "session.json")


def test_stop_marks_both_recordings_complete_after_the_recorders_stop(tmp_path):
    session = Session.create(tmp_path, FakeMedia(), 16000)
    session.start()
    session.stop()
    assert events == [("stop", "audio.pcm", False), ("stop", "video.ts", False)]
    # Even though nothing was recorded: the services stop waiting for the files.
    assert read(done_marker(session.audio_pcm)) == {"status": "ok"}
    assert read(done_marker(session.video_ts)) == {"status": "ok"}
    session.stop()  # twice is fine


def test_stop_marks_the_recordings_complete_even_if_a_recorder_fails(tmp_path, monkeypatch):
    session = Session.create(tmp_path, FakeMedia(), 16000)

    def fail():
        raise RuntimeError("stuck")

    monkeypatch.setattr(session._audio, "stop", fail)
    with pytest.raises(RuntimeError):
        session.stop()
    assert done_marker(session.audio_pcm).exists() and done_marker(session.video_ts).exists()


def test_abort_unfinished_marks_the_recordings_a_previous_run_left_unfinished(tmp_path, caplog):
    caplog.set_level(logging.WARNING, logger="server.sessions")
    cut_off = Session.create(tmp_path, FakeMedia(), 16000)  # the server died while recording it
    cut_off.audio_pcm.write_bytes(b"\0" * 100)
    half = Session.create(tmp_path, FakeMedia(), 16000)  # died between the two markers
    from common.io import write_done
    write_done(half.audio_pcm)
    stopped = Session.create(tmp_path, FakeMedia(), 16000)
    stopped.stop()
    (tmp_path / "before_session_json" / "audio").mkdir(parents=True)

    assert abort_unfinished(tmp_path) == sorted([cut_off.id, half.id])
    assert read(done_marker(cut_off.audio_pcm)) == read(done_marker(cut_off.video_ts)) == {"status": "aborted"}
    assert read(done_marker(half.audio_pcm)) == {"status": "ok"}
    assert read(done_marker(half.video_ts)) == {"status": "aborted"}
    assert read(done_marker(stopped.audio_pcm)) == read(done_marker(stopped.video_ts)) == {"status": "ok"}
    assert not any((tmp_path / "before_session_json").rglob("*.done"))
    assert f"session {cut_off.id} was left recording by a previous run" in caplog.text
    assert abort_unfinished(tmp_path) == []  # nothing left to mark
    assert abort_unfinished(tmp_path / "missing") == []


def test_the_server_marks_unfinished_recordings_when_it_starts(tmp_path, monkeypatch):
    left = Session.create(tmp_path, FakeMedia(), 16000)
    coordinator = Coordinator(Config(sessions_dir=str(tmp_path), services=(), monitor=False))
    started = []
    monkeypatch.setattr(coordinator._media, "start", lambda: started.append(done_marker(left.audio_pcm).exists()))
    coordinator.start()
    assert started == [True]  # before anything else starts
    assert read(done_marker(left.video_ts)) == {"status": "aborted"}


# --- heartbeats -------------------------------------------------------------------------

def beat(path, state="idle", age=0.0, session=None):
    path.write_text(json.dumps({"service": path.stem, "pid": 1, "state": state, "session": session,
                                "updated": time.time() - age}))


@pytest.mark.parametrize("state, age, up", [
    ("idle", 0, True),
    ("working", 5, True),
    ("loading", 0, False),  # can't work on a session yet
    ("stopped", 0, False),
    ("idle", 30, False),  # stale: the service died without saying so
])
def test_a_service_is_up_while_its_heartbeat_is_fresh_and_it_can_work(tmp_path, state, age, up):
    path = tmp_path / "asr_service.json"
    beat(path, state, age)
    client = WorkerServiceClient(path, status_timeout=15)
    assert client.name == "asr_service"
    assert client.check() is up and client.up is up


def test_a_service_without_a_readable_heartbeat_is_down(tmp_path):
    path = tmp_path / "asr_service.json"
    client = WorkerServiceClient(path, status_timeout=15)
    assert client.check() is False and client.status() is None
    path.write_text("{half")
    assert client.check() is False


def test_up_and_down_are_logged_once_per_change(tmp_path, caplog):
    caplog.set_level(logging.INFO, logger="server.services")
    path = tmp_path / "video_service.json"
    client = WorkerServiceClient(path, status_timeout=15)
    beat(path)
    client.check()
    client.check()
    beat(path, "stopped")
    client.check()
    assert caplog.text.count("video_service is up") == 1
    assert caplog.text.count("video_service is down (stopped)") == 1


def test_the_manager_reports_every_service(tmp_path):
    paths = [tmp_path / f"{name}.json" for name in ("asr_service", "llm_extraction_service", "video_service")]
    beat(paths[0])
    beat(paths[2], "working", session="s1")
    manager = WorkerServiceManager(paths, status_timeout=15, monitor_interval=60, monitor=False)
    assert manager.names == ["asr_service", "llm_extraction_service", "video_service"]
    assert manager.status() == {"asr_service": None, "llm_extraction_service": None, "video_service": None}
    expected = {"asr_service": True, "llm_extraction_service": False, "video_service": True}
    assert manager.start() == expected
    assert manager.status() == expected
    manager.stop()
