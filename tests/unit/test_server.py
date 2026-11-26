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


# --- exports ----------------------------------------------------------------------------

from server import coordinator as server_coordinator  # noqa: E402
from server.sessions import export, unexported  # noqa: E402


def recorded(sessions_dir, name, audio=b"pcm", video=b"ts", marked=True, exported=()):
    """A session folder whose recordings hold `audio` / `video` (None: no file), marked complete
    if `marked`, with the exports named in `exported` ("audio.mp3", "video.mp4")."""
    from conftest import make_session

    folder = make_session(sessions_dir, name)
    for rel, data in (("audio/audio.pcm", audio), ("video/video.ts", video)):
        if data is not None:
            (folder / rel).write_bytes(data)
        if marked:
            (folder / f"{rel}.done").write_text('{"status": "ok"}')
    for name in exported:
        (folder / ("audio" if name.endswith("mp3") else "video") / name).write_bytes(b"x")
    return folder


def test_unexported_finds_complete_recordings_without_their_export(tmp_path):
    recorded(tmp_path, "both_missing")
    recorded(tmp_path, "video_missing", exported=["audio.mp3"])
    recorded(tmp_path, "exported", exported=["audio.mp3", "video.mp4"])
    recorded(tmp_path, "still_recording", marked=False)
    recorded(tmp_path, "nothing_published", audio=b"", video=None)
    (tmp_path / "not_a_session" / "audio").mkdir(parents=True)
    assert [f.name for f in unexported(tmp_path)] == ["both_missing", "video_missing"]
    assert unexported(tmp_path / "missing") == []


def test_export_converts_both_recordings_and_can_skip_existing_exports(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(server_sessions, "pcm_to_mp3", lambda src, dst, rate, log: calls.append((dst.name, rate)) or True)
    monkeypatch.setattr(server_sessions, "ts_to_mp4", lambda src, dst, log: calls.append((dst.name, None)) or False)
    folder = recorded(tmp_path, "s1", exported=["audio.mp3"])
    assert export(folder, 16000) is False  # the video failed, but the audio was still tried
    assert calls == [("audio.mp3", 16000), ("video.mp4", None)]
    calls.clear()
    assert export(folder, 16000, skip_existing=True) is False
    assert calls == [("video.mp4", None)]


class FakeMediaClient:
    def add_path(self, name):
        pass

    def delete_path(self, name):
        pass


@pytest.fixture
def coordinator(tmp_path, monkeypatch):
    c = Coordinator(Config(sessions_dir=str(tmp_path), services=(), monitor=False))
    c._media.client = FakeMediaClient()
    monkeypatch.setattr(c._media, "start", lambda: None)
    monkeypatch.setattr(c._media, "stop", lambda: None)
    yield c
    c._exporter.shutdown(wait=False, cancel_futures=True)


def test_stop_session_replies_before_the_export_and_it_runs_in_the_background(coordinator, monkeypatch):
    import threading

    release, exported = threading.Event(), []

    def slow_export(folder, rate, skip_existing=False):
        release.wait(5)
        exported.append((folder.name, threading.current_thread().name))
        return True

    monkeypatch.setattr(server_coordinator, "export", slow_export)
    session = coordinator.start_session().session
    assert coordinator.stop_session() is session  # returns while the export waits
    assert done_marker(session.audio_pcm).exists() and done_marker(session.video_ts).exists()
    assert exported == []
    release.set()
    coordinator._exporter.shutdown(wait=True)
    assert exported[0][0] == session.id and exported[0][1].startswith("export")


def test_a_failing_export_is_logged_and_doesnt_fail_stop_session(coordinator, monkeypatch, caplog):
    def broken(folder, rate, skip_existing=False):
        raise OSError("disk full")

    monkeypatch.setattr(server_coordinator, "export", broken)
    session = coordinator.start_session().session
    coordinator.stop_session()
    coordinator._exporter.shutdown(wait=True)
    assert f"exporting session {session.id} failed" in caplog.text


def test_start_exports_what_a_previous_run_left_unexported(tmp_path, coordinator, monkeypatch):
    cut_off = recorded(tmp_path, "cut_off", marked=False)  # abort_unfinished marks it first
    half = recorded(tmp_path, "half", exported=["audio.mp3"])
    recorded(tmp_path, "done", exported=["audio.mp3", "video.mp4"])
    exported = []
    monkeypatch.setattr(server_coordinator, "export",
                        lambda folder, rate, skip_existing=False: exported.append((folder.name, skip_existing)) or True)
    coordinator.start()
    coordinator._exporter.shutdown(wait=True)  # every queued export done
    assert sorted(exported) == [(cut_off.name, True), (half.name, True)]


def test_stop_finishes_the_export_under_way_and_drops_the_queued_ones(tmp_path, coordinator, monkeypatch):
    import threading

    for name in ("a", "b", "c"):
        recorded(tmp_path, name)
    started, release, exported = threading.Event(), threading.Event(), []

    def slow_export(folder, rate, skip_existing=False):
        started.set()
        release.wait(5)
        exported.append(folder.name)
        return True

    monkeypatch.setattr(server_coordinator, "export", slow_export)
    coordinator.start()
    assert started.wait(5)
    threading.Timer(0.1, release.set).start()
    coordinator.stop()
    assert exported == ["a"]  # b and c are left for the next start
