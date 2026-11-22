"""The dashboard's view of a session (dashboard/server.py, Sessions): its state from the done
markers, what it waits for, and the problems the markers report."""

import json
import os
import time

import pytest

from conftest import make_session
from dashboard.server import Sessions

SERVICES = ("asr_service", "llm_extraction_service", "video_service")
RECORDINGS = ("audio/audio.pcm", "video/video.ts")
OUTPUTS = ("transcript/segments.json", "extractions/extractions", "video/wounds")


def mark(session, rel, status="ok"):
    (session / f"{rel}.done").write_text(json.dumps({"status": status}))


def touch(session, rel, age=0.0):
    path = session / rel
    path.touch()
    then = time.time() - age
    os.utime(path, (then, then))


def age_everything(session, seconds):
    """Makes every file and folder of the session look `seconds` old."""
    then = time.time() - seconds
    for path in [session, *session.rglob("*")]:
        os.utime(path, (then, then))


@pytest.fixture
def sessions(tmp_path):
    return Sessions(tmp_path, live_window=60)


def test_a_new_session_is_recording(tmp_path, sessions):
    s = make_session(tmp_path, "s1", services=SERVICES)
    touch(s, "audio/audio.pcm")
    summary = sessions.summary(s)
    assert summary["state"] == "recording" and summary["live"]
    assert summary["waiting"] == ["audio recording", "video recording", "transcription", "extraction",
                                  "wound detection"]
    assert summary["problems"] == []


def test_once_the_recordings_are_marked_it_is_processing_until_every_output_is(tmp_path, sessions):
    s = make_session(tmp_path, "s1", services=SERVICES)
    touch(s, "transcript/segments.json")
    for rel in RECORDINGS:
        mark(s, rel)
    assert sessions.summary(s)["state"] == "processing"
    mark(s, OUTPUTS[0])
    mark(s, OUTPUTS[2])
    summary = sessions.summary(s)
    assert summary["state"] == "processing" and summary["waiting"] == ["extraction"]
    mark(s, OUTPUTS[1])
    summary = sessions.summary(s)
    assert summary["state"] == "done" and not summary["live"] and summary["waiting"] == []


def test_only_the_services_the_session_is_for_are_waited_for(tmp_path, sessions):
    s = make_session(tmp_path, "s1", services=["asr_service"])
    for rel in RECORDINGS:
        mark(s, rel)
    mark(s, "transcript/segments.json")
    assert sessions.summary(s)["state"] == "done"


def test_a_session_json_without_a_services_list_waits_for_every_service(tmp_path, sessions):
    s = make_session(tmp_path, "s1")  # no "services"
    for rel in RECORDINGS:
        mark(s, rel)
    mark(s, "transcript/segments.json")
    assert sessions.summary(s)["waiting"] == ["extraction", "wound detection"]


def test_a_session_nothing_changed_in_for_the_live_window_is_stalled(tmp_path, sessions):
    s = make_session(tmp_path, "s1", services=SERVICES)
    for rel in RECORDINGS:
        mark(s, rel)
    touch(s, "transcript/segments.json")
    age_everything(s, 120)
    summary = sessions.summary(s)
    assert summary["state"] == "stalled" and not summary["live"]
    assert summary["waiting"] == ["transcription", "extraction", "wound detection"]


def test_markers_that_arent_ok_are_problems(tmp_path, sessions):
    s = make_session(tmp_path, "s1", services=SERVICES)
    mark(s, "audio/audio.pcm", "aborted")
    mark(s, "video/video.ts")
    mark(s, "transcript/segments.json", "failed")
    mark(s, "extractions/extractions", "abandoned")
    mark(s, "video/wounds")
    summary = sessions.summary(s)
    assert summary["state"] == "done"  # ended, but not cleanly
    assert summary["problems"] == [
        {"what": "audio recording", "status": "aborted"},
        {"what": "transcription", "status": "failed"},
        {"what": "extraction", "status": "abandoned"},
    ]


def test_only_folders_with_session_json_are_sessions(tmp_path, sessions):
    make_session(tmp_path, "s1")
    (tmp_path / "not_a_session" / "audio").mkdir(parents=True)
    assert [s["id"] for s in sessions.list()] == ["s1"]
    assert sessions.folder("s1") is not None
    assert sessions.folder("not_a_session") is None


def test_the_etag_changes_with_the_state_and_problems(tmp_path, sessions):
    s = make_session(tmp_path, "s1", services=["asr_service"])
    for rel in RECORDINGS:
        mark(s, rel)
    _, processing = sessions.detail(s)
    mark(s, "transcript/segments.json")
    _, done = sessions.detail(s)
    mark(s, "transcript/segments.json", "failed")
    _, failed = sessions.detail(s)
    assert len({processing, done, failed}) == 3


# --- the download ------------------------------------------------------------------------

def test_the_recordings_can_be_downloaded_once_the_session_has_stopped_recording(tmp_path, sessions):
    s = make_session(tmp_path, "s1", services=SERVICES)
    (s / "audio" / "audio.mp3").write_bytes(b"mp3")  # as if exported, though recording hasn't stopped
    detail, _ = sessions.detail(s)
    assert detail["download"] is None and sessions.exports(s) == []
    for rel in RECORDINGS:
        mark(s, rel)
    detail, _ = sessions.detail(s)
    assert detail["download"] == {"url": "/api/sessions/s1/download", "files": ["audio.mp3"]}
    (s / "video" / "video.mp4").write_bytes(b"mp4")
    detail, _ = sessions.detail(s)
    assert detail["download"]["files"] == ["audio.mp3", "video.mp4"]


def test_a_session_stopped_without_exports_has_nothing_to_download(tmp_path, sessions):
    s = make_session(tmp_path, "s1", services=SERVICES)
    for rel in RECORDINGS:
        mark(s, rel)
    assert sessions.detail(s)[0]["download"] is None


def test_the_download_is_a_zip_of_the_exports(tmp_path, sessions):
    import io
    import zipfile

    from dashboard.server import write_zip

    s = make_session(tmp_path, "s1", services=SERVICES)
    for rel in RECORDINGS:
        mark(s, rel)
    (s / "audio" / "audio.mp3").write_bytes(b"a" * 100)
    (s / "video" / "video.mp4").write_bytes(b"v" * 200_000)

    class Unseekable(io.RawIOBase):  # like the socket the server writes to
        def __init__(self):
            self.data = bytearray()

        def writable(self):
            return True

        def write(self, b):
            self.data += b
            return len(b)

    out = Unseekable()
    write_zip(out, sessions.exports(s))
    with zipfile.ZipFile(io.BytesIO(bytes(out.data))) as zf:
        assert zf.namelist() == ["audio.mp3", "video.mp4"]
        assert zf.read("audio.mp3") == b"a" * 100 and zf.read("video.mp4") == b"v" * 200_000
