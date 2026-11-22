"""Integration test for the server, on its own: drives it over its control port as the phone app
does, publishes short clips to it over RTSP, and checks what it recorded and how it answers.

Needs only the server running: no GPU and no model services.

    docker compose up -d server
    docker compose run --rm server_test
    docker compose run --rm server_test --seconds 10

Run it only while no real session is in progress: start_session ends the session that is
active. Sessions made by the server name every model service in session.json, so any model
service that is running works on the test's sessions too (a few seconds of media each).
The session folders are left in place for inspection; their ids are in the log.

Checks, in order:

    protocol    an unknown command, a line that isn't JSON and JSON that isn't an object
                each get {"ok": false, ...} and the connection stays open; a second client
                is told "busy" and disconnected
    recording   start_session replies with the session's id, an RTSP URL for it and which
                services are up; the folder has its subfolders and session.json; MediaMTX
                has a path for it. After publishing --seconds of audio and video and
                stop_session: audio/audio.pcm holds about that much 16 kHz mono 16-bit
                audio, video/video.ts is H.264 of about that length, both are marked
                complete {"status": "ok"}, audio.mp3 and video.mp4 were exported (in the
                background, after the reply), and the MediaMTX path is gone
    reconnect   a publisher that drops and comes back within one session is appended to
                the same recordings
    replacing   start_session while a session is active ends that one first: its
                recordings are marked complete and its MediaMTX path is removed
    no session  stop_session with no session active fails

Output goes to a log file (testlog.py), with ffmpeg's; the exit code says whether every check
passed. Only the standard library, ffmpeg and ffprobe are used.
"""

import argparse
import json
import shlex
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

INTEGRATION = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(INTEGRATION))
import testlog  # noqa: E402
from harness import Checks  # noqa: E402
from publish import publish_cmd  # noqa: E402
from testlog import fail, log  # noqa: E402

AUDIO = INTEGRATION / "asr" / "inputs" / "audio.mp3"
VIDEO = INTEGRATION / "video" / "inputs" / "video_short.mp4"
SERVICES = ("asr_service", "llm_extraction_service", "video_service")
SUBDIRS = ("audio", "transcript", "video", "extractions")
SAMPLING_RATE = 16000  # the server's default SAMPLING_RATE
# The recordings may miss up to this many seconds of each publish: the server starts ffmpeg once
# it sees the publisher (and video starts at a keyframe), and stops it a moment after the end.
SLACK = 2.5


class Client:
    """One connection to the control port: a JSON object per line each way."""

    def __init__(self, host: str, port: int, timeout: float):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.reader = self.sock.makefile("rb")

    def send_line(self, raw: bytes) -> Optional[dict]:
        """Sends one raw line; returns the reply, or None if the server closed the connection."""
        self.sock.sendall(raw + b"\n")
        return self.reply()

    def reply(self) -> Optional[dict]:
        line = self.reader.readline()
        return json.loads(line) if line else None

    def command(self, name: str) -> dict:
        reply = self.send_line(json.dumps({"command": name}).encode())
        log.info("%s -> %s", name, reply)
        if reply is None:
            fail("the server closed the connection")
        return reply

    def close(self) -> None:
        self.reader.close()
        self.sock.close()


class Media:
    """The MediaMTX Control API, to see which paths exist. It only answers on the server's machine."""

    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")

    def has_path(self, name: str) -> Optional[bool]:
        """Whether MediaMTX has a path named `name`; None if its API can't be reached."""
        try:
            with urllib.request.urlopen(f"{self.api_url}/v3/config/paths/get/{name}", timeout=5):
                return True
        except urllib.error.HTTPError as exc:
            return False if exc.code == 404 else None
        except (urllib.error.URLError, OSError) as exc:
            log.warning("MediaMTX API not reachable at %s: %s", self.api_url, exc)
            return None


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def probe(path: Path) -> tuple[Optional[float], Optional[str]]:
    """(duration in seconds, first stream's codec) of a media file, by ffprobe; None where unknown."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_name",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    info = json.loads(result.stdout or "{}") if result.returncode == 0 else {}
    try:
        duration = float(info.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        duration = None
    streams = info.get("streams") or [{}]
    return duration, streams[0].get("codec_name")


def frame_rate(path: Path) -> Optional[float]:
    """The frame rate of a file's first video stream, by ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True,
    )
    try:
        num, den = (int(x) for x in result.stdout.strip().split("/"))
        return num / den
    except ValueError:
        return None


def video_seconds(path: Path, fps: float) -> tuple[Optional[float], Optional[str]]:
    """(length in seconds, codec) of a file's first video stream: its frames over fps.

    Neither ffprobe's duration nor its frame rate: each run of the server's recorder starts
    the timestamps again, so after a reconnect the duration of video.ts spans only one run
    and the frame rate guessed from the timestamps can come out doubled. fps is the
    publisher's, which the recording keeps.
    """
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,nb_read_packets", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    streams = (json.loads(result.stdout or "{}") if result.returncode == 0 else {}).get("streams") or [{}]
    try:
        return int(streams[0]["nb_read_packets"]) / fps, streams[0].get("codec_name")
    except (KeyError, ValueError):
        return None, streams[0].get("codec_name")


def pcm_seconds(path: Path) -> float:
    return path.stat().st_size / (2 * SAMPLING_RATE) if path.is_file() else 0.0


def publish(args: argparse.Namespace, url: str, seconds: float) -> int:
    """Publishes `seconds` of the test audio and video to url in real time; returns ffmpeg's exit code."""
    cmd = publish_cmd(args.ffmpeg, "both", url, args.audio_file, args.video_file, seconds=seconds)
    log.info("publishing %gs to %s: %s", seconds, url, shlex.join(cmd))
    return subprocess.run(cmd, stderr=testlog.stream()).returncode


def start(client: Client, args: argparse.Namespace, checks: Checks) -> tuple[str, Path, str]:
    """start_session, checking its reply; returns the session's id, its folder and where to publish."""
    reply = client.command("start_session")
    if reply.get("ok") is not True or not reply.get("session_id"):
        fail(f"could not start a session: {reply}")
    session_id = reply["session_id"]
    checks.check(reply.get("streaming_url", "").startswith("rtsp://")
                 and reply["streaming_url"].endswith(f"/{session_id}"),
                 f"start_session gives an RTSP URL for the session: {reply.get('streaming_url')}")
    services = reply.get("services")
    checks.check(isinstance(services, dict) and set(services) == set(SERVICES)
                 and all(isinstance(up, bool) for up in services.values()),
                 f"start_session says which services are up: {services}")
    return session_id, args.sessions_dir / session_id, reply.get("streaming_url", "")


def stop(client: Client, session_id: str, checks: Checks) -> None:
    reply = client.command("stop_session")
    checks.check(reply.get("ok") is True and reply.get("session_id") == session_id,
                 f"stop_session stops session {session_id}")


def wait_for_files(paths: list[Path], timeout: float) -> float:
    """Waits until every path exists (the server exports in the background, after stop_session
    replies, and writes each export atomically); returns the seconds it took, or -1 on timeout."""
    start = time.monotonic()
    while not all(p.exists() for p in paths):
        if time.monotonic() - start > timeout:
            return -1
        time.sleep(0.1)
    return time.monotonic() - start


def check_marked(session_dir: Path, checks: Checks) -> None:
    for name in ("audio/audio.pcm", "video/video.ts"):
        marker = read_json(session_dir / f"{name}.done")
        checks.check(marker == {"status": "ok"}, f"{name} is marked complete: {marker}")


def check_protocol(args: argparse.Namespace, client: Client, checks: Checks) -> None:
    log.info("--- protocol")
    reply = client.command("no_such_command")
    checks.check(reply.get("ok") is False and set(reply.get("commands") or {}) == {"start_session", "stop_session"},
                 "an unknown command gets an error listing the commands")
    reply = client.send_line(b"{not json")
    checks.check(reply is not None and reply.get("ok") is False, f"a line that isn't JSON gets an error: {reply}")
    reply = client.send_line(b"[1, 2]")
    checks.check(reply is not None and reply.get("ok") is False, f"JSON that isn't an object gets an error: {reply}")

    second = Client(args.host, args.port, args.timeout)
    try:
        busy = second.reply()
        closed = second.reply() is None
    finally:
        second.close()
    checks.check(busy is not None and busy.get("ok") is False and "busy" in busy.get("error", ""),
                 f"a second client is told the server is busy: {busy}")
    checks.check(closed, "and is disconnected")
    reply = client.command("no_such_command")
    checks.check(reply.get("ok") is False, "the first client's connection still works")


def check_recording(args: argparse.Namespace, client: Client, media: Media, checks: Checks) -> None:
    log.info("--- recording")
    session_id, session_dir, url = start(client, args, checks)
    log.info("session %s at %s", session_id, session_dir)
    info = read_json(session_dir / "session.json")
    checks.check(isinstance(info, dict) and info.get("id") == session_id and bool(info.get("created"))
                 and info.get("services") == list(SERVICES),
                 f"session.json names the session and its services: {info}")
    checks.check(all((session_dir / sub).is_dir() for sub in SUBDIRS), f"the folder has {', '.join(SUBDIRS)}")
    checks.check(media.has_path(session_id) is True, "MediaMTX has a path for the session")

    code = publish(args, url, args.seconds)
    checks.check(code == 0, f"the publisher ran to the end (exit code {code})")
    time.sleep(args.linger)
    stop(client, session_id, checks)

    audio = session_dir / "audio" / "audio.pcm"
    seconds = pcm_seconds(audio)
    checks.check(args.seconds - SLACK <= seconds <= args.seconds + 0.5,
                 f"audio.pcm holds {seconds:.2f}s of 16 kHz mono audio, about the {args.seconds:g}s published")
    duration, codec = video_seconds(session_dir / "video" / "video.ts", args.fps)
    checks.check(codec == "h264" and duration is not None and args.seconds - SLACK <= duration <= args.seconds + 0.5,
                 f"video.ts is {codec}, {duration}s of frames, about the {args.seconds:g}s published")
    check_marked(session_dir, checks)
    waited = wait_for_files([session_dir / "audio" / "audio.mp3", session_dir / "video" / "video.mp4"], args.timeout)
    checks.check(waited >= 0, f"the recordings were exported in the background, {waited:.1f}s after stop_session replied"
                 if waited >= 0 else f"the recordings weren't exported within {args.timeout:g}s")
    mp3, _ = probe(session_dir / "audio" / "audio.mp3")
    checks.check(mp3 is not None and abs(mp3 - seconds) < 0.5, f"audio.mp3 was exported ({mp3}s)")
    mp4, codec = probe(session_dir / "video" / "video.mp4")
    checks.check(mp4 is not None and codec == "h264" and duration is not None and abs(mp4 - duration) < 0.5,
                 f"video.mp4 was exported ({mp4}s, {codec})")
    checks.check(media.has_path(session_id) is False, "the MediaMTX path is removed")


def check_reconnect(args: argparse.Namespace, client: Client, checks: Checks) -> None:
    log.info("--- reconnect")
    session_id, session_dir, url = start(client, args, checks)
    audio = session_dir / "audio" / "audio.pcm"
    publish(args, url, args.seconds)
    time.sleep(args.linger)
    first = pcm_seconds(audio)
    publish(args, url, args.seconds)  # the publisher comes back
    time.sleep(args.linger)
    stop(client, session_id, checks)
    total = pcm_seconds(audio)
    checks.check(first >= args.seconds - SLACK and total >= first + args.seconds - SLACK,
                 f"the second publish is appended to audio.pcm ({first:.2f}s, then {total:.2f}s)")
    duration, _ = video_seconds(session_dir / "video" / "video.ts", args.fps)
    checks.check(duration is not None and duration >= 2 * (args.seconds - SLACK),
                 f"and to video.ts ({duration}s of frames)")
    check_marked(session_dir, checks)


def check_replacing(args: argparse.Namespace, client: Client, media: Media, checks: Checks) -> None:
    log.info("--- replacing")
    first_id, first_dir, _ = start(client, args, checks)
    second_id, second_dir, _ = start(client, args, checks)  # without stopping the first
    checks.check(second_id != first_id, "start_session while a session is active starts a new one")
    log.info("the first session, %s, should be ended:", first_id)
    check_marked(first_dir, checks)
    checks.check(media.has_path(first_id) is False, "its MediaMTX path is removed")
    checks.check(media.has_path(second_id) is True, "the new session has its path")
    stop(client, second_id, checks)
    check_marked(second_dir, checks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1", help="server address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="server control port (default: 8000)")
    parser.add_argument("--media-api", default="http://127.0.0.1:9997",
                        help="MediaMTX Control API, which only answers locally (default: http://127.0.0.1:9997)")
    parser.add_argument("--sessions-dir", type=Path, default=Path("/sessions"),
                        help="the server's sessions folder, as seen here (default: /sessions)")
    parser.add_argument("--audio-file", type=Path, default=AUDIO, help=f"audio to publish (default: {AUDIO})")
    parser.add_argument("--video-file", type=Path, default=VIDEO,
                        help=f"video to publish, looped as needed (default: {VIDEO})")
    parser.add_argument("--seconds", type=float, default=6.0, help="seconds each publish lasts (default: 6)")
    parser.add_argument("--linger", type=float, default=2.0,
                        help="seconds to wait after a publish, so the server records the last of it (default: 2)")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="seconds to wait for a reply, stop_session's export included (default: 60)")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg binary to publish with (default: ffmpeg)")
    testlog.add_args(parser)
    args = parser.parse_args()
    if args.seconds <= SLACK:
        parser.error(f"--seconds must be more than {SLACK:g}")
    testlog.setup("server", args.test_log_dir)
    for path in (args.audio_file, args.video_file):
        if not path.is_file():
            fail(f"no such file: {path}")
    if not args.sessions_dir.is_dir():
        fail(f"no such folder: {args.sessions_dir}")
    args.fps = frame_rate(args.video_file)
    if not args.fps:
        fail(f"could not read the frame rate of {args.video_file}")

    checks = Checks()
    media = Media(args.media_api)
    try:
        client = Client(args.host, args.port, args.timeout)
    except OSError as exc:
        fail(f"could not reach the server at {args.host}:{args.port}: {exc}")
    try:
        check_protocol(args, client, checks)
        check_recording(args, client, media, checks)
        check_reconnect(args, client, checks)
        check_replacing(args, client, media, checks)
        log.info("--- no session")
        reply = client.command("stop_session")  # every session of this test has been stopped
        checks.check(reply.get("ok") is False and "no active session" in reply.get("error", ""),
                     "stop_session without an active session fails")
    finally:
        client.close()

    if checks.failed:
        fail(f"{checks.failed} check(s) failed")
    log.info("all checks passed")


if __name__ == "__main__":
    main()
