"""Shared helpers for the per-service integration clients (asr/main.py, video/main.py,
extraction/main.py, aggregation/main.py).

An integration client tests one model service on its own, without the server:
it creates a session folder the way the server does, drives the service over
its bridge socket (queue_session / end_session), writes the service's input
into the session as it would arrive live (with ffmpeg for audio and video, as
the server's recorder would; segment by segment for the extraction service, as
the ASR would; record by record for the aggregation service, as the extraction
service would), then waits for the service to finish the session and checks its output.

Each service has its own folder: main.py (the client), inputs/ (what the client feeds
the service) and ground_truth/ (what the output is compared against).

Output goes to a log file, not the console (testlog.py). Only the standard library is used. The clients need the same view of the
shared volumes as the services (the sessions folder, the bridge sockets and
the logs), and root to connect to the root-owned sockets, so on the Jetson
they run in the test container (tests/Dockerfile, see docker-compose.yml).
"""

import argparse
import json
import logging
import secrets
import shlex
import socket
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import testlog
from testlog import fail, log

# The subfolders every session has (server/sessions.py).
SUBDIRS = ("audio", "transcript", "video", "structured")


# Writes the service's input into a session folder; returns (ok, description of what it did).
Feed = Callable[[Path], tuple[bool, str]]


def add_common_args(parser: argparse.ArgumentParser, service: str, log_name: str,
                    timeout: float = 300.0, ffmpeg: bool = True) -> None:
    """Options every integration client takes. Defaults are the container paths of .env.example."""
    parser.add_argument("--sessions-dir", type=Path, default=Path("/sessions"),
                        help="the shared sessions folder (default: /sessions)")
    parser.add_argument("--bridge", type=Path, default=Path(f"/run/bridge/{service}.sock"),
                        help=f"the service's bridge socket (default: /run/bridge/{service}.sock)")
    parser.add_argument("--log-file", type=Path, default=Path(f"/logs/{log_name}.log"),
                        help=f"the service's log, watched for the end of the session (default: /logs/{log_name}.log)")
    parser.add_argument("--timeout", type=float, default=timeout,
                        help=f"seconds to wait for the service to finish after end_session (default: {timeout:g})")
    if ffmpeg:
        parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg binary (default: ffmpeg)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print what would be fed to the service without touching it")
    testlog.add_args(parser)


def setup_log(service: str, args: argparse.Namespace) -> None:
    """Starts this run's log file, unless it's a --dry-run (which prints instead)."""
    if not args.dry_run:
        testlog.setup(service, args.test_log_dir)


def require_file(path: Path) -> Path:
    """`path`, or exits if it isn't a file."""
    if not path.is_file():
        fail(f"no such file: {path}")
    return path


def create_session(sessions_dir: Path) -> tuple[str, Path]:
    """Makes a new session folder with the server's layout; returns (session_id, path).

    Ids are prefixed with "it-" so test sessions are easy to tell apart.
    """
    sessions_dir.mkdir(parents=True, exist_ok=True)
    while True:
        session_id = f"it-{secrets.token_hex(4)}"
        path = sessions_dir / session_id
        try:
            path.mkdir()
        except FileExistsError:
            continue
        for name in SUBDIRS:
            (path / name).mkdir()
        return session_id, path


def bridge_send(socket_path: Path, message: dict, timeout: float = 10.0) -> dict:
    """Sends one message to a service's bridge and returns its reply."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        try:
            sock.connect(str(socket_path))
        except OSError as exc:
            fail(f"can't reach the service at {socket_path}: {exc}")
        sock.sendall(json.dumps(message).encode() + b"\n")
        line = sock.makefile("rb").readline()
    if not line:
        fail(f"{socket_path} closed the connection")
    reply = json.loads(line)
    log.info("%s -> %s", message["command"], reply)
    return reply


def record(cmd: list[str], output: Path) -> int:
    """Runs ffmpeg with its stdout written to `output`, like the server's recorder, and its stderr to the log.

    Ctrl-C stops it early. Returns ffmpeg's exit code (None counts as 0).
    """
    log.info("recording to %s", output)
    log.info("  %s", shlex.join(cmd))
    with open(output, "wb") as out:
        try:
            proc = subprocess.Popen(cmd, stdout=out, stderr=testlog.stream())
        except FileNotFoundError:
            log.error("ffmpeg not found: %s", cmd[0])
            return 1
        try:
            return proc.wait()
        except KeyboardInterrupt:
            log.warning("interrupted, stopping ffmpeg")
            proc.terminate()
            try:
                return proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                return proc.wait()


def ffmpeg_feed(cmd: list[str], input_name: str) -> tuple[Feed, str]:
    """A feed that records ffmpeg's stdout into <session>/input_name, and its dry-run description."""
    def feed(session_dir: Path) -> tuple[bool, str]:
        code = record(cmd, session_dir / input_name)
        log.info("ffmpeg finished (exit code %s)", code)
        return code == 0, f"ffmpeg wrote {input_name} (exit code {code})"
    return feed, f"{shlex.join(cmd)} > <session>/{input_name}"


def write_json_atomic(path: Path, obj) -> None:
    """Writes obj as JSON via a temp file and a rename, so readers never see a partial write."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def session_log(log_file: Path, session_id: str) -> list[str]:
    """The service's log lines from "session <id> started" through "session <id> finished"."""
    try:
        lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return []
    start = next((i for i, line in enumerate(lines) if f"session {session_id} started" in line), None)
    if start is None:
        return []
    end = next((i for i, line in enumerate(lines[start:], start) if f"session {session_id} finished" in line),
               len(lines) - 1)
    return lines[start:end + 1]


def wait_for_session_end(log_file: Path, session_id: str, timeout: float) -> bool:
    """Waits for the service to log that the session finished (common/service.py).

    Returns False if it logged "session <id> failed" or the timeout passed.
    """
    finished = f"session {session_id} finished"
    failed = f"session {session_id} failed"
    log.info("waiting up to %gs for %r in %s", timeout, finished, log_file)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            text = log_file.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            text = ""
        if failed in text:
            log.error("the service logged %r; see %s", failed, log_file)
            return False
        if finished in text:
            return True
        time.sleep(1.0)
    log.error("timed out; see %s", log_file)
    return False


class Checks:
    """Collects pass/fail results and logs them as they come."""

    def __init__(self):
        self.failed = 0

    def check(self, ok: bool, message: str) -> bool:
        log.log(logging.INFO if ok else logging.ERROR, "  [%s] %s", "PASS" if ok else "FAIL", message)
        if not ok:
            self.failed += 1
        return ok


def run(service: str, args: argparse.Namespace, feed: Feed, dry_run: str,
        verify: Callable[[Path, str, Checks], None]) -> None:
    """The whole test: session, queue_session, feed, end_session, wait, verify(session_dir, session_id, checks).

    Exits 0 if every check passed, 1 otherwise. With --dry-run, prints `dry_run` instead.
    Everything else goes to this run's log file (testlog.setup, unless the caller set it up already).
    """
    if args.dry_run:
        print(dry_run)
        return

    if not log.handlers:
        testlog.setup(service, args.test_log_dir)
    session_id, session_dir = create_session(args.sessions_dir)
    log.info("session %s at %s", session_id, session_dir)
    queued = bridge_send(args.bridge, {"command": "queue_session", "session_id": session_id})
    if queued.get("ok") is not True:
        fail(f"{service} did not accept the session: {queued.get('error')}")

    fed, fed_message = False, "the input was not written"
    try:
        fed, fed_message = feed(session_dir)
    except KeyboardInterrupt:
        log.warning("interrupted, ending the session")
        fed_message = "interrupted while writing the input"
    finally:
        bridge_send(args.bridge, {"command": "end_session"})

    checks = Checks()
    log.info("results:")
    checks.check(fed, fed_message)
    if checks.check(wait_for_session_end(args.log_file, session_id, args.timeout),
                    f"{service} finished session {session_id}"):
        verify(session_dir, session_id, checks)

    if checks.failed:
        fail(f"{service}: {checks.failed} check(s) failed, session {session_id}")
    log.info("%s: all checks passed, session %s", service, session_id)
