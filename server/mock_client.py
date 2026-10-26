"""Mock client for the server: starts a session, publishes an audio+video file to
MediaMTX over RTSP in real time, then stops the session and checks the result.

    python mock_client.py --host 192.168.4.1
    python mock_client.py --sessions-dir ./sessions --api-url http://10.79.128.41:9997

The checks on the session folder and the recorded audio need --sessions-dir
(the server's sessions volume as seen from here). The check on the MediaMTX
path needs --api-url (its Control API only listens on localhost). Without
them those checks are skipped. Exits non-zero if any check fails.
"""

import argparse
import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

SESSION_SUBDIRS = ("audio", "transcript", "video", "structured")
SAMPLE_RATE = 16000  # the server's reader writes 16 kHz, 16-bit mono PCM
AUDIO_TOLERANCE = 0.10  # recorded length may differ from the input's by this much

results: list[tuple[str, bool]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f" ({detail})" if detail else ""))
    return ok


def skip(name: str, reason: str) -> None:
    print(f"  SKIP  {name} ({reason})")


def command(sock_file, sock, message: dict) -> dict:
    sock.sendall(json.dumps(message).encode() + b"\n")
    reply = json.loads(sock_file.readline())
    print(f"{message['command']} -> {reply}")
    return reply


def media_path(api_url: str, name: str):
    """The MediaMTX path's state, or None if it doesn't exist."""
    try:
        with urllib.request.urlopen(f"{api_url.rstrip('/')}/v3/paths/get/{name}", timeout=3) as r:
            return json.load(r)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def wait_for(condition, timeout: float, interval: float = 0.25) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(interval)
    return condition()


def reference_pcm(audio_file: str) -> bytes:
    """The audio file decoded to the PCM the server should end up recording."""
    return subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", audio_file,
         "-vn", "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"],
        capture_output=True, check=True,
    ).stdout


def wait_until_size_settles(path: Path, quiet: float = 1.0, timeout: float = 10.0) -> None:
    """Waits until the file stops growing, so the server's reader has drained."""
    deadline = time.monotonic() + timeout
    last, since = -1, time.monotonic()
    while time.monotonic() < deadline:
        size = path.stat().st_size if path.exists() else 0
        if size != last:
            last, since = size, time.monotonic()
        elif time.monotonic() - since >= quiet:
            return
        time.sleep(0.25)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="server control port")
    parser.add_argument("--video-file", default=str(Path(__file__).parent / "sample-30s.mp4"),
                         help="video-only source (its own audio track, if any, is dropped)")
    parser.add_argument("--audio-file", default=str(Path(__file__).parent / "obama_16k.mp3"),
                         help="audio source muxed in alongside the video")
    parser.add_argument("--sessions-dir", help="the server's sessions folder, as seen from here")
    args = parser.parse_args()

    sessions_dir = Path(args.sessions_dir) if args.sessions_dir else None
    expected = reference_pcm(args.audio_file) if sessions_dir else None

    with socket.create_connection((args.host, args.port), timeout=10) as sock:
        sock_file = sock.makefile("rb")

        print("start_session")
        session = command(sock_file, sock, {"command": "start_session"})
        if not check("start_session replied ok", session.get("ok") is True, session.get("error", "")):
            raise SystemExit("could not start session")
        session_id = session["session_id"]
        check("reply has a url ending in the session id", str(session.get("streaming_url", "")).endswith(f"/{session_id}"))

        session_dir = sessions_dir / session_id if sessions_dir else None
        audio = session_dir / "audio" / "audio.pcm" if session_dir else None
        video = session_dir / "video" / "video.ts" if session_dir else None
        if session_dir:
            check("session folder exists", session_dir.is_dir(), str(session_dir))
            missing = [name for name in SESSION_SUBDIRS if not (session_dir / name).is_dir()]
            check("session subfolders exist", not missing, f"missing: {missing}" if missing else "")
        else:
            skip("session folder", "no --sessions-dir")

        url = session["streaming_url"]
        print(f"publishing {args.video_file} (video) + {args.audio_file} (audio) to {url}")
        # -re (once per input) paces each file at its native rate, like a live
        # camera+mic. The video file's own audio track (if any) is dropped via
        # the -map selection; audio comes only from --audio-file, re-encoded to
        # AAC since RTSP/RTP needs a payload-able codec. Each stream runs to
        # its own end (no -shortest), so a duration mismatch just means one
        # stream keeps going alone after the other finishes.
        #
        # Video is re-encoded to H.264 (not stream-copied) with a short, fixed
        # keyframe interval (-g, one per second at 30fps) rather than whatever
        # GOP structure the source file happens to have - VideoReader's -c:v
        # copy on the server side can only start recording from a keyframe, so
        # a short source GOP matters more here than for a casual test file.
        publisher = subprocess.Popen([
            "ffmpeg", "-re", "-i", args.video_file,
            "-re", "-i", args.audio_file,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-bf", "0",
            "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
            "-c:a", "aac",
            "-f", "rtsp", url,
        ])
        try:
            check("publisher finished cleanly", publisher.wait() == 0, f"exit code {publisher.returncode}")
        except KeyboardInterrupt:
            print("interrupted")
            publisher.terminate()
        finally:
            if audio:
                wait_until_size_settles(audio)
            if video:
                wait_until_size_settles(video)
            print("stop_session")
            stopped = command(sock_file, sock, {"command": "stop_session"})
            check("stop_session replied ok", stopped.get("ok") is True, stopped.get("error", ""))
            check("stopped the same session", stopped.get("session_id") == session_id)

    if audio:
        size = audio.stat().st_size if audio.exists() else 0
        low, high = len(expected) * (1 - AUDIO_TOLERANCE), len(expected) * (1 + AUDIO_TOLERANCE)
        check("recorded audio length matches the input", low <= size <= high,
              f"{size} bytes, expected {len(expected)} ±{AUDIO_TOLERANCE:.0%}")

    if video:
        size = video.stat().st_size if video.exists() else 0
        check("recorded video is non-empty", size > 0, f"{size} bytes")

    failed = [name for name, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")

    publisher.wait()


    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
