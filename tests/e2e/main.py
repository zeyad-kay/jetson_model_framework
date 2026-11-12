"""Mock client that streams test media to the server, like the phone app would.

It starts a session over the control port, publishes the test media
(tests/integration/asr/inputs/audio.mp3 and video/inputs/video.mp4) to the returned RTSP URL in
real time, then stops the session.

Modes:
    audio   only the .mp3
    video   only the .mp4 (its own audio track, if any, is dropped)
    both    the .mp4's video and the .mp3's audio together (default)

    python tests/e2e/main.py --host 192.168.4.1 --mode audio
    python tests/e2e/main.py --host 192.168.4.1 --mode both --audio-file clip.mp3 --video-file clip.mp4
    python tests/e2e/main.py --mode video --dry-run      # print the ffmpeg command only

Ctrl-C stops streaming early; the session is still stopped cleanly. Needs an
ffmpeg built with RTSP output, the mp3/h264 decoders, and the aac and libx264
encoders (a stock distro ffmpeg has them); pass another binary with --ffmpeg.
Only the standard library is used, so it runs on any machine that can reach the
server. Output, ffmpeg's included, goes to a log file in logs/tests (see
tests/integration/testlog.py; --test-log-dir to change it), not the console.

On the Jetson, whose own ffmpeg can't do this, run it in its container
(tests/Dockerfile), which reaches the server on the host network:

    docker compose run --rm client --mode audio
"""

import argparse
import json
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path

# The test media, and testlog.py, are shared with the integration tests.
INTEGRATION = Path(__file__).resolve().parent.parent / "integration"
sys.path.insert(0, str(INTEGRATION))
import testlog  # noqa: E402
from testlog import fail, log  # noqa: E402

AUDIO = INTEGRATION / "asr" / "inputs" / "audio.mp3"
VIDEO = INTEGRATION / "video" / "inputs" / "video.mp4"
MODES = ("audio", "video", "both")


def publish_cmd(ffmpeg: str, mode: str, url: str, audio: Path, video: Path) -> list[str]:
    """The ffmpeg command that publishes the inputs for `mode` to `url`.

    -re paces each input at its native rate, like a live camera and mic.
    Video is re-encoded to H.264 with a keyframe every second and no B-frames:
    the server records it with -c:v copy, which can only start at a keyframe,
    so the source file's own (possibly long) GOP would delay the recording.
    Audio is re-encoded to AAC, which RTSP carries everywhere. Each stream runs
    to its own end, so in "both" mode the longer file keeps going alone.
    """
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "warning"]
    if mode in ("video", "both"):
        cmd += ["-re", "-i", str(video)]
    if mode in ("audio", "both"):
        cmd += ["-re", "-i", str(audio)]

    if mode == "audio":
        cmd += ["-map", "0:a:0"]
    elif mode == "video":
        cmd += ["-map", "0:v:0"]
    else:
        cmd += ["-map", "0:v:0", "-map", "1:a:0"]

    if mode in ("video", "both"):
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
                "-pix_fmt", "yuv420p", "-bf", "0",
                "-force_key_frames", "expr:gte(t,n_forced*1)", "-sc_threshold", "0"]
    if mode in ("audio", "both"):
        cmd += ["-c:a", "aac", "-b:a", "64k"]

    cmd += ["-f", "rtsp", "-rtsp_transport", "tcp", url]
    return cmd


def command(sock: socket.socket, reader, message: dict) -> dict:
    """Sends one control message and returns the server's reply."""
    sock.sendall(json.dumps(message).encode() + b"\n")
    line = reader.readline()
    if not line:
        fail("server closed the connection")
    reply = json.loads(line)
    log.info("%s -> %s", message["command"], reply)
    return reply


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=MODES, default="both", help="what to stream (default: both)")
    parser.add_argument("--host", default="127.0.0.1", help="server address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="server control port (default: 8000)")
    parser.add_argument("--audio-file", type=Path, default=AUDIO, help=f"audio source (default: {AUDIO})")
    parser.add_argument("--video-file", type=Path, default=VIDEO, help=f"video source (default: {VIDEO})")
    parser.add_argument("--linger", type=float, default=2.0,
                        help="seconds to wait after streaming before stopping the session, so the "
                             "server records the last of the stream (default: 2)")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg binary to publish with (default: ffmpeg)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the ffmpeg command without contacting the server")
    testlog.add_args(parser)
    args = parser.parse_args()
    if not args.dry_run:
        testlog.setup("e2e", args.test_log_dir)

    audio = video = None
    if args.mode in ("audio", "both"):
        audio = args.audio_file
    if args.mode in ("video", "both"):
        video = args.video_file
    for path in (audio, video):
        if path is not None and not path.is_file():
            fail(f"no such file: {path}")

    if args.dry_run:
        url = f"rtsp://{args.host}:8554/<session_id>"
        print(shlex.join(publish_cmd(args.ffmpeg, args.mode, url, audio, video)))
        return

    with socket.create_connection((args.host, args.port), timeout=10) as sock:
        reader = sock.makefile("rb")
        session = command(sock, reader, {"command": "start_session"})
        if session.get("ok") is not True:
            fail(f"could not start a session: {session.get('error')}")
        down = [name for name, ok in (session.get("services") or {}).items() if not ok]
        if down:
            log.warning("these services did not accept the session: %s", ", ".join(down))

        url = session["streaming_url"]
        sources = " + ".join(f"{p.name} ({kind})" for p, kind in ((video, "video"), (audio, "audio")) if p)
        log.info("streaming %s to %s", sources, url)
        cmd = publish_cmd(args.ffmpeg, args.mode, url, audio, video)
        log.info("  %s", shlex.join(cmd))
        try:
            publisher = subprocess.Popen(cmd, stderr=testlog.stream())
        except FileNotFoundError:
            command(sock, reader, {"command": "stop_session"})
            fail(f"ffmpeg not found: {args.ffmpeg}")

        code = None
        try:
            code = publisher.wait()
            log.info("publisher finished (exit code %s), stopping the session in %gs", code, args.linger)
            time.sleep(args.linger)
        except KeyboardInterrupt:
            log.warning("interrupted, stopping the stream")
            publisher.terminate()
            try:
                publisher.wait(timeout=5)
            except subprocess.TimeoutExpired:
                publisher.kill()
                publisher.wait()
        finally:
            stopped = command(sock, reader, {"command": "stop_session"})

    if stopped.get("ok") is not True:
        fail(f"could not stop the session: {stopped.get('error')}")
    if code not in (0, None):
        fail(f"session {session['session_id']} done, but the publisher exited with code {code}")
    log.info("session %s done", session["session_id"])


if __name__ == "__main__":
    main()
