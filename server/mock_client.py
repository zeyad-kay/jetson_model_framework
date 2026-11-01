"""Mock client for the server: starts a session, publishes an audio+video file to
MediaMTX over RTSP in real time, then stops the session and checks the result.

    python mock_client.py --host 192.168.4.1
    python mock_client.py --sessions-dir ./sessions --api-url http://10.79.128.41:9997
    python mock_client.py --sessions-dir ./sessions --drops 2 --drop-interval 10 --drop-seconds 5

The checks on the session folder and the recorded audio need --sessions-dir
(the server's sessions volume as seen from here). The check on the MediaMTX
path needs --api-url (its Control API only listens on localhost). Without
them those checks are skipped. Exits non-zero if any check fails.

--drops N simulates N network drops: every --drop-interval seconds the
publisher is killed without a clean shutdown, stays offline for --drop-seconds,
then reconnects to the same URL. Like a live camera and mic, the source keeps
running while offline, so the audio from the outage is lost and the recording
should be shorter by about --drop-seconds per drop.
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
SAMPLE_RATE = 16000  # the server's audio listener writes 16 kHz, 16-bit mono PCM
BYTES_PER_SECOND = SAMPLE_RATE * 2
RECONNECT_SLACK = 3.0  # extra seconds the server may take to notice a reconnect, per drop
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
    """Waits until the file stops growing, so the server's listener has drained."""
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
    parser.add_argument("--drops", type=int, default=0, help="number of network drops to simulate")
    parser.add_argument("--drop-interval", type=float, default=10.0,
                         help="seconds of publishing before each drop")
    parser.add_argument("--drop-seconds", type=float, default=5.0,
                         help="seconds the publisher stays offline per drop")
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
        # GOP structure the source file happens to have - VideoListener's -c:v
        # copy on the server side can only start recording from a keyframe, so
        # a short source GOP matters more here than for a casual test file.
        def start_publisher(offset: float) -> subprocess.Popen:
            # -ss before each -i seeks both files to where a live source would be.
            return subprocess.Popen([
                "ffmpeg", "-ss", str(offset), "-re", "-i", args.video_file,
                "-ss", str(offset), "-re", "-i", args.audio_file,
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-preset", "veryfast", "-bf", "0",
                "-g", "30", "-keyint_min", "30", "-sc_threshold", "0",
                "-c:a", "aac",
                "-f", "rtsp", url,
            ])

        drops_done = 0
        sizes_at_reconnect = []
        offset = 0.0
        publisher = start_publisher(offset)
        try:
            while True:
                if drops_done < args.drops:
                    try:
                        publisher.wait(timeout=args.drop_interval)
                    except subprocess.TimeoutExpired:
                        drops_done += 1
                        print(f"network drop {drops_done}/{args.drops}: offline for {args.drop_seconds}s")
                        publisher.kill()  # no clean RTSP teardown, like a lost connection
                        publisher.wait()
                        time.sleep(args.drop_seconds)
                        offset += args.drop_interval + args.drop_seconds
                        sizes_at_reconnect.append(audio.stat().st_size if audio and audio.exists() else 0)
                        print(f"reconnecting, resuming the source at {offset}s")
                        publisher = start_publisher(offset)
                        continue
                else:
                    publisher.wait()
                break
            check("simulated every network drop", drops_done == args.drops,
                  f"{drops_done}/{args.drops}; the input ended too soon" if drops_done != args.drops else "")
            check("publisher finished cleanly", publisher.returncode == 0, f"exit code {publisher.returncode}")
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
        # Each drop loses the outage's audio, plus a little while the server notices the reconnect.
        target = len(expected) - drops_done * args.drop_seconds * BYTES_PER_SECOND
        low = target * (1 - AUDIO_TOLERANCE) - drops_done * RECONNECT_SLACK * BYTES_PER_SECOND
        high = target * (1 + AUDIO_TOLERANCE)
        check("recorded audio length matches the input", low <= size <= high,
              f"{size} bytes, expected {int(target)} ±{AUDIO_TOLERANCE:.0%}"
              + (f" after {drops_done} drops" if drops_done else ""))
        if sizes_at_reconnect:
            # Bytes written before the reconnect can't include audio from after it.
            check("recording resumed after the last drop", size > sizes_at_reconnect[-1],
                  f"{sizes_at_reconnect[-1]} -> {size} bytes")

    if video:
        size = video.stat().st_size if video.exists() else 0
        check("recorded video is non-empty", size > 0, f"{size} bytes")

    failed = [name for name, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")

    publisher.wait()


    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
