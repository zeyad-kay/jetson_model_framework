"""Integration test for the video service (video_service), on its own.

Creates a session, queues it on the video service over its bridge, then
writes video/inputs/video.mp4's video in real time into <session>/video/video.ts as
H.264 in MPEG-TS, as the server records it. When the file has been played
out it ends the session, waits for the service to finish, and checks the
overlay it wrote: <session>/video/overlay.mp4.

    docker compose run --rm video_test
    docker compose run --rm video_test --video-file /tests/integration/video/inputs/clip.mp4
    python tests/integration/video/main.py --dry-run

Exits non-zero if any check fails. See harness.py for how it runs.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # tests/integration, for harness.py
import harness  # noqa: E402

VIDEO = HERE / "inputs" / "video.mp4"


def ffmpeg_cmd(ffmpeg: str, video: Path) -> list[str]:
    """Encodes `video`'s first video stream at its native rate (-re) to H.264 MPEG-TS on stdout.

    Like the e2e client, a keyframe every second and no B-frames, so the
    stream matches what the phone publishes and the server copies to video.ts.
    The audio track is dropped.
    """
    return [
        ffmpeg, "-hide_banner", "-loglevel", "warning",
        "-re", "-i", str(video),
        "-map", "0:v:0",
        "-c:v", "libx264", "-preset", "veryfast", "-tune", "zerolatency",
        "-pix_fmt", "yuv420p", "-bf", "0",
        "-force_key_frames", "expr:gte(t,n_forced*1)", "-sc_threshold", "0",
        "-f", "mpegts", "-flush_packets", "1",
        "pipe:1",
    ]


def probe(path: Path) -> dict:
    """ffprobe's view of the first video stream: duration and frame count; {} if unreadable."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=nb_read_packets,width,height:format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=120,
        )
        info = json.loads(out.stdout or "{}")
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {}
    stream = (info.get("streams") or [{}])[0]
    return {
        "duration": float((info.get("format") or {}).get("duration") or 0),
        "frames": int(stream.get("nb_read_packets") or 0),
        "size": f"{stream.get('width')}x{stream.get('height')}",
    }


def verify(session_dir: Path, session_id: str, checks: harness.Checks) -> None:
    source = probe(session_dir / "video" / "video.ts")
    overlay_path = session_dir / "video" / "overlay.mp4"
    if not checks.check(overlay_path.is_file() and overlay_path.stat().st_size > 0, "overlay.mp4 was written"):
        return
    overlay = probe(overlay_path)
    checks.check(overlay.get("frames", 0) > 0,
                 f"overlay.mp4 is playable ({overlay.get('frames', 0)} frames, "
                 f"{overlay.get('duration', 0):.1f}s, {overlay.get('size')})")
    if source.get("frames"):
        # The service may stop a little short of the recording; most of it should be there.
        ratio = overlay.get("frames", 0) / source["frames"]
        checks.check(ratio >= 0.9, f"overlay covers {ratio:.0%} of video.ts's {source['frames']} frames")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video-file", type=Path, default=VIDEO, help=f"video source (default: {VIDEO})")
    harness.add_common_args(parser, service="video_service", log_name="video")
    args = parser.parse_args()
    harness.setup_log("video_service", args)

    video = harness.require_file(args.video_file)
    feed, dry_run = harness.ffmpeg_feed(ffmpeg_cmd(args.ffmpeg, video), "video/video.ts")
    harness.run("video_service", args, feed, dry_run, verify)


if __name__ == "__main__":
    main()
