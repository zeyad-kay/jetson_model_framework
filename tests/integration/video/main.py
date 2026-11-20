"""Integration test for the video service (video_service), on its own.

Creates a session, queues it on the video service over its bridge, then
writes video/inputs/video.mp4's video in real time into <session>/video/video.ts as
H.264 in MPEG-TS, as the server records it. When the file has been played
out it ends the session, waits for the service to finish, and checks the
frames it saved: <session>/video/wounds/*.jpg.

    docker compose run --rm video_test
    docker compose run --rm video_test --video-file /tests/integration/video/inputs/clip.mp4
    python tests/integration/video/main.py --dry-run

Exits non-zero if any check fails. See harness.py for how it runs.
"""

import argparse
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


def verify(session_dir: Path, session_id: str, checks: harness.Checks) -> None:
    frames = sorted((session_dir / "video" / "wounds").glob("*.jpg"))
    if not checks.check(len(frames) > 0, f"saved {len(frames)} frames with a detection to video/wounds/"):
        return
    # A JPEG starts with the SOI marker and, written completely, ends with EOI.
    broken = [f.name for f in frames if not (f.read_bytes()[:2] == b"\xff\xd8" and f.read_bytes()[-2:] == b"\xff\xd9")]
    checks.check(not broken, f"every frame is a complete JPEG ({len(broken)} broken: {broken[:5]})")

    # Each frame's <frame>.box: one "x0 y0 x1 y1" line of pixels per detected object.
    bad = []
    for frame in frames:
        box_path = frame.with_suffix(".box")
        lines = box_path.read_text().splitlines() if box_path.is_file() else []
        boxes = [line.split() for line in lines]
        if not boxes or any(len(b) != 4 or not all(v.isdigit() for v in b) or int(b[0]) > int(b[2])
                            or int(b[1]) > int(b[3]) for b in boxes):
            bad.append(box_path.name)
    checks.check(not bad, f"every frame has a .box file of x0 y0 x1 y1 lines ({len(bad)} bad: {bad[:5]})")


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
