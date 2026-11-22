"""Publishing test media to the server over RTSP, as the phone app would: shared by the mock
client (tests/e2e/main.py) and the server's integration test (tests/integration/server/main.py).

Only the standard library is used.
"""

from pathlib import Path
from typing import Optional


def publish_cmd(ffmpeg: str, mode: str, url: str, audio: Optional[Path], video: Optional[Path],
                seconds: Optional[float] = None) -> list[str]:
    """The ffmpeg command that publishes the inputs for `mode` ("audio", "video" or "both") to `url`.

    -re paces each input at its native rate, like a live camera and mic.
    Video is re-encoded to H.264 with a keyframe every second and no B-frames:
    the server records it with -c:v copy, which can only start at a keyframe,
    so the source file's own (possibly long) GOP would delay the recording.
    Audio is re-encoded to AAC, which RTSP carries everywhere. Each stream runs
    to its own end, so in "both" mode the longer file keeps going alone. With seconds, the
    publish stops after that many seconds, and the video loops if it is shorter.
    """
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "warning"]
    if mode in ("video", "both"):
        loop = ["-stream_loop", "-1"] if seconds is not None else []
        cmd += ["-re", *loop, "-i", str(video)]
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

    if seconds is not None:
        cmd += ["-t", f"{seconds:g}"]
    cmd += ["-f", "rtsp", "-rtsp_transport", "tcp", url]
    return cmd
