import subprocess
from pathlib import Path

from common.service import WorkerService
from video_service.config import Config
from common.log import get_logger
from video_service.model import Sam3

log = get_logger(__name__)


class VideoService(WorkerService):
    """Segments each session's video as it is recorded.

    Idles until queue_session() names a session, then reads <session>/video/video.ts,
    segments it with SAM 3 and writes the masks drawn over the frames to
    <session>/video/overlay.mp4 until the session ends (end_session(), or queue_session() of
    the next session). shutdown() ends the current session and makes run() return.
    """

    def __init__(self, config: Config):
        super().__init__(config.sessions_dir, config.poll_interval, log)
        self.config = config

    def _load_model(self) -> Sam3:
        cfg = self.config
        model = Sam3(
            prompt=cfg.prompt,
            idle_timeout=cfg.idle_timeout,
            probe_timeout=cfg.probe_timeout,
            drain_idle=cfg.drain_idle,
            alpha=cfg.alpha,
        )
        log.info("model ready: %s (prompt %r)", type(model).__name__, model.prompt)
        return model

    def _process_session(self, session_dir: Path) -> None:
        video_path = session_dir / "video" / "video.ts"
        output_path = session_dir / "video" / "overlay.mp4"
        log.info("segmenting %s, writing %s", video_path, output_path)
        encoder = None
        num_frames = 0
        try:
            if not self._wait_for_file(video_path):
                return
            for frame, object_ids, masks, seconds in self._model.segment(str(video_path)):
                if encoder is None:
                    encoder = self._open_encoder(output_path, frame.shape[1], frame.shape[0], self._model.fps)
                overlaid = self._model.overlay_masks(frame, object_ids, masks)
                encoder.stdin.write(overlaid.tobytes())
                num_frames += 1
                if num_frames % 100 == 0:
                    log.debug("segmented %d frames (%.2fs)", num_frames, seconds)
        finally:
            # Close even on a crash so the partial video stays playable.
            if encoder is not None:
                encoder.stdin.close()
                encoder.wait()
            log.info("wrote %d frames to %s", num_frames, output_path)

    @staticmethod
    def _open_encoder(output_path: Path, width: int, height: int, fps: float) -> subprocess.Popen:
        """Starts ffmpeg reading raw RGB frames on stdin and encoding them to output_path."""
        return subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # yuv420p needs even dimensions
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output_path)],
            stdin=subprocess.PIPE,
        )
