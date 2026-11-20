from pathlib import Path

from common.io import write_jpeg, write_text
from common.service import WorkerService
from video_service.config import Config
from common.log import get_logger
from video_service.model import Sam3

log = get_logger(__name__)


class VideoService(WorkerService):
    """Segments each session's video as it is recorded.

    Idles until queue_session() names a session, then reads <session>/video/video.ts,
    segments it with SAM 3 and saves each frame in which an object is detected with a score of at
    least min_score to <session>/video/wounds/<seconds>.jpg (the folder is made with the first one),
    with those objects' bounding boxes in <seconds>.box, one "x0 y0 x1 y1" line of pixels per box.
    <seconds> is the frame's position in the video, zero-padded with milliseconds (000012.340), so
    the names sort in video order. It runs until the session ends (end_session(), or
    queue_session() of the next session). shutdown() ends the current session and makes run() return.
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
        )
        log.info("model ready: %s (prompt %r)", type(model).__name__, model.prompt)
        return model

    def _process_session(self, session_dir: Path) -> None:
        video_path = session_dir / "video" / "video.ts"
        output_dir = session_dir / "video" / "wounds"
        min_score = self.config.min_score
        log.info("segmenting %s, writing frames with a detection scoring >= %.2f to %s", video_path, min_score, output_dir)
        num_frames = 0
        num_saved = 0
        try:
            if not self._wait_for_file(video_path):
                return
            for frame, _object_ids, scores, boxes, seconds in self._model.segment(str(video_path)):
                kept = [box for score, box in zip(scores, boxes) if float(score) >= min_score]
                if kept:
                    # Boxes first, so whoever sees the image can already read its boxes.
                    name = f"{seconds:010.3f}"
                    lines = "".join(" ".join(str(round(float(v))) for v in box) + "\n" for box in kept)
                    write_text(output_dir / f"{name}.box", lines)
                    write_jpeg(output_dir / f"{name}.jpg", frame, self.config.jpeg_quality)
                    num_saved += 1
                num_frames += 1
                if num_frames % 100 == 0:
                    log.debug("segmented %d frames (%.2fs), saved %d", num_frames, seconds, num_saved)
        finally:
            log.info("segmented %d frames, saved %d to %s", num_frames, num_saved, output_dir)

