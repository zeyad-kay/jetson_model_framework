from pathlib import Path
from typing import Any, Optional

from common.io import write_jpeg, write_text
from common.log import get_logger
from common.service import WorkerService
from video_service.config import Config
from video_service.reader import FrameReader
from video_service.model import Sam3

log = get_logger(__name__)


class VideoService(WorkerService):
    """Segments each session's video as it is recorded.

    Works through the sessions the server creates (see WorkerService), in its steps:

    reader   a FrameReader: the next frame of <session>/video/video.ts, which the server
             appends to as the session goes on (H.264 in MPEG-TS): {"frame": HxWx3 uint8
             RGB array, "seconds": its position in the video}.
    predict  Sam3.predict(): the objects SAM 3 tracks for the prompt on the frame.
    save     if an object is detected with a score of at least min_score, writes the frame to
             <session>/video/wounds/<seconds>.jpg (the folder is made with the first one), with
             those objects' bounding boxes in <seconds>.box, one "x0 y0 x1 y1" line of pixels
             per box. <seconds> is zero-padded with milliseconds (000012.340), so the names
             sort in video order.
    cleanup  (the base's) stops the model's video and writes video/wounds.done ({"status":
             "ok", "failed" or "abandoned"}), which marks the frames in wounds/ final, even if
             the session failed (its output marker).

    The session ends once the server marks video.ts complete (video.ts.done) and its last
    frame is segmented. It also ends if the stream turns out broken, or once nothing in the
    session has changed for abandon_timeout seconds without that marker ("abandoned").
    """

    def __init__(self, config: Config):
        super().__init__("video_service", config.sessions_dir, config.poll_interval, log,
                         status_path=config.status_path, max_session_age=config.max_session_age,
                         abandon_timeout=config.abandon_timeout)
        self.config = config
        # The current session's state, set by open_reader() and setup().
        self._frames: Optional[FrameReader] = None
        self._output_dir: Optional[Path] = None
        self._num_segmented = 0
        self._num_saved = 0

    def load_model(self) -> Sam3:
        model = Sam3(prompt=self.config.prompt)
        log.info("model ready: %s (prompt %r)", type(model).__name__, model.prompt)
        return model

    def output_path(self, session_dir: Path) -> Path:
        return session_dir / "video" / "wounds"

    def open_reader(self, session_dir: Path) -> FrameReader:
        self._frames = FrameReader(session_dir / "video" / "video.ts", probe_timeout=self.config.probe_timeout)
        return self._frames

    def setup(self, session_dir: Path) -> None:
        self._output_dir = session_dir / "video" / "wounds"
        self._num_segmented = self._num_saved = 0
        log.info("segmenting %s, writing frames with a detection scoring >= %.2f to %s",
                 self._frames.path, self.config.min_score, self._output_dir)
        self._model.start()

    def save(self, inputs: dict, outputs: dict[str, Any]) -> None:
        if self._num_segmented == 0:
            log.info("video stream: %dx%d at %.2f fps", self._frames.width, self._frames.height, self._frames.fps)
        min_score = self.config.min_score
        kept = [box for score, box in zip(outputs["scores"], outputs["boxes"]) if float(score) >= min_score]
        if kept:
            # Boxes first, so whoever sees the image can already read its boxes.
            name = f"{inputs['seconds']:010.3f}"
            lines = "".join(" ".join(str(round(float(v))) for v in box) + "\n" for box in kept)
            write_text(self._output_dir / f"{name}.box", lines)
            write_jpeg(self._output_dir / f"{name}.jpg", inputs["frame"], self.config.jpeg_quality)
            self._num_saved += 1
        self._num_segmented += 1
        if self._num_segmented % 100 == 0:
            log.debug("segmented %d frames (%.2fs), saved %d", self._num_segmented, inputs["seconds"], self._num_saved)
