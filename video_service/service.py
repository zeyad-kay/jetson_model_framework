import queue
import subprocess
import threading
import time
from pathlib import Path

from common.service import WorkerService
from video_service.config import Config
from video_service.log import get_logger
from video_service.model import Sam3

log = get_logger(__name__)


class VideoService(WorkerService):
    """Segments each session's video as it is recorded.

    Idles until start() names a session, then reads <session>/video/video.ts,
    segments it with SAM 3 and writes the masks drawn over the frames to
    <session>/video/overlay.mp4 until the session ends (stop(), or start() of
    the next session). shutdown() ends the current session and makes run() return.
    """

    def __init__(self, config: Config):
        self.config = config
        self._sessions = queue.Queue()  # session ids waiting to be processed; None = shut down
        self._end_session = threading.Event()
        self._shutdown = threading.Event()
        self._model = None

    def start(self, session_id: str) -> None:
        """Queues a session; any session in progress is finished first.

        Raises ValueError if session_id isn't an existing session folder.
        Safe to call from any thread.
        """
        if Path(session_id).name != session_id or not self._session_dir(session_id).is_dir():
            raise ValueError(f"unknown session: {session_id!r}")
        # Order matters: run() clears the flag after taking an id off the queue.
        self._end_session.set()
        self._sessions.put(session_id)

    def stop(self) -> None:
        """Finishes the current session (processing its remaining frames). Safe from any thread."""
        self._end_session.set()

    def shutdown(self) -> None:
        """Finishes the current session and makes run() return. Safe from any thread or signal handler."""
        self._shutdown.set()
        self._end_session.set()
        self._sessions.put(None)

    def _session_dir(self, session_id: str) -> Path:
        return Path(self.config.sessions_dir) / session_id

    def run(self) -> None:
        cfg = self.config
        self._model = Sam3(
            idle_timeout=cfg.idle_timeout,
            probe_timeout=cfg.probe_timeout,
            drain_idle=cfg.drain_idle,
            alpha=cfg.alpha,
        )
        log.info("model ready: %s (prompt %r)", type(self._model).__name__, self._model.prompt)
        log.info("waiting for a session")
        while True:
            session_id = self._sessions.get()
            if session_id is None or self._shutdown.is_set():
                return
            self._end_session.clear()
            self._run_session(session_id)
            log.info("waiting for a session")

    def _run_session(self, session_id: str) -> None:
        video_dir = self._session_dir(session_id) / "video"
        video_path = video_dir / "video.ts"
        output_path = video_dir / "overlay.mp4"
        log.info("session %s started", session_id)
        log.info("segmenting %s, writing %s", video_path, output_path)
        worker = threading.Thread(target=self._segment, args=(video_path, output_path), daemon=True)
        worker.start()
        self._end_session.wait()
        # Let the model drain the frames it has not processed yet. finish() is a
        # no-op if segment() hasn't started yet, so repeat it until the worker
        # exits; fall back to stop() if draining takes too long.
        deadline = time.monotonic() + self.config.drain_timeout
        while worker.is_alive():
            if time.monotonic() < deadline:
                self._model.finish()
            else:
                log.warning("draining timed out, stopping segmentation")
                self._model.stop()
            worker.join(timeout=0.5)
        log.info("session %s finished", session_id)

    def _segment(self, video_path: Path, output_path: Path) -> None:
        encoder = None
        num_frames = 0
        try:
            # The recorder creates the file shortly after the session starts.
            while not video_path.exists():
                if self._end_session.wait(self.config.poll_interval):
                    return
            for frame, object_ids, masks, seconds in self._model.segment(str(video_path)):
                if encoder is None:
                    encoder = self._open_encoder(output_path, frame.shape[1], frame.shape[0], self._model.fps)
                overlaid = self._model.overlay_masks(frame, object_ids, masks)
                encoder.stdin.write(overlaid.tobytes())
                num_frames += 1
                if num_frames % 100 == 0:
                    log.debug("segmented %d frames (%.2fs)", num_frames, seconds)
        except Exception:
            log.exception("segmentation failed for %s", video_path)
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
