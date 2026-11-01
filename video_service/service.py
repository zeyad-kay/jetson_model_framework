import threading

from common.service import WorkerService
from video_service.config import Config
from video_service.log import get_logger

log = get_logger(__name__)


class VideoService(WorkerService):
    """Placeholder: tracks the current session so the server sees the service as up. It does no work yet."""

    def __init__(self, config: Config):
        self.config = config
        self._shutdown = threading.Event()

    def start(self, session_id: str) -> None:
        log.info("session %s started", session_id)

    def stop(self) -> None:
        log.info("session stopped")

    def shutdown(self) -> None:
        self._shutdown.set()

    def run(self) -> None:
        log.info("waiting for a session")
        self._shutdown.wait()
