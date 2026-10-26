import os
import signal
import subprocess
import threading
from pathlib import Path
from urllib.parse import urlsplit

from server.config import Config
from server.media import AudioReader, MediaServerClient, MediaServerError, VideoReader
from server.log import get_logger, setup_logging
from server.services import ServiceManager
from server.sessions import SessionManager
from server.streaming import StreamingServer

log = get_logger(__name__)

# Where the Dockerfile extracts the MediaMTX release (relative to /app).
MEDIAMTX_DIR = Path("mediamtx")


def _url_host(host: str) -> str:
    """Brackets an IPv6 address so it can go in a URL."""
    return f"[{host}]" if ":" in host else host


class Server:
    """Ties the pieces together: the remote client's connection, the session
    storage and the other services. Handles the client's commands."""

    def __init__(self, config: Config):
        self.sessions = SessionManager(config.sessions_dir)
        self.services = ServiceManager(
            config.services, config.ping_timeout, config.ping_interval, config.monitor
        )
        self.streamer = StreamingServer(
            config.streaming_host, config.streaming_port, self.handle_message
        )
        # Every command we accept from the client: name -> (description, handler).
        self.media = MediaServerClient(config.media_api_url)
        self.media_rtsp_url = config.media_rtsp_url.rstrip("/")
        self.media_port = urlsplit(config.media_rtsp_url).port or 8554
        self.audio_reader: AudioReader | None = None
        self.video_reader: VideoReader | None = None
        self.media_server: subprocess.Popen | None = None
        self.commands = {
            "start_session": (
                "Creates a new session folder, starts it on all services and "
                "returns the URL to publish the stream to.",
                self.start_session,
            ),
            "stop_session": (
                "Stops the active session on all services and closes it.",
                self.stop_session,
            ),
        }

    def start_session(self, message: dict, local_host: str) -> dict:
        self._stop_reader()
        session = self.sessions.start()
        log.info("created session %s", session.id)
        # self.services.start_session(session.id)
        try:
            self.media.add_path(session.id)
        except MediaServerError as exc:
            log.warning("could not create media path %s: %s", session.id, exc)
        stream_url = f"{self.media_rtsp_url}/{session.id}"
        self.audio_reader = AudioReader(
            stream_url, self.media, session.id, session.audio_dir / "audio.pcm"
        )
        self.audio_reader.start()
        self.video_reader = VideoReader(
            stream_url, self.media, session.id, session.video_dir / "video.ts"
        )
        self.video_reader.start()
        return {
            "ok": True,
            "session_id": session.id,
            # Where the client can publish: the address it reached us at.
            "streaming_url": f"rtsp://{_url_host(local_host)}:{self.media_port}/{session.id}",
        }

    def stop_session(self, message: dict, local_host: str) -> dict:
        session = self.sessions.session
        if session is None:
            return {"ok": False, "error": "no active session"}
        self._stop_reader()
        # self.services.stop_session()
        try:
            self.media.delete_path(session.id)
        except MediaServerError as exc:
            log.warning("could not delete media path %s: %s", session.id, exc)
        self.sessions.stop()
        log.info("stopped session %s", session.id)
        return {"ok": True, "session_id": session.id}

    def handle_message(self, message: dict, local_host: str) -> dict:
        """Dispatches a client message to its command handler and returns the reply."""
        log.info("client message: %s", message)
        command = message.get("command")
        entry = self.commands.get(command)
        if entry is None:
            return {
                "ok": False,
                "error": f"unknown command: {command!r}",
                "commands": {name: desc for name, (desc, _) in self.commands.items()},
            }
        return entry[1](message, local_host)

    def init_media_server(self) -> None:
        """Runs the mediamtx/mediamtx binary as a child process."""
        if self.media_server is not None and self.media_server.poll() is None:
            return
        binary = MEDIAMTX_DIR / "mediamtx"
        # Run from its own folder so it picks up the mediamtx.yml next to it.
        self.media_server = subprocess.Popen(
            [str(binary.resolve())],
            cwd=binary.parent,
            env={**os.environ},
        )
        log.info("started media server (pid %d)", self.media_server.pid)

    def _stop_reader(self) -> None:
        if self.audio_reader is not None:
            self.audio_reader.stop()
            self.audio_reader = None
        if self.video_reader is not None:
            self.video_reader.stop()
            self.video_reader = None

    def _stop_media_server(self) -> None:
        self._stop_reader()
        if self.media_server is None:
            return
        self.media_server.terminate()
        try:
            self.media_server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.media_server.kill()
            self.media_server.wait()
        self.media_server = None

    def start(self) -> None:
        self.init_media_server()
        # self.services.start()
        self.streamer.start()

    def stop(self) -> None:
        self.streamer.close()
        self._stop_reader()
        # self.services.stop()
        self.sessions.stop()
        self._stop_media_server()

    def __enter__(self) -> "Server":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()


def main() -> None:
    config = Config.load()
    setup_logging(level=config.log_level, log_file=config.log_file)
    log.info("starting server")

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    with Server(config):
        stop.wait()

    log.info("server stopped")


if __name__ == "__main__":
    main()
