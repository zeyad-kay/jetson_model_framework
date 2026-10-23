import signal
import subprocess
import threading
from pathlib import Path

from server.auth import authenticate
from server.config import Config
from server.log import get_logger, setup_logging
from server.services import ServiceManager
from server.sessions import SessionManager
from server.streaming import UnAuthorized, StreamingServer

log = get_logger(__name__)

# Where the Dockerfile extracts the MediaMTX release (relative to /app).
MEDIAMTX_DIR = Path("mediamtx")


class Server:
    """Ties the pieces together: the remote client's connection, the session
    storage and the other services. Handles the client's commands."""

    def __init__(self, config: Config):
        self.private_key = config.private_key
        self.sessions = SessionManager(config.sessions_dir)
        self.services = ServiceManager(
            config.services, config.ping_timeout, config.ping_interval, config.monitor
        )
        self.streamer = StreamingServer(
            config.streaming_host, config.streaming_port, self.handle_message
        )
        # Every command we accept from the client: name -> (description, handler).
        self.media_server: subprocess.Popen | None = None
        self.commands = {
            "establish_connection": (
                "Authenticates the client with the key sent in the request.",
                self.establish_connection,
            ),
            "start_session": (
                "Creates a new session folder and starts it on all services.",
                self.start_session,
            ),
            "stop_session": (
                "Stops the active session on all services and closes it.",
                self.stop_session,
            ),
        }

    def establish_connection(self, message: dict) -> dict:
        key = message.get("key")
        if not isinstance(key, str) or not authenticate(key, self.private_key):
            log.warning("authentication failed")
            raise UnAuthorized({"ok": False, "error": "authentication failed"})
        log.info("client authenticated")
        return {"ok": True}

    def start_session(self, message: dict) -> dict:
        session = self.sessions.start()
        log.info("created session %s", session.id)
        delivered = self.services.start_session(session.id)
        return {
            "ok": True,
            "session_id": session.id,
            "path": str(session.path),
            "delivered": delivered,
        }

    def stop_session(self, message: dict) -> dict:
        session = self.sessions.session
        if session is None:
            return {"ok": False, "error": "no active session"}
        delivered = self.services.stop_session()
        self.sessions.stop()
        log.info("stopped session %s", session.id)
        return {"ok": True, "session_id": session.id, "delivered": delivered}

    def handle_message(self, message: dict) -> dict:
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
        return entry[1](message)

    def init_media_server(self) -> None:
        """Runs the mediamtx/mediamtx binary as a child process."""
        if self.media_server is not None and self.media_server.poll() is None:
            return
        binary = MEDIAMTX_DIR / "mediamtx"
        # Run from its own folder so it picks up the mediamtx.yml next to it.
        self.media_server = subprocess.Popen([str(binary.resolve())], cwd=binary.parent)
        log.info("started media server (pid %d)", self.media_server.pid)

    def _stop_media_server(self) -> None:
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
        self.services.start()
        self.streamer.start()

    def stop(self) -> None:
        self.streamer.close()
        self.services.stop()
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
