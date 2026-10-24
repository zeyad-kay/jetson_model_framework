import json
import socketserver
import threading
from collections.abc import Callable

from server.log import get_logger

log = get_logger(__name__)


class UnAuthorized(Exception):
    """Raised by a message handler to send `reply` and then drop the connection."""

    def __init__(self, reply: dict):
        super().__init__(reply)
        self.reply = reply


class _Handler(socketserver.StreamRequestHandler):
    """Reads newline-delimited JSON messages and writes one JSON reply per message."""

    def handle(self) -> None:
        for line in self.rfile:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
                if not isinstance(message, dict):
                    raise ValueError("message must be a JSON object")
                reply = self.server.on_message(message, self.request.getsockname()[0])
            except UnAuthorized as close:
                self.wfile.write(json.dumps(close.reply).encode() + b"\n")
                return
            except Exception as exc:
                log.warning("bad client message %r: %s", line[:200], exc)
                reply = {"ok": False, "error": str(exc)}
            self.wfile.write(json.dumps(reply).encode() + b"\n")


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


class StreamingServer:
    """Listens on a TCP port for a remote client (e.g. a phone).

    Messages are one JSON object per line, e.g. {"command": "start_session"}.
    Each is passed to on_message (called on a connection thread) along with the
    local address the client connected to, and the returned dict is sent back as
    the reply.

    This class does no authentication itself: on_message decides what to accept
    (and can raise UnAuthorized to reject a client and drop the connection).
    Traffic is unencrypted, so only expose the port to a trusted network.
    """

    def __init__(self, host: str, port: int, on_message: Callable[[dict, str], dict]):
        self.host = host
        self.port = port
        self._on_message = on_message
        self._server = None
        self._thread = None

    def start(self) -> None:
        self._server = _Server((self.host, self.port), _Handler)
        self._server.on_message = self._on_message
        self.port = self._server.server_address[1]  # resolves port 0
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="streaming-server", daemon=True
        )
        self._thread.start()
        log.info("streaming server listening on %s:%d", self.host, self.port)

    def close(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()
        self._server = None

    def __enter__(self) -> "StreamingServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
