import json
import socketserver
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Union

from asr_service.log import get_logger

log = get_logger(__name__)


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
                reply = self.server.on_message(message)
            except Exception as exc:
                log.warning("bad bridge message %r: %s", line[:200], exc)
                reply = {"ok": False, "error": str(exc)}
            self.wfile.write(json.dumps(reply).encode() + b"\n")


class _Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


class Bridge:
    """Listens on a Unix socket for JSON messages from other processes.

    Messages are one JSON object per line, e.g. {"command": "stop"}. Each is
    passed to on_message (called on a listener thread), and the returned dict
    is sent back as the reply.

    There is no authentication; access is limited by who can reach the socket
    file, so keep it in a directory shared only with trusted containers.
    """

    def __init__(self, path: Union[str, Path], on_message: Callable[[dict], dict]):
        self.path = Path(path)
        self._on_message = on_message
        self._server = None
        self._thread = None

    def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)  # stale socket from a previous run
        self._server = _Server(str(self.path), _Handler)
        self._server.on_message = self._on_message
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="bridge", daemon=True
        )
        self._thread.start()
        log.info("listening for bridge messages on %s", self.path)

    def close(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self.path.unlink(missing_ok=True)
        self._thread.join()
        self._server = None

    def __enter__(self) -> "Bridge":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
