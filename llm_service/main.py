"""Placeholder llm_service: answers the server's bridge messages (start, stop, ping)
with ok so the server sees it as up. It does no work yet."""

import json
import logging
import os
import signal
import socketserver
import threading
from pathlib import Path

BRIDGE_PATH = Path(os.environ.get("LLM_BRIDGE_PATH", "/run/bridge/llm_service.sock"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("llm_service")


class Handler(socketserver.StreamRequestHandler):
    """Reads newline-delimited JSON messages and writes one JSON reply per message."""

    def handle(self) -> None:
        for line in self.rfile:
            line = line.strip()
            if not line:
                continue
            try:
                command = json.loads(line).get("command")
                log.info("bridge message: %s", line.decode(errors="replace"))
                if command == "ping":
                    reply = {"ok": True, "reply": "pong"}
                elif command in ("start", "stop"):
                    reply = {"ok": True}
                else:
                    reply = {"ok": False, "error": f"unknown command: {command!r}"}
            except (ValueError, AttributeError) as exc:
                reply = {"ok": False, "error": str(exc)}
            self.wfile.write(json.dumps(reply).encode() + b"\n")


class Server(socketserver.ThreadingUnixStreamServer):
    daemon_threads = True


def main() -> None:
    BRIDGE_PATH.parent.mkdir(parents=True, exist_ok=True)
    BRIDGE_PATH.unlink(missing_ok=True)  # stale socket from a previous run
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    with Server(str(BRIDGE_PATH), Handler) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        log.info("listening for bridge messages on %s", BRIDGE_PATH)
        stop.wait()
        server.shutdown()
    BRIDGE_PATH.unlink(missing_ok=True)
    log.info("stopped")


if __name__ == "__main__":
    main()
