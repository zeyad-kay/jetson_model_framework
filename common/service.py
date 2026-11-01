import logging
import signal
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Union

from common.bridge import Bridge


class WorkerService(ABC):
    """What every service (asr, llm, video) offers the server: work on one session at a time.

    The server drives it over the bridge with start, stop and ping messages;
    run() is the service's main loop and blocks until shutdown().
    """

    @abstractmethod
    def start(self, session_id: str) -> None:
        """Starts working on a session, finishing any in progress first.

        Raises ValueError if the session is unknown. Safe to call from any thread.
        """

    @abstractmethod
    def stop(self) -> None:
        """Finishes the current session, then waits for the next. Safe from any thread."""

    @abstractmethod
    def shutdown(self) -> None:
        """Finishes the current session and makes run() return. Safe from any thread or signal handler."""

    @abstractmethod
    def run(self) -> None:
        """Runs the service until shutdown()."""


def make_message_handler(service: WorkerService, log: logging.Logger) -> Callable[[dict], dict]:
    """Builds the bridge callback: dispatches {"command": ...} messages to the service."""

    def cmd_start(message: dict) -> dict:
        session_id = message.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            return {"ok": False, "error": "start needs a session_id"}
        try:
            service.start(session_id)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True}

    def cmd_stop(message: dict) -> dict:
        service.stop()
        return {"ok": True}

    def cmd_ping(message: dict) -> dict:
        return {"ok": True, "reply": "pong"}

    # Every bridge command we accept: name -> (description, handler).
    commands = {
        "start": ("Work on the session named by session_id.", cmd_start),
        "stop": ("Finish the current session, then wait for the next.", cmd_stop),
        "ping": ("Health check; replies with 'pong'.", cmd_ping),
    }

    def handle_message(message: dict) -> dict:
        log.info("bridge message: %s", message)
        command = message.get("command")
        entry = commands.get(command)
        if entry is None:
            return {
                "ok": False,
                "error": f"unknown command: {command!r}",
                "commands": {name: desc for name, (desc, _) in commands.items()},
            }
        return entry[1](message)

    return handle_message


def run_service(service: WorkerService, bridge_path: Union[str, Path], log: logging.Logger) -> None:
    """Serves the bridge and runs the service. Ctrl-C and SIGTERM finish the current session, then return."""
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: service.shutdown())
    with Bridge(bridge_path, make_message_handler(service, log), log):
        service.run()
