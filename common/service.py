import logging
import queue
import signal
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, Union

from common.bridge import Bridge


class WorkerService(ABC):
    """What every service (asr, llm, video) offers the server: work on one session at a time.

    The server drives it over the bridge with queue_session, end_session and ping messages;
    run() is the service's main loop and blocks until shutdown().

    run() loads the model once (_load_model()), then idles until queue_session()
    names a session and runs _process_session() for it on a worker thread.
    The session ends at end_session(), queue_session() of the next session, or shutdown();
    the model's finish() is then called until the worker returns, so the
    input already recorded is processed before the next session begins.
    Subclasses implement _load_model() and _process_session().
    """

    def __init__(self, sessions_dir: Union[str, Path], poll_interval: float, log: logging.Logger):
        self._sessions_dir = Path(sessions_dir)
        self._poll_interval = poll_interval
        self._log = log
        self._sessions = queue.Queue()  # session ids waiting to be processed; None = shut down
        self._end_session_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._model = None

    @abstractmethod
    def _load_model(self) -> Any:
        """Loads and returns the model. It must have finish(), which asks the
        work in progress to process the input already recorded, then end."""

    @abstractmethod
    def _process_session(self, session_dir: Path) -> None:
        """Processes one session's input until it is drained. Runs on a worker thread.

        Exceptions are logged by the caller. Use _wait_for_file() to wait for
        an input the recorder has not created yet.
        """

    def queue_session(self, session_id: str) -> None:
        """Queues a session; any session in progress is finished first.

        Raises ValueError if session_id isn't an existing session folder.
        Safe to call from any thread.
        """
        if Path(session_id).name != session_id or not self._session_dir(session_id).is_dir():
            raise ValueError(f"unknown session: {session_id!r}")
        # Order matters: run() clears the flag after taking an id off the queue.
        self._end_session_event.set()
        self._sessions.put(session_id)

    def end_session(self) -> None:
        """Ends the current session; the service keeps running and waits for the next.

        The session's remaining input is processed first. Safe from any thread.
        """
        self._end_session_event.set()

    def shutdown(self) -> None:
        """Finishes the current session and makes run() return. Safe from any thread or signal handler."""
        self._shutdown_event.set()
        self._end_session_event.set()
        self._sessions.put(None)

    def run(self) -> None:
        """Runs the service until shutdown()."""
        self._model = self._load_model()
        self._log.info("waiting for a session")
        while True:
            session_id = self._sessions.get()
            if session_id is None or self._shutdown_event.is_set():
                return
            self._end_session_event.clear()
            self._supervise_session(session_id)
            self._log.info("waiting for a session")

    def _session_dir(self, session_id: str) -> Path:
        return self._sessions_dir / session_id

    def _supervise_session(self, session_id: str) -> None:
        self._log.info("session %s started", session_id)
        session_dir = self._session_dir(session_id)

        def worker_main() -> None:
            try:
                self._process_session(session_dir)
            except Exception:
                self._log.exception("session %s failed", session_id)

        worker = threading.Thread(target=worker_main, daemon=True)
        worker.start()
        self._end_session_event.wait()
        # Let the model process all the input it has not processed yet, however
        # long that takes. finish() is a no-op if the model hasn't started yet,
        # so repeat it until the worker exits.
        while worker.is_alive():
            self._model.finish()
            worker.join(timeout=0.5)
        self._log.info("session %s finished", session_id)

    def _wait_for_file(self, path: Path) -> bool:
        """Blocks until path exists; returns False if the session ended first.

        The recorder creates its files shortly after the session starts.
        """
        while not path.exists():
            if self._end_session_event.wait(self._poll_interval):
                return path.exists()  # it may have appeared just as the session ended
        return True


def make_message_handler(service: WorkerService, log: logging.Logger) -> Callable[[dict], dict]:
    """Builds the bridge callback: dispatches {"command": ...} messages to the service."""

    def cmd_queue_session(message: dict) -> dict:
        session_id = message.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            return {"ok": False, "error": "queue_session needs a session_id"}
        try:
            service.queue_session(session_id)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True}

    def cmd_end_session(message: dict) -> dict:
        service.end_session()
        return {"ok": True}

    def cmd_ping(message: dict) -> dict:
        return {"ok": True, "reply": "pong"}

    # Every bridge command we accept: name -> (description, handler).
    commands = {
        "queue_session": ("Work on the session named by session_id, ending the current one first.", cmd_queue_session),
        "end_session": ("Finish the current session, then wait for the next.", cmd_end_session),
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
