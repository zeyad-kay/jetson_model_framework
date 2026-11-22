import json
import logging
import os
import signal
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union

from common.io import done_marker, write_done, write_json


class SessionDrained(Exception):
    """Raised by WorkerService.poll() when the session will give no more input: its reader
    found the input finished, the service gave up waiting for it, or the service is
    shutting down."""


def last_change(folder: Path) -> float:
    """The newest modification time (Unix time) of folder and everything under it; 0 if it doesn't exist."""
    newest = 0.0
    try:
        entries = list(os.scandir(folder))
        newest = folder.stat().st_mtime
    except OSError:
        return newest
    for entry in entries:
        try:
            if entry.is_dir(follow_symlinks=False):
                newest = max(newest, last_change(Path(entry.path)))
            else:
                newest = max(newest, entry.stat(follow_symlinks=False).st_mtime)
        except OSError:
            continue  # removed meanwhile
    return newest


class WorkerService(ABC):
    """What every service (asr, llm, video) runs: the sessions it finds on disk, one at a time.

    run() loads the model once (load_model()), then works through the pending sessions under
    sessions_dir, oldest first, until shutdown(). A session is pending for this service if
    its folder has session.json (the server writes it when it creates the session), that
    file is at most max_session_age seconds old, its "services" list (if any) names this
    service, and the service has not marked its output complete yet (the done marker of
    output_path(), see common.io.done_marker). So a service that was down or restarted picks
    up the sessions it missed, and deleting a session's output marker makes the service redo
    that session.

    Each session runs in the same steps in every service:

        open_reader(session_dir)           the reader of the session's input
        setup(session_dir)                 open the session's outputs, model.start()
        loop:
            inputs = poll()                the reader's next input (a dict), or None if there is none yet
            outputs = predict(inputs)      run the model on it
            save(inputs, outputs)          write what it produced to the session folder
        cleanup()                          close the reader, model.stop(), finalize_outputs() and
                                           write the output marker; always runs

    A service implements load_model, output_path, open_reader, setup and save, and
    finalize_outputs if it has outputs to finish; poll, predict and cleanup are the same for
    every service. The loop waits poll_interval seconds
    after a poll() that found nothing. The session ends when the reader finds the input
    finished: its upstream marked it complete (see common.io.done_marker).

    If the input never completes (whoever wrote it died without marking it), the service
    gives up on the session once the reader has had nothing new and nothing in the
    session's folder has changed for abandon_timeout seconds (session_abandoned):
    cleanup() marks its outputs "abandoned" and the service goes on with the next session.

    An exception in any step ends the session: it is logged ("session <id> failed"),
    cleanup() still runs (and marks the outputs failed), and the service goes on with the
    next session. A session is tried once per run of the service. The session's start and
    end are logged as "session <id> started" and "session <id> finished".

    shutdown() ends the current session once the reader has nothing new, without marking
    its outputs (session_interrupted), so the service redoes it when it runs again.

    If status_path is set, the service writes its heartbeat there every heartbeat_interval
    seconds: {"service", "pid", "state" ("loading", "idle", "working" or "stopped"),
    "session", "updated" (Unix time)}. The server reads it to tell which services are up.
    """

    def __init__(self, name: str, sessions_dir: Union[str, Path], poll_interval: float, log: logging.Logger,
                 status_path: Optional[Union[str, Path]] = None, max_session_age: float = 24 * 3600,
                 abandon_timeout: float = 1800.0, scan_interval: float = 1.0, heartbeat_interval: float = 5.0):
        self.name = name
        self._sessions_dir = Path(sessions_dir)
        self._poll_interval = poll_interval
        self._log = log
        self._status_path = Path(status_path) if status_path else None
        self._max_session_age = max_session_age
        self._abandon_timeout = abandon_timeout
        self._scan_interval = scan_interval  # seconds between looks for a pending session
        self._heartbeat_interval = heartbeat_interval
        self._shutdown_event = threading.Event()
        self._model = None
        self._attempted: set[str] = set()  # sessions this run has worked on: never retried
        self._state = "loading"
        self._session_id: Optional[str] = None
        self._status_lock = threading.Lock()
        # The current session's folder and reader, and how the session went.
        self._current_dir: Optional[Path] = None
        self._reader = None
        self._session_failed = False  # whether a step of the current session raised
        self._input_finished = False  # whether the input will give no more (finished, or given up on)
        self._input_abandoned = False  # whether the service gave up waiting for the input
        self._last_activity = 0.0  # the last new input, or change in the session's folder, seen (Unix time)

    # --- the steps each service implements -------------------------------------------

    @abstractmethod
    def load_model(self) -> Any:
        """Loads and returns the model (a StreamingModel), once at startup. run() keeps it as self._model."""

    @abstractmethod
    def output_path(self, session_dir: Path) -> Path:
        """The session's output (a file or a folder). cleanup() marks it complete once the service
        is done with the session (its output marker, common.io.done_marker of it); the session is
        pending until the marker exists."""

    @abstractmethod
    def open_reader(self, session_dir: Path) -> Any:
        """Returns the reader of the session's input, an object with

            read()     the next input recorded, a dict, or None if there is none yet. Never
                       blocks for long: the loop does the waiting. An input file the recorder
                       has not created yet counts as nothing new. Raises EOFError once the
                       input has finished for good (its upstream marked it complete).
            close()    releases what it holds (processes, files); called before cleanup().
        """

    @abstractmethod
    def setup(self, session_dir: Path) -> None:
        """Prepares a session: resets the per-session state, opens its outputs and calls model.start()."""

    def poll(self) -> Optional[dict]:
        """Returns the reader's next input, or None if it has none yet.

        Raises SessionDrained when the reader finds the input finished (EOFError), when the
        reader has nothing new and the session has been idle for abandon_timeout seconds
        (see session_abandoned), or, once shutdown() has been called, when the reader has
        nothing new.
        """
        stopping = self._shutdown_event.is_set()  # read before the reader, so an input arriving now isn't missed
        try:
            inputs = self._reader.read()
        except EOFError as exc:
            self._input_finished = True
            self._log.info("the session's input has finished: %s", exc)
            raise SessionDrained from exc
        if inputs is not None:
            self._last_activity = time.time()
        elif stopping or self._give_up():
            raise SessionDrained
        return inputs

    def predict(self, inputs: dict) -> Any:
        """Runs the model on one input from poll() and returns its outputs: model.predict(inputs).

        Override it only when a service needs more than its model's predict().
        """
        return self._model.predict(inputs)

    @abstractmethod
    def save(self, inputs: dict, outputs: Any) -> None:
        """Writes one prediction's outputs to the session folder."""

    def finalize_outputs(self) -> None:
        """Finishes the session's outputs before the output marker is written: closes the files
        setup() opened, writes what save() held back. Does nothing unless a service overrides it.

        Called by cleanup(), so it runs after a failure too, including one in open_reader() or
        setup(); it must cope with a setup() that stopped part way. If it raises, the output
        marker is not written.
        """

    def cleanup(self) -> None:
        """Ends the session: calls model.stop(), finalize_outputs() and, unless session_interrupted,
        writes the output marker, {"status": session_status}.

        Always runs, after the reader is closed, and after a failure too, including one
        in open_reader() or setup(), so the session reads as ended even then.
        """
        if self._model is not None:
            self._model.stop()
        self.finalize_outputs()
        if not self.session_interrupted:
            write_done(self.output_path(self._current_dir), self.session_status)

    # --- the session lifecycle, shared by every service --------------------------------

    @property
    def session_failed(self) -> bool:
        """Whether a step of the current session raised; cleanup() marks its outputs so."""
        return self._session_failed

    @property
    def session_abandoned(self) -> bool:
        """Whether the service gave up waiting for the current session's input, which nothing
        marked complete and nothing changed for abandon_timeout seconds; cleanup() marks its
        outputs "abandoned"."""
        return self._input_abandoned

    @property
    def session_status(self) -> str:
        """The status cleanup() writes to the output marker: "failed" (session_failed),
        "abandoned" (session_abandoned) or "ok"."""
        if self._session_failed:
            return "failed"
        return "abandoned" if self._input_abandoned else "ok"

    @property
    def session_interrupted(self) -> bool:
        """Whether shutdown() ended the current session before its input finished; cleanup()
        then leaves the outputs unmarked, so the session is redone when the service runs again."""
        return self._shutdown_event.is_set() and not self._input_finished

    def shutdown(self) -> None:
        """Ends the current session (see session_interrupted) and makes run() return.

        Safe from any thread or signal handler (it takes no lock).
        """
        self._shutdown_event.set()

    def run(self) -> None:
        """Runs the service until shutdown()."""
        heartbeat = threading.Thread(target=self._heartbeat, name="heartbeat", daemon=True)
        heartbeat.start()
        try:
            self._set_state("loading")
            self._model = self.load_model()
            self._set_state("idle")
            self._log.info("waiting for a session")
            while not self._shutdown_event.is_set():
                session_id = self._next_session()
                if session_id is None:
                    self._shutdown_event.wait(self._scan_interval)
                    continue
                self._attempted.add(session_id)
                self._set_state("working", session_id)
                self._run_session(session_id)
                self._set_state("idle")
                self._log.info("waiting for a session")
        finally:
            self._shutdown_event.set()  # stops the heartbeat
            heartbeat.join(timeout=5)
            self._set_state("stopped")

    def _session_dir(self, session_id: str) -> Path:
        return self._sessions_dir / session_id

    def _next_session(self) -> Optional[str]:
        """The oldest pending session (see the class docstring), or None."""
        now = time.time()
        pending = []
        try:
            folders = list(self._sessions_dir.iterdir())
        except FileNotFoundError:
            return None
        for folder in folders:
            if folder.name in self._attempted:
                continue
            info_path = folder / "session.json"
            try:
                created = info_path.stat().st_mtime
            except (FileNotFoundError, NotADirectoryError):
                continue
            if now - created > self._max_session_age or done_marker(self.output_path(folder)).exists():
                continue
            try:
                services = json.loads(info_path.read_text(encoding="utf-8")).get("services")
            except (OSError, ValueError, AttributeError):
                continue  # half-written: next time
            if services is not None and self.name not in services:
                continue
            pending.append((created, folder.name))
        return min(pending)[1] if pending else None

    def _start_session(self, session_id: str) -> Path:
        """Resets the per-session state of the base class; returns the session's folder."""
        self._log.info("session %s started", session_id)
        self._current_dir = self._session_dir(session_id)
        self._reader = None
        self._session_failed = False
        self._input_finished = False
        self._input_abandoned = False
        self._last_activity = time.time()
        return self._current_dir

    def _give_up(self) -> bool:
        """Whether to give up on the current session: neither a new input nor a change in its
        folder for abandon_timeout seconds. Sets session_abandoned if so.

        The folder is only looked at once that long has passed since the last activity seen.
        """
        now = time.time()
        if now - self._last_activity < self._abandon_timeout:
            return False
        self._last_activity = max(self._last_activity, last_change(self._current_dir))
        if now - self._last_activity < self._abandon_timeout:
            return False
        self._input_abandoned = self._input_finished = True
        self._log.warning("giving up on session %s: its input isn't marked complete, and nothing in it "
                          "has changed for %gs", self._current_dir.name, self._abandon_timeout)
        return True

    def _run_session(self, session_id: str) -> None:
        """open_reader and setup, then poll/predict/save until drained, then close the reader and
        cleanup. Logs failures instead of raising."""
        session_dir = self._start_session(session_id)
        try:
            self._reader = self.open_reader(session_dir)
            self.setup(session_dir)
            while True:
                try:
                    inputs = self.poll()
                except SessionDrained:
                    break
                if inputs is None:
                    self._shutdown_event.wait(self._poll_interval)
                    continue
                self.save(inputs, self.predict(inputs))
        except Exception:
            self._session_failed = True
            self._log.exception("session %s failed", session_id)
        finally:
            try:
                if self._reader is not None:
                    self._reader.close()
            except Exception:
                self._log.exception("session %s failed to close its reader", session_id)
            self._reader = None
            try:
                self.cleanup()
            except Exception:
                self._log.exception("session %s failed to clean up", session_id)
        if self.session_interrupted:
            self._log.info("session %s interrupted by shutdown; it is redone on the next run", session_id)
        self._log.info("session %s finished", session_id)

    # --- the heartbeat ------------------------------------------------------------------

    def _set_state(self, state: str, session_id: Optional[str] = None) -> None:
        with self._status_lock:
            self._state, self._session_id = state, session_id
        self._write_status()

    def _write_status(self) -> None:
        if self._status_path is None:
            return
        # Held while writing: the heartbeat thread and a state change share write_json's temp file.
        with self._status_lock:
            status = {"service": self.name, "pid": os.getpid(), "state": self._state,
                      "session": self._session_id, "updated": time.time()}
            try:
                write_json(self._status_path, status)
            except OSError as exc:
                self._log.warning("could not write the status file %s: %s", self._status_path, exc)

    def _heartbeat(self) -> None:
        while not self._shutdown_event.wait(self._heartbeat_interval):
            self._write_status()


def run_service(service: WorkerService) -> None:
    """Runs the service until Ctrl-C or SIGTERM, which end the current session (see
    WorkerService.session_interrupted), then returns."""
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: service.shutdown())
    service.run()
