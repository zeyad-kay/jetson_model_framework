import json
import threading
import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Optional, Union

from common.log import get_logger

log = get_logger(__name__)


class WorkerServiceClient:
    """What the server knows about one model service: its heartbeat file.

    The service rewrites the file every few seconds (see WorkerService):
    {"service", "pid", "state", "session", "updated"}. The name is the file's stem
    (asr_service.json -> "asr_service"). The service is up if the file was updated within
    status_timeout seconds and its state is "idle" or "working". `up` is the result of the
    last check, or None before the first one.
    """

    def __init__(self, path: Union[str, Path], status_timeout: float):
        self.path = Path(path)
        self.name = self.path.stem
        self._status_timeout = status_timeout
        self.up: Optional[bool] = None

    def status(self) -> Optional[dict]:
        """The service's last heartbeat, or None if it has none (or it can't be read right now)."""
        try:
            status = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return status if isinstance(status, dict) else None

    def check(self) -> bool:
        """Reads the heartbeat, records whether the service is up in `up`, and logs when it changes."""
        status = self.status()
        up = (
            status is not None
            and status.get("state") in ("idle", "working")
            and isinstance(status.get("updated"), (int, float))
            and time.time() - status["updated"] <= self._status_timeout
        )
        if self.up != up:
            self.up = up
            if up:
                log.info("%s is up", self.name)
            else:
                state = status.get("state") if status else "no heartbeat"
                log.warning("%s is down (%s)", self.name, state)
        return up


class WorkerServiceManager:
    """Every WorkerServiceClient: the services the server creates sessions for, and their health.

    If monitor is True, start() also starts checking every service's heartbeat in the
    background, every monitor_interval seconds (logging when one goes up or down), and
    stop() stops it.
    """

    def __init__(
        self,
        paths: Iterable[Union[str, Path]],
        status_timeout: float,
        monitor_interval: float,
        monitor: bool = True,
    ):
        self._services: dict[str, WorkerServiceClient] = {}
        for path in paths:
            service = WorkerServiceClient(path, status_timeout)
            if service.name in self._services:
                log.warning("ignoring %s: a service named %r already exists", path, service.name)
                continue
            self._services[service.name] = service
        self.monitor_interval = monitor_interval
        self.monitor = monitor
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __getitem__(self, name: str) -> WorkerServiceClient:
        return self._services[name]

    def __iter__(self) -> Iterator[WorkerServiceClient]:
        return iter(self._services.values())

    @property
    def names(self) -> list[str]:
        """The services' names, as session.json lists the services a session is for."""
        return list(self._services)

    def status(self) -> dict[str, Optional[bool]]:
        """Returns name -> last known up/down (None if not checked yet)."""
        return {service.name: service.up for service in self}

    def check(self) -> dict[str, bool]:
        """Checks every service's heartbeat now. Returns name -> up."""
        return {service.name: service.check() for service in self}

    def start(self) -> dict[str, bool]:
        """Checks every service's heartbeat. Returns name -> up. Also starts monitoring if the monitor flag is set."""
        up = self.check()
        if self.monitor:
            self.start_monitoring()
        return up

    def stop(self) -> None:
        """Stops monitoring, if it is on."""
        if self.monitor:
            self.stop_monitoring()

    def start_monitoring(self) -> None:
        """Checks every service each monitor_interval in a background thread. A no-op if already running."""
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor, name="monitor", daemon=True)
        self._thread.start()

    def stop_monitoring(self) -> None:
        """Stops the monitor thread. A no-op if it isn't running."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _monitor(self) -> None:
        while True:
            self.check()
            if self._stop_event.wait(self.monitor_interval):
                return
