import json
import socket
import threading
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Optional, Union

from server.log import get_logger

log = get_logger(__name__)


class Service:
    """A handle on one other service, reached through its bridge socket.

    The name is the socket file's stem (asr_service.sock -> "asr_service").
    `up` is the result of the last ping, or None before the first one.
    """

    def __init__(self, path: Union[str, Path], timeout: float):
        self.path = Path(path)
        self.name = self.path.stem
        self.timeout = timeout
        self.up: Optional[bool] = None

    def send(self, message: dict) -> Optional[dict]:
        """Sends one JSON message and returns the reply, or None on failure."""
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.timeout)
                sock.connect(str(self.path))
                sock.sendall(json.dumps(message).encode() + b"\n")
                return json.loads(sock.makefile("rb").readline())
        except (OSError, ValueError) as exc:
            log.debug("message to %s failed: %s", self.name, exc)
            return None

    def start(self) -> bool:
        """Checks that the service is running and answering pings."""
        return self.ping()

    def start_session(self, session_id: str) -> bool:
        """Tells the service to start working on a session. True if it replied ok."""
        reply = self.send({"command": "start", "session_id": session_id})
        return self._ok("start", reply)

    def stop_session(self) -> bool:
        """Tells the service to stop its current session. True if it replied ok."""
        return self._ok("stop", self.send({"command": "stop"}))

    def _ok(self, command: str, reply: Optional[dict]) -> bool:
        ok = reply is not None and reply.get("ok") is True
        if not ok:
            log.warning("%s: %s failed (%s)", self.name, command, reply)
        return ok

    def ping(self) -> bool:
        """Pings the service, records the result in `up`, and logs when it changes."""
        reply = self.send({"command": "ping"})
        up = reply is not None and reply.get("ok") is True and reply.get("reply") == "pong"
        if self.up != up:
            self.up = up
            if up:
                log.info("%s is up", self.name)
            else:
                log.warning("%s is down", self.name)
        return up


class ServiceManager:
    """Owns every Service: starting and stopping them together and monitoring their health.

    If monitor is True, start_services also starts pinging every service in the
    background and stop_services stops it.
    """

    def __init__(
        self,
        paths: Iterable[Union[str, Path]],
        timeout: float,
        ping_interval: float,
        monitor: bool = True,
    ):
        self._services: dict[str, Service] = {}
        for path in paths:
            service = Service(path, timeout)
            if service.name in self._services:
                log.warning("ignoring %s: a service named %r already exists", path, service.name)
                continue
            self._services[service.name] = service
        self.ping_interval = ping_interval
        self.monitor = monitor
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def __getitem__(self, name: str) -> Service:
        return self._services[name]

    def __iter__(self) -> Iterator[Service]:
        return iter(self._services.values())

    def status(self) -> dict[str, Optional[bool]]:
        """Returns name -> last known up/down (None if not checked yet)."""
        return {service.name: service.up for service in self}

    def send(self, message: str, service_name: str) -> Optional[dict]:
        """Sends `message` as a command to the named service and returns its reply.

        Returns None if there is no such service or it doesn't answer.
        """
        service = self._services.get(service_name)
        if service is None:
            log.warning("no service named %r (have: %s)", service_name, ", ".join(self._services))
            return None
        return service.send({"command": message})

    def start(self) -> dict[str, bool]:
        """Makes sure every service is running and answers a ping. Returns name -> up.

        Also starts monitoring if the monitor flag is set.
        """
        up = {service.name: service.start() for service in self}
        if self.monitor:
            self.start_monitoring()
        return up

    def stop(self) -> dict[str, bool]:
        """Stops the current session on every service, and monitoring if it is on."""
        delivered = self.stop_session()
        if self.monitor:
            self.stop_monitoring()
        return delivered

    def start_session(self, session_id: str) -> dict[str, bool]:
        """Starts the session on every service. Returns name -> delivered."""
        return {service.name: service.start_session(session_id) for service in self}

    def stop_session(self) -> dict[str, bool]:
        """Stops the current session on every service. Returns name -> delivered."""
        return {service.name: service.stop_session() for service in self}

    def start_monitoring(self) -> None:
        """Pings every service each ping_interval in a background thread. A no-op if already running."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._monitor, name="monitor", daemon=True)
        self._thread.start()

    def stop_monitoring(self) -> None:
        """Stops the monitor thread. A no-op if it isn't running."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _monitor(self) -> None:
        while True:
            for service in self:
                service.ping()
            if self._stop.wait(self.ping_interval):
                return
