import json
import re
import subprocess
import threading
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Optional
from urllib.parse import quote

from server.log import get_logger

log = get_logger(__name__)

# Connection kinds that MediaMTX can kick, as named in its Control API.
KICKABLE = ("rtspsessions", "rtspsconns", "rtmpconns", "webrtcsessions", "srtconns")


class MediaServerError(Exception):
    """Raised when the MediaMTX Control API can't be reached or rejects a call."""


class MediaServerClient:
    """Thin wrapper over the MediaMTX Control API (v3).

    The API only listens on localhost by default and must be enabled (api: yes).
    """

    def __init__(self, base_url: str = "http://127.0.0.1:9997", timeout: float = 3.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, body: Optional[dict] = None):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(
            f"{self.base_url}/v3/{path}",
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:200]
            raise MediaServerError(f"{method} {path}: HTTP {exc.code} {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise MediaServerError(f"{method} {path}: {exc}") from exc
        return json.loads(raw) if raw else None

    def ping(self) -> bool:
        """True if the Control API answers."""
        try:
            self._request("GET", "paths/list")
            return True
        except MediaServerError as exc:
            log.debug("media server not reachable: %s", exc)
            return False

    def add_path(self, name: str, **config) -> None:
        """Creates a path. `config` takes MediaMTX path settings, e.g. record=True."""
        self._request("POST", f"config/paths/add/{quote(name, safe='')}", config)

    def delete_path(self, name: str) -> None:
        self._request("DELETE", f"config/paths/delete/{quote(name, safe='')}")

    def list_paths(self) -> list[dict]:
        """Runtime state of every path (source, readers, ready...)."""
        return self._request("GET", "paths/list").get("items", [])

    def get_path(self, name: str) -> Optional[dict]:
        """Runtime state of one path, or None if it doesn't exist."""
        try:
            return self._request("GET", f"paths/get/{quote(name, safe='')}")
        except MediaServerError as exc:
            if "HTTP 404" in str(exc):
                return None
            raise

    def is_streaming(self, name: str) -> bool:
        """True if a publisher is connected and the stream is ready."""
        path = self.get_path(name)
        return bool(path and path.get("ready"))

    def kick(self, kind: str, connection_id: str) -> None:
        """Disconnects one client. `kind` is one of KICKABLE."""
        if kind not in KICKABLE:
            raise ValueError(f"unknown connection kind: {kind!r}")
        self._request("POST", f"{kind}/kick/{quote(connection_id, safe='')}")


class StreamReader:
    """Runs ffmpeg to read a stream from MediaMTX and hands its audio to `on_audio`.

    The audio is decoded to raw 16-bit mono PCM. The publisher usually isn't
    connected yet when the session starts, so ffmpeg is restarted until stop()
    is called (this also covers the publisher reconnecting).
    """

    def __init__(
        self,
        url: str,
        on_audio: Callable[[bytes], None],
        sample_rate: int = 16000,
        retry_delay: float = 1.0,
    ):
        self.url = url
        self._on_audio = on_audio
        self._sample_rate = sample_rate
        self._retry_delay = retry_delay
        self._stopped = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._last_error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="stream-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stopped.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _run(self) -> None:
        cmd = [
            "ffmpeg", "-loglevel", "error",
            "-rtsp_transport", "tcp",
            "-i", self.url,
            "-vn", "-f", "s16le", "-ac", "1", "-ar", str(self._sample_rate),
            "pipe:1",
        ]
        while not self._stopped.is_set():
            self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            log.debug("ffmpeg reading %s (pid %d)", self.url, self._proc.pid)
            if self._stopped.is_set():  # stop() ran before _proc was set
                self._proc.terminate()
            threading.Thread(target=self._log_errors, args=(self._proc,), daemon=True).start()
            while chunk := self._proc.stdout.read1(65536):
                self._last_error = None  # streaming fine, so report the next failure
                self._on_audio(chunk)
            self._proc.wait()
            self._stopped.wait(self._retry_delay)

    def _log_errors(self, proc: subprocess.Popen) -> None:
        """Logs ffmpeg's stderr. A repeat of the last error (e.g. while waiting
        for the publisher) is skipped so the retry loop doesn't flood the log."""
        for raw in proc.stderr:
            line = raw.decode(errors="replace").strip()
            # Compare without ffmpeg's memory addresses, which change on every run.
            error = re.sub(r"0x[0-9a-f]+", "0x?", line)
            if line and error != self._last_error:
                self._last_error = error
                log.warning("ffmpeg (%s): %s", self.url, line)
