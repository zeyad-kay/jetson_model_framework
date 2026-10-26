import json
import re
import subprocess
import threading
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional, Union
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


class _FfmpegReader:
    """Base for running ffmpeg against a MediaMTX stream and writing straight to disk.

    ffmpeg owns the write to disk entirely (Python never touches the decoded
    bytes). The publisher usually isn't connected yet when the session starts,
    so this polls the Control API's `is_streaming` until a publisher is
    actually live before launching ffmpeg - avoiding ffmpeg's own
    failed-connect/retry churn (a full process spawned just to hit a 404).
    There's no retry after that: a mid-session disconnect is assumed not to
    happen, so ffmpeg runs once for the life of the reader.

    stop() always ends ffmpeg with a hard terminate (no graceful shutdown to
    wait on), so subclasses must pass `-flush_packets 1` in their command:
    without it, ffmpeg buffers its output and a short session can be killed
    before anything is ever written, leaving an empty file.
    """

    def __init__(self, url: str, media: "MediaServerClient", path: str, poll_interval: float = 0.1):
        self.url = url
        self._media = media
        self._path = path
        self._poll_interval = poll_interval
        self._stopped = threading.Event()
        self._proc: Optional[subprocess.Popen] = None
        self._last_error: Optional[str] = None
        self._thread: Optional[threading.Thread] = None

    def _build_cmd(self) -> list[str]:
        """The ffmpeg command to run. Subclasses fill in the output options."""
        raise NotImplementedError

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=type(self).__name__, daemon=True)
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
        while not self._stopped.is_set() and not self._media.is_streaming(self._path):
            self._stopped.wait(self._poll_interval)
        if self._stopped.is_set():
            return
        self._proc = subprocess.Popen(
            self._build_cmd(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        log.debug("ffmpeg reading %s (pid %d)", self.url, self._proc.pid)
        if self._stopped.is_set():  # stop() ran before _proc was set
            self._proc.terminate()
        threading.Thread(target=self._log_errors, args=(self._proc,), daemon=True).start()
        self._proc.wait()

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


class AudioReader(_FfmpegReader):
    """Reads a MediaMTX stream's audio, decoded to raw 16-bit mono PCM, into `output_path`."""

    def __init__(
        self,
        url: str,
        media: "MediaServerClient",
        path: str,
        output_path: Union[str, Path],
        sample_rate: int = 16000,
    ):
        super().__init__(url, media, path)
        self.output_path = Path(output_path)
        self._sample_rate = sample_rate

    def _build_cmd(self) -> list[str]:
        return [
            "ffmpeg", "-loglevel", "error", "-y",
            "-rtsp_transport", "tcp",
            "-i", self.url,
            "-vn", "-f", "s16le", "-ac", "1", "-ar", str(self._sample_rate),
            "-flush_packets", "1",
            str(self.output_path),
        ]


class VideoReader(_FfmpegReader):
    """Reads a MediaMTX stream's video into `output_path`, copying the encoded
    packets as-is (no decoding/re-encoding).

    MPEG-TS is used as the container regardless of `output_path`'s extension,
    since it needs no trailer/index to be valid, unlike e.g. MP4 - so a file
    terminated mid-stream (publisher drop, reconnect, session stop) is still
    playable up to the point it was cut off.
    """

    def __init__(
        self,
        url: str,
        media: "MediaServerClient",
        path: str,
        output_path: Union[str, Path],
    ):
        super().__init__(url, media, path)
        self.output_path = Path(output_path)

    def _build_cmd(self) -> list[str]:
        return [
            "ffmpeg", "-loglevel", "error", "-y",
            "-rtsp_transport", "tcp",
            "-i", self.url,
            "-an", "-c:v", "copy", "-flush_packets", "1", "-f", "mpegts",
            str(self.output_path),
        ]

# ffmpeg -rtsp_transport tcp -i rtsp://<hotspot_ip>:8554/<session_id> -c:v libx265 -crf 28 -c:a copy -movflags frag_keyframe+empty_moof+default_base_moof output.mp4