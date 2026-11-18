"""The dashboard's web server: the page in static/ and a read-only JSON view of the sessions.

    GET /                         the page (static/index.html)
    GET /static/<file>            its scripts and styles
    GET /api/sessions             [{id, started, updated, live, has_epcr}], newest first
    GET /api/sessions/<id>        {id, started, updated, live, epcr, wounds}, where epcr is built
                                  from <session>/structured/ (<category>.json and
                                  medication_administrations.json, see epcr.py), null until the
                                  extraction service has taken in its first segment, and wounds
                                  lists the wound images a model has predicted,
                                  <session>/video/wounds/*.png, oldest first:
                                  [{name, time, url}] (time: the file's modification time)
    GET /api/sessions/<id>/wounds/<name>.png
                                  one of those images

Times are Unix seconds. /api/sessions/<id> carries an ETag, so a page polling it gets
304 Not Modified until the session changes. Nothing is ever written.
"""

import json
import mimetypes
import os
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from common.log import get_logger
from dashboard.epcr import build, fingerprint

log = get_logger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Files whose changes mean the session is still being recorded or processed.
ACTIVITY_FILES = ("audio/audio.pcm", "video/video.ts", "transcript/segments.json",
                  "structured/progress.json")
# Files written once a session is over: the server's exports at stop_session, and the
# extraction service's mark that it has extracted the last segment.
END_FILES = ("audio/audio.mp3", "video/video.mp4", "structured/extractions.done")
# Where a model writes its predicted wound images, inside a session's folder.
WOUNDS_DIR = Path("video") / "wounds"


def mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


class Sessions:
    """Read-only access to the session folders under sessions_dir."""

    def __init__(self, sessions_dir: Path, live_window: float):
        self.sessions_dir = sessions_dir
        self.live_window = live_window

    def folder(self, session_id: str) -> Path | None:
        """The session's folder, or None if session_id isn't a session (or tries to leave sessions_dir)."""
        if not session_id or Path(session_id).name != session_id or session_id.startswith("."):
            return None
        path = self.sessions_dir / session_id
        return path if path.is_dir() else None

    def summary(self, path: Path) -> dict:
        times = [t for t in (mtime(path / f) for f in ACTIVITY_FILES) if t is not None]
        started = mtime(path)  # the server creates the folder and its subfolders when the session starts
        updated = max(times) if times else started
        ended = any((path / f).exists() for f in END_FILES)
        live = not ended and updated is not None and time.time() - updated <= self.live_window
        return {
            "id": path.name,
            "started": started,
            "updated": updated,
            "live": live,
            "has_epcr": (path / "structured" / "progress.json").is_file(),
        }

    def list(self) -> list[dict]:
        try:
            folders = [p for p in self.sessions_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
        except FileNotFoundError:
            return []
        return sorted((self.summary(p) for p in folders), key=lambda s: s["started"] or 0, reverse=True)

    def wounds(self, path: Path) -> "list[dict]":  # quoted: list is a method of this class
        """The session's wound images, oldest first, each {name, time, url}; the url changes
        when the file does, so the page never shows a stale cached copy."""
        try:
            files = [f for f in (path / WOUNDS_DIR).iterdir()
                     if f.suffix.lower() == ".png" and not f.name.startswith(".") and f.is_file()]
        except OSError:
            return []
        out = []
        for f in files:
            t = mtime(f)
            if t is not None:
                out.append({"name": f.name, "time": t,
                            "url": f"/api/sessions/{quote(path.name)}/wounds/{quote(f.name)}?v={int(t * 1000)}"})
        return sorted(out, key=lambda w: (w["time"], w["name"]))

    def wound_file(self, path: Path, name: str) -> Path | None:
        """A wound image of the session, or None if name isn't one (or tries to leave the folder)."""
        if Path(name).name != name or name.startswith(".") or not name.lower().endswith(".png"):
            return None
        file = path / WOUNDS_DIR / name
        return file if file.is_file() else None

    def detail(self, path: Path) -> tuple[dict, str]:
        """The session's summary with its ePCR and wound images, and an ETag for it."""
        structured = path / "structured"
        summary = self.summary(path)
        epcr = build(structured)
        wounds = self.wounds(path)
        summary.pop("has_epcr")
        changes = (fingerprint(structured), summary["live"], tuple((w["name"], w["time"]) for w in wounds))
        etag = f'"{hash(changes) & 0xFFFFFFFFFFFF:x}"'
        return {**summary, "epcr": epcr, "wounds": wounds}, etag


def make_handler(sessions: Sessions) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "dashboard"

        def do_GET(self) -> None:
            path = unquote(urlsplit(self.path).path)
            if path in ("/", "/index.html"):
                return self.send_static("index.html")
            if path.startswith("/static/"):
                return self.send_static(path[len("/static/"):])
            if path == "/api/sessions":
                return self.send_json(sessions.list())
            if path.startswith("/api/sessions/"):
                parts = path[len("/api/sessions/"):].split("/")
                folder = sessions.folder(parts[0])
                if folder is None:
                    return self.send_error(HTTPStatus.NOT_FOUND, "no such session")
                if len(parts) == 3 and parts[1] == "wounds":
                    file = sessions.wound_file(folder, parts[2])
                    if file is None:
                        return self.send_error(HTTPStatus.NOT_FOUND, "no such wound image")
                    return self.send_file(file)
                if len(parts) != 1:
                    return self.send_error(HTTPStatus.NOT_FOUND)
                body, etag = sessions.detail(folder)
                if self.headers.get("If-None-Match") == etag:
                    self.send_response(HTTPStatus.NOT_MODIFIED)
                    self.send_header("ETag", etag)
                    self.end_headers()
                    return
                return self.send_json(body, etag)
            self.send_error(HTTPStatus.NOT_FOUND)

        def send_static(self, name: str) -> None:
            file = (STATIC_DIR / name).resolve()
            if STATIC_DIR not in file.parents or not file.is_file():
                return self.send_error(HTTPStatus.NOT_FOUND)
            self.send_file(file)

        def send_file(self, file: Path) -> None:
            try:
                data = file.read_bytes()
            except OSError:  # removed since it was listed
                return self.send_error(HTTPStatus.NOT_FOUND)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(file.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, obj, etag: str | None = None) -> None:
            data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            if etag:
                self.send_header("ETag", etag)
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, fmt: str, *args) -> None:
            log.debug("%s %s", self.address_string(), fmt % args)  # the page polls; keep INFO quiet

    return Handler


def serve(sessions_dir: str | os.PathLike, host: str, port: int, live_window: float) -> ThreadingHTTPServer:
    """Builds the HTTP server; call serve_forever() on the result to run it."""
    httpd = ThreadingHTTPServer((host, port), make_handler(Sessions(Path(sessions_dir), live_window)))
    httpd.daemon_threads = True
    return httpd
