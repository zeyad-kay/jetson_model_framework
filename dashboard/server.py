"""The dashboard's web server: the page in static/ and a read-only JSON view of the sessions.

    GET /                         the page (static/index.html)
    GET /static/<file>            its scripts and styles
    GET /api/sessions             [{id, started, updated, state, live, waiting, problems, has_epcr}],
                                  newest first (see Sessions.summary)
    GET /api/sessions/<id>        {id, started, updated, state, live, waiting, problems, epcr, wounds,
                                  download}, where epcr is built
                                  from <session>/extractions/ (<category>.json and
                                  medication_administrations.json, see epcr.py), null until the
                                  extraction service has taken in its first segment, and wounds
                                  lists the wound images a model has predicted,
                                  <session>/video/wounds/*.{jpg,jpeg,png}, earliest first:
                                  [{name, time, seconds, url, boxes}] (time: the file's
                                  modification time; seconds: the frame's position in the video,
                                  from a name like 000012.340.jpg, else null; ordered by seconds,
                                  then time; boxes: the url of its <stem>.box, else null)
    GET /api/sessions/<id>/wounds/<name>
                                  one of those images, or a .box file: its bounding boxes, one
                                  "x0 y0 x1 y1" line of image pixels per box
    GET /api/sessions/<id>/download
                                  session-<id>.zip of the session's recordings as the server
                                  exported them when it stopped recording (audio/audio.mp3,
                                  video/video.mp4; those that exist), streamed; 404 until the
                                  session has stopped recording and one exists. The detail's
                                  download is {url, files} then, else null.

A session is a folder with session.json (the server writes it when it creates the session).
Times are Unix seconds. /api/sessions/<id> carries an ETag, so a page polling it gets
304 Not Modified until the session changes. Nothing is ever written.
"""

import json
import mimetypes
import os
import re
import time
import zipfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlsplit

from common.log import get_logger
from dashboard.epcr import build, fingerprint, read_json

log = get_logger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Files (and folders) whose changes mean the session is still being recorded or processed.
ACTIVITY_FILES = ("audio/audio.pcm", "video/video.ts", "transcript/transcript.jsonl",
                  "transcript/segments.json", "extractions/progress.json", "video/wounds")
# What each part of a session writes, and its name on the page. Each is marked complete by a
# done marker next to it, <path>.done = {"status": "ok", "failed", "abandoned" or "aborted"}
# (common.io.done_marker): the recordings by the server when recording stops (or "aborted" on
# its next start, if it was cut off), and each output by its model service when it is done.
RECORDINGS = {"audio/audio.pcm": "audio recording", "video/video.ts": "video recording"}
OUTPUTS = {  # by the service name session.json lists
    "asr_service": ("transcript/segments.json", "transcription"),
    "llm_extraction_service": ("extractions/extractions", "extraction"),
    "video_service": ("video/wounds", "wound detection"),
}
# Where a model writes its predicted wound images, inside a session's folder.
WOUNDS_DIR = Path("video") / "wounds"
WOUND_SUFFIXES = (".jpg", ".jpeg", ".png")
# The recordings the server exports when it stops recording, as the download names them. The
# exports are written atomically (common.io), so one that exists is complete.
EXPORTS = {"audio/audio.mp3": "audio.mp3", "video/video.mp4": "video.mp4"}
VIDEO_SECONDS_NAME = re.compile(r"\d+(\.\d+)?")  # the video service's names: 000012.340
BOX_SUFFIX = ".box"  # an image's bounding boxes, written by the video service next to it
mimetypes.add_type("text/plain", BOX_SUFFIX)


def video_seconds(file: Path) -> float | None:
    """The video position in a wound image's name (000012.340.jpg -> 12.34); None if it has none."""
    # Strict: float() would also take "000000_2" (underscores), "inf" or "1e3".
    return float(file.stem) if VIDEO_SECONDS_NAME.fullmatch(file.stem) else None


def marker_status(path: Path) -> str | None:
    """The status in path's done marker ("ok", "failed", ...); None while there is no marker."""
    marker = path.with_name(path.name + ".done")
    if not marker.exists():
        return None
    data = read_json(marker)
    status = data.get("status") if isinstance(data, dict) else None
    return status if isinstance(status, str) else "unknown"


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
        return path if (path / "session.json").is_file() else None

    @staticmethod
    def outputs(path: Path) -> list[tuple[str, str]]:
        """(path, name) of the outputs the session waits for: those of the services its session.json
        lists, or of every service if it lists none."""
        info = read_json(path / "session.json")
        services = info.get("services") if isinstance(info, dict) else None
        if not isinstance(services, list):
            services = list(OUTPUTS)
        return [OUTPUTS[s] for s in OUTPUTS if s in services]

    def summary(self, path: Path) -> dict:
        """The session at a glance:

            state     "recording" until the server marks both recordings complete, then
                      "processing" until every service the session is for marks its output
                      complete, then "done"; "stalled" instead of recording or processing once
                      nothing has changed for live_window seconds (e.g. a service is down)
            live      whether it is recording or processing
            waiting   the names of what isn't marked complete yet ("extraction", ...)
            problems  [{"what", "status"}] for each marker whose status isn't "ok"
        """
        times = [t for t in (mtime(path / f) for f in ACTIVITY_FILES) if t is not None]
        started = mtime(path)  # the server creates the folder and its subfolders when the session starts
        updated = max(times) if times else started
        recordings = [(name, marker_status(path / f)) for f, name in RECORDINGS.items()]
        outputs = [(name, marker_status(path / f)) for f, name in self.outputs(path)]
        waiting = [name for name, status in recordings + outputs if status is None]
        problems = [{"what": name, "status": status} for name, status in recordings + outputs
                    if status not in (None, "ok")]
        if not waiting:
            state = "done"
        elif updated is not None and time.time() - updated > self.live_window:
            state = "stalled"
        else:
            state = "recording" if any(status is None for _, status in recordings) else "processing"
        return {
            "id": path.name,
            "started": started,
            "updated": updated,
            "state": state,
            "live": state in ("recording", "processing"),
            "waiting": waiting,
            "problems": problems,
            "has_epcr": (path / "extractions" / "progress.json").is_file(),
        }

    def list(self) -> list[dict]:
        try:
            folders = [p for p in self.sessions_dir.iterdir()
                       if not p.name.startswith(".") and (p / "session.json").is_file()]
        except FileNotFoundError:
            return []
        return sorted((self.summary(p) for p in folders), key=lambda s: s["started"] or 0, reverse=True)

    def wounds(self, path: Path) -> "list[dict]":  # quoted: list is a method of this class
        """The session's wound images, earliest in the video first, each {name, time, seconds, url, boxes}.

        seconds is the video position the video service puts in the name (000012.340.jpg), or None
        for another name; those follow, oldest first. The url changes when the file does, so the
        page never shows a stale cached copy."""
        try:
            files = [f for f in (path / WOUNDS_DIR).iterdir()
                     if f.suffix.lower() in WOUND_SUFFIXES and not f.name.startswith(".") and f.is_file()]
        except OSError:
            return []
        out = []
        for f in files:
            t = mtime(f)
            if t is not None:
                out.append({"name": f.name, "time": t, "seconds": video_seconds(f),
                            "url": self.wound_url(path, f, t), "boxes": self.wound_url(path, f.with_suffix(BOX_SUFFIX))})
        return sorted(out, key=lambda w: (w["seconds"] is None, w["seconds"] or 0, w["time"], w["name"]))

    @staticmethod
    def wound_url(path: Path, file: Path, t: float | None = None) -> str | None:
        """The url of one of the session's wound files, or None if it doesn't exist. The url
        carries the file's modification time t, so it changes when the file does."""
        if t is None:
            t = mtime(file)
            if t is None:
                return None
        return f"/api/sessions/{quote(path.name)}/wounds/{quote(file.name)}?v={int(t * 1000)}"

    def wound_file(self, path: Path, name: str) -> Path | None:
        """A wound image or .box file of the session, or None if name isn't one (or tries to leave the folder)."""
        if Path(name).name != name or name.startswith(".") or not name.lower().endswith(WOUND_SUFFIXES + (BOX_SUFFIX,)):
            return None
        file = path / WOUNDS_DIR / name
        return file if file.is_file() else None

    @staticmethod
    def exports(path: Path) -> "list[tuple[Path, str]]":  # quoted: list is a method of this class
        """(file, name in the zip) of the session's exported recordings, once it has stopped
        recording (both recordings marked complete; the server exports them after that).
        Empty before, or if neither was exported."""
        if any(marker_status(path / f) is None for f in RECORDINGS):
            return []
        return [(path / f, name) for f, name in EXPORTS.items() if (path / f).is_file()]

    def detail(self, path: Path) -> tuple[dict, str]:
        """The session's summary with its ePCR, wound images and download, and an ETag for it."""
        extractions = path / "extractions"
        summary = self.summary(path)
        epcr = build(extractions)
        wounds = self.wounds(path)
        exports = [name for _, name in self.exports(path)]
        download = {"url": f"/api/sessions/{quote(path.name)}/download", "files": exports} if exports else None
        summary.pop("has_epcr")
        changes = (fingerprint(extractions), summary["state"], tuple(summary["waiting"]),
                   tuple((p["what"], p["status"]) for p in summary["problems"]),
                   tuple((w["name"], w["time"]) for w in wounds), tuple(exports))
        etag = f'"{hash(changes) & 0xFFFFFFFFFFFF:x}"'
        return {**summary, "epcr": epcr, "wounds": wounds, "download": download}, etag


def write_zip(out, files: "list[tuple[Path, str]]") -> None:
    """Writes a zip of files ((file, name in the zip)) to the stream `out`, which needn't be
    seekable, reading each file in chunks. Stored, not compressed: mp3 and mp4 already are."""
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
        for file, name in files:
            zf.write(file, arcname=name)


def make_handler(sessions: Sessions) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "dashboard"

        def handle(self) -> None:
            # The browser drops a response it no longer wants (an image stepped past before
            # it loaded, a page closed mid-poll); without this each one prints a traceback.
            try:
                super().handle()
            except (BrokenPipeError, ConnectionResetError):
                log.debug("%s closed the connection before the response was sent", self.address_string())

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
                if len(parts) == 2 and parts[1] == "download":
                    files = sessions.exports(folder)
                    if not files:
                        return self.send_error(HTTPStatus.NOT_FOUND, "no recordings to download yet")
                    return self.send_zip(f"session-{folder.name}.zip", files)
                if len(parts) == 3 and parts[1] == "wounds":
                    file = sessions.wound_file(folder, parts[2])
                    if file is None:
                        return self.send_error(HTTPStatus.NOT_FOUND, "no such wound file")
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

        def send_zip(self, filename: str, files: "list[tuple[Path, str]]") -> None:
            """Streams a zip of files as a download. Its length isn't known up front, so the
            response ends when the connection closes (the server speaks HTTP/1.0)."""
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                write_zip(self.wfile, files)
            except (BrokenPipeError, ConnectionResetError):
                log.info("download of %s cancelled", filename)

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
