import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from common.io import done_marker, pcm_to_mp3, ts_to_mp4, write_done, write_json
from common.log import get_logger
from server.media import AudioListener, MediaServer, VideoListener

log = get_logger(__name__)

# The subfolders every session has.
_SUBDIRS = ("audio", "transcript", "video", "extractions")
# The recordings of a session's stream, relative to its folder.
_RECORDINGS = ("audio/audio.pcm", "video/video.ts")
# What each recording is exported to once it is complete, next to it.
_EXPORTS = {"audio/audio.pcm": "audio/audio.mp3", "video/video.ts": "video/video.mp4"}


def abort_unfinished(sessions_dir: Union[str, Path]) -> list[str]:
    """Marks the recordings a previous run of the server left unfinished complete, as
    {"status": "aborted"}; returns the ids of the sessions it marked.

    Called when the server starts, so no session is being recorded: a session (a folder
    with session.json) whose audio.pcm or video.ts has no done marker was cut off by a crash
    or a kill. Without the markers the services would wait for the rest of its recordings
    until they gave up on it.
    """
    aborted = []
    try:
        folders = sorted(Path(sessions_dir).iterdir())
    except FileNotFoundError:
        return aborted
    for folder in folders:
        if not (folder / "session.json").is_file():
            continue
        unmarked = [folder / name for name in _RECORDINGS if not done_marker(folder / name).exists()]
        for path in unmarked:
            write_done(path, "aborted")
        if unmarked:
            aborted.append(folder.name)
            log.warning("session %s was left recording by a previous run: marked %s aborted",
                        folder.name, ", ".join(path.name for path in unmarked))
    return aborted


def export(folder: Union[str, Path], sampling_rate: int, skip_existing: bool = False) -> bool:
    """Converts a session's recordings to audio/audio.mp3 and video/video.mp4, each written
    atomically (see common.io), so a file that exists is complete. Call once recording has
    stopped. skip_existing leaves an export that already exists alone.

    Tries both even if the first fails. True if both succeeded (or were skipped); failures are
    logged.
    """
    folder = Path(folder)
    audio_pcm, audio_mp3 = folder / "audio" / "audio.pcm", folder / "audio" / "audio.mp3"
    video_ts, video_mp4 = folder / "video" / "video.ts", folder / "video" / "video.mp4"
    audio = (skip_existing and audio_mp3.exists()) or pcm_to_mp3(audio_pcm, audio_mp3, sampling_rate, log)
    video = (skip_existing and video_mp4.exists()) or ts_to_mp4(video_ts, video_mp4, log)
    return audio and video


def unexported(sessions_dir: Union[str, Path]) -> list[Path]:
    """The sessions with a complete recording that hasn't been exported: one that is marked
    complete, isn't empty, and has no export next to it.

    Called when the server starts (after abort_unfinished), for the sessions a previous run
    didn't export: cut off by a crash, killed while exporting, or still recording when the
    server was stopped.
    """
    try:
        folders = sorted(Path(sessions_dir).iterdir())
    except FileNotFoundError:
        return []
    out = []
    for folder in folders:
        if not (folder / "session.json").is_file():
            continue
        for recording, exported in _EXPORTS.items():
            path = folder / recording
            try:
                recorded = done_marker(path).exists() and path.stat().st_size > 0
            except OSError:  # no recording
                continue
            if recorded and not (folder / exported).exists():
                out.append(folder)
                break
    return out


class Session:
    """One session: its folder on disk and the recording of its stream.

    Sessions live at <sessions_dir>/<random_8_char_id>/ with audio, transcript,
    video and extractions subdirectories. Create one with Session.create().

    create() writes session.json ({"id", "created", "services"}), which marks the folder as a
    session for the services it names; they find it on disk.
    start() begins recording the session's stream to audio/audio.pcm and
    video/video.ts (written directly by ffmpeg, see AudioListener and
    VideoListener), stop() ends it and marks both files complete (audio.pcm.done,
    video.ts.done; see common.io.done_marker). export() (this module's) then converts the
    raw files to audio.mp3 and video.mp4; the Coordinator runs it in the background. A
    session is used once: start, then stop.
    """

    def __init__(self, session_id: str, path: Path, media: MediaServer, sampling_rate: int):
        self.id = session_id
        self.path = path
        stream_url = media.read_url(session_id)
        self._audio = AudioListener(
            stream_url, media.client, session_id, self.audio_pcm, sampling_rate=sampling_rate
        )
        self._video = VideoListener(stream_url, media.client, session_id, self.video_ts)

    @classmethod
    def create(
        cls, sessions_dir: Union[str, Path], media: MediaServer, sampling_rate: int,
        services: Optional[list[str]] = None,
    ) -> "Session":
        """Makes a new session folder, with a unique random id, under sessions_dir, for the
        named services (None: every service)."""
        sessions_dir = Path(sessions_dir)
        sessions_dir.mkdir(parents=True, exist_ok=True)
        while True:
            session_id = secrets.token_hex(4)
            path = sessions_dir / session_id
            try:
                path.mkdir()
            except FileExistsError:
                continue
            break
        for name in _SUBDIRS:
            (path / name).mkdir()
        # Last, so a session.json names a folder that is ready.
        info = {"id": session_id, "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        if services is not None:
            info["services"] = list(services)
        write_json(path / "session.json", info)
        return cls(session_id, path, media, sampling_rate)

    @property
    def audio_dir(self) -> Path:
        return self.path / "audio"

    @property
    def transcript_dir(self) -> Path:
        return self.path / "transcript"

    @property
    def video_dir(self) -> Path:
        return self.path / "video"

    @property
    def extractions_dir(self) -> Path:
        return self.path / "extractions"

    @property
    def audio_pcm(self) -> Path:
        return self.audio_dir / "audio.pcm"

    @property
    def audio_mp3(self) -> Path:
        return self.audio_dir / "audio.mp3"

    @property
    def video_ts(self) -> Path:
        return self.video_dir / "video.ts"

    @property
    def video_mp4(self) -> Path:
        return self.video_dir / "video.mp4"

    def start(self) -> None:
        """Starts recording the session's stream."""
        self._audio.start()
        self._video.start()

    def stop(self) -> None:
        """Stops recording, then marks audio.pcm and video.ts complete. Safe to call more than once.

        The markers come after the recorders have stopped (each stop() waits for its ffmpeg),
        so they are written after the last byte, and before the export, which can take a while:
        the services finish the session without waiting for it. Written even if a file was
        never created (no stream was published), so the services stop waiting for it.
        """
        try:
            self._audio.stop()
        finally:
            try:
                self._video.stop()
            finally:
                write_done(self.audio_pcm)
                write_done(self.video_ts)
