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


class Session:
    """One session: its folder on disk and the recording of its stream.

    Sessions live at <sessions_dir>/<random_8_char_id>/ with audio, transcript,
    video and extractions subdirectories. Create one with Session.create().

    create() writes session.json ({"id", "created", "services"}), which marks the folder as a
    session for the services it names; they find it on disk.
    start() begins recording the session's stream to audio/audio.pcm and
    video/video.ts (written directly by ffmpeg, see AudioListener and
    VideoListener), stop() ends it and marks both files complete (audio.pcm.done,
    video.ts.done; see common.io.done_marker), and export() then converts the raw files
    to audio.mp3 and video.mp4. A session is used once: start, stop, export.
    """

    def __init__(self, session_id: str, path: Path, media: MediaServer, sampling_rate: int):
        self.id = session_id
        self.path = path
        self._sampling_rate = sampling_rate
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
        so they are written after the last byte, and before export(), which can take a while:
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

    def export(self) -> bool:
        """Converts the raw recordings to mp3 and mp4. Call after stop().

        Tries both even if the first fails. True if both succeeded; failures are logged.
        """
        audio = pcm_to_mp3(self.audio_pcm, self.audio_mp3, self._sampling_rate, log)
        video = ts_to_mp4(self.video_ts, self.video_mp4, log)
        return audio and video
