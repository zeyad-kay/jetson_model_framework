import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from server.io import AudioWriter, VideoWriter
from server.log import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Session:
    id: str
    path: Path

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
    def structured_dir(self) -> Path:
        return self.path / "structured"


class SessionManager:
    """Owns the server's session storage: one active session and its writers.

    Sessions live at <sessions_dir>/<random_8_char_id>/ with audio, transcript,
    video and structured subdirectories. Starting a session closes the previous
    one. Not thread-safe.
    """

    def __init__(self, sessions_dir: Union[str, Path]):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._session: Optional[Session] = None
        self._audio: Optional[AudioWriter] = None
        self._video: Optional[VideoWriter] = None

    @property
    def session(self) -> Optional[Session]:
        return self._session

    def start(self) -> Session:
        """Creates a new session folder, makes it active, and opens its audio and video files."""
        self._close()
        session = self._create_dirs()
        self._audio = AudioWriter(session.audio_dir / "audio.pcm")
        self._video = VideoWriter(session.video_dir / "video.bin")
        self._session = session
        return session

    def write_audio(self, chunk: Union[bytes, bytearray, memoryview]) -> None:
        """Appends a chunk to the active session's audio file."""
        if self._audio is None:
            raise RuntimeError("no active session")
        self._audio.write(chunk)

    def write_video(self, chunk: Union[bytes, bytearray, memoryview]) -> None:
        """Appends a chunk to the active session's video file."""
        if self._video is None:
            raise RuntimeError("no active session")
        self._video.write(chunk)

    def stop(self) -> None:
        """Closes the active session's writers. A no-op if there is none."""
        self._close()

    def _close(self) -> None:
        if self._audio is not None:
            self._audio.close()
            self._audio = None
        if self._video is not None:
            self._video.close()
            self._video = None
        if self._session is not None:
            log.info("closed session %s", self._session.id)
            self._session = None

    def _create_dirs(self) -> Session:
        while True:
            session_id = secrets.token_hex(4)
            path = self.sessions_dir / session_id
            try:
                path.mkdir()
            except FileExistsError:
                continue
            break
        for name in ("audio", "transcript", "video", "structured"):
            (path / name).mkdir()
        return Session(session_id, path)
