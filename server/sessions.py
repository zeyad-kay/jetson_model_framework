import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

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
    """Owns the server's session storage: one active session and its folders.

    Sessions live at <sessions_dir>/<random_8_char_id>/ with audio, transcript,
    video and structured subdirectories. The audio and video files are written
    directly by ffmpeg (see AudioListener and VideoListener), not by this class.
    Starting a session closes the previous one. Not thread-safe.
    """

    def __init__(self, sessions_dir: Union[str, Path]):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self._session: Optional[Session] = None

    @property
    def session(self) -> Optional[Session]:
        return self._session

    def start(self) -> Session:
        """Creates a new session folder and makes it active."""
        self._close()
        session = self._create_dirs()
        self._session = session
        return session

    def stop(self) -> None:
        """Closes the active session. A no-op if there is none."""
        self._close()

    def _close(self) -> None:
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
