import secrets
from pathlib import Path
from typing import Union

from common.io import pcm_to_mp3, ts_to_mp4
from server.log import get_logger
from server.media import AudioListener, MediaServer, VideoListener

log = get_logger(__name__)

# The subfolders every session has.
_SUBDIRS = ("audio", "transcript", "video", "structured")


class Session:
    """One session: its folder on disk and the recording of its stream.

    Sessions live at <sessions_dir>/<random_8_char_id>/ with audio, transcript,
    video and structured subdirectories. Create one with Session.create().

    start() begins recording the session's stream to audio/audio.pcm and
    video/video.ts (written directly by ffmpeg, see AudioListener and
    VideoListener), stop() ends it, and export() then converts the raw files to
    audio.mp3 and video.mp4. A session is used once: start, stop, export.
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
        cls, sessions_dir: Union[str, Path], media: MediaServer, sampling_rate: int
    ) -> "Session":
        """Makes a new session folder, with a unique random id, under sessions_dir."""
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
    def structured_dir(self) -> Path:
        return self.path / "structured"

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
        """Stops recording. Safe to call more than once."""
        self._audio.stop()
        self._video.stop()

    def export(self) -> bool:
        """Converts the raw recordings to mp3 and mp4. Call after stop().

        Tries both even if the first fails. True if both succeeded; failures are logged.
        """
        audio = pcm_to_mp3(self.audio_pcm, self.audio_mp3, self._sampling_rate, log)
        video = ts_to_mp4(self.video_ts, self.video_mp4, log)
        return audio and video
