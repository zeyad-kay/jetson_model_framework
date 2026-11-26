from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from common.config import env_getter, service_settings, shared_paths


@dataclass(frozen=True)
class Config:
    # Root of the shared sessions volume; each session is <sessions_dir>/<session_id>/.
    sessions_dir: str = "/sessions"
    # Where the service writes its heartbeat, which the server reads.
    status_path: Optional[str] = "/run/status/video_service.json"
    # Sessions created longer ago than this (seconds) are never picked up.
    max_session_age: float = 24 * 3600
    # Seconds without a new input or any change in a session's folder, while its input isn't
    # marked complete, before the service gives up on it.
    abandon_timeout: float = 1800.0
    # Text prompt naming what SAM 3 segments.
    prompt: str = "wound"
    # Seconds to wait for the video stream to become probeable.
    probe_timeout: float = 60.0
    # Minimum detection score (0-1) of an object for its frame to be saved.
    min_score: float = 0.9
    # JPEG quality of the saved frames, 1-95.
    jpeg_quality: int = 90
    poll_interval: float = 0.05
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @classmethod
    def load(cls, env_file: Optional[Union[str, Path]] = None) -> "Config":
        """Loads settings from a .env file, overridden by real environment variables.

        The file is `env_file`, else $ENV_FILE, else ./.env. The directories (SESSIONS_DIR, LOGS_DIR, STATUS_DIR) and LOG_LEVEL
        are shared; the rest are VIDEO_-prefixed.
        """
        get = env_getter(env_file)
        defaults = cls()
        return cls(
            **service_settings(get, "video_service", defaults),
            **shared_paths(get, "video", defaults),
            prompt=get("VIDEO_PROMPT", str, defaults.prompt),
            probe_timeout=get("VIDEO_PROBE_TIMEOUT", float, defaults.probe_timeout),
            min_score=get("VIDEO_MIN_SCORE", float, defaults.min_score),
            jpeg_quality=get("VIDEO_JPEG_QUALITY", int, defaults.jpeg_quality),
            poll_interval=get("VIDEO_POLL_INTERVAL", float, defaults.poll_interval),
        )
