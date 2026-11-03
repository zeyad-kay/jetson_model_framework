from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from common.config import env_getter, shared_paths


@dataclass(frozen=True)
class Config:
    # Root of the shared sessions volume; each session is <sessions_dir>/<session_id>/.
    sessions_dir: str = "/sessions"
    bridge_path: str = "/run/bridge/video_service.sock"
    # Seconds without a new frame before the video is considered finished.
    idle_timeout: float = 10.0
    # Seconds the file must stop growing, after the session ends, before segmentation finishes.
    drain_idle: float = 1.0
    # Seconds to wait for the video stream to become probeable.
    probe_timeout: float = 60.0
    # Opacity of the mask overlay, 0-1.
    alpha: float = 0.5
    poll_interval: float = 0.05
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @classmethod
    def load(cls, env_file: Optional[Union[str, Path]] = None) -> "Config":
        """Loads settings from a .env file, overridden by real environment variables.

        The file is `env_file`, else $ENV_FILE, else ./.env. The directories (SESSIONS_DIR, LOGS_DIR, BRIDGE_DIR), LOG_LEVEL
        and SAMPLING_RATE are shared; the rest are VIDEO_-prefixed.
        """
        get = env_getter(env_file)
        defaults = cls()
        return cls(
            **shared_paths(get, "video_service", "video", defaults),
            idle_timeout=get("VIDEO_IDLE_TIMEOUT", float, defaults.idle_timeout),
            drain_idle=get("VIDEO_DRAIN_IDLE", float, defaults.drain_idle),
            probe_timeout=get("VIDEO_PROBE_TIMEOUT", float, defaults.probe_timeout),
            alpha=get("VIDEO_ALPHA", float, defaults.alpha),
            poll_interval=get("VIDEO_POLL_INTERVAL", float, defaults.poll_interval),
        )
