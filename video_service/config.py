from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from common.config import env_getter


@dataclass(frozen=True)
class Config:
    # Root of the shared sessions volume; each session is <sessions_dir>/<session_id>/.
    sessions_dir: str = "/sessions"
    bridge_path: str = "/run/bridge/video_service.sock"
    log_level: str = "INFO"

    @classmethod
    def load(cls, env_file: Optional[Union[str, Path]] = None) -> "Config":
        """Loads settings from a .env file, overridden by real environment variables.

        The file is `env_file`, else $VIDEO_ENV_FILE, else ./.env. Every setting
        is a VIDEO_-prefixed upper-case variable, e.g. VIDEO_BRIDGE_PATH.
        """
        get = env_getter("VIDEO_", env_file)
        defaults = cls()
        return cls(
            sessions_dir=get("sessions_dir", str, defaults.sessions_dir),
            bridge_path=get("bridge_path", str, defaults.bridge_path),
            log_level=get("log_level", str, defaults.log_level),
        )
