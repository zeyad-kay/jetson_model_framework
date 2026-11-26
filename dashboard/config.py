from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from common.config import env_getter, shared_paths


@dataclass(frozen=True)
class Config:
    # Root of the shared sessions volume; each session is <sessions_dir>/<session_id>/.
    sessions_dir: str = "/sessions"
    # Address and port the dashboard's web page is served on.
    host: str = "0.0.0.0"
    port: int = 8080
    # A session that is still recording or processing counts as stalled once none of its
    # files has changed for this many seconds.
    live_window: float = 120.0
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @classmethod
    def load(cls, env_file: Optional[Union[str, Path]] = None) -> "Config":
        """Loads settings from a .env file, overridden by real environment variables.

        The file is `env_file`, else $ENV_FILE, else ./.env. SESSIONS_DIR, LOGS_DIR and
        LOG_LEVEL are shared; the rest are DASHBOARD_-prefixed.
        """
        get = env_getter(env_file)
        defaults = cls()
        return cls(
            **shared_paths(get, "dashboard", defaults),
            host=get("DASHBOARD_HOST", str, defaults.host),
            port=get("DASHBOARD_PORT", int, defaults.port),
            live_window=get("DASHBOARD_LIVE_WINDOW", float, defaults.live_window),
        )
