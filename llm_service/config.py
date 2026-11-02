from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from common.config import env_getter, shared_paths


@dataclass(frozen=True)
class Config:
    # Root of the shared sessions volume; each session is <sessions_dir>/<session_id>/.
    sessions_dir: str = "/sessions"
    bridge_path: str = "/run/bridge/llm_service.sock"
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @classmethod
    def load(cls, env_file: Optional[Union[str, Path]] = None) -> "Config":
        """Loads settings from a .env file, overridden by real environment variables.

        The file is `env_file`, else $ENV_FILE, else ./.env. The directories (SESSIONS_DIR, LOGS_DIR, BRIDGE_DIR), LOG_LEVEL
        and SAMPLING_RATE are shared; the rest are LLM_-prefixed.
        """
        get = env_getter(env_file)
        defaults = cls()
        return cls(
            **shared_paths(get, "llm_service", "llm", defaults),
        )
