from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from common.config import env_getter



@dataclass(frozen=True)
class Config:
    # Root of the shared sessions volume; each session is <sessions_dir>/<session_id>/.
    sessions_dir: str = "/sessions"
    bridge_path: str = "/run/bridge/asr_service.sock"
    sampling_rate: int = 16000
    chunk_seconds: float = 1.0
    poll_interval: float = 0.05
    # Max seconds to keep transcribing after a session ends before cutting off.
    drain_timeout: float = 30.0
    # A pause of at least this many seconds starts a new transcript segment.
    segment_pause: float = 20.0
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @classmethod
    def load(cls, env_file: Optional[Union[str, Path]] = None) -> "Config":
        """Loads settings from a .env file, overridden by real environment variables.

        The file is `env_file`, else $ASR_ENV_FILE, else ./.env. A missing
        file is fine (defaults and the environment are used). Every setting
        is an ASR_-prefixed upper-case variable, e.g. ASR_BRIDGE_PATH.
        """
        get = env_getter("ASR_", env_file)
        defaults = cls()

        return cls(
            sessions_dir=get("sessions_dir", str, defaults.sessions_dir),
            bridge_path=get("bridge_path", str, defaults.bridge_path),
            sampling_rate=get("sampling_rate", int, defaults.sampling_rate),
            chunk_seconds=get("chunk_seconds", float, defaults.chunk_seconds),
            poll_interval=get("poll_interval", float, defaults.poll_interval),
            drain_timeout=get("drain_timeout", float, defaults.drain_timeout),
            segment_pause=get("segment_pause", float, defaults.segment_pause),
            log_level=get("log_level", str, defaults.log_level),
            log_file=get("log_file", str, defaults.log_file),
        )
