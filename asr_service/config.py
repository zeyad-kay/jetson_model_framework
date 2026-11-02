from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from common.config import env_getter, shared_paths



@dataclass(frozen=True)
class Config:
    # Root of the shared sessions volume; each session is <sessions_dir>/<session_id>/.
    sessions_dir: str = "/sessions"
    bridge_path: str = "/run/bridge/asr_service.sock"
    sampling_rate: int = 16000
    # Transcription language passed to the model.
    language: str = "en-US"
    # Seconds the model stays behind the writer so it only reads flushed audio.
    delay: int = 2
    # Lookahead tokens of the streaming model (higher = more accurate, more latency).
    num_lookahead_tokens: int = 6
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

        The file is `env_file`, else $ENV_FILE, else ./.env. A missing
        file is fine (defaults and the environment are used). The directories (SESSIONS_DIR, LOGS_DIR, BRIDGE_DIR), LOG_LEVEL
        and SAMPLING_RATE are shared; the rest are ASR_-prefixed.
        """
        get = env_getter(env_file)
        defaults = cls()

        return cls(
            **shared_paths(get, "asr_service", "asr", defaults),
            sampling_rate=get("SAMPLING_RATE", int, defaults.sampling_rate),
            language=get("ASR_LANGUAGE", str, defaults.language),
            delay=get("ASR_DELAY", int, defaults.delay),
            num_lookahead_tokens=get("ASR_NUM_LOOKAHEAD_TOKENS", int, defaults.num_lookahead_tokens),
            poll_interval=get("ASR_POLL_INTERVAL", float, defaults.poll_interval),
            drain_timeout=get("ASR_DRAIN_TIMEOUT", float, defaults.drain_timeout),
            segment_pause=get("ASR_SEGMENT_PAUSE", float, defaults.segment_pause),
        )
