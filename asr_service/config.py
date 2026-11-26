from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from common.config import env_getter, service_settings, shared_paths



@dataclass(frozen=True)
class Config:
    # Root of the shared sessions volume; each session is <sessions_dir>/<session_id>/.
    sessions_dir: str = "/sessions"
    # Where the service writes its heartbeat, which the server reads.
    status_path: Optional[str] = "/run/status/asr_service.json"
    # Sessions created longer ago than this (seconds) are never picked up.
    max_session_age: float = 24 * 3600
    # Seconds without a new input or any change in a session's folder, while its input isn't
    # marked complete, before the service gives up on it.
    abandon_timeout: float = 1800.0
    sampling_rate: int = 16000
    # Transcription language passed to the model.
    language: str = "en-US"
    # Lookahead tokens of the streaming model (higher = more accurate, more latency).
    num_lookahead_tokens: int = 6
    poll_interval: float = 0.05
    # A pause of at least this many seconds of audio ends a transcript segment.
    segment_pause: float = 20.0
    # A sentence end ends a transcript segment once it holds at least this many seconds of audio.
    segment_min_length: float = 20.0
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @classmethod
    def load(cls, env_file: Optional[Union[str, Path]] = None) -> "Config":
        """Loads settings from a .env file, overridden by real environment variables.

        The file is `env_file`, else $ENV_FILE, else ./.env. A missing
        file is fine (defaults and the environment are used). The directories (SESSIONS_DIR, LOGS_DIR, STATUS_DIR), LOG_LEVEL
        and SAMPLING_RATE are shared; the rest are ASR_-prefixed.
        """
        get = env_getter(env_file)
        defaults = cls()

        return cls(
            **service_settings(get, "asr_service", defaults),
            **shared_paths(get, "asr", defaults),
            sampling_rate=get("SAMPLING_RATE", int, defaults.sampling_rate),
            language=get("ASR_LANGUAGE", str, defaults.language),
            num_lookahead_tokens=get("ASR_NUM_LOOKAHEAD_TOKENS", int, defaults.num_lookahead_tokens),
            poll_interval=get("ASR_POLL_INTERVAL", float, defaults.poll_interval),
            segment_pause=get("ASR_SEGMENT_PAUSE", float, defaults.segment_pause),
            segment_min_length=get("ASR_SEGMENT_MIN_LENGTH", float, defaults.segment_min_length),
        )
