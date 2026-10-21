import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

ENV_FILE_VAR = "ASR_ENV_FILE"


def _parse_env_file(path: Path) -> dict[str, str]:
    """Parses KEY=VALUE lines. Skips blanks and comments, strips optional quotes."""
    values = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip()] = value
    return values


@dataclass(frozen=True)
class Config:
    # Root of the shared sessions volume; each session is <sessions_dir>/<session_id>/.
    sessions_dir: str = "/sessions"
    bridge_path: str = "/run/bridge/asr_service.sock"
    sampling_rate: int = 16000
    chunk_seconds: float = 1.0
    poll_interval: float = 0.05
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @classmethod
    def load(cls, env_file: Optional[Union[str, Path]] = None) -> "Config":
        """Loads settings from a .env file, overridden by real environment variables.

        The file is `env_file`, else $ASR_ENV_FILE, else ./.env. A missing
        file is fine (defaults and the environment are used). Every setting
        is an ASR_-prefixed upper-case variable, e.g. ASR_BRIDGE_PATH.
        """
        if env_file is None:
            env_file = os.environ.get(ENV_FILE_VAR, ".env")
        path = Path(env_file)
        env = _parse_env_file(path) if path.is_file() else {}
        env.update({k: v for k, v in os.environ.items() if k.startswith("ASR_")})

        defaults = cls()

        def get(name, cast, default):
            value = env.get(f"ASR_{name.upper()}", "")
            return cast(value) if value != "" else default

        return cls(
            sessions_dir=get("sessions_dir", str, defaults.sessions_dir),
            bridge_path=get("bridge_path", str, defaults.bridge_path),
            sampling_rate=get("sampling_rate", int, defaults.sampling_rate),
            chunk_seconds=get("chunk_seconds", float, defaults.chunk_seconds),
            poll_interval=get("poll_interval", float, defaults.poll_interval),
            log_level=get("log_level", str, defaults.log_level),
            log_file=get("log_file", str, defaults.log_file),
        )
