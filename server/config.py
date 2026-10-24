import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

ENV_PREFIX = "SERVER_"
ENV_FILE_VAR = "SERVER_ENV_FILE"


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


def _parse_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Config:
    streaming_host: str = "0.0.0.0"
    streaming_port: int = 8000
    # Bridge sockets of the services to health-check (the file stem is the name).
    services: tuple[str, ...] = (
        "/run/bridge/asr_service.sock",
        "/run/bridge/llm_service.sock",
        "/run/bridge/video_service.sock",
    )
    # Root of the shared volume that holds one folder per session.
    sessions_dir: str = "/sessions"
    # Ping the services in the background while a session is running.
    monitor: bool = True
    ping_interval: float = 20.0
    ping_timeout: float = 2.0
    log_level: str = "INFO"
    log_file: Optional[str] = None
    # MediaMTX Control API (localhost only).
    media_api_url: str = "http://127.0.0.1:9997"
    # Where ffmpeg reads the published streams from (<url>/<session_id>).
    media_rtsp_url: str = "rtsp://127.0.0.1:8554"

    @classmethod
    def load(cls, env_file: Optional[Union[str, Path]] = None) -> "Config":
        """Loads settings from a .env file, overridden by real environment variables.

        The file is `env_file`, else $SERVER_ENV_FILE, else ./.env. A missing
        file is fine (defaults and the environment are used). Every setting
        is a SERVER_-prefixed upper-case variable, e.g. SERVER_STREAMING_PORT.
        """
        if env_file is None:
            env_file = os.environ.get(ENV_FILE_VAR, ".env")
        path = Path(env_file)
        env = _parse_env_file(path) if path.is_file() else {}
        env.update({k: v for k, v in os.environ.items() if k.startswith(ENV_PREFIX)})

        defaults = cls()

        def get(name, cast, default):
            value = env.get(f"{ENV_PREFIX}{name.upper()}", "")
            return cast(value) if value != "" else default

        return cls(
            streaming_host=get("streaming_host", str, defaults.streaming_host),
            streaming_port=get("streaming_port", int, defaults.streaming_port),
            services=get("services", _parse_list, defaults.services),
            sessions_dir=get("sessions_dir", str, defaults.sessions_dir),
            monitor=get("monitor", _parse_bool, defaults.monitor),
            ping_interval=get("ping_interval", float, defaults.ping_interval),
            ping_timeout=get("ping_timeout", float, defaults.ping_timeout),
            log_level=get("log_level", str, defaults.log_level),
            log_file=get("log_file", str, defaults.log_file),
            media_api_url=get("media_api_url", str, defaults.media_api_url),
            media_rtsp_url=get("media_rtsp_url", str, defaults.media_rtsp_url),
        )
