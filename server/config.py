from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from common.config import env_getter, parse_bool, parse_list, shared_paths


@dataclass(frozen=True)
class Config:
    command_host: str = "0.0.0.0"
    command_port: int = 8000
    # Bridge sockets of the services to health-check (the file stem is the name).
    services: tuple[str, ...] = (
        "/run/bridge/asr_service.sock",
        "/run/bridge/llm_service.sock",
        "/run/bridge/video_service.sock",
    )
    # Root of the shared volume that holds one folder per session.
    sessions_dir: str = "./sessions"
    # Ping the services in the background while a session is running.
    monitor: bool = True
    # Audio sample rate (Hz) of the recorded PCM; must match the ASR service's sampling rate.
    sampling_rate: int = 16000
    ping_interval: float = 20.0
    ping_timeout: float = 2.0
    log_level: str = "INFO"
    log_file: Optional[str] = "server.log"
    # MediaMTX Control API (localhost only).
    media_api_url: str = "http://127.0.0.1:9997"
    # Where ffmpeg reads the published streams from (<url>/<session_id>).
    media_rtsp_url: str = "rtsp://127.0.0.1:8554"
    # The MediaMTX binary (the Dockerfile extracts it to /app/mediamtx) and the config it runs with.
    media_binary: str = "mediamtx/mediamtx"
    media_config: str = str(Path(__file__).resolve().parent / "mediamtx.yml")

    @classmethod
    def load(cls, env_file: Optional[Union[str, Path]] = None) -> "Config":
        """Loads settings from a .env file, overridden by real environment variables.

        The file is `env_file`, else $ENV_FILE, else ./.env. A missing
        file is fine (defaults and the environment are used). The directories (SESSIONS_DIR, LOGS_DIR, BRIDGE_DIR), LOG_LEVEL
        and SAMPLING_RATE are shared; the rest are SERVER_-prefixed.
        """
        get = env_getter(env_file)
        defaults = cls()

        shared = shared_paths(get, "server", "server", defaults)
        shared.pop("bridge_path")  # the server has no bridge socket of its own
        bridge_dir = get("BRIDGE_DIR", str, None)
        services = (
            tuple(f"{bridge_dir}/{name}.sock" for name in ("asr_service", "llm_service", "video_service"))
            if bridge_dir else defaults.services
        )
        return cls(
            **shared,
            services=services,
            command_host=get("SERVER_COMMAND_HOST", str, defaults.command_host),
            command_port=get("SERVER_COMMAND_PORT", int, defaults.command_port),
            monitor=get("SERVER_MONITOR", parse_bool, defaults.monitor),
            sampling_rate=get("SAMPLING_RATE", int, defaults.sampling_rate),
            ping_interval=get("SERVER_PING_INTERVAL", float, defaults.ping_interval),
            ping_timeout=get("SERVER_PING_TIMEOUT", float, defaults.ping_timeout),
            media_api_url=get("SERVER_MEDIA_API_URL", str, defaults.media_api_url),
            media_rtsp_url=get("SERVER_MEDIA_RTSP_URL", str, defaults.media_rtsp_url),
            media_binary=get("SERVER_MEDIA_BINARY", str, defaults.media_binary),
            media_config=get("SERVER_MEDIA_CONFIG", str, defaults.media_config),
        )
