import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional, Union


def parse_env_file(path: Path) -> dict[str, str]:
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


def parse_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def env_getter(env_file: Optional[Union[str, Path]] = None) -> Callable[[str, Callable, Any], Any]:
    """Returns get(name, cast, default) reading settings by their exact variable name.

    There is no prefixing: every variable in the file is visible to every caller,
    so names must be unique across services (e.g. ASR_LOG_LEVEL, not LOG_LEVEL).

    Settings come from a .env file, overridden by real environment variables.
    The file is `env_file`, else $ENV_FILE, else ./.env. A missing file is
    fine. An unset or empty setting gives `default`.
    """
    if env_file is None:
        env_file = os.environ.get("ENV_FILE", ".env")
    path = Path(env_file)
    env = parse_env_file(path) if path.is_file() else {}
    env.update(os.environ)

    def get(name: str, cast: Callable, default: Any) -> Any:
        value = env.get(name, "")
        return cast(value) if value != "" else default

    return get


def shared_paths(get: Callable, log_name: str, defaults) -> dict[str, Any]:
    """Settings every process (the model services, the server, the dashboard) derives from
    the shared .env settings.

    SESSIONS_DIR is the sessions root and LOGS_DIR holds <log_name>.log; a directory that
    is unset keeps the default. Returns sessions_dir, log_level and log_file for Config(**...).
    """
    logs_dir = get("LOGS_DIR", str, None)
    return {
        "sessions_dir": get("SESSIONS_DIR", str, defaults.sessions_dir),
        "log_level": get("LOG_LEVEL", str, defaults.log_level),
        "log_file": os.path.join(logs_dir, f"{log_name}.log") if logs_dir else defaults.log_file,
    }


def status_path(status_dir: str, service: str) -> str:
    """Where a model service writes its heartbeat: <STATUS_DIR>/<service>.json."""
    return os.path.join(status_dir, f"{service}.json")


def service_settings(get: Callable, service: str, defaults) -> dict[str, Any]:
    """Settings every model service shares: status_path (its heartbeat, in STATUS_DIR; unset
    keeps the default), max_session_age (SESSION_MAX_AGE, seconds) and abandon_timeout
    (SESSION_ABANDON_TIMEOUT, seconds)."""
    status_dir = get("STATUS_DIR", str, None)
    return {
        "status_path": status_path(status_dir, service) if status_dir else defaults.status_path,
        "max_session_age": get("SESSION_MAX_AGE", float, defaults.max_session_age),
        "abandon_timeout": get("SESSION_ABANDON_TIMEOUT", float, defaults.abandon_timeout),
    }
