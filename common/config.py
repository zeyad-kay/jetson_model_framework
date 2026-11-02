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


def shared_paths(get: Callable, service: str, log_name: str, defaults) -> dict[str, Any]:
    """Settings every service derives from the shared .env directories.

    SESSIONS_DIR is the sessions root, BRIDGE_DIR holds <service>.sock and
    LOGS_DIR holds <log_name>.log. A directory that is unset keeps the service's default.
    Returns sessions_dir, bridge_path, log_level and log_file for Config(**...);
    callers without a bridge socket (the server) drop bridge_path.
    """
    bridge_dir = get("BRIDGE_DIR", str, None)
    logs_dir = get("LOGS_DIR", str, None)
    return {
        "sessions_dir": get("SESSIONS_DIR", str, defaults.sessions_dir),
        "bridge_path": os.path.join(bridge_dir, f"{service}.sock") if bridge_dir else getattr(defaults, "bridge_path", None),
        "log_level": get("LOG_LEVEL", str, defaults.log_level),
        "log_file": os.path.join(logs_dir, f"{log_name}.log") if logs_dir else defaults.log_file,
    }
