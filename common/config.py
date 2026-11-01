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


def env_getter(prefix: str, env_file: Optional[Union[str, Path]] = None) -> Callable[[str, Callable, Any], Any]:
    """Returns get(name, cast, default) reading <prefix><NAME> settings.

    Settings come from a .env file, overridden by real environment variables.
    The file is `env_file`, else $<prefix>ENV_FILE, else ./.env. A missing file
    is fine. An unset or empty setting gives `default`.
    """
    if env_file is None:
        env_file = os.environ.get(f"{prefix}ENV_FILE", ".env")
    path = Path(env_file)
    env = parse_env_file(path) if path.is_file() else {}
    env.update({k: v for k, v in os.environ.items() if k.startswith(prefix)})

    def get(name: str, cast: Callable, default: Any) -> Any:
        value = env.get(f"{prefix}{name.upper()}", "")
        return cast(value) if value != "" else default

    return get
