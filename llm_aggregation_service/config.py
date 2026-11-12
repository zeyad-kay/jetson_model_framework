from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from common.config import env_getter, shared_paths


@dataclass(frozen=True)
class Config:
    # Root of the shared sessions volume; each session is <sessions_dir>/<session_id>/.
    sessions_dir: str = "/sessions"
    bridge_path: str = "/run/bridge/llm_aggregation_service.sock"
    # The aggregation prompt ({entities} is replaced with one drug's extracted mentions).
    prompt_path: str = str(Path(__file__).resolve().parent / "prompts" / "aggregate.txt")
    # Hugging Face id of the aggregation model.
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    # How often extractions.json is checked for new records, in seconds.
    poll_interval: float = 0.5
    # After the session ends, the aggregator waits for the extraction service to mark
    # the extractions complete. If no mark comes (the extraction service is down), it
    # gives up once extractions.json has been unchanged this many seconds. Must exceed
    # the extraction service's drain (LLM_EXTRACTION_DRAIN_IDLE) plus the time it takes
    # to extract the session's last segment.
    drain_timeout: float = 600.0
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @classmethod
    def load(cls, env_file: Optional[Union[str, Path]] = None) -> "Config":
        """Loads settings from a .env file, overridden by real environment variables.

        The file is `env_file`, else $ENV_FILE, else ./.env. The directories (SESSIONS_DIR,
        LOGS_DIR, BRIDGE_DIR) and LOG_LEVEL are shared; the rest are LLM_AGGREGATION_-prefixed.
        """
        get = env_getter(env_file)
        defaults = cls()
        return cls(
            **shared_paths(get, "llm_aggregation_service", "llm_aggregation", defaults),
            prompt_path=get("LLM_AGGREGATION_PROMPT", str, defaults.prompt_path),
            model=get("LLM_AGGREGATION_MODEL", str, defaults.model),
            poll_interval=get("LLM_AGGREGATION_POLL_INTERVAL", float, defaults.poll_interval),
            drain_timeout=get("LLM_AGGREGATION_DRAIN_TIMEOUT", float, defaults.drain_timeout),
        )
