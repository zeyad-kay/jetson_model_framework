import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from common.config import env_getter, parse_bool, shared_paths


@dataclass(frozen=True)
class Config:
    # Root of the shared sessions volume; each session is <sessions_dir>/<session_id>/.
    sessions_dir: str = "/sessions"
    bridge_path: str = "/run/bridge/llm_service.sock"
    # The service's own directory (holds terminology/index) and its prompts directory
    # (separated/ for extraction, aggregate.txt for aggregation, _shared.yaml for both).
    llm_dir: str = str(Path(__file__).resolve().parent)
    prompts_dir: str = str(Path(__file__).resolve().parent / "prompts")
    # Hugging Face ids of the extraction (per segment) and aggregation (per session) models.
    extract_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    aggregate_model: str = "Qwen/Qwen2.5-7B-Instruct"
    # Attach ATC / ICD-10-PCS codes to extracted entities; needs the code index in
    # llm_service/terminology/index. The aggregator groups medications by ATC name.
    link: bool = True
    # How often segments.json is checked for new segments, in seconds.
    poll_interval: float = 0.5
    # After the session ends, seconds segments.json must stay unchanged before extraction ends.
    # Must exceed the ASR's drain (transcribing its leftover audio before it writes the
    # last segment), or that segment is missed.
    drain_idle: float = 30.0
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @classmethod
    def load(cls, env_file: Optional[Union[str, Path]] = None) -> "Config":
        """Loads settings from a .env file, overridden by real environment variables.

        The file is `env_file`, else $ENV_FILE, else ./.env. The directories (SESSIONS_DIR, LOGS_DIR, BRIDGE_DIR) and LOG_LEVEL
        are shared; the rest are LLM_-prefixed.
        """
        get = env_getter(env_file)
        defaults = cls()
        llm_dir = get("LLM_DIR", str, defaults.llm_dir)
        return cls(
            **shared_paths(get, "llm_service", "llm", defaults),
            llm_dir=llm_dir,
            prompts_dir=get("LLM_PROMPTS_DIR", str, os.path.join(llm_dir, "prompts")),
            extract_model=get("LLM_EXTRACT_MODEL", str, defaults.extract_model),
            aggregate_model=get("LLM_AGGREGATE_MODEL", str, defaults.aggregate_model),
            link=get("LLM_LINK", parse_bool, defaults.link),
            poll_interval=get("LLM_POLL_INTERVAL", float, defaults.poll_interval),
            drain_idle=get("LLM_DRAIN_IDLE", float, defaults.drain_idle),
        )
