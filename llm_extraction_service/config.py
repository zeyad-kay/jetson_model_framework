import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from common.config import env_getter, parse_bool, service_settings, shared_paths


@dataclass(frozen=True)
class Config:
    # Root of the shared sessions volume; each session is <sessions_dir>/<session_id>/.
    sessions_dir: str = "/sessions"
    # Where the service writes its heartbeat, which the server reads.
    status_path: Optional[str] = "/run/status/llm_extraction_service.json"
    # Sessions created longer ago than this (seconds) are never picked up.
    max_session_age: float = 24 * 3600
    # Seconds without a new input or any change in a session's folder, while its input isn't
    # marked complete, before the service gives up on it.
    abandon_timeout: float = 1800.0
    # The service's own directory (holds terminology/index) and its prompts directory
    # (separated/ for extraction, _shared.yaml for the fragments its prompts use).
    llm_dir: str = str(Path(__file__).resolve().parent)
    prompts_dir: str = str(Path(__file__).resolve().parent / "prompts")
    # Hugging Face id of the extraction model (one run per segment).
    extract_model: str = "Qwen/Qwen3-4B-Instruct-2507"
    # Attach ATC / ICD-10-PCS codes to extracted entities; needs the code index in
    # llm_extraction_service/terminology/index. The dashboard names medications by their ATC concept.
    link: bool = True
    # How often segments.json is checked for new segments, in seconds.
    poll_interval: float = 0.5
    log_level: str = "INFO"
    log_file: Optional[str] = None

    @classmethod
    def load(cls, env_file: Optional[Union[str, Path]] = None) -> "Config":
        """Loads settings from a .env file, overridden by real environment variables.

        The file is `env_file`, else $ENV_FILE, else ./.env. The directories (SESSIONS_DIR, LOGS_DIR, STATUS_DIR) and LOG_LEVEL
        are shared; the rest are LLM_EXTRACTION_-prefixed.
        """
        get = env_getter(env_file)
        defaults = cls()
        llm_dir = get("LLM_EXTRACTION_DIR", str, defaults.llm_dir)
        return cls(
            **service_settings(get, "llm_extraction_service", defaults),
            **shared_paths(get, "llm_extraction", defaults),
            llm_dir=llm_dir,
            prompts_dir=get("LLM_EXTRACTION_PROMPTS_DIR", str, os.path.join(llm_dir, "prompts")),
            extract_model=get("LLM_EXTRACTION_MODEL", str, defaults.extract_model),
            link=get("LLM_EXTRACTION_LINK", parse_bool, defaults.link),
            poll_interval=get("LLM_EXTRACTION_POLL_INTERVAL", float, defaults.poll_interval),
        )
