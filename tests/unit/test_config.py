"""Every process's Config.load() (the model services, the server, the dashboard) reads the shared
.env settings: SESSIONS_DIR, LOGS_DIR and, for the model services and the server, STATUS_DIR."""

import importlib

import pytest

SHARED = ("SESSIONS_DIR", "LOGS_DIR", "STATUS_DIR", "LOG_LEVEL", "SESSION_MAX_AGE", "SESSION_ABANDON_TIMEOUT")
SERVICES = {"asr_service": "asr", "llm_extraction_service": "llm_extraction", "video_service": "video"}


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    for name in SHARED:
        monkeypatch.delenv(name, raising=False)
    path = tmp_path / ".env"
    path.write_text("SESSIONS_DIR=/data/sessions\nLOGS_DIR=/data/logs\nSTATUS_DIR=/data/status\n"
                    "SESSION_MAX_AGE=60\nSESSION_ABANDON_TIMEOUT=120\n")
    return path


@pytest.mark.parametrize("service", SERVICES)
def test_a_model_service_writes_its_heartbeat_in_status_dir(env_file, service):
    config = importlib.import_module(f"{service}.config").Config.load(env_file)
    assert config.sessions_dir == "/data/sessions"
    assert config.log_file == f"/data/logs/{SERVICES[service]}.log"
    assert config.status_path == f"/data/status/{service}.json"
    assert (config.max_session_age, config.abandon_timeout) == (60, 120)


@pytest.mark.parametrize("service", SERVICES)
def test_without_status_dir_a_model_service_keeps_its_default_heartbeat(tmp_path, monkeypatch, service):
    for name in SHARED:
        monkeypatch.delenv(name, raising=False)
    config = importlib.import_module(f"{service}.config").Config.load(tmp_path / "missing.env")
    assert config.status_path == f"/run/status/{service}.json"


def test_the_server_reads_the_heartbeats_in_status_dir(env_file):
    from server.config import Config

    config = Config.load(env_file)
    assert config.sessions_dir == "/data/sessions" and config.log_file == "/data/logs/server.log"
    assert config.services == tuple(f"/data/status/{name}.json" for name in SERVICES)


def test_the_dashboard_loads_the_shared_settings(env_file):
    from dashboard.config import Config

    config = Config.load(env_file)
    assert config.sessions_dir == "/data/sessions" and config.log_file == "/data/logs/dashboard.log"
