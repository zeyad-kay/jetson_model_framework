import signal
from collections.abc import Callable

from asr_service.config import Config
from asr_service.bridge import Bridge
from asr_service.log import get_logger, setup_logging
from asr_service.service import ASRService

log = get_logger(__name__)


def make_message_handler(service: ASRService) -> Callable[[dict], dict]:
    """Builds the bridge callback: dispatches {"command": ...} messages to the service."""

    def cmd_start(message: dict) -> dict:
        session_id = message.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            return {"ok": False, "error": "start needs a session_id"}
        try:
            service.start(session_id)
        except ValueError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True}

    def cmd_stop(message: dict) -> dict:
        service.stop()
        return {"ok": True}

    def cmd_ping(message: dict) -> dict:
        return {"ok": True, "reply": "pong"}

    # Every bridge command we accept: name -> (description, handler).
    commands = {
        "start": ("Transcribe the session named by session_id.", cmd_start),
        "stop": ("Process the current session's remaining audio, then wait for the next.", cmd_stop),
        "ping": ("Health check; replies with 'pong'.", cmd_ping),
    }

    def handle_message(message: dict) -> dict:
        log.info("bridge message: %s", message)
        command = message.get("command")
        entry = commands.get(command)
        if entry is None:
            return {
                "ok": False,
                "error": f"unknown command: {command!r}",
                "commands": {name: desc for name, (desc, _) in commands.items()},
            }
        return entry[1](message)

    return handle_message


def main() -> None:
    config = Config.load()
    setup_logging(level=config.log_level, log_file=config.log_file)
    log.info("starting asr_service")

    service = ASRService(config)
    # Ctrl-C and SIGTERM finish the current session, then exit.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: service.shutdown())

    with Bridge(config.bridge_path, make_message_handler(service)):
        service.run()

    log.info("asr_service stopped")


if __name__ == "__main__":
    main()
