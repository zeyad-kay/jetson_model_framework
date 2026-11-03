import signal
import threading

from server.client_listener import ClientListener
from server.command_handler import CommandHandler
from server.config import Config
from server.coordinator import Coordinator
from server.log import get_logger, setup_logging

log = get_logger(__name__)


def main() -> None:
    config = Config.load()
    setup_logging(level=config.log_level, log_file=config.log_file)
    log.info("starting server")

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())

    # The coordinator starts first and stops last; the listener is the reverse.
    with Coordinator(config) as coordinator, ClientListener(
        config.command_host, config.command_port, CommandHandler(coordinator).handle_message
    ):
        stop.wait()

    log.info("server stopped")


if __name__ == "__main__":
    main()
