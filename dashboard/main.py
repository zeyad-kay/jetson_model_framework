import signal
import threading

from common.log import get_logger, setup_logging
from dashboard.config import Config
from dashboard.server import serve

log = get_logger("dashboard.main")


def main() -> None:
    config = Config.load()
    setup_logging("dashboard", level=config.log_level, log_file=config.log_file, also_stderr=True)
    httpd = serve(config.sessions_dir, config.host, config.port, config.live_window)
    # SIGTERM (docker stop) and Ctrl-C stop the server cleanly; shutdown() must come from another thread.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: threading.Thread(target=httpd.shutdown, daemon=True).start())
    log.info("serving %s on http://%s:%d", config.sessions_dir, config.host, config.port)
    httpd.serve_forever()
    httpd.server_close()
    log.info("dashboard stopped")


if __name__ == "__main__":
    main()
