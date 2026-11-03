import threading
from typing import NamedTuple

from server.config import Config
from server.log import get_logger
from server.media import MediaServer, MediaServerError
from server.services import WorkerServiceManager
from server.sessions import Session

log = get_logger(__name__)


class SessionError(Exception):
    """A session operation failed; the message is safe to show the client."""


class NoActiveSession(SessionError):
    """Raised when stopping a session while none is running."""


class SessionStartError(SessionError):
    """Raised when a session could not be started; nothing is left running."""


class Started(NamedTuple):
    session: Session
    # Service name -> whether it accepted the session. A service that is down
    # doesn't stop the session; the caller decides what to do with the result.
    services: dict[str, bool]


class Coordinator:
    """Ties the pieces together: the other services, the media server and the
    active session. Runs the session workflow.

    start_session, stop_session and stop are serialised by a lock, since client
    commands arrive on separate connection threads.
    """

    def __init__(self, config: Config):
        self._config = config
        self._services = WorkerServiceManager(
            config.services, config.ping_timeout, config.ping_interval, config.monitor
        )
        self._media = MediaServer(config.media_api_url, config.media_rtsp_url, config.media_binary, config.media_config)
        self._active: Session | None = None
        self._lock = threading.Lock()

    def start_session(self) -> Started:
        """Creates a session, starts it on every service and begins recording its stream.

        A session still running is ended properly first (exported, services
        stopped, media path removed). Raises SessionStartError, after undoing
        whatever was set up, if the media path can't be created or recording
        can't begin.
        """
        with self._lock:
            if self._active is not None:
                log.info("session %s still active, ending it first", self._active.id)
                self._end_active()
            session = Session.create(self._config.sessions_dir, self._media, self._config.sampling_rate)
            log.info("created session %s", session.id)
            delivered = self._services.start_session(session.id)
            try:
                self._media.client.add_path(session.id)
                session.start()
            except Exception as exc:
                log.exception("could not start session %s", session.id)
                self._release(session)
                raise SessionStartError(f"could not start session: {exc}") from exc
            self._active = session
            return Started(session, delivered)

    def stop_session(self) -> Session:
        """Stops the active session everywhere and exports its recording.

        Raises NoActiveSession if there is none. The session is always fully
        released, even if exporting fails.
        """
        with self._lock:
            session = self._active
            if session is None:
                raise NoActiveSession("no active session")
            try:
                self._end_active()
            except Exception as exc:
                log.exception("error while stopping session %s", session.id)
                raise SessionError(f"session {session.id} was released but stopping it failed: {exc}") from exc
            return session

    def _end_active(self) -> None:
        """Stops, exports and releases the active session. Always leaves none active."""
        session = self._active
        try:
            session.stop()
            if not session.export():
                log.warning("session %s was not fully exported", session.id)
        finally:
            self._active = None
            self._release(session)

    def _release(self, session: Session) -> None:
        """Tells the services the session is over and removes its media path. Never raises."""
        session.stop()
        self._services.stop_session()
        try:
            self._media.client.delete_path(session.id)
        except MediaServerError as exc:
            log.warning("could not delete media path %s: %s", session.id, exc)
        log.info("stopped session %s", session.id)

    def publish_url(self, session: Session, host: str) -> str:
        """Where a remote client publishes the session's stream, given the address it reached us at.

        The one deliberate window onto the media server for the protocol layer;
        everything else on the Coordinator is private.
        """
        return self._media.publish_url(session.id, host)

    def start(self) -> None:
        self._media.start()
        self._services.start()

    def stop(self) -> None:
        with self._lock:
            if self._active is not None:
                self._active.stop()
                self._active = None
            self._services.stop()
            self._media.stop()

    def __enter__(self) -> "Coordinator":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
