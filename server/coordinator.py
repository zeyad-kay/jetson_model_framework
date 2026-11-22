import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

from server.config import Config
from common.log import get_logger
from server.media import MediaServer, MediaServerError
from server.services import WorkerServiceManager
from server.sessions import Session, abort_unfinished, export, unexported

log = get_logger(__name__)


class SessionError(Exception):
    """A session operation failed; the message is safe to show the client."""


class NoActiveSession(SessionError):
    """Raised when stopping a session while none is running."""


class SessionStartError(SessionError):
    """Raised when a session could not be started; nothing is left running."""


class Started(NamedTuple):
    session: Session
    # Service name -> whether it is up (its heartbeat is fresh), so it will pick the session
    # up. A service that is down doesn't stop the session: it catches up once it is back.
    services: dict[str, bool]


class Coordinator:
    """Ties the pieces together: the other services, the media server and the
    active session. Runs the session workflow.

    start_session, stop_session and stop are serialised by a lock, since client
    commands arrive on separate connection threads.

    A stopped session's recordings are exported (audio.mp3, video.mp4) in the background, one
    session at a time, so stop_session replies as soon as recording has stopped. stop() lets
    the export under way finish and drops those queued; start() exports whatever a previous
    run left unexported.
    """

    def __init__(self, config: Config):
        self._config = config
        self._services = WorkerServiceManager(
            config.services, config.status_timeout, config.monitor_interval, config.monitor
        )
        self._media = MediaServer(config.media_api_url, config.media_rtsp_url, config.media_binary, config.media_config)
        self._active_session: Session | None = None
        self._lock = threading.Lock()
        # One worker: exports run one after another, beside the recording of the next session.
        self._exporter = ThreadPoolExecutor(max_workers=1, thread_name_prefix="export")

    def start_session(self) -> Started:
        """Creates a session (the services find it on disk) and begins recording its stream.

        A session still running is ended properly first (recording stopped, its export
        queued, media path removed). Raises SessionStartError, after undoing
        whatever was set up, if the media path can't be created or recording
        can't begin.
        """
        with self._lock:
            if self._active_session is not None:
                log.info("session %s still active, ending it first", self._active_session.id)
                self._end_active_session()
            session = Session.create(self._config.sessions_dir, self._media, self._config.sampling_rate,
                                     services=self._services.names)
            log.info("created session %s", session.id)
            self._active_session = session
            up = self._services.check()
            try:
                self._media.client.add_path(session.id)
                session.start()
            except Exception as exc:
                log.exception("could not start session %s", session.id)
                try:
                    self._end_active_session(export=False)
                except Exception:
                    log.exception("could not stop session %s", session.id)
                raise SessionStartError(f"could not start session: {exc}") from exc
            return Started(session, up)

    def stop_session(self) -> Session:
        """Stops recording the active session (which ends it on the services) and queues
        its export.

        Raises NoActiveSession if there is none. The session is always fully
        released, even if stopping it fails.
        """
        with self._lock:
            session = self._active_session
            if session is None:
                raise NoActiveSession("no active session")
            try:
                self._end_active_session()
            except Exception as exc:
                log.exception("error while stopping session %s", session.id)
                raise SessionError(f"session {session.id} was released but stopping it failed: {exc}") from exc
            return session

    def _end_active_session(self, export: bool = True) -> None:
        """Ends the active session.

        Stops recording (which marks the session's streams complete, ending it on the
        services), removes its media path and, if `export`, queues the export of its
        recordings. The path is removed, and the export queued, even if stopping raises; the
        exception then propagates.
        """
        session, self._active_session = self._active_session, None
        try:
            session.stop()
        finally:
            try:
                self._media.client.delete_path(session.id)
            except MediaServerError as exc:
                log.warning("could not delete media path %s: %s", session.id, exc)
            log.info("stopped session %s", session.id)
            if export:
                self._queue_export(session.path)

    def _queue_export(self, folder: Path, skip_existing: bool = False) -> None:
        """Exports a session's recordings in the background (see sessions.export)."""
        self._exporter.submit(self._export, folder, skip_existing)

    def _export(self, folder: Path, skip_existing: bool) -> None:
        """Runs on the export worker. Logs the outcome instead of raising."""
        try:
            if export(folder, self._config.sampling_rate, skip_existing):
                log.info("exported session %s", folder.name)
            else:
                log.warning("session %s was not fully exported", folder.name)
        except Exception:
            log.exception("exporting session %s failed", folder.name)

    def publish_url(self, session: Session, host: str) -> str:
        """Where a remote client publishes the session's stream, given the address it reached us at.

        The one deliberate window onto the media server for the protocol layer;
        everything else on the Coordinator is private.
        """
        return self._media.publish_url(session.id, host)

    def start(self) -> None:
        # Before any session of this run exists: whatever is unfinished was left by the last run.
        abort_unfinished(self._config.sessions_dir)
        for folder in unexported(self._config.sessions_dir):
            log.info("session %s was not exported by a previous run; exporting it", folder.name)
            self._queue_export(folder, skip_existing=True)
        self._media.start()
        self._services.start()

    def stop(self) -> None:
        with self._lock:
            if self._active_session is not None:
                self._active_session.stop()
                self._active_session = None
            self._services.stop()
            self._media.stop()
        # Outside the lock: lets the export under way finish; the queued ones are dropped (the
        # next start exports them, as it does the active session's).
        self._exporter.shutdown(wait=True, cancel_futures=True)

    def __enter__(self) -> "Coordinator":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()
