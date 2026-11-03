from server.coordinator import Coordinator, SessionError
from server.log import get_logger

log = get_logger(__name__)


class CommandHandler:
    """The client's protocol: maps its {"command": ...} messages onto the Coordinator.

    handle_message is the callback for a ClientListener. It knows the command
    names and the shape of the replies; the Coordinator knows nothing about them.
    """

    def __init__(self, coordinator: Coordinator):
        self._coordinator = coordinator
        # Every command we accept from the client: name -> (description, handler).
        self.commands = {
            "start_session": (
                "Creates a new session folder, starts it on all services and "
                "returns the URL to publish the stream to.",
                self._start_session,
            ),
            "stop_session": (
                "Stops the active session on all services and closes it.",
                self._stop_session,
            ),
        }

    def handle_message(self, message: dict, local_host: str) -> dict:
        """Dispatches a client message to its command handler and returns the reply."""
        log.info("client message: %s", message)
        command = message.get("command")
        entry = self.commands.get(command)
        if entry is None:
            return {
                "ok": False,
                "error": f"unknown command: {command!r}",
                "commands": {name: desc for name, (desc, _) in self.commands.items()},
            }
        return entry[1](message, local_host)

    def _start_session(self, message: dict, local_host: str) -> dict:
        try:
            session, services = self._coordinator.start_session()
        except SessionError as exc:
            return {"ok": False, "error": str(exc)}
        return {
            "ok": True,
            "session_id": session.id,
            # Where the client can publish: the address it reached us at.
            "streaming_url": self._coordinator.publish_url(session, local_host),
            # Which services accepted the session; a false means that service is not working on it.
            "services": services,
        }

    def _stop_session(self, message: dict, local_host: str) -> dict:
        try:
            session = self._coordinator.stop_session()
        except SessionError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "session_id": session.id}
