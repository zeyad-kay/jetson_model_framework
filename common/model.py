from typing import Any


class StreamingModel:
    """The interface every service's model offers: one session at a time, one input at a time.

        model.start()            begins a session: resets the state kept between inputs
        model.predict(inputs)    runs on the session's next input (a dict), in order, and returns the outputs
        model.stop()             ends the session: releases its state (threads, GPU memory)

    A model never reads files or waits for input: its service polls the session's inputs
    and hands each to predict() (see WorkerService in common/service.py). Weights load in
    __init__, once; start() and stop() run once per session.

    This is a plain class rather than an ABC so a model hierarchy can use SingletonMeta as
    its metaclass (SingletonMeta scopes to the topmost class that uses it, so this base
    doesn't change that scope). predict() raises NotImplementedError until overridden.
    """

    def start(self) -> None:
        """Begins a session: resets the state predict() keeps between inputs."""

    def predict(self, inputs: dict) -> Any:
        """Runs the model on the session's next input, a dict, and returns what it produced."""
        raise NotImplementedError

    def stop(self) -> None:
        """Ends the session and releases its state. Safe to call more than once, or without start()."""
