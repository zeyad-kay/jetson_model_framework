import os
import threading
from typing import Any
from collections.abc import Generator

class _SingletonMeta(type):
    """Allows only one model instance across the whole ASRModel hierarchy.

    Using a metaclass (rather than __new__) means __init__ runs only once,
    so model weights are not reloaded on repeated construction.
    """

    _instance = None
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        with _SingletonMeta._lock:
            if _SingletonMeta._instance is None:
                _SingletonMeta._instance = super().__call__(*args, **kwargs)
            elif not isinstance(_SingletonMeta._instance, cls):
                raise RuntimeError(
                    f"A {type(_SingletonMeta._instance).__name__} instance "
                    f"already exists; cannot create {cls.__name__}"
                )
        return _SingletonMeta._instance


class ASRModel(metaclass=_SingletonMeta):
    """Blueprint for ASR models. Subclasses override transcribe."""

    def __init__(self, sampling_rate: int = 16000, channel: str = "mono", bytes_per_sample: int = 2, language: str = "en-US"):
        self._sampling_rate = sampling_rate
        self._channel = channel
        self._bytes_per_sample = bytes_per_sample
        self._language = language

    @property
    def sampling_rate(self) -> int:
        """Expected audio sample rate in Hz."""
        return self._sampling_rate

    @property
    def channel(self) -> str:
        """Expected audio channel layout, e.g. "mono"."""
        return self._channel

    @property
    def language(self) -> str:
        """Expected transcription language."""
        return self._language

    @property
    def _bytes_per_sample(self) -> int:
        """Expected bytes per sample"""
        return self.__bytes_per_sample

    def transcribe(self, audio_path: str) -> Generator[str, None, None]:
        """Transcribe the audio file at audio_path and yields the text.

        The base implementation only validates that the path exists.
        Subclasses should call super().transcribe(audio_path) before
        running inference.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        yield ""

