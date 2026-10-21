import wave
from pathlib import Path
from typing import Union


class AudioWriter:
    """Writes a realtime audio stream to a raw (headerless) PCM file.

    Each chunk is flushed as it arrives, so the file on disk always reflects
    everything received so far.
    """

    def __init__(self, path: Union[str, Path], append: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, "ab" if append else "wb")
        self.bytes_written = 0

    def write(self, chunk: Union[bytes, bytearray, memoryview]) -> None:
        self._file.write(chunk)
        self._file.flush()
        self.bytes_written += len(chunk)

    def close(self) -> None:
        self._file.close()

    def to_wav(
        self,
        wav_path: Union[str, Path, None] = None,
        sampling_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
    ) -> Path:
        """Writes a WAV copy of the PCM file and returns its path.

        The PCM file is left untouched. `wav_path` defaults to the PCM path
        with a .wav suffix. Flushes first, so it also works mid-stream.
        """
        self._file.flush()
        wav_path = Path(wav_path) if wav_path else self.path.with_suffix(".wav")
        with open(self.path, "rb") as pcm, wave.open(str(wav_path), "wb") as wav:
            wav.setnchannels(channels)
            wav.setsampwidth(sample_width)
            wav.setframerate(sampling_rate)
            while chunk := pcm.read(1 << 20):
                wav.writeframes(chunk)
        return wav_path

    def __enter__(self) -> "AudioWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class VideoWriter:
    """Writes a realtime video stream to a file. Not implemented yet."""

    def __init__(self, path: Union[str, Path], append: bool = False):
        self.path = Path(path)
        self.append = append

    def write(self, chunk: Union[bytes, bytearray, memoryview]) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass
