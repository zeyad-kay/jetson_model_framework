import time
from collections.abc import Generator
from pathlib import Path
from typing import Optional, Union


class AudioReader:
    """Reads fixed-length chunks from a raw PCM file that is still being written.

    The chunk size in bytes is derived from the sampling rate:
    sampling_rate * chunk_seconds frames, each of channels * sample_width bytes.
    """

    def __init__(
        self,
        path: Union[str, Path],
        sampling_rate: int = 16000,
        chunk_seconds: float = 1.0,
        channels: int = 1,
        sample_width: int = 2,
    ):
        self.path = Path(path)
        self.sampling_rate = sampling_rate
        self.chunk_seconds = chunk_seconds
        frame_size = channels * sample_width
        self.chunk_size = int(sampling_rate * chunk_seconds) * frame_size
        if self.chunk_size <= 0:
            raise ValueError("chunk_seconds is too small for the sampling rate")
        self._file = None

    def _open(self) -> bool:
        """Opens the file if the writer has created it yet."""
        if self._file is None:
            try:
                self._file = open(self.path, "rb")
            except FileNotFoundError:
                return False
        return True

    def read_chunk(self) -> Optional[bytes]:
        """Returns the next full chunk, or None if not enough data is available yet.

        Does not block and does not consume partial data.
        """
        if not self._open():
            return None
        start = self._file.tell()
        data = self._file.read(self.chunk_size)
        if len(data) < self.chunk_size:
            self._file.seek(start)  # leave the partial data for the next call
            return None
        return data

    def read_remaining(self) -> bytes:
        """Returns whatever is left after the last full chunk (may be shorter)."""
        if not self._open():
            return b""
        return self._file.read()

    def stream(
        self, poll_interval: float = 0.05, idle_timeout: Optional[float] = None
    ) -> Generator[bytes, None, None]:
        """Yields chunks as the file grows.

        Polls every poll_interval seconds while waiting for data. If
        idle_timeout is set and no full chunk arrives within that many
        seconds, the leftover partial chunk (if any) is yielded and the
        stream ends. With idle_timeout=None it waits forever.
        """
        last_data = time.monotonic()
        while True:
            chunk = self.read_chunk()
            if chunk is not None:
                last_data = time.monotonic()
                yield chunk
                continue
            if idle_timeout is not None and time.monotonic() - last_data >= idle_timeout:
                rest = self.read_remaining()
                if rest:
                    yield rest
                return
            time.sleep(poll_interval)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "AudioReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class TranscriptWriter:
    """Writes transcript text from an ASR model to a text file as it is generated.

    Each piece of text is followed by `separator` and flushed immediately, so
    the file always reflects everything transcribed so far.
    """

    def __init__(
        self, path: Union[str, Path], separator: str = "\n", append: bool = False
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.separator = separator
        self._file = open(self.path, "a" if append else "w", encoding="utf-8")

    def write(self, text: str) -> None:
        self._file.write(text + self.separator)
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "TranscriptWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
