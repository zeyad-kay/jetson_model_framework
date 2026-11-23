import os
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

from common.io import done_marker


class PcmReader:
    """Reads raw 16-bit mono PCM (s16le) chunk by chunk as its recorder appends to it, without
    ever blocking.

        reader = PcmReader(path, model.chunk_span)
        inputs = reader.read()   # {"samples", "final"}, or None if the next chunk isn't recorded yet
        reader.close()           # closes the file

    span(index) gives the samples [begin, end) of the audio that chunk index (from 0) covers;
    chunks may overlap, and the reader reads the overlap from the file again. samples is a
    float32 array in [-1, 1) of the next chunk, once the file holds all of it. Once the
    recorder has marked the file complete (<path>.done, see common.io.done_marker) and the
    rest no longer fills a chunk, the final read (final True) hands out the rest of the audio
    from the next chunk's begin: fewer samples than a chunk, maybe none. The read after it
    raises EOFError. A file that doesn't exist yet has no audio yet; one marked complete
    without ever being created gives a single final read with no samples.

    An odd trailing byte (half a sample still being written) is not counted until it is whole.
    """

    BYTES_PER_SAMPLE = 2

    def __init__(self, path: Union[str, Path], span: Callable[[int], tuple[int, int]]):
        self.path = Path(path)
        self.done_path = done_marker(self.path)
        self._span = span
        self._index = 0  # the next chunk to hand out
        self._file = None
        self._finished = False  # whether the final read was made

    def read(self) -> Optional[dict]:
        if self._finished:
            raise EOFError(f"{self.path} is complete")
        complete = self.done_path.exists()  # before reading, so the last samples aren't missed
        try:
            received = os.path.getsize(self.path) // self.BYTES_PER_SAMPLE
        except FileNotFoundError:
            received = 0
        begin, end = self._span(self._index)
        if received >= end:
            self._index += 1
            return {"samples": self._read(begin, end), "final": False}
        if not complete:
            return None
        self._finished = True
        return {"samples": self._read(begin, max(received, begin)), "final": True}

    def _read(self, begin: int, end: int) -> np.ndarray:
        """The samples [begin, end), which the file holds."""
        data = b""
        if end > begin:
            if self._file is None:
                self._file = open(self.path, "rb")
            self._file.seek(begin * self.BYTES_PER_SAMPLE)
            data = self._file.read((end - begin) * self.BYTES_PER_SAMPLE)
        return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
