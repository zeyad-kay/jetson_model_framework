import collections
from pathlib import Path
from typing import Optional, Union

from common.io import done_marker
from common.utils import read_segments


class SegmentReader:
    """Reads an ASR segments.json as the ASR service adds segments to it: each segment once, in order.

        reader = SegmentReader(path)
        segment = reader.read()   # the next new segment, or None if there is none yet
        reader.close()            # nothing to release

    The ASR service rewrites the file (atomically) with every segment closed so far, so the
    file is read again once the segments read before have all been handed out. A missing or
    half-written file has no segments yet. read() raises EOFError once the ASR service has
    marked the file complete (segments.json.done, see common.io.done_marker) and every
    segment has been handed out.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.done_path = done_marker(self.path)
        self._pending = collections.deque()  # segments read but not handed out yet
        self._last_id = 0  # segment_id of the last segment read

    def read(self) -> Optional[dict]:
        if not self._pending:
            complete = self.done_path.exists()  # before reading, so the last segment isn't missed
            new = [s for s in read_segments(self.path) if s.get("segment_id", 0) > self._last_id]
            if new:
                self._pending.extend(new)
                self._last_id = new[-1]["segment_id"]
            elif complete:
                raise EOFError(f"{self.path} is complete")
        return self._pending.popleft() if self._pending else None

    def close(self) -> None:
        pass
