import json
import os
import re
import threading
from pathlib import Path
from typing import Optional, Union


class SegmentTracker:
    """Groups timestamped transcript fragments into segments of words, live.

    A segment ends at a pause of at least min_pause seconds of audio with no
    text. Because silence produces no fragments, tick() is how the tracker
    learns the audio position advanced and closes the segment without waiting
    for the next fragment.

    Only closed segments are written to the JSON file, as a list of
    {"segment_id": n, "start", "end", "text": [{"word", "start", "end"}]},
    with segment_id counting up from 1. The segment still growing is held back
    until it closes. The file is replaced atomically, so readers never see a
    partial write, and it is rewritten only when a segment closes.

    Word times are estimates. Fragments arrive stamped with the end of the
    audio chunk (chunk_seconds long) they were decoded from, so every fragment
    of a chunk shares one stamp. Once a chunk is complete its span is divided
    evenly between its fragments, and a word runs from the start of its first
    fragment to the end of its last. A segment runs from its first word's start
    to its last word's end. Pauses are still judged on the raw stamps.
    """

    def __init__(self, path: Union[str, Path], min_pause: float, chunk_seconds: float):
        self.path = Path(path)
        self.min_pause = min_pause
        self.chunk_seconds = chunk_seconds
        self._segments = []
        self._open = False  # whether the last segment (not written yet) can still grow
        self._last_t = 0.0  # stamp of the last chunk added to the open segment
        self._word: Optional[dict] = None  # the word still being built, in the open segment
        self._boundary = True  # whitespace was seen, so the next text starts a new word
        self._chunk_t = 0.0  # fragments of the chunk still being received
        self._chunk_fragments = []
        self._dirty = True  # write an initial empty file
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush()

    def add(self, text: str, t: float) -> None:
        with self._lock:
            if self._chunk_fragments and t != self._chunk_t:
                self._flush_chunk()
            self._chunk_t = t
            self._chunk_fragments.append(text)

    def tick(self, t: float) -> None:
        """Reports the audio position; closes the open segment if the pause is long enough."""
        with self._lock:
            if self._chunk_fragments and t > self._chunk_t:
                self._flush_chunk()  # the audio moved on, so that chunk is complete
            if not self._chunk_fragments and self._open and t - self._last_t >= self.min_pause:
                self._close_open()
            self._write_if_dirty()

    def close(self) -> None:
        """Ends the open segment (the session is over) and writes the file."""
        with self._lock:
            self._flush_chunk()
            self._close_open()
            self._write_if_dirty()

    def flush(self) -> None:
        with self._lock:
            self._write_if_dirty()

    def __len__(self) -> int:
        """Number of segments written (closed)."""
        return len(self._segments) - self._open

    def _flush_chunk(self) -> None:
        """Turns the received chunk's fragments into words, spreading them over the chunk's span."""
        fragments, t = self._chunk_fragments, self._chunk_t
        self._chunk_fragments = []
        if not fragments:
            return
        if self._open and t - self._last_t >= self.min_pause:
            self._close_open()
        start = max(t - self.chunk_seconds, 0.0)
        step = (t - start) / len(fragments)
        for i, fragment in enumerate(fragments):
            for piece in re.findall(r"\s+|\S+", fragment):
                self._feed(piece, start + i * step, start + (i + 1) * step)
        if self._open:
            self._last_t = t

    def _feed(self, piece: str, start: float, end: float) -> None:
        if piece.isspace():
            self._boundary = True
            return
        if self._word is not None and not self._boundary:
            self._word["word"] += piece
            self._word["end"] = end
        else:
            if not self._open:
                self._segments.append(
                    {"segment_id": len(self._segments) + 1, "start": start, "end": end, "text": []}
                )
                self._open = True
            self._word = {"word": piece, "start": start, "end": end}
            self._segments[-1]["text"].append(self._word)
            self._boundary = False
        self._segments[-1]["end"] = end

    def _close_open(self) -> None:
        if self._open:
            self._open = False
            self._word = None
            self._boundary = True
            self._dirty = True

    def _write_if_dirty(self) -> None:
        if not self._dirty:
            return
        closed = self._segments[:-1] if self._open else self._segments
        out = [
            {
                "segment_id": s["segment_id"],
                "start": round(s["start"], 3),
                "end": round(s["end"], 3),
                "text": [
                    {"word": w["word"], "start": round(w["start"], 3), "end": round(w["end"], 3)}
                    for w in s["text"]
                ],
            }
            for s in closed
        ]
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)
        self._dirty = False
