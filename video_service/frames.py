import json
import queue
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np

from common.io import done_marker


def probe(path: Union[str, Path]) -> Optional[tuple[int, int, float]]:
    """(width, height, fps) of the file's first video stream, or None if it has no readable one (yet)."""
    try:
        if Path(path).stat().st_size == 0:
            return None
    except FileNotFoundError:
        return None
    res = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        return None
    try:
        # ffprobe can list the stream twice (program + top level); use the first
        stream = json.loads(res.stdout)["streams"][0]
        width, height = int(stream["width"]), int(stream["height"])
        num, _, den = stream.get("r_frame_rate", "0/1").partition("/")
        fps = float(num) / float(den or 1) if float(num) else 30.0
    except (ValueError, KeyError, IndexError, ZeroDivisionError):
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height, fps


class FrameReader:
    """Decodes an MPEG-TS file while it is still being written, without ever blocking the caller.

        reader = FrameReader(path)
        inputs = reader.read()   # the next frame, {"frame", "seconds"}, or None if none is decoded yet
        reader.close()           # stops the decoder; safe to call more than once

    frame is an HxWx3 uint8 RGB array, seconds its position in the video (its index over
    the stream's frame rate).

    read() first waits for the file to have a readable video stream: until then it returns
    None, probing the file (ffprobe) at most every probe_interval seconds, and raises
    TimeoutError if the file has existed for probe_timeout seconds without one. It then
    starts ffmpeg, which a thread feeds the file's bytes as they are written, and whose
    frames another thread queues for read() to take.

    read() raises EOFError when the video has finished: once the file is marked complete
    (<path>.done, see common.io.done_marker), the feeder closes ffmpeg's input after the
    last byte, ffmpeg flushes the frames it held back, and read() raises EOFError after
    the last of them. Also if the decoder stops (the stream is broken), or if the file is
    marked complete without a readable video stream.
    """

    def __init__(self, path: Union[str, Path], probe_timeout: float = 60.0, probe_interval: float = 0.5, feed_interval: float = 0.05, queue_size: int = 8):
        self.path = Path(path)
        self.done_path = done_marker(self.path)
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.fps: Optional[float] = None  # frame rate of the stream; set once it is opened
        self.opened_at: Optional[float] = None  # when decoding started (monotonic); None until then
        self._probe_timeout = probe_timeout
        self._probe_interval = probe_interval
        self._feed_interval = feed_interval  # seconds between checks for new bytes
        self._num_read = 0  # frames read
        self._first_seen: Optional[float] = None  # when the file was first seen to exist
        self._last_probe: Optional[float] = None
        # Bounded: backpressures the feeder and ffmpeg if inference is slower than the video.
        self._frames: queue.Queue = queue.Queue(maxsize=queue_size)  # raw frames; None = decoder stopped
        self._ended: Optional[str] = None  # why the video finished, once it has
        self._stop = threading.Event()  # set by close(): the feeder stops
        self._fed_all = threading.Event()  # set by the feeder once it has fed the complete file
        self._decoder: Optional[subprocess.Popen] = None
        self._feeder_thread: Optional[threading.Thread] = None
        self._reader_thread: Optional[threading.Thread] = None

    @property
    def opened(self) -> bool:
        """Whether the stream was found and is being decoded."""
        return self.opened_at is not None

    def read(self) -> Optional[dict]:
        """The next decoded frame, {"frame", "seconds"}, or None if there is none yet. Never blocks for long."""
        if self._ended:
            raise EOFError(self._ended)
        if not self.opened and not self._open():
            return None
        try:
            buf = self._frames.get_nowait()
        except queue.Empty:
            return None
        if buf is None:
            # The feeder closed ffmpeg's input after the last byte, or the decoder failed.
            complete = self._fed_all.is_set()
            self._ended = f"{self.path} is complete" if complete else f"the decoding of {self.path} stopped"
            raise EOFError(self._ended)
        seconds = self._num_read / self.fps
        self._num_read += 1
        return {"frame": np.frombuffer(buf, np.uint8).reshape(self.height, self.width, 3), "seconds": seconds}

    def close(self) -> None:
        """Stops the decoder and its threads. The frames not read yet are dropped."""
        self._stop.set()
        if self._decoder is None:
            return
        self._decoder.kill()
        self._feeder_thread.join(timeout=5)  # its next write fails, or its wait ends
        # Keep the queue empty so the reader thread, maybe blocked on put(), can finish:
        # once the decoder is dead it makes at most two more puts (a pending frame, then None).
        deadline = time.monotonic() + 5
        while self._reader_thread.is_alive() and time.monotonic() < deadline:
            while True:
                try:
                    self._frames.get_nowait()
                except queue.Empty:
                    break
            self._reader_thread.join(timeout=0.05)
        self._decoder.wait()
        self._decoder.stdout.close()
        self._decoder = self._feeder_thread = self._reader_thread = None

    def _open(self) -> bool:
        """Starts decoding if the file has a readable video stream by now; returns whether it has."""
        now = time.monotonic()
        if self._last_probe is not None and now - self._last_probe < self._probe_interval:
            return False
        self._last_probe = now
        complete = self.done_path.exists()  # before probing, so a stream written just before the marker is found
        info = probe(self.path)
        if info is None:
            if complete:
                self._ended = f"{self.path} is complete without a readable video stream"
                raise EOFError(self._ended)
            if self._first_seen is None:
                if self.path.exists():
                    self._first_seen = now
            elif now - self._first_seen >= self._probe_timeout:
                raise TimeoutError(f"no video stream found in {self.path} after {self._probe_timeout}s")
            return False
        self.width, self.height, self.fps = info
        self._start_decoder()
        self.opened_at = time.monotonic()
        return True

    def _start_decoder(self) -> None:
        frame_bytes = self.width * self.height * 3
        # While the file grows, its end is not the stream's, so ffmpeg's defaults would hold back
        # frames: it buffers ~5 MB / 5 s to analyse the stream before decoding (a short recording
        # yields nothing), and frame-threaded decoding keeps the newest ~15 frames. A tiny probe and
        # one decoder thread emit each frame as it arrives; passthrough keeps ffmpeg from
        # duplicating frames to fill the frame rate it then can't measure. The demuxer still holds
        # the last packets until more input comes, or the input ends: the feeder ends it once the
        # file is complete.
        self._decoder = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-probesize", "32768", "-analyzeduration", "0", "-threads", "1",
             "-f", "mpegts", "-i", "-",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-fps_mode", "passthrough", "-"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        )
        decoder, frames, stop, fed_all = self._decoder, self._frames, self._stop, self._fed_all
        path, done_path, feed_interval = self.path, self.done_path, self._feed_interval

        def feeder() -> None:
            """Writes the file's bytes to ffmpeg as they are written; closes its input after the last."""
            try:
                with open(path, "rb") as f:
                    while not stop.is_set():
                        complete = done_path.exists()  # before reading, so the last bytes are read
                        chunk = f.read(1 << 16)
                        if chunk:
                            decoder.stdin.write(chunk)
                            decoder.stdin.flush()
                        elif complete:
                            fed_all.set()
                            break
                        else:
                            stop.wait(feed_interval)
            except (OSError, ValueError):
                pass  # the decoder was killed (close())
            finally:
                try:
                    decoder.stdin.close()
                except OSError:
                    pass

        def reader() -> None:
            while True:
                buf = decoder.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                frames.put(buf)
            frames.put(None)

        self._feeder_thread = threading.Thread(target=feeder, daemon=True)
        self._reader_thread = threading.Thread(target=reader, daemon=True)
        self._feeder_thread.start()
        self._reader_thread.start()
