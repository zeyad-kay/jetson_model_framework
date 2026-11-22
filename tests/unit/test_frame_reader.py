"""FrameReader (video_service/frames.py): decoding an MPEG-TS file while it is written.

Needs an ffmpeg that reads from pipes and encodes H.264 (the host's minimal build doesn't);
the tests skip without one. The video service's image has it:

    docker run --rm -v "$PWD":/repo -w /repo --entrypoint python3 \\
        jetson_model_framework-video_service -m pytest -p no:cacheprovider tests/unit/test_frame_reader.py
"""

import shutil
import subprocess
import threading
import time

import pytest

np = pytest.importorskip("numpy")

from common.io import write_done  # noqa: E402
from video_service.frames import FrameReader  # noqa: E402

WIDTH, HEIGHT, FPS = 128, 64, 10


def frame(value, rng) -> "np.ndarray":
    """Noise in the top half, so the stream is as large as a camera's (ffmpeg decodes nothing until it
    has read -probesize bytes, 32 KB); grey `value` in the bottom half, which value_of() reads back."""
    image = np.full((HEIGHT, WIDTH, 3), value, np.uint8)
    image[: HEIGHT // 2] = rng.integers(0, 256, (HEIGHT // 2, WIDTH, 3), np.uint8)
    return image


def value_of(image) -> int:
    """The grey value frame() put in the bottom half, read away from the noise (H.264 blocks are 16 px)."""
    return round(float(image[HEIGHT * 3 // 4:].mean()) / 10) * 10


def encode(values) -> bytes:
    """An MPEG-TS stream with one frame() per value, at FPS: H.264 like the phone's, a keyframe every 5 frames.

    libx264rgb, since the video service's image has no libx264; the decoder reads it the same way.
    """
    rng = np.random.default_rng(0)
    raw = b"".join(frame(v, rng).tobytes() for v in values)
    res = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264rgb", "-g", "5", "-bf", "0",
         "-f", "mpegts", "-"],
        input=raw, capture_output=True,
    )
    if res.returncode != 0:
        pytest.skip(f"this ffmpeg can't make a test stream: {res.stderr.decode(errors='replace').strip()[-200:]}")
    return res.stdout


@pytest.fixture(autouse=True)
def needs_ffmpeg():
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe") and shutil.which("tail")):
        pytest.skip("needs ffmpeg, ffprobe and tail")


def read_frames(reader, count, timeout=10.0):
    """Reads until count frames have come out or timeout seconds pass."""
    frames = []
    deadline = time.monotonic() + timeout
    while len(frames) < count and time.monotonic() < deadline:
        inputs = reader.read()
        if inputs is None:
            time.sleep(0.01)
        else:
            frames.append(inputs)
    return frames


def test_reads_the_frames_of_a_file_as_it_is_written_until_it_is_complete(tmp_path):
    values = [10 * i for i in range(20)]
    stream = encode(values)
    path = tmp_path / "video.ts"
    path.touch()  # the recorder creates the file before it has a stream

    def record():
        with open(path, "ab") as f:
            for i in range(0, len(stream), 4096):
                f.write(stream[i:i + 4096])
                f.flush()
                time.sleep(0.05)

    reader = FrameReader(path, probe_interval=0.05, feed_interval=0.01)
    assert reader.read() is None and not reader.opened
    recorder = threading.Thread(target=record)
    recorder.start()
    try:
        # While the file grows the demuxer holds back the last few frames: their packets
        # aren't known to be complete until more is written, or the input ends.
        frames = read_frames(reader, len(values) - 5, timeout=5)
        recorder.join()
        time.sleep(0.3)
        frames += read_frames(reader, len(values), timeout=0.3)
        assert len(frames) < len(values)  # some are held back
        write_done(path)  # the recorder has stopped: the reader ends the decoder's input
        frames += read_frames(reader, len(values) - len(frames), timeout=5)
        with pytest.raises(EOFError, match="is complete"):
            read_until_eof(reader)
    finally:
        recorder.join()
        reader.close()
    # Not fps: probed from the first few KB, a synthetic stream's frame rate can come out wrong.
    assert (reader.width, reader.height) == (WIDTH, HEIGHT)
    assert all(f["frame"].shape == (HEIGHT, WIDTH, 3) and f["frame"].dtype == np.uint8 for f in frames)
    # Every frame, in order: the end of the input flushed the ones held back.
    assert [value_of(f["frame"]) for f in frames] == values
    assert [f["seconds"] for f in frames] == [i / reader.fps for i in range(len(frames))]


def read_until_eof(reader, timeout=5.0):
    """Reads until read() raises (the EOFError the test expects); fails if it doesn't in time."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        reader.read()
        time.sleep(0.01)
    raise AssertionError("no EOFError")


def test_a_file_complete_from_the_start_is_read_whole(tmp_path):
    path = tmp_path / "video.ts"
    path.write_bytes(encode(range(0, 200, 10)))
    write_done(path)
    reader = FrameReader(path, probe_interval=0.01)
    try:
        frames = read_frames(reader, 20)
        assert [value_of(f["frame"]) for f in frames] == list(range(0, 200, 10))
        with pytest.raises(EOFError):
            read_until_eof(reader)
        with pytest.raises(EOFError):
            reader.read()  # and stays ended
    finally:
        reader.close()


@pytest.mark.parametrize("content", [None, b"", b"not a video" * 100])
def test_a_file_complete_without_a_video_stream_ends_at_once(tmp_path, content):
    path = tmp_path / "video.ts"
    if content is not None:
        path.write_bytes(content)
    write_done(path)  # e.g. no stream was ever published
    reader = FrameReader(path, probe_timeout=60, probe_interval=0.01)
    with pytest.raises(EOFError, match="without a readable video stream"):
        reader.read()
    reader.close()


def test_waits_for_the_file_to_exist(tmp_path):
    path = tmp_path / "video.ts"
    reader = FrameReader(path, probe_timeout=0.2, probe_interval=0.01)
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:  # longer than probe_timeout: it only counts once the file exists
        assert reader.read() is None
    assert not reader.opened
    path.write_bytes(encode(range(0, 200, 10)))
    try:
        assert len(read_frames(reader, 2)) == 2
        assert reader.opened
        assert (reader.width, reader.height, reader.fps) == (WIDTH, HEIGHT, FPS)
    finally:
        reader.close()


def test_times_out_on_a_file_without_a_video_stream(tmp_path):
    path = tmp_path / "video.ts"
    path.write_bytes(b"not a video" * 1000)
    reader = FrameReader(path, probe_timeout=0.3, probe_interval=0.05)
    with pytest.raises(TimeoutError):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            assert reader.read() is None
            time.sleep(0.01)
    reader.close()


def test_close_stops_a_decoder_blocked_on_a_full_queue(tmp_path):
    path = tmp_path / "video.ts"
    path.write_bytes(encode(range(0, 250, 5)))
    reader = FrameReader(path, probe_interval=0.01, queue_size=1)
    assert len(read_frames(reader, 1)) == 1
    time.sleep(0.3)  # the decoder fills the queue and blocks
    decoder, feeder = reader._decoder, reader._feeder_thread
    started = time.monotonic()
    reader.close()
    reader.close()  # twice is fine
    assert time.monotonic() - started < 5
    assert decoder.poll() is not None and not feeder.is_alive()


def test_a_decoder_that_stops_raises_eof_after_its_frames(tmp_path):
    path = tmp_path / "video.ts"
    path.write_bytes(encode(range(0, 200, 10)))
    reader = FrameReader(path, probe_interval=0.01)
    try:
        assert len(read_frames(reader, 2)) == 2
        reader._decoder.kill()  # as if the stream were broken
        with pytest.raises(EOFError):
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                reader.read()
                time.sleep(0.01)
        with pytest.raises(EOFError):
            reader.read()
    finally:
        reader.close()
