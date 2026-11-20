import os
import queue
import subprocess
import threading
import time
from typing import Any
from collections.abc import Generator

from common.singleton import SingletonMeta


class VideoModel(metaclass=SingletonMeta):
    """Blueprint for video segmentation models. Subclasses override segment."""

    def __init__(self, prompt: str = "wound", pixel_format: str = "rgb24", idle_timeout: float = 10.0):
        self._prompt = prompt
        self._pixel_format = pixel_format
        self._idle_timeout = idle_timeout

    @property
    def prompt(self) -> str:
        """Text prompt naming what the model segments."""
        return self._prompt

    @property
    def pixel_format(self) -> str:
        """Pixel format of the frames yielded by segment()."""
        return self._pixel_format

    @property
    def idle_timeout(self) -> float:
        """Seconds without a new frame before the input is considered finished."""
        return self._idle_timeout

    def segment(self, video_path: str) -> Generator[tuple[Any, list[int], Any, Any, float], None, None]:
        """Segment the video at video_path and yield (frame, object_ids, scores, boxes, seconds).

        frame is an HxWx3 uint8 RGB numpy array, object_ids the ids of the
        tracked objects, scores one detection score (0-1) per id and boxes one
        (x0, y0, x1, y1) pixel box per id. seconds is the position of the frame
        in the video, from its start.

        The base implementation only validates that the path exists.
        Subclasses should call super().segment(video_path) before
        running inference.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        yield None, [], None, None, 0.0


class Sam3(VideoModel):
    """Realtime streaming video segmentation using facebook/sam3.

    segment() follows an MPEG-TS file that is still being written, decodes
    frames as they arrive, and yields the masks SAM 3 tracks for the text
    prompt on each frame.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam3",
        prompt: str = "wound",
        device: str = "cuda",
        dtype: str = "float16",  # ~2.7x faster than fp32; mean mask IoU vs fp32 0.998 (bf16: 0.975)
        idle_timeout: float = 10.0,
        probe_timeout: float = 60.0,
        drain_idle: float = 1.0,
    ):
        super().__init__(prompt, "rgb24", idle_timeout)
        # Heavy imports are deferred so importing this module stays cheap.
        import torch
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        self._torch = torch
        torch.set_grad_enabled(False)  # inference only; avoids building autograd graphs

        self._device = device
        self._dtype = getattr(torch, dtype)
        self._probe_timeout = probe_timeout
        self._drain_idle = drain_idle
        self._fps = None
        self._finish_event = threading.Event()
        self._processor = Sam3VideoProcessor.from_pretrained(model_id)
        self._model = Sam3VideoModel.from_pretrained(model_id, device_map=device, dtype=self._dtype).eval()

    @property
    def fps(self) -> float | None:
        """Frame rate of the video being segmented; None until segment() has probed it."""
        return self._fps

    def finish(self) -> None:
        """Ask a running segment() to process the frames already written, then end.

        The frames still queued are processed; segment() ends once
        the file has stopped growing for `drain_idle` seconds.
        """
        self._finish_event.set()

    def _probe_stream(self, path: str) -> tuple[int, int, float]:
        """Wait until the growing file has a readable video stream; return (width, height, fps)."""
        deadline = time.time() + self._probe_timeout
        while time.time() < deadline:
            if os.path.getsize(path) > 0:
                res = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0",
                     "-show_entries", "stream=width,height,r_frame_rate", "-of", "csv=p=0", path],
                    capture_output=True, text=True,
                )
                # ffprobe can list the stream twice (program + top level); use the first line
                parts = res.stdout.strip().split("\n")[0].split(",")
                if res.returncode == 0 and len(parts) == 3 and parts[0].isdigit():
                    num, _, den = parts[2].partition("/")
                    fps = float(num) / float(den or 1) if float(num) else 30.0
                    return int(parts[0]), int(parts[1]), fps
            time.sleep(0.5)
        raise TimeoutError(f"No video stream found in {path} after {self._probe_timeout}s")

    def _follow_frames(self, path: str, width: int, height: int):
        """Yield HxWx3 uint8 RGB frames from a growing MPEG-TS file until it stops growing."""
        import numpy as np

        frame_bytes = width * height * 3
        # tail -f keeps feeding bytes as the writer appends; ffmpeg decodes from stdin
        tail = subprocess.Popen(["tail", "-c", "+1", "-f", path], stdout=subprocess.PIPE)
        # The input never ends, so ffmpeg's defaults would hold back frames: it buffers ~5 MB / 5 s
        # to analyse the stream before decoding (a short recording yields nothing), and frame-
        # threaded decoding keeps the newest ~15 frames. A tiny probe and one decoder thread emit
        # each frame as it arrives; passthrough keeps ffmpeg from duplicating frames to fill the
        # frame rate it then can't measure.
        decoder = subprocess.Popen(
            ["ffmpeg", "-loglevel", "error", "-probesize", "32768", "-analyzeduration", "0", "-threads", "1",
             "-f", "mpegts", "-i", "-",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-fps_mode", "passthrough", "-"],
            stdin=tail.stdout, stdout=subprocess.PIPE,
        )
        tail.stdout.close()

        frames = queue.Queue(maxsize=8)  # bounded: backpressures tail/ffmpeg if inference is slower

        def reader():
            while True:
                buf = decoder.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                frames.put(buf)
            frames.put(None)

        threading.Thread(target=reader, daemon=True).start()
        last_frame = time.monotonic()
        try:
            while True:
                try:
                    buf = frames.get(timeout=0.5)
                except queue.Empty:
                    idle = time.monotonic() - last_frame
                    limit = self._drain_idle if self._finish_event.is_set() else self._idle_timeout
                    if idle >= limit:
                        return
                    continue
                if buf is None:
                    return
                last_frame = time.monotonic()
                yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
        finally:
            decoder.kill()
            tail.kill()

    def segment(self, video_path: str) -> Generator[tuple[Any, list[int], Any, Any, float], None, None]:
        """Yield (frame, object_ids, scores, boxes, seconds) as the (possibly still growing) MPEG-TS file is decoded.

        frame is the original HxWx3 uint8 RGB frame, scores SAM 3's detection
        score per object and boxes its (x0, y0, x1, y1) box per object, in the
        frame's pixels. seconds is the
        frame's position in the video, derived from the frame count and the
        stream's frame rate.

        Runs until finish() is called and the file is
        drained, no frame arrives for idle_timeout seconds, or the generator is closed.
        """
        for _ in super().segment(video_path):  # validates the path exists
            pass

        self._finish_event.clear()
        processor = self._processor
        model = self._model

        width, height, fps = self._probe_stream(video_path)
        self._fps = fps

        # A fresh session per call: tracking state must not leak between videos.
        session = processor.init_video_session(
            inference_device=self._device,
            processing_device=self._device,
            video_storage_device=self._device,
            dtype=self._dtype,
        )
        session = processor.add_text_prompt(inference_session=session, text=self._prompt)

        for frame_idx, frame in enumerate(self._follow_frames(video_path, width, height)):
            inputs = processor(images=frame, device=self._device, return_tensors="pt").to(model.device)
            outputs = model(
                inference_session=session,
                frame=inputs.pixel_values[0].to(self._dtype),  # passing a frame enables streaming mode
                reverse=False,
            )
            processed = processor.postprocess_outputs(
                session,
                outputs,
                original_sizes=inputs.original_sizes,  # required for streaming inference
            )
            yield frame, processed["object_ids"], processed["scores"], processed["boxes"], frame_idx / fps
