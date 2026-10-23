import os
import queue
import threading
import time
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

    def __init__(self, sampling_rate: int = 16000, channel: str = "mono", bytes_per_sample: int = 2, language: str = "en-US", delay: int = 4):
        self._sampling_rate = sampling_rate
        self._channel = channel
        self._bytes_per_sample = bytes_per_sample
        self._language = language
        self._delay = delay

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
    def bytes_per_sample(self) -> int:
        """Expected bytes per sample"""
        return self._bytes_per_sample
    
    @property
    def delay(self) -> int:
        """Delay in seconds between what the model is transcribing and the realtime audio"""
        return self._delay

    def transcribe(self, audio_path: str) -> Generator[str, None, None]:
        """Transcribe the audio file at audio_path and yields the text.

        The base implementation only validates that the path exists.
        Subclasses should call super().transcribe(audio_path) before
        running inference.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        yield ""


class _StopStreaming(Exception):
    """Raised inside the audio reader when a stop was requested."""


class Nemotron(ASRModel):
    """Realtime streaming ASR using nvidia/nemotron-3.5-asr-streaming-0.6b.

    transcribe() follows a file that is still being written (e.g. a growing
    WAV/PCM file), staying `delay` seconds behind the writer so it only reads
    bytes that have already been flushed, and yields text as it is decoded.
    """

    WAV_HEADER_BYTES = 44

    def __init__(
        self,
        model_id: str = "nvidia/nemotron-3.5-asr-streaming-0.6b",
        num_lookahead_tokens: int = 6,
        sampling_rate: int = 16000,
        channel: str = "mono",
        bytes_per_sample: int = 2,
        language: str = "en-US",
        delay: int = 2,
        device: str = "cuda",
    ):
        super().__init__(sampling_rate, channel, bytes_per_sample, language, delay)
        # Heavy imports are deferred so importing this module stays cheap.
        from transformers import AutoModelForRNNT, AutoProcessor

        self._stop_event = threading.Event()
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForRNNT.from_pretrained(model_id).to(device)
        self._processor.set_num_lookahead_tokens(num_lookahead_tokens)

    @property
    def streaming_latency_ms(self) -> int:
        """Model-side streaming latency in milliseconds."""
        return self._processor.streaming_latency_ms

    def stop(self) -> None:
        """Ask a running transcribe() to finish."""
        self._stop_event.set()

    def _wait_for_samples(self, audio_path: str, end_sample: int) -> None:
        """Block until the file holds `end_sample` samples plus the delay margin."""
        needed_bytes = (
            self.WAV_HEADER_BYTES
            + (end_sample + int(self._delay * self._sampling_rate)) * self._bytes_per_sample
        )
        while os.path.getsize(audio_path) < needed_bytes:
            if self._stop_event.is_set():
                raise _StopStreaming
            time.sleep(0.1)

    def _read_samples(self, f, audio_path: str, start_sample: int, num_samples: int):
        import numpy as np

        self._wait_for_samples(audio_path, start_sample + num_samples)
        f.seek(self.WAV_HEADER_BYTES + start_sample * self._bytes_per_sample)
        data = f.read(num_samples * self._bytes_per_sample)
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    def _process(self, audio, is_first: bool):
        inputs = self._processor(
            audio,
            sampling_rate=self._sampling_rate,
            is_streaming=True,
            is_first_audio_chunk=is_first,
            language=self._language,
            return_tensors="pt",
        )
        return inputs.to(self._model.device, dtype=self._model.dtype)

    def transcribe(self, audio_path: str) -> Generator[str, None, None]:
        """Yield text chunks as the (possibly still growing) 16-bit PCM WAV file is decoded.

        Runs until stop() is called or the generator is closed.
        """
        from transformers import TextIteratorStreamer

        for _ in super().transcribe(audio_path):  # validates the path exists
            pass

        self._stop_event.clear()
        processor = self._processor
        model = self._model

        with open(audio_path, "rb") as f:
            try:
                first_audio = self._read_samples(
                    f, audio_path, 0, processor.num_samples_first_audio_chunk
                )
            except _StopStreaming:
                return
            first_inputs = self._process(first_audio, is_first=True)

            def input_features_generator():
                yield first_inputs.input_features[:, : processor.num_mel_frames_first_audio_chunk, :]

                mel_frame_idx = processor.num_mel_frames_first_audio_chunk
                hop_length = processor.feature_extractor.hop_length
                n_fft = processor.feature_extractor.n_fft

                start_idx = mel_frame_idx * hop_length - n_fft // 2
                while not self._stop_event.is_set():
                    end_idx = start_idx + processor.num_samples_per_audio_chunk
                    begin = max(start_idx, 0)
                    try:
                        chunk_audio = self._read_samples(f, audio_path, begin, end_idx - begin)
                    except _StopStreaming:
                        return
                    yield self._process(chunk_audio, is_first=False).input_features

                    mel_frame_idx += processor.num_mel_frames_per_audio_chunk
                    start_idx = mel_frame_idx * hop_length - n_fft // 2

            streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True, timeout=0.5)
            thread = threading.Thread(
                target=model.generate,
                kwargs={
                    **first_inputs,
                    "input_features": input_features_generator(),
                    "streamer": streamer,
                },
                daemon=True,
            )
            thread.start()

            try:
                stream_iter = iter(streamer)
                while not self._stop_event.is_set():
                    try:
                        yield next(stream_iter)
                    except queue.Empty:
                        continue  # no new text yet; re-check the stop flag
                    except StopIteration:
                        break
            finally:
                self._stop_event.set()
                thread.join(timeout=5)

