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

    def transcribe(self, audio_path: str) -> Generator[tuple[str, float], None, None]:
        """Transcribe the audio file at audio_path and yields (text, seconds).

        seconds is the position in the audio, from its start, up to which the
        model had consumed audio when the text came out. Implementations may
        also yield empty text as a heartbeat while no text is coming.

        The base implementation only validates that the path exists.
        Subclasses should call super().transcribe(audio_path) before
        running inference.
        """
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        yield "", 0.0


class _StopStreaming(Exception):
    """Raised inside the audio reader when a stop was requested."""


class _TimedTokenStreamer:
    """generate() streamer that emits text as soon as each token is decoded.

    TextIteratorStreamer holds back the last word until the next word starts,
    so after a pause the word before it would come out stamped with the time
    of the word after it. This streamer decodes every token immediately and
    stamps it with position() as read when the token arrived.
    Iterating yields (text, seconds); a read raises queue.Empty after `timeout`
    seconds without text.
    """

    def __init__(self, tokenizer, position, timeout: float):
        self._tokenizer = tokenizer
        self._position = position
        self._timeout = timeout
        self._queue = queue.Queue()
        self._ids = []
        self._prefix_offset = 0
        self._read_offset = 0

    def _emit(self, final: bool) -> None:
        decode = lambda ids: self._tokenizer.decode(ids, skip_special_tokens=True)
        prefix_text = decode(self._ids[self._prefix_offset : self._read_offset])
        new_text = decode(self._ids[self._prefix_offset :])
        # A trailing replacement char is a UTF-8 sequence still split across tokens.
        if len(new_text) > len(prefix_text) and (final or not new_text.endswith("\ufffd")):
            self._queue.put((new_text[len(prefix_text) :], self._position()))
            self._prefix_offset = self._read_offset
            self._read_offset = len(self._ids)

    def put(self, value) -> None:
        self._ids.extend(value.reshape(-1).tolist())
        self._emit(final=False)

    def end(self) -> None:
        self._emit(final=True)
        self._queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get(timeout=self._timeout)
        if item is None:
            raise StopIteration
        return item


class Nemotron(ASRModel):
    """Realtime streaming ASR using nvidia/nemotron-3.5-asr-streaming-0.6b.

    transcribe() follows a file that is still being written (e.g. a growing
    WAV/PCM file), staying `delay` seconds behind the writer so it only reads
    bytes that have already been flushed, and yields text as it is decoded.
    """

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
        header_bytes: int = 0,
    ):
        super().__init__(sampling_rate, channel, bytes_per_sample, language, delay)
        self._header_bytes = header_bytes
        # Heavy imports are deferred so importing this module stays cheap.
        from transformers import AutoModelForRNNT, AutoProcessor

        self._stop_event = threading.Event()
        self._finish_event = threading.Event()
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForRNNT.from_pretrained(model_id).to(device)
        self._processor.set_num_lookahead_tokens(num_lookahead_tokens)

    @property
    def chunk_seconds(self) -> float:
        """Length of the audio chunks the model decodes at a time."""
        return self._processor.num_samples_per_audio_chunk / self._sampling_rate

    @property
    def streaming_latency_ms(self) -> int:
        """Model-side streaming latency in milliseconds."""
        return self._processor.streaming_latency_ms

    def stop(self) -> None:
        """Ask a running transcribe() to finish."""
        self._stop_event.set()

    def finish(self) -> None:
        """Ask a running transcribe() to decode the rest of the file, then end.

        Unlike stop(), the delay margin is dropped, so the audio the writer
        flushed in the last `delay` seconds is transcribed too.
        """
        self._finish_event.set()

    def _wait_for_samples(self, audio_path: str, end_sample: int) -> None:
        """Block until the file holds `end_sample` samples plus the delay margin.

        Returns early, possibly short, once finish() is called.
        """
        while True:
            if self._stop_event.is_set():
                raise _StopStreaming
            finishing = self._finish_event.is_set()
            margin = 0 if finishing else int(self._delay * self._sampling_rate)
            needed_bytes = self._header_bytes + (end_sample + margin) * self._bytes_per_sample
            if finishing or os.path.getsize(audio_path) >= needed_bytes:
                return
            time.sleep(0.1)

    def _read_samples(self, f, audio_path: str, start_sample: int, num_samples: int):
        import numpy as np

        self._wait_for_samples(audio_path, start_sample + num_samples)
        f.seek(self._header_bytes + start_sample * self._bytes_per_sample)
        data = f.read(num_samples * self._bytes_per_sample)
        data = data[: len(data) // self._bytes_per_sample * self._bytes_per_sample]
        return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    def _process(self, audio, is_first: bool, num_samples: int):
        import numpy as np

        # Only short once finish() has cut the wait: pad the tail with silence.
        audio = np.pad(audio, (0, num_samples - len(audio)))
        inputs = self._processor(
            audio,
            sampling_rate=self._sampling_rate,
            is_streaming=True,
            is_first_audio_chunk=is_first,
            language=self._language,
            return_tensors="pt",
        )
        return inputs.to(self._model.device, dtype=self._model.dtype)

    def transcribe(self, audio_path: str) -> Generator[tuple[str, float], None, None]:
        """Yield (text, seconds) as the (possibly still growing) 16-bit PCM WAV file is decoded.

        seconds is the end of the audio fed to the model so far, so it is
        accurate to about one audio chunk (the model decodes chunk by chunk).
        While no text is coming it yields ("", seconds) about every 0.5 s.

        Runs until stop() is called, finish() is called and the file is
        drained, or the generator is closed.
        """
        for _ in super().transcribe(audio_path):  # validates the path exists
            pass

        self._stop_event.clear()
        self._finish_event.clear()
        processor = self._processor
        model = self._model
        audio_end = 0  # samples handed to the model so far

        with open(audio_path, "rb") as f:
            try:
                first_audio = self._read_samples(
                    f, audio_path, 0, processor.num_samples_first_audio_chunk
                )
            except _StopStreaming:
                return
            if len(first_audio) == 0:  # finish() before any audio arrived
                return
            first_inputs = self._process(
                first_audio, is_first=True, num_samples=processor.num_samples_first_audio_chunk
            )
            audio_end = len(first_audio)

            def input_features_generator():
                nonlocal audio_end
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
                    if begin + len(chunk_audio) <= mel_frame_idx * hop_length:
                        return  # finish() and no audio left that isn't already decoded
                    audio_end = begin + len(chunk_audio)
                    yield self._process(
                        chunk_audio, is_first=False, num_samples=end_idx - begin
                    ).input_features
                    if len(chunk_audio) < end_idx - begin:
                        return  # that was the padded final chunk

                    mel_frame_idx += processor.num_mel_frames_per_audio_chunk
                    start_idx = mel_frame_idx * hop_length - n_fft // 2

            streamer = _TimedTokenStreamer(
                processor.tokenizer, lambda: audio_end / self._sampling_rate, timeout=0.5
            )
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
                        text, seconds = next(stream_iter)
                    except queue.Empty:
                        # No new text yet: yield an empty heartbeat so the caller sees
                        # the audio position advance, then re-check the stop flag.
                        yield "", audio_end / self._sampling_rate
                        continue
                    except StopIteration:
                        break
                    yield text, seconds
            finally:
                self._stop_event.set()
                thread.join(timeout=5)

