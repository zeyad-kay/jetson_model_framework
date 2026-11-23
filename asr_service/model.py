import queue
import threading
from typing import Any, Optional

import numpy as np

from common.model import StreamingModel
from common.singleton import SingletonMeta


class _FragmentStreamer:
    """generate() streamer that decodes each token as soon as it arrives, stamped with position().

    TextIteratorStreamer holds back the last word until the next word starts, so after a
    pause the word before it would come out stamped with the time of the word after it.
    This streamer decodes every token immediately and stamps it with position() as read
    when the token arrived. take() returns the (text, seconds) fragments decoded since
    the last take().
    """

    def __init__(self, tokenizer, position):
        self._tokenizer = tokenizer
        self._position = position
        self._lock = threading.Lock()
        self._fragments: list[tuple[str, float]] = []
        self._ids: list[int] = []
        self._prefix_offset = 0
        self._read_offset = 0

    def _emit(self, final: bool) -> None:
        decode = lambda ids: self._tokenizer.decode(ids, skip_special_tokens=True)
        prefix_text = decode(self._ids[self._prefix_offset : self._read_offset])
        new_text = decode(self._ids[self._prefix_offset :])
        # A trailing replacement char is a UTF-8 sequence still split across tokens.
        if len(new_text) > len(prefix_text) and (final or not new_text.endswith("�")):
            with self._lock:
                self._fragments.append((new_text[len(prefix_text) :], self._position()))
            self._prefix_offset = self._read_offset
            self._read_offset = len(self._ids)

    def put(self, value) -> None:
        self._ids.extend(value.reshape(-1).tolist())
        self._emit(final=False)

    def end(self) -> None:
        self._emit(final=True)

    def take(self) -> list[tuple[str, float]]:
        with self._lock:
            fragments, self._fragments = self._fragments, []
        return fragments


class _ChunkedGenerate:
    """model.generate() on its own thread, over mel chunks handed to it one at a time.

    Streaming generate() pulls the next chunk from its input_features generator once it has
    decoded every encoder frame of the chunks before it. feed() hands it a chunk and returns
    once generate() asks for the next one (or has ended), so that chunk is decoded. The
    token of the decoding step that asked comes out once the next chunk is fed, or at end().
    """

    def __init__(self, model, inputs: dict):
        self._chunks: queue.Queue = queue.Queue()
        self._changed = threading.Condition()
        self._requested = 0  # chunks generate() has asked for
        self._fed = 0  # chunks handed to it
        self._done = False  # whether generate() has returned
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._run, args=(model, {**inputs, "input_features": self._features()}),
            name="asr-generate", daemon=True,
        )

    def _features(self):
        while True:
            with self._changed:
                self._requested += 1
                self._changed.notify_all()
            chunk = self._chunks.get()
            if chunk is None:
                return
            yield chunk

    def _run(self, model, kwargs: dict) -> None:
        try:
            model.generate(**kwargs)
        except BaseException as exc:  # handed to the caller of feed() / end()
            self._error = exc
        finally:
            with self._changed:
                self._done = True
                self._changed.notify_all()

    def _raise_error(self) -> None:
        if self._error is not None:
            raise RuntimeError("the ASR model's generate() failed") from self._error

    def feed(self, chunk) -> None:
        """Hands generate() the next chunk and waits until it is decoded."""
        self._chunks.put(chunk)
        self._fed += 1
        if self._fed == 1:
            self._thread.start()
        with self._changed:
            self._changed.wait_for(lambda: self._requested > self._fed or self._done)
        self._raise_error()

    def end(self, timeout: Optional[float] = None) -> None:
        """Tells generate() there are no more chunks and waits for it to decode the rest and return."""
        self._chunks.put(None)
        if self._fed == 0:
            return  # never started
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise TimeoutError(f"the ASR model's generate() did not end within {timeout:g}s")
        self._raise_error()


class Nemotron(StreamingModel, metaclass=SingletonMeta):
    """Realtime streaming ASR with nvidia/nemotron-3.5-asr-streaming-0.6b.

        model.start()
        model.predict({"samples": ..., "final": False})   # {"fragments": [(text, seconds)], "seconds"}
        ...
        model.predict({"samples": ..., "final": True})    # the last fragments
        model.stop()

    The model decodes the audio in chunks, chunk_span(index) of the session's audio: the
    first one, then chunk_seconds long ones that overlap by the feature extractor's window.
    Each predict() is handed the next chunk, float32 mono samples at sampling_rate in
    [-1, 1) (see PcmReader), decodes it, and returns the text fragments (tokens) decoded
    meanwhile, each stamped with the end of the audio fed to the model when it came out (so
    accurate to about one chunk), and seconds, the end of the audio fed so far. A fragment
    can come out one chunk late: the last token of a chunk's decoding comes with the next
    chunk. With final, samples are the rest of the audio from the next chunk's begin (fewer
    than a chunk, maybe none): it is padded with silence into a last chunk and decoded if
    any of it hasn't been yet, and the decoding ends.

    The model's streaming generate() runs on a thread per session, pulling the chunks
    predict() hands it (see _ChunkedGenerate); predict() returns once they are decoded.
    """

    def __init__(
        self,
        model_id: str = "nvidia/nemotron-3.5-asr-streaming-0.6b",
        num_lookahead_tokens: int = 6,
        sampling_rate: int = 16000,
        language: str = "en-US",
        device: str = "cuda",
        end_timeout: float = 60.0,
    ):
        # Heavy imports are deferred so importing this module stays cheap.
        from transformers import AutoModelForRNNT, AutoProcessor

        self.sampling_rate = sampling_rate
        self.language = language
        self._end_timeout = end_timeout  # seconds the decoding of the final chunk may take
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForRNNT.from_pretrained(model_id).to(device)
        self._processor.set_num_lookahead_tokens(num_lookahead_tokens)
        self._reset()

    def _reset(self) -> None:
        self._started = False
        self._chunks = 0  # chunks fed to the model
        self._audio_end = 0  # samples fed to the model
        self._streamer: Optional[_FragmentStreamer] = None
        self._generate: Optional[_ChunkedGenerate] = None

    @property
    def chunk_seconds(self) -> float:
        """Length of the audio chunks the model decodes at a time."""
        return self._processor.num_samples_per_audio_chunk / self.sampling_rate

    @property
    def streaming_latency_ms(self) -> int:
        """Model-side streaming latency in milliseconds."""
        return self._processor.streaming_latency_ms

    def chunk_span(self, index: int) -> tuple[int, int]:
        """The samples [begin, end) of the session's audio that chunk index (from 0) covers.

        A chunk after the first starts n_fft // 2 samples before its first mel frame, whose
        window reaches back that far, so it overlaps the chunk before it."""
        processor = self._processor
        if index == 0:
            return 0, processor.num_samples_first_audio_chunk
        extractor = processor.feature_extractor
        start = self._mel_frames(index) * extractor.hop_length - extractor.n_fft // 2
        return max(start, 0), start + processor.num_samples_per_audio_chunk

    def _mel_frames(self, chunks: int) -> int:
        """Mel frames fed to the model once it has been fed chunks chunks."""
        processor = self._processor
        if chunks == 0:
            return 0
        return processor.num_mel_frames_first_audio_chunk + (chunks - 1) * processor.num_mel_frames_per_audio_chunk

    def start(self) -> None:
        self.stop()
        self._started = True
        self._streamer = _FragmentStreamer(self._processor.tokenizer, lambda: self._audio_end / self.sampling_rate)

    def predict(self, inputs: dict) -> dict[str, Any]:
        if not self._started:
            raise RuntimeError("Nemotron.predict() before start()")
        samples = np.asarray(inputs["samples"], dtype=np.float32)
        final = bool(inputs.get("final"))
        begin, end = self.chunk_span(self._chunks)
        if len(samples) > end - begin or (not final and len(samples) != end - begin):
            raise ValueError(f"chunk {self._chunks} is {end - begin} samples, not {len(samples)}"
                             + (" (or fewer, with final)" if final else ""))
        if final:
            # Fed only if it reaches past the mel frames already decoded.
            decoded = self._mel_frames(self._chunks) * self._processor.feature_extractor.hop_length
            if begin + len(samples) > decoded:
                self._feed(samples, begin, end)
            generate, self._generate = self._generate, None
            if generate is not None:
                generate.end(timeout=self._end_timeout)
        else:
            self._feed(samples, begin, end)
        return {"fragments": self._streamer.take(), "seconds": self._audio_end / self.sampling_rate}

    def stop(self) -> None:
        """Ends the session's decoding (dropping what it hasn't output) and releases its state."""
        generate = self._generate
        self._reset()
        if generate is not None:
            try:
                generate.end(timeout=5)
            except Exception:
                pass  # the session is over: its outputs are no longer wanted

    def _feed(self, samples: np.ndarray, begin: int, end: int) -> None:
        """Feeds samples, the session's audio from begin, as the chunk [begin, end) (padded with
        silence if fewer) and waits for it to be decoded."""
        processor = self._processor
        is_first = self._chunks == 0
        features = processor(
            np.pad(samples, (0, end - begin - len(samples))),
            sampling_rate=self.sampling_rate,
            is_streaming=True,
            is_first_audio_chunk=is_first,
            language=self.language,
            return_tensors="pt",
        ).to(self._model.device, dtype=self._model.dtype)
        self._audio_end = begin + len(samples)
        if is_first:
            self._generate = _ChunkedGenerate(self._model, {**features, "streamer": self._streamer})
            self._generate.feed(features["input_features"][:, : processor.num_mel_frames_first_audio_chunk, :])
        else:
            self._generate.feed(features["input_features"])
        self._chunks += 1
