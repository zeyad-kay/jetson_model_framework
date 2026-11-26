"""The ASR service (asr_service/): PcmReader reading audio.pcm chunk by chunk as it grows, Nemotron
decoding the chunks it is fed (with stand-ins for the processor and generate()), and ASRService's
steps and output marker (with a stand-in model)."""

import json
import logging
import threading

import numpy as np
import pytest

from conftest import make_session, wait_until
from asr_service.reader import PcmReader
from asr_service.config import Config
from asr_service.model import Nemotron
from asr_service.service import ASRService
from common.io import done_marker, write_done
from common.model import StreamingModel


def read(path):
    return json.loads(path.read_text())


def pcm(*values) -> bytes:
    return np.array(values, dtype="<i2").tobytes()


# --- PcmReader --------------------------------------------------------------------------

def span(index):
    """Chunks of 4 samples, every 3: they overlap by 1."""
    return 3 * index, 3 * index + 4


def test_pcm_reader_hands_out_each_chunk_once_recorded_then_the_rest(tmp_path):
    path = tmp_path / "audio.pcm"
    reader = PcmReader(path, span)
    assert reader.read() is None  # not created yet
    path.write_bytes(pcm(16384, -16384, 0) + b"\x01")  # and half of the next sample
    assert reader.read() is None  # not a chunk yet
    with path.open("ab") as f:
        f.write(b"\x00" + pcm(*range(5, 10)))  # completes the sample, then 5 more: 9
    first = reader.read()
    assert first["samples"].dtype == np.float32
    assert first["samples"].tolist() == [0.5, -0.5, 0.0, 1 / 32768] and first["final"] is False
    second = reader.read()
    assert (second["samples"] * 32768).tolist() == [1, 5, 6, 7] and second["final"] is False
    assert reader.read() is None  # samples 6 to 8 are fewer than a chunk
    write_done(path)
    last = reader.read()
    assert (last["samples"] * 32768).tolist() == [7, 8, 9] and last["final"] is True  # the rest
    with pytest.raises(EOFError):
        reader.read()
    reader.close()


def test_pcm_reader_ending_on_a_whole_chunk_still_hands_out_the_rest(tmp_path):
    path = tmp_path / "audio.pcm"
    path.write_bytes(pcm(*range(1, 5)))
    write_done(path)
    reader = PcmReader(path, span)
    assert reader.read()["final"] is False
    last = reader.read()
    assert (last["samples"] * 32768).tolist() == [4] and last["final"] is True  # the overlap
    reader.close()


def test_pcm_reader_of_a_file_complete_without_audio_gives_one_final_read(tmp_path):
    path = tmp_path / "audio.pcm"
    write_done(path)  # no stream was ever published
    reader = PcmReader(path, span)
    inputs = reader.read()
    assert inputs["final"] is True and len(inputs["samples"]) == 0
    with pytest.raises(EOFError):
        reader.read()
    reader.close()


# --- Nemotron's chunking ----------------------------------------------------------------

class FakeExtractor:
    hop_length = 10
    n_fft = 20


class FakeTokenizer:
    def decode(self, ids, skip_special_tokens=True):
        return "".join(f" w{i}" for i in ids if i != 0)  # 0: the prompt, a special token


class FakeFeatures(dict):
    def to(self, device, dtype=None):
        return self


class FakeProcessor:
    """Chunks of 25 samples (3 mel frames), then 40 (2 frames) every 20 samples: they overlap by 20."""

    num_samples_first_audio_chunk = 25
    num_mel_frames_first_audio_chunk = 3
    num_samples_per_audio_chunk = 40
    num_mel_frames_per_audio_chunk = 2
    feature_extractor = FakeExtractor()
    tokenizer = FakeTokenizer()

    def __init__(self):
        self.calls = []  # (audio, is_first)

    def __call__(self, audio, sampling_rate, is_streaming, is_first_audio_chunk, language, return_tensors):
        self.calls.append((audio.tolist(), is_first_audio_chunk))
        return FakeFeatures(input_features=audio.reshape(1, -1, 1), num_lookahead_tokens=6)


class FakeRNNT:
    """Stands in for the model's streaming generate(): one token per chunk, numbered from 1."""

    device = "cpu"
    dtype = None

    def __init__(self, fail_on=None):
        self.fail_on = fail_on
        self.chunks = []
        self.kwargs = None
        self.ended = threading.Event()

    def generate(self, input_features, streamer, **kwargs):
        self.kwargs = kwargs
        streamer.put(np.array([0]))  # the prompt
        for chunk in input_features:
            self.chunks.append(chunk.reshape(-1).tolist())
            if len(self.chunks) == self.fail_on:
                raise ValueError("decoding failed")
            streamer.put(np.array([len(self.chunks)]))
        streamer.end()
        self.ended.set()


def make_nemotron(rnnt=None) -> Nemotron:
    model = object.__new__(Nemotron)  # without loading weights
    model.sampling_rate = 10  # so seconds = samples / 10
    model.language = "en-US"
    model._end_timeout = 5
    model._processor = FakeProcessor()
    model._model = rnnt or FakeRNNT()
    model._reset()
    return model


def samples(start, stop, final=False):
    return {"samples": np.arange(start, stop, dtype=np.float32), "final": final}


def chunk(model, index):
    """Chunk index of the audio 0, 1, 2, ..."""
    return samples(*model.chunk_span(index))


def test_nemotron_chunk_spans_overlap_by_the_extractors_window():
    model = make_nemotron()
    # the first chunk; then from its 3 mel frames (30 samples) less n_fft // 2, every 2 frames
    assert [model.chunk_span(i) for i in range(4)] == [(0, 25), (20, 60), (40, 80), (60, 100)]


def test_nemotron_decodes_each_chunk_as_it_is_fed():
    model = make_nemotron()
    model.start()
    assert model.predict(chunk(model, 0)) == {"fragments": [(" w1", 2.5)], "seconds": 2.5}
    assert model.predict(chunk(model, 1)) == {"fragments": [(" w2", 6.0)], "seconds": 6.0}
    assert model.predict(chunk(model, 2)) == {"fragments": [(" w3", 8.0)], "seconds": 8.0}
    calls = model._processor.calls
    assert [(audio[0], len(audio), first) for audio, first in calls] == [(0, 25, True), (20, 40, False), (40, 40, False)]
    assert model._model.chunks[0] == [0, 1, 2]  # the first chunk's mel frames only
    assert model._model.kwargs == {"num_lookahead_tokens": 6}

    # The rest, from the next chunk's begin, padded with silence; then the decoding ends.
    out = model.predict(samples(60, 85, final=True))
    assert out == {"fragments": [(" w4", 8.5)], "seconds": 8.5}
    assert calls[-1][0] == list(range(60, 85)) + [0] * 15
    assert model._model.ended.is_set()
    model.stop()


def test_nemotron_rejects_samples_that_are_not_the_next_chunk():
    model = make_nemotron()
    model.start()
    with pytest.raises(ValueError, match="chunk 0 is 25 samples, not 10"):
        model.predict(samples(0, 10))
    with pytest.raises(ValueError, match="or fewer"):
        model.predict(samples(0, 30, final=True))
    assert model._processor.calls == []
    model.stop()


def test_nemotron_final_with_less_than_a_chunk_pads_the_first_chunk():
    model = make_nemotron()
    model.start()
    out = model.predict(samples(1, 11, final=True))
    assert out == {"fragments": [(" w1", 1.0)], "seconds": 1.0}
    assert model._processor.calls == [(list(range(1, 11)) + [0] * 15, True)]


def test_nemotron_final_without_new_audio_feeds_no_chunk():
    model = make_nemotron()
    model.start()
    model.predict(chunk(model, 0))  # samples 0 to 24, whose mel frames reach 30
    out = model.predict(samples(20, 25, final=True))  # the rest: only the overlap
    assert out == {"fragments": [], "seconds": 2.5}
    assert len(model._model.chunks) == 1 and model._model.ended.is_set()


def test_nemotron_final_without_any_audio_never_starts_decoding():
    model = make_nemotron()
    model.start()
    assert model.predict(samples(0, 0, final=True)) == {"fragments": [], "seconds": 0.0}
    assert model._model.kwargs is None
    model.stop()


def test_nemotron_raises_when_generate_fails():
    model = make_nemotron(FakeRNNT(fail_on=2))
    model.start()
    model.predict(chunk(model, 0))
    with pytest.raises(RuntimeError, match="generate"):
        model.predict(chunk(model, 1))
    model.stop()


def test_nemotron_stop_ends_the_decoding_and_start_begins_from_scratch():
    first = FakeRNNT()
    model = make_nemotron(first)
    with pytest.raises(RuntimeError, match="before start"):
        model.predict(chunk(model, 0))
    model.start()
    model.predict(chunk(model, 0))
    model.predict(chunk(model, 1))
    model.stop()
    assert first.ended.wait(5)
    model.stop()  # twice is fine

    model._model = second = FakeRNNT()
    model.start()
    assert model.predict(chunk(model, 0)) == {"fragments": [(" w1", 2.5)], "seconds": 2.5}
    assert second.chunks == [[0, 1, 2]]
    model.stop()


# --- ASRService -------------------------------------------------------------------------

class FakeNemotron(StreamingModel):
    """Stands in for Nemotron, decoding 1 s chunks that don't overlap: "Patient is stable." at 1 s
    once it has any audio. With fail, the predict() after that raises."""

    chunk_seconds = 1.0
    sampling_rate = 16000
    language = "en-US"

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []
        self.received = 0
        self.said = False

    def chunk_span(self, index):
        size = int(self.chunk_seconds * self.sampling_rate)
        return index * size, (index + 1) * size

    def start(self):
        self.calls.append("start")
        self.received, self.said = 0, False

    def predict(self, inputs):
        if self.said and self.fail:
            raise RuntimeError("model failed")
        self.received += len(inputs["samples"])
        fragments = []
        if self.received and not self.said:
            fragments, self.said = [("Patient ", 1.0), ("is stable.", 1.0)], True
        if inputs["final"]:
            self.calls.append("final")
        return {"fragments": fragments, "seconds": self.received / self.sampling_rate}

    def stop(self):
        self.calls.append("stop")


def make_asr_service(sessions_dir, model, abandon_timeout=1800.0) -> ASRService:
    service = ASRService(Config(sessions_dir=str(sessions_dir), poll_interval=0.01, status_path=None,
                                abandon_timeout=abandon_timeout))
    service.load_model = lambda: model
    service._scan_interval = 0.01
    return service


@pytest.fixture(autouse=True)
def logs(caplog):
    caplog.set_level(logging.INFO, logger="asr_service")


def finished(caplog):
    return "session s1 finished" in caplog.text


@pytest.fixture
def recorded(tmp_path):
    session_dir = make_session(tmp_path, "s1")
    (session_dir / "audio" / "audio.pcm").write_bytes(b"\0" * 32000)  # 1 s: a chunk
    return session_dir


def segment_ids(session_dir):
    return [s["segment_id"] for s in read(session_dir / "transcript" / "segments.json")]


def test_transcribes_until_audio_pcm_is_complete_and_marks_segments_json(recorded, run_service, caplog):
    model = FakeNemotron()
    run_service(make_asr_service(recorded.parent, model))
    transcript = recorded / "transcript"
    def text():  # the transcript: transcript.jsonl's fragments joined as they are
        path = transcript / "transcript.jsonl"
        return "".join(json.loads(line)["text"] for line in path.read_text().splitlines()) if path.exists() else ""

    assert wait_until(lambda: text() == "Patient is stable.")
    assert not (transcript / "transcript.txt").exists()
    assert not wait_until(lambda: finished(caplog), timeout=0.3)  # still recording
    with (recorded / "audio" / "audio.pcm").open("ab") as f:
        f.write(b"\0" * 3200)  # less than a chunk: handed out once the audio is complete
    assert not wait_until(lambda: model.received > 16000, timeout=0.3)
    write_done(recorded / "audio" / "audio.pcm")  # the server stops recording
    assert wait_until(lambda: finished(caplog))

    assert model.calls == ["start", "final", "stop"]
    assert model.received == 17600  # every sample, once
    assert read(done_marker(transcript / "segments.json")) == {"status": "ok"}
    assert segment_ids(recorded) == [1]  # the last segment, written before the marker
    lines = [json.loads(line) for line in (transcript / "transcript.jsonl").read_text().splitlines()]
    assert lines == [{"t": 1.0, "text": "Patient "}, {"t": 1.0, "text": "is stable."}]


def test_a_failing_model_marks_segments_json_failed(recorded, run_service, caplog):
    model = FakeNemotron(fail=True)
    run_service(make_asr_service(recorded.parent, model))
    assert wait_until(lambda: model.said)
    with (recorded / "audio" / "audio.pcm").open("ab") as f:
        f.write(b"\0" * 32000)  # the next chunk, whose predict() fails
    assert wait_until(lambda: finished(caplog))
    assert "session s1 failed" in caplog.text
    assert model.calls[-1] == "stop"
    assert read(recorded / "transcript" / "segments.json.done") == {"status": "failed"}
    assert segment_ids(recorded) == [1]  # what was transcribed before the failure


def test_audio_complete_without_being_recorded_ends_at_once(tmp_path, run_service, caplog):
    session = make_session(tmp_path, "s1")
    write_done(session / "audio" / "audio.pcm")  # no stream was ever published
    model = FakeNemotron()
    run_service(make_asr_service(tmp_path, model))
    assert wait_until(lambda: finished(caplog))
    assert model.calls == ["start", "final", "stop"]
    assert read(session / "transcript" / "segments.json.done") == {"status": "ok"}
    assert segment_ids(session) == []


def test_audio_never_marked_complete_is_given_up_on(recorded, run_service, caplog):
    run_service(make_asr_service(recorded.parent, FakeNemotron(), abandon_timeout=0.3))
    assert wait_until(lambda: finished(caplog))
    assert "giving up on session s1" in caplog.text
    assert read(recorded / "transcript" / "segments.json.done") == {"status": "abandoned"}
    assert segment_ids(recorded) == [1]


def test_shutdown_before_the_audio_is_complete_leaves_the_session_to_redo(recorded, run_service, caplog):
    model = FakeNemotron()
    runner = run_service(make_asr_service(recorded.parent, model))
    assert wait_until(lambda: model.said)
    runner.stop()
    assert finished(caplog) and "interrupted by shutdown" in caplog.text
    assert model.calls == ["start", "stop"]
    assert segment_ids(recorded) == [1]  # the audio so far was transcribed
    assert not (recorded / "transcript" / "segments.json.done").exists()


def test_sessions_already_transcribed_are_skipped(recorded, run_service, caplog):
    write_done(recorded / "transcript" / "segments.json")
    model = FakeNemotron()
    run_service(make_asr_service(recorded.parent, model))
    assert wait_until(lambda: "waiting for a session" in caplog.text)
    assert not wait_until(lambda: "session s1 started" in caplog.text, timeout=0.2)
    assert model.calls == []
