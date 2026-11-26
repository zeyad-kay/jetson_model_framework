"""LLMExtractionService (llm_extraction_service/service.py) with a fake model: following
segments.json, writing extractions/, draining at the end of the session."""

import json
import logging

import pytest

from conftest import make_session, wait_until
from common.io import write_done, write_json
from llm_extraction_service.config import Config
from llm_extraction_service.reader import SegmentReader
from llm_extraction_service.service import LLMExtractionService

CATEGORIES = {"medications": True, "vital_signs": True}


class FakeExtractor:
    """Stands in for QwenExtractor: each segment adds one record to every category."""

    CATEGORIES = CATEGORIES

    def __init__(self, fail_on=None):
        self.fail_on = fail_on  # segment_id whose predict() raises
        self.calls = []
        self.parse_failures = 0
        self.validation_errors = 0
        self._records = {}

    def start(self):
        self.calls.append("start")
        self._records = {c: [] for c in CATEGORIES}

    def predict(self, segment):
        sid = segment["segment_id"]
        self.calls.append(("predict", sid))
        if sid == self.fail_on:
            raise RuntimeError("model failed")
        for c in CATEGORIES:
            self._records[c].append({"segment": sid, "extractions": [{"entity": c}]})
        return {"segment": sid, "start": segment["start"], "end": segment["end"],
                **{c: list(r) for c, r in self._records.items()},
                "medication_administrations": [{"id": f"m-{sid}"}]}

    def stop(self):
        self.calls.append("stop")

    def predicted(self):
        return [c[1] for c in self.calls if isinstance(c, tuple)]


def segment(n: int) -> dict:
    return {"segment_id": n, "start": float(n), "end": n + 0.5,
            "text": [{"word": f"word{n}", "start": float(n), "end": n + 0.5}]}


def write_segments(session_dir, ids) -> None:
    """Rewrites segments.json atomically with these segments, as the ASR service does."""
    write_json(session_dir / "transcript" / "segments.json", [segment(n) for n in ids])


@pytest.fixture
def session(tmp_path):
    return make_session(tmp_path, "s1", age=10)


def make_service(sessions_dir, model, abandon_timeout=1800.0) -> LLMExtractionService:
    config = Config(sessions_dir=str(sessions_dir), poll_interval=0.01, status_path=None,
                    abandon_timeout=abandon_timeout)
    service = LLMExtractionService(config)
    service.load_model = lambda: model
    service._scan_interval = 0.01
    return service


def read(path):
    return json.loads(path.read_text())


def finished(caplog, session_id="s1"):
    return f"session {session_id} finished" in caplog.text


def complete(session_dir) -> None:
    """Marks segments.json complete, as the ASR service does after its last segment."""
    write_done(session_dir / "transcript" / "segments.json")


@pytest.fixture(autouse=True)
def logs(caplog):
    caplog.set_level(logging.INFO, logger="llm_extraction_service")


def test_follows_segments_and_writes_the_epcr(session, run_service, caplog):
    model = FakeExtractor()
    service = make_service(session.parent, model)
    run_service(service)
    write_segments(session, [1, 2])
    assert wait_until(lambda: model.predicted() == [1, 2])
    write_segments(session, [1, 2, 3])  # the file is rewritten with every closed segment so far
    assert wait_until(lambda: model.predicted() == [1, 2, 3])
    assert not wait_until(lambda: finished(caplog), timeout=0.2)  # not complete yet: keeps following
    write_segments(session, [1, 2, 3, 4])
    complete(session)
    assert wait_until(lambda: finished(caplog))

    assert model.calls[0] == "start" and model.calls[-1] == "stop"
    assert model.predicted() == [1, 2, 3, 4]  # each segment once, in order, the last one too
    out = session / "extractions"
    assert [r["segment"] for r in read(out / "medications.json")] == [1, 2, 3, 4]
    assert [r["segment"] for r in read(out / "vital_signs.json")] == [1, 2, 3, 4]
    assert read(out / "medication_administrations.json") == [{"id": "m-4"}]
    assert read(out / "progress.json") == {"segment": 4, "start": 4.0, "end": 4.5}
    assert read(out / "extractions.done") == {"status": "ok"}


def test_a_transcript_complete_without_segments_ends_at_once(session, run_service, caplog):
    model = FakeExtractor()
    service = make_service(session.parent, model)
    complete(session)  # e.g. no audio was recorded
    run_service(service)
    assert wait_until(lambda: finished(caplog))
    assert model.calls == ["start", "stop"]
    assert read(session / "extractions" / "extractions.done") == {"status": "ok"}


def test_a_failing_model_marks_the_extraction_failed(session, run_service, caplog):
    model = FakeExtractor(fail_on=2)
    service = make_service(session.parent, model)
    write_segments(session, [1, 2, 3])
    complete(session)
    run_service(service)
    assert wait_until(lambda: finished(caplog))
    assert "session s1 failed" in caplog.text
    assert model.predicted() == [1, 2]
    assert model.calls[-1] == "stop"
    assert read(session / "extractions" / "progress.json")["segment"] == 1
    assert read(session / "extractions" / "extractions.done") == {"status": "failed"}


def test_shutdown_before_the_transcript_is_complete_leaves_the_session_to_redo(session, run_service, caplog):
    model = FakeExtractor()
    service = make_service(session.parent, model)
    runner = run_service(service)
    write_segments(session, [1])
    assert wait_until(lambda: model.predicted() == [1])
    runner.stop()
    assert model.calls[-1] == "stop"
    assert not (session / "extractions" / "extractions.done").exists()


def test_a_transcript_never_marked_complete_is_given_up_on(session, run_service, caplog):
    model = FakeExtractor()
    service = make_service(session.parent, model, abandon_timeout=0.3)
    write_segments(session, [1, 2])  # and the ASR service dies before marking segments.json
    run_service(service)
    assert wait_until(lambda: finished(caplog))
    assert model.predicted() == [1, 2]
    assert read(session / "extractions" / "extractions.done") == {"status": "abandoned"}


def test_each_session_starts_from_scratch(tmp_path, run_service, caplog):
    s1 = make_session(tmp_path, "s1", age=20)
    s2 = make_session(tmp_path, "s2", age=10)
    model = FakeExtractor()
    service = make_service(tmp_path, model)
    write_segments(s1, [1, 2])
    write_segments(s2, [1])
    complete(s1)
    complete(s2)
    run_service(service)
    assert wait_until(lambda: finished(caplog, "s2"))
    # s2's segment 1 is new to s2, though s1 had a segment 1 too.
    assert model.predicted() == [1, 2, 1]
    assert model.calls.count("start") == 2 and model.calls.count("stop") == 2
    assert read(s2 / "extractions" / "progress.json")["segment"] == 1


def test_only_sessions_for_the_extraction_service_are_picked_up(tmp_path, run_service, caplog):
    other = make_session(tmp_path, "asr_only", age=20, services=["asr_service"])
    mine = make_session(tmp_path, "mine", age=10, services=["llm_extraction_service"])
    complete(other)
    complete(mine)
    model = FakeExtractor()
    service = make_service(tmp_path, model)
    run_service(service)
    assert wait_until(lambda: finished(caplog, "mine"))
    assert "session asr_only started" not in caplog.text
    assert not (other / "extractions" / "extractions.done").exists()


def test_segment_reader_hands_out_each_segment_once_in_order(session):
    path = session / "transcript" / "segments.json"
    reader = SegmentReader(path)
    assert reader.read() is None  # no file yet
    path.write_text('[{"segment_id": 1')  # half-written: no segments yet
    assert reader.read() is None
    write_segments(session, [2, 1])
    assert [reader.read()["segment_id"], reader.read()["segment_id"], reader.read()] == [1, 2, None]
    write_segments(session, [1, 2])  # rewritten, nothing new
    assert reader.read() is None
    write_segments(session, [1, 2, 3, 4])
    assert [reader.read()["segment_id"], reader.read()["segment_id"], reader.read()] == [3, 4, None]
