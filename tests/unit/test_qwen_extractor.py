"""QwenExtractor's StreamingModel steps (start / predict / stop), with _llm_call() stubbed:
no weights are loaded."""

import json

import pytest

from llm_extraction_service.model import QwenExtractor


class Replies:
    """A stand-in for _llm_call(): replies per category, recording which categories were prompted."""

    def __init__(self, extractor, replies=None, default='{"extractions": []}'):
        self._category_of = {p: c for c, p in extractor._prompts.items()}
        self.replies = replies or {}  # category -> reply text, or a list of them, one per call
        self.default = default
        self.prompted = []

    def __call__(self, messages):
        category = self._category_of[messages[0]["content"]]
        self.prompted.append(category)
        reply = self.replies.get(category, self.default)
        return reply.pop(0) if isinstance(reply, list) else reply


@pytest.fixture
def extractor():
    """A QwenExtractor without weights, prompts or linker (bypassing SingletonMeta and __init__)."""
    ex = object.__new__(QwenExtractor)
    ex._prompts = {c: f"prompt for {c}" for c in QwenExtractor.CATEGORIES}
    ex._linker = None
    ex._rxnav = None
    ex._link_cache = {}
    ex.start()
    return ex


def segment(n: int, text: str = "patient has high blood pressure") -> dict:
    words = [{"word": w, "start": n * 10 + i, "end": n * 10 + i + 0.5} for i, w in enumerate(text.split())]
    return {"segment_id": n, "start": n * 10.0, "end": n * 10.0 + len(words), "text": words}


HISTORY = json.dumps({"extractions": [{"entity": "patient_history", "quotes": ["high blood pressure"]}]})


def test_predict_returns_every_category_for_the_segment(extractor):
    extractor._llm_call = Replies(extractor)
    record = extractor.predict(segment(1))
    assert record["segment"] == 1 and record["start"] == 10.0 and record["end"] == 15.0
    assert set(record) == {"segment", "start", "end", "medication_administrations", *QwenExtractor.CATEGORIES}
    assert all(record[c] == [] for c in QwenExtractor.CATEGORIES)
    assert record["medication_administrations"] == []


def test_records_accumulate_across_predictions(extractor):
    extractor._llm_call = Replies(extractor, {"medical_history": [HISTORY, '{"extractions": []}', HISTORY]})
    extractor.predict(segment(1))
    extractor.predict(segment(2, "nothing new here"))
    record = extractor.predict(segment(3))
    history = record["medical_history"]
    assert [r["segment"] for r in history] == [1, 3]
    quote = history[0]["extractions"][0]["quotes"][0]
    assert quote == {"text": "high blood pressure", "time": 12}  # timed from the segment's words


def test_predict_returns_a_copy(extractor):
    extractor._llm_call = Replies(extractor, {"medical_history": HISTORY})
    record = extractor.predict(segment(1))
    record["medical_history"].clear()
    assert len(extractor.predict(segment(2))["medical_history"]) == 2


def test_an_empty_segment_prompts_nothing(extractor):
    extractor._llm_call = replies = Replies(extractor)
    record = extractor.predict({"segment_id": 4, "start": 1.0, "end": 2.0, "text": []})
    assert replies.prompted == []
    assert record["segment"] == 4


def test_start_resets_the_session(extractor):
    extractor._llm_call = Replies(extractor, {"medical_history": HISTORY}, default="not json")
    extractor.predict(segment(1))
    assert extractor.parse_failures == len(QwenExtractor.CATEGORIES) - 1
    extractor.stop()
    assert extractor.parse_failures == len(QwenExtractor.CATEGORIES) - 1  # stop() keeps the counts

    extractor.start()
    assert extractor.parse_failures == 0 and extractor.validation_errors == 0
    extractor._llm_call = Replies(extractor)
    record = extractor.predict(segment(1))
    assert record["medical_history"] == []  # the previous session's records are gone


def test_start_can_limit_the_categories(extractor):
    extractor.start(categories=["vital_signs", "medical_history"])
    extractor._llm_call = replies = Replies(extractor, {"medical_history": HISTORY})
    record = extractor.predict(segment(1))
    assert sorted(replies.prompted) == ["medical_history", "vital_signs"]
    assert len(record["medical_history"]) == 1
    assert record["medications"] == []

    extractor.start()  # back to every category
    extractor._llm_call = replies = Replies(extractor)
    extractor.predict(segment(1))
    assert sorted(replies.prompted) == sorted(QwenExtractor.CATEGORIES)


def test_start_rejects_unknown_categories(extractor):
    with pytest.raises(ValueError, match="unknown categories: nope"):
        extractor.start(categories=["nope"])
