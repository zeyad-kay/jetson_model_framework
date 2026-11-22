"""The done marker helpers, and the extraction service's reader ending on segments.json.done.
(The server's markers: test_server.py; the ASR service's: test_asr_service.py.)"""

import json

import pytest

from conftest import make_session
from common.io import done_marker, write_done
from llm_extraction_service.reader import SegmentReader
from test_llm_extraction_service import write_segments


def read(path):
    return json.loads(path.read_text())


def test_done_marker_helpers(tmp_path):
    path = tmp_path / "audio.pcm"
    assert done_marker(path) == tmp_path / "audio.pcm.done"
    write_done(path)
    assert read(done_marker(path)) == {"status": "ok"}
    write_done(path, "failed")
    assert read(done_marker(path)) == {"status": "failed"}


# --- the extraction service: segments.json.done ----------------------------------------

@pytest.fixture
def transcript(tmp_path):
    return make_session(tmp_path, "s1")


def test_segment_reader_hands_out_every_segment_before_eof(transcript):
    path = transcript / "transcript" / "segments.json"
    reader = SegmentReader(path)
    write_segments(transcript, [1, 2])
    write_done(path)
    assert [reader.read()["segment_id"], reader.read()["segment_id"]] == [1, 2]
    with pytest.raises(EOFError):
        reader.read()


def test_segment_reader_ends_when_the_transcript_is_complete_without_segments(transcript):
    path = transcript / "transcript" / "segments.json"
    reader = SegmentReader(path)
    assert reader.read() is None
    write_done(path, "failed")  # e.g. the ASR service failed before writing any
    with pytest.raises(EOFError):
        reader.read()
