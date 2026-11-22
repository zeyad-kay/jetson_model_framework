"""Sam3 (video_service/model.py) as a StreamingModel, with SAM 3's processor and model stubbed:
start() opens a fresh tracking session, predict() feeds it one frame, stop() drops it."""

import pytest

from video_service.model import Sam3

HISTORY = 3


class FakeTensor:
    def to(self, *_):
        return self


class FakeSession:
    """The per-frame state Sam3._pop_cached_frames() trims, keyed by frame index."""

    def __init__(self, prompt=None):
        self.prompt = prompt
        self.processed_frames = {}
        self.output_dict_per_obj = {1: {"non_cond_frame_outputs": {}}}
        self.frames_tracked_per_obj = {1: {}}
        self.obj_id_to_tracker_score_frame_wise = {}
        self.suppressed_obj_ids = {}

    def frames(self):
        return sorted(self.processed_frames)


class FakeInputs:
    pixel_values = [FakeTensor()]
    original_sizes = [(2, 4)]

    def to(self, _):
        return self


class FakeProcessor:
    def __init__(self):
        self.sessions = []

    def init_video_session(self, **_):
        session = FakeSession()
        self.sessions.append(session)
        return session

    def add_text_prompt(self, inference_session, text):
        inference_session.prompt = text
        return inference_session

    def __call__(self, images, device, return_tensors):
        return FakeInputs()

    def postprocess_outputs(self, session, outputs, original_sizes):
        return {"object_ids": [1], "scores": [0.9], "boxes": [[0, 0, 4, 2]], "masks": "unused"}


class FakeModel:
    device = "cpu"

    def __call__(self, inference_session, frame, reverse):
        # Streaming mode appends the frame to the session's state, under the next frame index.
        idx = inference_session.frames_seen = getattr(inference_session, "frames_seen", -1) + 1
        inference_session.processed_frames[idx] = frame
        inference_session.output_dict_per_obj[1]["non_cond_frame_outputs"][idx] = "out"
        inference_session.frames_tracked_per_obj[1][idx] = "tracked"
        inference_session.obj_id_to_tracker_score_frame_wise[idx] = {1: 0.9}
        inference_session.suppressed_obj_ids[idx] = set()
        return "outputs"


@pytest.fixture
def sam3():
    model = object.__new__(Sam3)  # skips loading SAM 3
    model._prompt = "wound"
    model._device = "cpu"
    model._dtype = None
    model._processor = FakeProcessor()
    model._model = FakeModel()
    model._history = HISTORY
    model._session = None
    model._frame_idx = 0
    return model


def test_predict_returns_the_objects_tracked_on_the_frame(sam3):
    sam3.start()
    assert sam3.predict({"frame": "frame 0", "seconds": 0.0}) == {
        "object_ids": [1], "scores": [0.9], "boxes": [[0, 0, 4, 2]]}
    assert sam3._processor.sessions[0].prompt == "wound"


def test_predict_drops_the_state_of_frames_the_tracker_no_longer_reads(sam3):
    sam3.start()
    for i in range(10):
        sam3.predict({"frame": f"frame {i}"})
    session = sam3._session
    kept = list(range(10 - HISTORY, 10))
    assert session.frames() == kept
    assert sorted(session.output_dict_per_obj[1]["non_cond_frame_outputs"]) == kept
    assert sorted(session.frames_tracked_per_obj[1]) == kept
    assert sorted(session.obj_id_to_tracker_score_frame_wise) == kept
    assert sorted(session.suppressed_obj_ids) == kept


def test_each_start_opens_a_fresh_session(sam3):
    sam3.start()
    for i in range(5):
        sam3.predict({"frame": i})
    first = sam3._session
    sam3.stop()
    assert sam3._session is None
    sam3.start()
    assert sam3._session is not first and sam3._session.frames() == []
    sam3.predict({"frame": 0})
    assert sam3._session.frames() == [0]


def test_predict_needs_start(sam3):
    with pytest.raises(RuntimeError):
        sam3.predict({"frame": 0})
    sam3.start()
    sam3.stop()
    sam3.stop()  # twice is fine
    with pytest.raises(RuntimeError):
        sam3.predict({"frame": 0})
