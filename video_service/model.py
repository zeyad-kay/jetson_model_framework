from typing import Any

from common.model import StreamingModel
from common.singleton import SingletonMeta


class Sam3(StreamingModel, metaclass=SingletonMeta):
    """Realtime streaming video segmentation using facebook/sam3, fed one frame at a time.

        model.start()                    begins a video: a fresh tracking state
        model.predict({"frame": frame})  segments the video's next frame, an HxWx3 uint8 RGB
                                         array, and returns {"object_ids", "scores", "boxes"}
        model.stop()                     ends the video and drops its tracking state

    predict() takes the video's frames in order and returns the objects SAM 3 tracks
    for the text prompt on each: object_ids the ids of the tracked objects, scores one
    detection score (0-1) per id and boxes one (x0, y0, x1, y1) box per id, in the
    frame's pixels.
    """

    def __init__(
        self,
        model_id: str = "facebook/sam3",
        prompt: str = "wound",
        device: str = "cuda",
        dtype: str = "float16",  # ~2.7x faster than fp32; mean mask IoU vs fp32 0.998 (bf16: 0.975)
    ):
        self._prompt = prompt
        # Heavy imports are deferred so importing this module stays cheap.
        import torch
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        self._torch = torch
        torch.set_grad_enabled(False)  # inference only; avoids building autograd graphs

        self._device = device
        self._dtype = getattr(torch, dtype)
        self._processor = Sam3VideoProcessor.from_pretrained(model_id)
        self._model = Sam3VideoModel.from_pretrained(model_id, device_map=device, dtype=self._dtype).eval()
        # How many past frames the tracker reads: mask memories (num_maskmem - 1 back) and
        # object pointers (max_object_pointers_in_encoder - 1 back). Older per-frame state is dropped.
        tracker_cfg = self._model.config.tracker_config
        self._history = max(tracker_cfg.num_maskmem, tracker_cfg.max_object_pointers_in_encoder)
        self._session = None  # the video's tracking state, from start() to stop()
        self._frame_idx = 0  # index of the video's next frame

    @property
    def prompt(self) -> str:
        """Text prompt naming what the model segments."""
        return self._prompt

    def start(self) -> None:
        """Begins a video. A fresh session each time: tracking state must not leak between videos."""
        session = self._processor.init_video_session(
            inference_device=self._device,
            processing_device=self._device,
            video_storage_device=self._device,
            dtype=self._dtype,
        )
        self._session = self._processor.add_text_prompt(inference_session=session, text=self._prompt)
        self._frame_idx = 0

    def predict(self, inputs: dict) -> dict[str, Any]:
        """Segments inputs["frame"], the video's next HxWx3 uint8 RGB frame; returns {"object_ids", "scores", "boxes"}."""
        session = self._session
        if session is None:
            raise RuntimeError("predict() before start()")
        processor = self._processor
        model = self._model
        frame_inputs = processor(images=inputs["frame"], device=self._device, return_tensors="pt").to(model.device)
        outputs = model(
            inference_session=session,
            frame=frame_inputs.pixel_values[0].to(self._dtype),  # passing a frame enables streaming mode
            reverse=False,
        )
        processed = processor.postprocess_outputs(
            session,
            outputs,
            original_sizes=frame_inputs.original_sizes,  # required for streaming inference
        )
        self._pop_cached_frames(session, self._frame_idx - self._history)
        self._frame_idx += 1
        return {"object_ids": processed["object_ids"], "scores": processed["scores"], "boxes": processed["boxes"]}

    def stop(self) -> None:
        """Ends the video: drops its tracking state, and with it the frames and features it holds on the GPU."""
        self._session = None
        self._frame_idx = 0

    @staticmethod
    def _pop_cached_frames(session, old_idx: int) -> None:
        """Drop the per-frame state of frame old_idx, which the tracker no longer reads.

        In streaming mode the session otherwise keeps every frame's pixels and outputs,
        so memory grows with the length of the video. Called once per frame, so each
        old frame is dropped exactly once. Conditioning frames are kept: the tracker
        picks among them by distance, not by a fixed window.
        """
        if old_idx < 0:
            return
        if session.processed_frames is not None:
            session.processed_frames.pop(old_idx, None)
        for out in session.output_dict_per_obj.values():
            out["non_cond_frame_outputs"].pop(old_idx, None)
        for frames in session.frames_tracked_per_obj.values():
            frames.pop(old_idx, None)
        # Despite their names these are keyed by frame index; the score dict is deep-copied every frame.
        session.obj_id_to_tracker_score_frame_wise.pop(old_idx, None)
        session.suppressed_obj_ids.pop(old_idx, None)
