import glob
import json
import os
import re
import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

from common.singleton import SingletonMeta


class LLMModel(metaclass=SingletonMeta):
    """Blueprint for LLM models. Subclasses override process.

    There is one instance per class rather than one across the whole hierarchy.
    llm_aggregation_service/model.py has its own copy, so the two services
    don't depend on each other.
    """

    _singleton_per_class = True

    def __init__(self, model_id: str, prompt_path: str, max_new_tokens: int = 4096, dtype: str = "bfloat16"):
        self._model_id = model_id
        self._prompt_path = prompt_path
        self._max_new_tokens = max_new_tokens
        self._dtype = dtype

    @property
    def model_id(self) -> str:
        """Hugging Face id of the model."""
        return self._model_id

    @property
    def prompt_path(self) -> str:
        """Prompt file (or directory of prompt files) the model is run with."""
        return self._prompt_path

    @property
    def max_new_tokens(self) -> int:
        """Cap on the tokens generated per call."""
        return self._max_new_tokens

    @property
    def dtype(self) -> str:
        """Torch dtype the weights are loaded in, e.g. "bfloat16"."""
        return self._dtype

    def generate(self, messages: list[dict]) -> str:
        """Reply to a chat, a list of {"role", "content"} messages. Subclasses implement it."""
        raise NotImplementedError

    def process(self, input_path: str) -> Generator[dict[str, Any], None, None]:
        """Run the model over the input at input_path and yield the records it produces.

        The base implementation only validates that the path exists.
        Subclasses should call super().process(input_path) before
        running inference.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input not found: {input_path}")
        yield {}


# ── Qwen extraction ──────────────────────────────────────────────────────────

class QwenExtractor(LLMModel):
    """Stage 1 of the ePCR pipeline: per-segment entity extraction with a Qwen model.

    The model is loaded on the GPU with transformers; decoding is greedy, and
    Qwen3 hybrid models are kept out of thinking mode.

    process() follows the segments.json the ASR service writes to
    <session>/transcript/ while it is still growing, and yields one
    extraction record per segment as each segment appears. The model reports
    what the segment says and the quotes it said it in; times are never taken
    from the model, they come from matching each quote back to the segment's
    word timestamps.
    """

    CATEGORIES = ["medications", "interventions", "vital_signs", "mechanism_of_injury",
                  "demographics", "general_notes"]
    # One entity per case, so stored as a single object rather than a list.
    OBJECT_CATEGORIES = {"mechanism_of_injury"}

    def __init__(
        self,
        model_id: str,
        prompt_path: str,
        max_new_tokens: int = 4096,
        dtype: str = "bfloat16",
        link: bool = True,
        link_threshold: float = 0.70,
        rxnav: bool = False,
        index_dir: str | None = None,
        poll_interval: float = 0.5,
        drain_idle: float = 5.0,
    ):
        """prompt_path is a directory (one call per .txt file in it, each
        returning one or more categories) or a single file (one call returning
        all of them). link attaches ATC (and with rxnav, RxNorm via the
        network) codes to medications and ICD-10-PCS codes to interventions,
        using the embedding index in index_dir; the aggregation service groups
        medications by their ATC name, so leave it on when aggregating.

        poll_interval is how often segments.json is re-read for new segments.
        After finish(), process() ends once there are no new segments and the
        file has not changed for drain_idle seconds, which gives the ASR
        service time to write the segment it closes at the end of a session.
        """
        super().__init__(model_id, prompt_path, max_new_tokens, dtype)
        self._poll_interval = poll_interval
        self._drain_idle = drain_idle
        self._finish_event = threading.Event()
        # Prompts and the linker load first, so a bad path fails before the weights load.
        self._parse_failures = 0
        self._prompts = self._load_prompts(prompt_path)
        self._linker = None
        self._rxnav = None
        if link:
            from llm_extraction_service.terminology_linker import Linker, rxnav_rxcui

            self._linker = Linker(threshold=link_threshold, index_dir=index_dir)
            self._rxnav = rxnav_rxcui if rxnav else None

        # Heavy imports are deferred so importing this module stays cheap.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA GPU available; the LLM models run on GPU only")
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=getattr(torch, dtype), device_map="auto"
        )

    def generate(self, messages: list[dict]) -> str:
        tok = self._tokenizer
        try:  # Qwen3 hybrid models: keep them out of thinking mode
            prompt = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:  # tokenizers that don't take the kwarg
            prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(self._model.device)
        out = self._model.generate(
            **inputs, max_new_tokens=self._max_new_tokens, do_sample=False,
            temperature=None, top_p=None, repetition_penalty=1.1,
        )
        return tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    @staticmethod
    def _render(text: str, shared_path: Path) -> str:
        """Substitutes the shared {fragment} placeholders of shared_path (_shared.yaml) into a prompt.

        Other braces (JSON, the runtime {entities} placeholder) are left untouched.
        """
        import yaml

        shared = yaml.safe_load(shared_path.read_text()) if shared_path.exists() else {}
        fragments = dict((shared or {}).get("fields") or {})
        fragments.update({k + "_def": v for k, v in ((shared or {}).get("definitions") or {}).items()})
        for k, v in fragments.items():
            text = text.replace("{" + k + "}", str(v))
        return text

    @staticmethod
    def _extract_json(reply: str) -> dict | None:
        """The first balanced {...} object in a model reply, parsed; None if there is none."""
        s = re.sub(r"<think>.*?</think>", "", reply, flags=re.S)
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.S)
        i = s.find("{")
        if i < 0:
            return None
        depth = 0
        for j in range(i, len(s)):
            if s[j] == "{":
                depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(s[i:j + 1])
                    except ValueError:
                        return None
                    return obj if isinstance(obj, dict) else None
        return None

    @staticmethod
    def _normalize(text: str) -> str:
        """Lowercase alphanumerics separated by single spaces, for fuzzy quote matching."""
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (text or "").lower())).strip()

    @staticmethod
    def items(record: dict, category: str) -> list[dict]:
        """The item dicts of a category, whether it is stored as one object or a list."""
        v = record.get(category)
        if isinstance(v, dict):
            return [v] if v else []
        return v if isinstance(v, list) else []

    @property
    def parse_failures(self) -> int:
        """Model replies with no parseable JSON object in the last process() run."""
        return self._parse_failures

    def finish(self) -> None:
        """Ask a running process() to extract the segments still to come, then end.

        It keeps reading segments.json until the file has stopped changing
        for drain_idle seconds, so segments the ASR service writes while
        finishing its own session are extracted too.
        """
        self._finish_event.set()

    @classmethod
    def _load_prompts(cls, prompt_path: str) -> dict[str, str]:
        """{name: rendered prompt}: one per .txt file of a directory, or the single file."""
        if os.path.isdir(prompt_path):
            files = sorted(glob.glob(os.path.join(prompt_path, "*.txt")))
            if not files:
                raise FileNotFoundError(f"no .txt prompt files in {prompt_path}")
        elif os.path.isfile(prompt_path):
            files = [prompt_path]
        else:
            raise FileNotFoundError(f"prompt not found: {prompt_path}")
        shared_path = Path(prompt_path).parent / "_shared.yaml"
        return {Path(f).stem: cls._render(Path(f).read_text(), shared_path) for f in files}

    @classmethod
    def _empty(cls, category: str):
        return {} if category in cls.OBJECT_CATEGORIES else []

    def process(self, segments_path: str) -> Generator[dict[str, Any], None, None]:
        """Yield one extraction record per segment of the (possibly still growing) segments.json, in order.

        A record is {"segment", "start", "end", <category>: [...], ...}; every
        category is present, empty when the segment says nothing about it.
        The ASR service writes only closed segments and replaces the file
        atomically, so each segment is read once, complete.

        Runs until finish() is called and the file is drained, or the
        generator is closed.
        """
        for _ in super().process(segments_path):  # validates the path exists
            pass

        self._finish_event.clear()
        self._parse_failures = 0
        last_id = 0  # segment_id of the last segment extracted
        last_change = time.monotonic()  # when segments.json was last seen to change
        last_mtime = None
        finish_time = None

        while True:
            new = [s for s in self._read_segments(segments_path) if s.get("segment_id", 0) > last_id]
            for segment in new:
                yield self.extract(segment)
                last_id = segment["segment_id"]
            if new:
                continue  # more may have arrived while the model was busy

            try:
                mtime = os.stat(segments_path).st_mtime_ns
            except FileNotFoundError:
                mtime = None
            if mtime != last_mtime:
                last_mtime, last_change = mtime, time.monotonic()

            if self._finish_event.is_set():
                if finish_time is None:
                    finish_time = time.monotonic()
                # Idle counts from finish() at the earliest, so the ASR service
                # gets drain_idle seconds to write its final segment.
                if time.monotonic() - max(last_change, finish_time) >= self._drain_idle:
                    return
            time.sleep(self._poll_interval)

    @staticmethod
    def _read_segments(segments_path: str) -> list[dict]:
        """The segments in segments.json, ordered by segment_id; [] if it can't be read right now."""
        try:
            with open(segments_path, encoding="utf-8") as f:
                segments = json.load(f)
        except (FileNotFoundError, ValueError):
            return []
        if not isinstance(segments, list):
            return []
        return sorted((s for s in segments if isinstance(s, dict)), key=lambda s: s.get("segment_id", 0))

    def extract(self, segment: dict) -> dict[str, Any]:
        """Extract one segment ({"segment_id", "start", "end", "text": [{"word", "start", "end"}]})."""
        words = segment.get("text") or []
        record = {"segment": segment.get("segment_id"), "start": segment.get("start"), "end": segment.get("end")}
        for c in self.CATEGORIES:
            record[c] = self._empty(c)
        transcript = " ".join(w["word"] for w in words).strip()
        if not transcript:
            return record

        for prompt in self._prompts.values():
            obj = self._extract_json(self.generate([
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"TRANSCRIPT:\n{transcript}"},
            ]))
            if obj is None:
                self._parse_failures += 1
                continue
            for c in self.CATEGORIES:  # keep whichever categories this call returned
                if c not in obj:
                    continue
                if c in self.OBJECT_CATEGORIES:
                    record[c] = obj[c] if isinstance(obj[c], dict) else {}
                else:
                    record[c] = obj[c] if isinstance(obj[c], list) else []

        if self._linker is not None:
            self._link(record)
        self._attach_times(record, words)
        return record

    def _link(self, record: dict) -> None:
        """Attach codes to medications and interventions, keyed on the entity name."""
        for category, system in (("medications", "atc"), ("interventions", "icd10pcs")):
            for item in self.items(record, category):
                if not isinstance(item, dict):
                    continue
                name = item.get("entity") or item.get("standard_name") or ""
                item[system] = self._linker.link(name, system)
                if category == "medications" and self._rxnav is not None:
                    item["rxnorm"] = self._rxnav(name)

    def _attach_times(self, record: dict, words: list[dict]) -> None:
        """Add "time" (earliest, seconds) and "times" (all, sorted) to each entity
        by matching its quotes back to the segment's words."""
        tokens = []  # (normalized token, index of the word it came from)
        for i, w in enumerate(words):
            tokens.extend((t, i) for t in self._normalize(w.get("word")).split())
        for c in self.CATEGORIES:
            for item in self.items(record, c):
                if not isinstance(item, dict):
                    continue
                times = {self._quote_time(q, tokens, words) for q in (item.get("quotes") or []) if isinstance(q, str)}
                times.discard(None)
                if times:
                    item["times"] = sorted(times)
                    item["time"] = item["times"][0]

    @classmethod
    def _quote_time(cls, quote: str, tokens: list[tuple[str, int]], words: list[dict]) -> float | None:
        """Start time of the run of words that best matches the quote, or None if nothing is close.

        Slides a quote-sized window over the segment's tokens and scores each by
        word overlap (Jaccard); the best window must score at least 0.5.
        """
        q = cls._normalize(quote).split()
        if not q or not tokens:
            return None
        qset = set(q)
        best, best_score = None, 0.0
        for i in range(max(1, len(tokens) - len(q) + 1)):
            window = {t for t, _ in tokens[i:i + len(q)]}
            score = len(window & qset) / len(window | qset)
            if score > best_score:
                best, best_score = i, score
        if best is None or best_score < 0.5:
            return None
        return words[tokens[best][1]].get("start")
