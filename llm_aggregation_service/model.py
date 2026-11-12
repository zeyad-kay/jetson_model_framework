import json
import os
import re
import threading
import time
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import Any

from common.singleton import SingletonMeta


class LLMModel(metaclass=SingletonMeta):
    """Blueprint for LLM models. Subclasses override process.

    There is one instance per class rather than one across the whole hierarchy.
    A copy of the one in llm_extraction_service/model.py, so the two services
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


# ── Qwen aggregation ─────────────────────────────────────────────────────────

class QwenAggregator(LLMModel):
    """Stage 2 of the ePCR pipeline: merges QwenExtractor's records into one with a Qwen model.

    Only medications go through the model, one call per drug: medications are
    grouped by their ATC concept name (falling back to the extracted name when
    unlinked) and the model turns each drug's mentions into events (single
    dose or infusion, dose, rate changes). Times are rebuilt from the
    extraction's transcript-matched times, never the model's. Every other
    category is merged without the model: identical items are collapsed and
    sorted by time.

    The model is loaded on the GPU with transformers; decoding is greedy, and
    Qwen3 hybrid models are kept out of thinking mode.

    process() follows the extractions.json the extraction service writes while it is
    still growing, and re-aggregates every record so far whenever records are
    added.
    """

    PASSTHROUGH_CATEGORIES = ["interventions", "vital_signs", "mechanism_of_injury",
                              "demographics", "general_notes"]
    DETAIL_FIELDS = ["dose", "route", "body_site", "device", "value"]
    # One entity per case, so stored as a single object rather than a list.
    OBJECT_CATEGORIES = {"mechanism_of_injury"}

    def __init__(
        self,
        model_id: str,
        prompt_path: str,
        max_new_tokens: int = 4096,
        dtype: str = "bfloat16",
        intervention_vocab: Sequence[str] | None = (
            "advanced airway", "mechanical ventilation", "oxygen therapy",
            "chest decompression", "ultrasound", "cardiac monitoring",
        ),
        poll_interval: float = 0.5,
        drain_timeout: float = 600.0,
    ):
        """intervention_vocab keeps only interventions whose category is in it,
        dropping ones the extractor invented outside that set; None keeps all.

        poll_interval is how often extractions.json is re-read for new records.
        After finish(), process() ends once the extractor has marked the
        extractions complete and every record is aggregated; if no mark comes,
        it ends once extractions.json has not changed for drain_timeout seconds.
        """
        super().__init__(model_id, prompt_path, max_new_tokens, dtype)
        # The prompt loads first, so a bad path fails before the weights load.
        if not os.path.isfile(prompt_path):
            raise FileNotFoundError(f"aggregator prompt not found: {prompt_path}")
        self._template = Path(prompt_path).read_text()
        self._vocab = {v.strip().lower() for v in intervention_vocab} if intervention_vocab else None
        self._raw_output = ""
        self._poll_interval = poll_interval
        self._drain_timeout = drain_timeout
        self._finish_event = threading.Event()
        self._timed_out = False

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
    def raw_output(self) -> str:
        """The model's raw replies (one block per drug) for the last aggregate."""
        return self._raw_output

    @property
    def timed_out(self) -> bool:
        """Whether the last process() run ended on drain_timeout rather than the extractor's mark."""
        return self._timed_out

    def finish(self) -> None:
        """Ask a running process() to aggregate the records still to come, then end.

        It keeps following extractions.json until the extractor marks it
        complete (or drain_timeout passes), since the extractor goes on
        extracting the session's last segments after the session ends.
        """
        self._finish_event.set()

    def process(self, extractions_path: str, done_path: str) -> Generator[tuple[list[dict], dict[str, Any]], None, None]:
        """Yield (records, aggregate of records) each time extractions.json gains records.

        extractions_path is the (possibly still growing, possibly not yet
        created) list of QwenExtractor records; the extractor replaces it
        atomically after each segment. done_path is the file the extractor
        creates once that list is final. If records arrive faster than they
        are aggregated, only the latest list is aggregated.

        The aggregate is {"medications": [events], "interventions": [...], ...,
        "mechanism_of_injury": {...}}. Runs until finish() is called and the
        list is final, or the generator is closed.
        """
        # Not super().process(): the extractor may not have written the file yet.
        self._finish_event.clear()
        self._timed_out = False
        aggregated = 0  # records in the last list aggregated
        last_change = time.monotonic()  # when extractions.json was last seen to change
        last_mtime = None
        finish_time = None

        while True:
            complete = os.path.exists(done_path)  # checked first: the mark follows the last write
            records = self._read_records(extractions_path)
            if len(records) > aggregated:
                aggregated = len(records)
                yield records, self.aggregate(records)
                continue  # more may have arrived while the model was busy

            try:
                mtime = os.stat(extractions_path).st_mtime_ns
            except FileNotFoundError:
                mtime = None
            if mtime != last_mtime:
                last_mtime, last_change = mtime, time.monotonic()

            if self._finish_event.is_set():
                if complete:
                    return
                if finish_time is None:
                    finish_time = time.monotonic()
                if time.monotonic() - max(last_change, finish_time) >= self._drain_timeout:
                    self._timed_out = True
                    return
            time.sleep(self._poll_interval)

    @staticmethod
    def _read_records(extractions_path: str) -> list[dict]:
        """The records in extractions.json; [] if it can't be read right now."""
        try:
            with open(extractions_path, encoding="utf-8") as f:
                records = json.load(f)
        except (FileNotFoundError, ValueError):
            return []
        return [r for r in records if isinstance(r, dict)] if isinstance(records, list) else []

    def aggregate(self, records: list[dict]) -> dict[str, Any]:
        """Aggregate QwenExtractor records (as yielded by its process())."""
        by_drug = {}
        for e in self._gather_meds(records):
            by_drug.setdefault(e["name"], []).append(e)

        medications, raws = [], []
        for drug, entities in by_drug.items():
            feed = [{k: e[k] for k in ("name", "time", "dose", "dose_unit", "route", "rate", "rate_unit", "quotes")}
                    for e in entities]
            prompt = self._template.replace("{entities}", json.dumps(feed, indent=1, ensure_ascii=False))
            raw = self.generate([{"role": "user", "content": prompt}])
            raws.append(f"===== {drug} =====\n{raw}")
            obj = self._extract_json(raw) or {}
            for event in obj.get("medications") or []:
                if not isinstance(event, dict):
                    continue
                event.pop("rate_changes", None)  # rebuilt from the event's own quotes
                self._derive_times_and_rates(event, entities)
                event.setdefault("entity", drug)
                event.setdefault("standard_name", drug)
                medications.append(event)
        self._raw_output = "\n\n".join(raws)

        medications.sort(key=lambda ev: self._time_key(ev.get("start_time")))
        result = {"medications": medications}
        result.update(self._gather_passthrough(records))
        return result

    @staticmethod
    def _time_key(t) -> float:
        """Sort key for a time in seconds; missing times sort last."""
        return float(t) if isinstance(t, (int, float)) else 1e9

    @classmethod
    def _gather_meds(cls, records: list[dict]) -> list[dict]:
        """Every extracted medication, named by its ATC concept (when linked), in time order."""
        entities = []
        for rec in records:
            seg = rec.get("segment")
            for j, it in enumerate(rec.get("medications") or []):
                if not isinstance(it, dict):
                    continue
                atc = it.get("atc") or {}
                name = atc.get("name") if atc.get("code") else (it.get("standard_name") or it.get("entity") or "")
                t = it.get("time", rec.get("start"))
                entities.append({
                    "id": f"m{seg}_{j}", "name": name, "time": t,
                    "dose": it.get("dose", ""), "dose_unit": it.get("dose_unit", ""),
                    "route": it.get("route", ""), "rate": it.get("rate") or [],
                    "rate_unit": it.get("rate_unit", ""), "quotes": it.get("quotes") or [],
                    "times": it.get("times") or ([t] if t is not None else []),
                })
        entities.sort(key=lambda e: cls._time_key(e["time"]))
        return entities

    @classmethod
    def _derive_times_and_rates(cls, event: dict, entities: list[dict]) -> None:
        """Fill an event's start/end time and rate changes from the entities it groups.

        The entities are found by matching the event's (verbatim) quotes to
        theirs, so every time is a transcript time.
        """
        quotes = {cls._normalize(q) for q in (event.get("quotes") or []) if isinstance(q, str)}
        matched = [e for e in entities if any(cls._normalize(q) in quotes for q in e["quotes"])]
        if not matched:
            return
        times = sorted({t for e in matched for t in e["times"] if t is not None}, key=cls._time_key)
        if times:
            event["start_time"] = times[0]
            event["end_time"] = times[-1] if times[-1] != times[0] else None
        event["rate_changes"] = [{"time": e["time"], "rate": r}
                                 for e in matched for r in e["rate"] if str(r).strip()]
        if not event.get("rate_unit"):
            event["rate_unit"] = next((e["rate_unit"] for e in matched if e["rate_unit"]), "")

    def _gather_passthrough(self, records: list[dict]) -> dict[str, Any]:
        """The non-medication categories, identical items merged (quotes and times unioned), in time order."""
        merged = {c: {} for c in self.PASSTHROUGH_CATEGORIES}  # category -> key -> merged item
        for rec in records:
            t0 = rec.get("start")
            for c in self.PASSTHROUGH_CATEGORIES:
                for it in self.items(rec, c):
                    if not isinstance(it, dict):
                        continue
                    if (c == "interventions" and self._vocab is not None
                            and str(it.get("category", "")).strip().lower() not in self._vocab):
                        continue
                    name = it.get("entity") or it.get("note") or ""
                    fields = {d: it[d] for d in self.DETAIL_FIELDS if it.get(d)}
                    fields.update({k: it[k] for k in ("standard_name", "category", "details") if it.get(k)})
                    key = (name.strip().lower(), json.dumps(fields, sort_keys=True, ensure_ascii=False))
                    t = it.get("time", t0)
                    quotes = [q for q in (it.get("quotes") or []) if isinstance(q, str) and q.strip()]
                    cur = merged[c].get(key)
                    if cur is None:
                        merged[c][key] = {"entity": name, "time": t,
                                          "times": list(it.get("times") or ([t] if t is not None else [])),
                                          "quotes": quotes, **fields}
                        continue
                    for q in quotes:
                        if q not in cur["quotes"]:
                            cur["quotes"].append(q)
                    if t is not None and t not in cur["times"]:
                        cur["times"].append(t)
                    if self._time_key(t) < self._time_key(cur["time"]):
                        cur["time"] = t

        result = {}
        for c in self.PASSTHROUGH_CATEGORIES:
            items = sorted(merged[c].values(), key=lambda it: self._time_key(it["time"]))
            for it in items:
                it["times"].sort(key=self._time_key)
            result[c] = (items[0] if items else {}) if c in self.OBJECT_CATEGORIES else items
        return result
