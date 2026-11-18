import copy
import glob
import json
import os
import re
import threading
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

from common.log import get_logger
from common.singleton import SingletonMeta
from common.utils import as_list, quote_time, read_segments, word_tokens
from llm_extraction_service import schema

log = get_logger(__name__)


class LLMModel(metaclass=SingletonMeta):
    """Blueprint for LLM models. Subclasses override extract.

    There is one instance per class rather than one across the whole hierarchy.
    """

    _singleton_per_class = True

    def __init__(self, model_id: str, prompt_path: str, max_new_tokens: int = 4096, dtype: str = "bfloat16"):
        """prompt_path is a directory (one prompt per .txt file in it) or a single prompt file;
        its prompts are loaded and rendered here (see _load_prompts), so a bad path or an
        unfilled placeholder fails before a subclass loads its weights."""
        self._model_id = model_id
        self._prompt_path = prompt_path
        self._max_new_tokens = max_new_tokens
        self._dtype = dtype
        self._prompts = self._load_prompts(prompt_path)

    @property
    def model_id(self) -> str:
        """Hugging Face id of the model."""
        return self._model_id

    @property
    def prompt_path(self) -> str:
        """Prompt file (or directory of prompt files) the model is run with."""
        return self._prompt_path

    @property
    def prompts(self) -> dict[str, str]:
        """{name: rendered prompt}, one per prompt file, named after the file without .txt."""
        return self._prompts

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
    def _prompt_fragments(cls, shared: dict) -> dict[str, str]:
        """{placeholder: text} a subclass adds to the shared fragments, given the parsed
        _shared.yaml; none by default."""
        return {}

    @classmethod
    def _render(cls, text: str, shared_path: Path) -> str:
        """Substitutes the shared {fragment} placeholders of shared_path (_shared.yaml) into a prompt.

        definitions.X becomes {X_def}; the entries of every *_fields section (common_fields,
        vital_sign_fields, ...) and of instructions become {X}. Subclasses add their own
        fragments with _prompt_fragments(). Fragments may use fragments.
        Braces that aren't a placeholder (the JSON in a prompt) are left untouched; a placeholder
        left unfilled is an error.
        """
        import yaml

        shared = (yaml.safe_load(shared_path.read_text()) if shared_path.exists() else None) or {}
        fragments = {}
        for section, entries in shared.items():
            if section.endswith("_fields") or section == "instructions":
                fragments.update(entries or {})
        fragments.update({k + "_def": v for k, v in (shared.get("definitions") or {}).items()})
        fragments.update(cls._prompt_fragments(shared))
        for _ in range(2):  # twice, for the placeholders inside fragments
            for k, v in fragments.items():
                text = text.replace("{" + k + "}", str(v).strip())
        unfilled = sorted(set(re.findall(r"\{([a-z_]+)\}", text)))
        if unfilled:
            raise ValueError(f"no fragment in {shared_path} for {', '.join('{' + u + '}' for u in unfilled)}")
        return text

    def extract(self, input_path: str) -> Generator[dict[str, Any], None, None]:
        """Run the model over the input at input_path and yield the records it produces.

        The base implementation only validates that the path exists.
        Subclasses should call super().extract(input_path) before
        running inference.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input not found: {input_path}")
        yield {}


# ── Qwen extraction ──────────────────────────────────────────────────────────

class QwenExtractor(LLMModel):
    """The ePCR of a session, extracted by a Qwen model one segment at a time.

    The model is loaded on the GPU with transformers; decoding is greedy, and
    Qwen3 hybrid models are kept out of thinking mode.

    extract() follows the segments.json the ASR service writes to
    <session>/transcript/ while it is still growing. There is one prompt per
    category, named after it (prompts/separated/medications.txt returns the
    medications). Each category's extractions are a log of records, one per
    segment that stated something new, as in templates/:

        [{"segment": 1, "extractions": [{...}, ...]}, {"segment": 3, "extractions": [...]}]

    Medications are tracked as administrations: each bolus given or infusion started
    gets an id ("ketamine-1"), and later segments report updates to it by that id
    (details stated later, a new rate, the infusion stopped); see schema.py.

    For each new segment, every prompt is given its category's earlier records (for
    medications, each administration's current state), as context, and the segment's
    transcript, and returns only what the new segment states: {"extractions": [...]} (a "segment" in the reply is ignored; the code
    sets it). The items are validated and, if any are left, pushed onto the
    category as a new record; earlier records are never changed. A reply that
    can't be parsed adds nothing.

    Every category is validated against its dataclass in schema.py: an item that doesn't fit,
    or a quote not found in the segment, is dropped and logged, and each quote is a
    {"text", "time"} object, its time found by matching it to the segment's words
    (never taken from the model). Once demographics or mechanism_of_injury is
    complete over all its records (schema.is_complete), and then QUIET_SEGMENTS
    segments in a row add nothing to it, its flag in CATEGORIES turns False and its
    prompt is no longer run for the rest of the session; the quiet segments leave
    room for a correction of what was just stated.
    """

    # Each category, in order, and whether it is still being extracted: all start True, and one
    # turns False once it is complete and QUIET_SEGMENTS segments have added nothing to it since.
    # Each extract() run works on its own copy (self._extracting), so every session starts here.
    CATEGORIES = {
        "medications": True,
        "interventions": True,
        "vital_signs": True,
        "mechanism_of_injury": True,
        "demographics": True,
        "medical_history": True,
        "patient_condition": True,
    }
    # Segments in a row that must add nothing to a complete category before its calls stop.
    QUIET_SEGMENTS = 2
    # Categories checked against their dataclass in schema.py.
    VALIDATED = set(schema.CATEGORIES)
    # Fields the code adds after the model replies; left out of the earlier records the model is
    # fed, so they look like what it reports. standard_name is too, except for medications and
    # patient_condition, where the model reports it (elsewhere schema.py fills it in).
    STATE_OMIT = ("time", "times", "atc", "icd10pcs", "rxnorm")
    REPORTS_STANDARD_NAME = {"medications", "patient_condition"}

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
        """prompt_path is a directory with exactly one prompt per category,
        <category>.txt; a file named after no category, or a category with no
        file, is an error. link attaches ATC (and with rxnav, RxNorm via the
        network) codes to medications and ICD-10-PCS codes to interventions,
        using the embedding index in index_dir; the dashboard names
        medications by their ATC concept when linked.

        poll_interval is how often segments.json is re-read for new segments.
        After finish(), extract() ends once there are no new segments and the
        file has not changed for drain_idle seconds, which gives the ASR
        service time to write the segment it closes at the end of a session.
        """
        super().__init__(model_id, prompt_path, max_new_tokens, dtype)
        self._poll_interval = poll_interval
        self._drain_idle = drain_idle
        self._finish_event = threading.Event()
        # The prompts are loaded by LLMModel; the linker loads next, both before the weights.
        self._parse_failures = 0
        self._validation_errors = 0
        self._extracting = dict(self.CATEGORIES)
        self._quiet = {c: 0 for c in self.CATEGORIES}  # segments in a row that added nothing to c
        # Each prompt returns the category it is named after, and every category has one.
        unknown = sorted(set(self._prompts) - set(self.CATEGORIES))
        missing = [c for c in self.CATEGORIES if c not in self._prompts]
        if unknown or missing:
            problems = ([f"named after no category: {', '.join(unknown)}"] if unknown else []) + \
                       ([f"missing: {', '.join(c + '.txt' for c in missing)}"] if missing else [])
            raise ValueError(f"prompts in {prompt_path}: {'; '.join(problems)}"
                             f" (there must be one <category>.txt for each of {', '.join(self.CATEGORIES)})")
        self._linker = None
        self._rxnav = None
        self._link_cache = {}  # (name, system) -> code
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

    @classmethod
    def _prompt_fragments(cls, shared: dict) -> dict[str, str]:
        """For each validated category with entities in schema.py:

        {<category>_entities}      the entities, comma-separated (e.g. {vital_signs_entities})
        {<category>_entity_list}   one "- <entity>: <description>" line each, from the
                                   category's entities in _shared.yaml, which must be exactly
                                   the schema's (a mismatch is an error)

        and, from intervention_categories in _shared.yaml, which must be exactly
        schema.Intervention.CATEGORIES:

        {intervention_list}        one "- <category>: <definition>" line each
        {intervention_categories}  the categories, comma-separated, underscores shown as spaces

        and, from medication_routes in _shared.yaml, which must be exactly the routes of
        schema.Medication.ROUTES:

        {medication_route_list}    one "- <route>: <description>" line each
        """
        fragments = {}
        categories = shared.get("intervention_categories") or {}
        if set(categories) != set(schema.Intervention.CATEGORIES):
            raise ValueError("_shared.yaml intervention_categories must be exactly schema.Intervention.CATEGORIES: "
                             f"missing {sorted(set(schema.Intervention.CATEGORIES) - set(categories))}, "
                             f"unknown {sorted(set(categories) - set(schema.Intervention.CATEGORIES))}")
        order = schema.Intervention.CATEGORIES
        fragments["intervention_list"] = "\n".join(f"- {c}: {categories[c]}" for c in order)
        fragments["intervention_categories"] = ", ".join(c.replace("_", " ") for c in order)
        routes = shared.get("medication_routes") or {}
        if set(routes) != set(schema.Medication.ROUTES):
            raise ValueError("_shared.yaml medication_routes must be exactly schema.Medication.ROUTES: "
                             f"missing {sorted(set(schema.Medication.ROUTES) - set(routes))}, "
                             f"unknown {sorted(set(routes) - set(schema.Medication.ROUTES))}")
        fragments["medication_route_list"] = "\n".join(f"- {r}: {routes[r]}" for r in schema.Medication.ROUTES)
        for c, entities in schema.ENTITIES.items():
            fragments[f"{c}_entities"] = ", ".join(entities)
            described = (shared.get(c) or {}).get("entities")
            if described is None:
                continue
            if set(described) != set(entities):
                raise ValueError(f"_shared.yaml {c}.entities must be exactly the entities schema.py accepts: "
                                 f"missing {sorted(set(entities) - set(described))}, "
                                 f"unknown {sorted(set(described) - set(entities))}")
            fragments[f"{c}_entity_list"] = "\n".join(f"- {e}: {described[e]}" for e in entities)
        return fragments

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

    @property
    def parse_failures(self) -> int:
        """Model replies with no parseable JSON object in the last extract() run."""
        return self._parse_failures

    @property
    def validation_errors(self) -> int:
        """Invalid items and quotes not found in the transcript, dropped by validation (schema.py),
        in the last extract() run."""
        return self._validation_errors

    def finish(self) -> None:
        """Ask a running extract() to take in the segments still to come, then end.

        It keeps reading segments.json until the file has stopped changing
        for drain_idle seconds, so segments the ASR service writes while
        finishing its own session are extracted too.
        """
        self._finish_event.set()

    def extract(self, segments_path: str) -> Generator[dict[str, Any], None, None]:
        """Yield the extractions after each segment of the (possibly still growing) segments.json,
        in order.

        A record is {"segment", "start", "end", <category>: [...], ...,
        "medication_administrations": [...]}: the segment's id and times, every
        category's records once that segment is taken in, and the current state of each
        medication administration (schema.merge_medications). The
        ASR service writes only closed segments and replaces the file
        atomically, so each segment is read once, complete.

        Runs until finish() is called and the file is drained, or the
        generator is closed.
        """
        for _ in super().extract(segments_path):  # validates the path exists
            pass

        self._finish_event.clear()
        self._parse_failures = 0
        self._validation_errors = 0
        self._extracting = dict(self.CATEGORIES)
        self._quiet = {c: 0 for c in self.CATEGORIES}  # segments in a row that added nothing to c
        extractions = {c: [] for c in self.CATEGORIES}
        last_id = 0  # segment_id of the last segment extracted
        last_change = time.monotonic()  # when segments.json was last seen to change
        last_mtime = None
        finish_time = None

        while True:
            new = [s for s in read_segments(segments_path) if s.get("segment_id", 0) > last_id]
            for segment in new:
                extractions = self.apply_segment(segment, extractions)
                yield {"segment": segment.get("segment_id"), "start": segment.get("start"),
                       "end": segment.get("end"), **copy.deepcopy(extractions),
                       "medication_administrations": schema.merge_medications(extractions["medications"])}
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

    def apply_segment(self, segment: dict, previous_extractions: dict[str, list]) -> dict[str, list]:
        """The extractions once segment ({"segment_id", "start", "end", "text": [{"word", "start",
        "end"}]}) is taken in: previous_extractions ({category: records}), with a new record
        pushed onto each category the segment states something new for. previous_extractions
        is left as it was.
        """
        words = segment.get("text") or []
        transcript = " ".join(w["word"] for w in words).strip()
        if not transcript:
            return previous_extractions

        segment_id = segment.get("segment_id")
        time_of = self._quote_timer(words)
        new = {c: list(records) for c, records in previous_extractions.items()}
        answered = set()  # categories the model gave a readable reply for
        for c, extracting in self._extracting.items():
            if not extracting:
                continue  # done: demographics or mechanism_of_injury
            obj = self._extract_json(self.generate([
                {"role": "system", "content": self._prompts[c]},
                {"role": "user", "content": self._user_message(transcript, previous_extractions[c], c)},
            ]))
            items = obj.get("extractions") if obj is not None else None
            if isinstance(items, dict):  # one item given as an object rather than a list
                items = [items]
            if not isinstance(items, list):
                self._parse_failures += 1  # no JSON, or no list of extractions in it
                continue
            answered.add(c)
            if c in self.VALIDATED:
                items = self._validate(c, items, time_of, segment_id)
            if c == "medications":
                items = self._resolve_medications(items, previous_extractions[c], segment_id)
            if items:
                if self._linker is not None:
                    self._link(c, items)
                new[c].append({"segment": segment_id, "extractions": items})

        for c in answered:
            self._update_quiet(c, new[c], added=len(new[c]) > len(previous_extractions[c]),
                               segment_id=segment_id)
        return new

    def _update_quiet(self, category: str, records: list[dict], added: bool, segment_id) -> None:
        """Counts the segments in a row that added nothing to a complete category, and turns the
        category off once there are QUIET_SEGMENTS of them. A segment that adds a record, or
        a category that isn't complete, restarts the count. Only segments the model gave a
        readable reply for are counted."""
        all_items = [it for record in records for it in record["extractions"]]
        if added or not schema.is_complete(category, all_items):
            self._quiet[category] = 0
            return
        self._quiet[category] += 1
        if self._quiet[category] >= self.QUIET_SEGMENTS:
            self._extracting[category] = False
            log.info("segment %s: %s is complete and %d segments added nothing; no more calls for it",
                     segment_id, category, self._quiet[category])

    def _resolve_medications(self, items: list[dict], records: list[dict], segment_id) -> list[dict]:
        """Gives each new administration its id ("<standard name>-<n>", n counting that medication's
        administrations from 1), and keeps an update only when its ref names an administration
        already extracted, an infusion for rate_changed and stopped; others are dropped and logged."""
        admins = {a["id"]: a for a in schema.merge_medications(records)}
        counts = {}
        for a in admins.values():
            counts[a["standard_name"].lower()] = counts.get(a["standard_name"].lower(), 0) + 1
        kept = []
        for it in items:
            if it["action"] in schema.Medication.ACTIONS:
                name = it["standard_name"].lower()
                counts[name] = counts.get(name, 0) + 1
                kept.append({"id": f"{re.sub(r'[^a-z0-9]+', '-', name).strip('-')}-{counts[name]}", **it})
                continue
            admin = admins.get(it["ref"])
            if admin is None:
                problem = f"ref {it['ref']!r} is not an administration (there are: {', '.join(admins) or 'none'})"
            elif it["action"] in ("rate_changed", "stopped") and admin["type"] != "infusion":
                problem = f"{it['action']} for {it['ref']!r}, which is a bolus, not an infusion"
            else:
                kept.append(it)
                continue
            self._validation_errors += 1
            log.warning("segment %s: medications: dropped an update: %s", segment_id, problem)
        return kept

    def _validate(self, category: str, items: list, time_of, segment_id) -> list[dict]:
        """The valid items of a validated category, their quotes timed with time_of, in the
        templates' format; each dropped item or quote is logged (see schema.validate)."""
        valid, errors = schema.validate(category, items, time_of)
        self._validation_errors += len(errors)
        for e in errors:
            log.warning("segment %s: %s", segment_id, e)
        return valid

    @staticmethod
    def _quote_timer(words: list[dict]):
        """time_of(text) for a segment: the start of the run of the segment's words the quote
        best matches, or None when it matches none."""
        tokens = word_tokens(words)
        return lambda text: quote_time(text, tokens, words)

    def _user_message(self, transcript: str, records: list[dict], category: str) -> str:
        """The category's earlier records (without the fields the code adds, and with each quote
        as its text alone), then the new segment's transcript. For medications, the current
        state of each administration instead (schema.merge_medications), without quotes.

        Both go in the user message, so the system prompt stays the same for every call.
        """
        def strip(it: dict) -> dict:
            omit = self.STATE_OMIT + (() if category in self.REPORTS_STANDARD_NAME else ("standard_name",))
            out = {k: v for k, v in it.items() if k not in omit}
            if isinstance(out.get("quotes"), list):
                out["quotes"] = [q.get("text") if isinstance(q, dict) else q for q in out["quotes"]]
            return out

        if category == "medications":
            keep = ("id", "type", "entity", "standard_name", *schema.MedicationUpdate.DETAILS, "status")
            earlier = [{k: a[k] for k in keep if a.get(k) is not None} for a in schema.merge_medications(records)]
        else:
            earlier = [{"segment": r.get("segment"),
                        "extractions": [strip(it) for it in as_list(r.get("extractions")) if isinstance(it, dict)]}
                       for r in records]
        return (f"PREVIOUS EXTRACTIONS:\n{json.dumps(earlier, ensure_ascii=False)}\n\n"
                f"TRANSCRIPT OF THE NEW SEGMENT:\n{transcript}")

    def _link(self, category: str, items: list[dict]) -> None:
        """Attach codes to new medication (ATC, and RxNorm with rxnav) and intervention
        (ICD-10-PCS) items, keyed on the entity name (cached per name)."""
        system = {"medications": "atc", "interventions": "icd10pcs"}.get(category)
        if system is None:
            return

        def cached(name: str, system: str, lookup):
            key = (name, system)
            if key not in self._link_cache:
                self._link_cache[key] = lookup(name)
            return self._link_cache[key]

        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("standard_name") or item.get("entity")  # "ketamine" links better than "ket"
            if not name:
                continue  # a medication update: its administration has the codes
            item[system] = cached(name, system, lambda n: self._linker.link(n, system))
            if category == "medications" and self._rxnav is not None:
                item["rxnorm"] = cached(name, "rxnorm", self._rxnav)
