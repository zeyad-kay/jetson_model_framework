"""Integration test for the LLM extraction service (llm_extraction_service), on its own.

Creates a session, queues it on the extraction service over its bridge, then writes
extraction/inputs/segments.json into <session>/transcript/segments.json one segment
at a time, --interval seconds apart, the way the ASR service writes closed
segments (the whole list so far, replaced atomically). After the last segment
it ends the session, waits for the service to finish (it extracts what is
left and waits LLM_EXTRACTION_DRAIN_IDLE seconds for more), and checks:

    structured/<category>.json        one per category: its records, [{"segment",
                                      "extractions": [{...}, ...]}, ...], one per input
                                      segment that added something, in segment order,
                                      none empty; not all empty; every quote time
                                      inside its record's segment
    structured/medications.json       its records hold actions: a new administration
                                      ("given", "started") has its own id, and an update
                                      ("update", "rate_changed", "stopped") names an
                                      administration of an earlier segment by its ref
    structured/medication_administrations.json
                                      each administration once, merged from those actions
    structured/progress.json          the last input segment's id, start and end
    structured/extractions.done       created, marking the files final

Then it logs a score that never fails the test. It compares all the extracted items
with extraction/ground_truth/<category>.json, which holds records like the service's
(segment numbers follow inputs/segments.json), and medications by administration, from
ground_truth/medication_administrations.json (schema.merge_medications of its
medications.json), as the service writes them: an item matches when their quotes
overlap (the words of a quote pair, Jaccard >= 0.5), or, where either has no quotes,
when the kind is the same. The score counts coverage, not one-to-one matches: per
category, the ground-truth items some extracted item matches (recall), the extracted
items that match some ground-truth item (precision), and how often a matched pair
agrees on the kind (KIND_FIELDS: the entity, a medication's standard name, an
intervention's category) and on the value (vital signs, demographics: their numbers,
or for sex its words; medications: the dose, each administration's merged state, with
the quotes of all its actions). Empty items are ignored. Unparseable replies
and the service's validation errors (invalid items and quotes not found in the
transcript, both dropped) are reported from its log.

Every category is validated by the service (llm_extraction_service/schema.py), so every
quote is a {"text", "time"} object.

    docker compose run --rm llm_extraction_test
    docker compose run --rm llm_extraction_test --interval 0       # write all segments at once
    docker compose run --rm llm_extraction_test --score /sessions/<id>   # only score an earlier session
    python tests/integration/extraction/main.py --dry-run

Exits non-zero if any check fails. See harness.py for how it runs.
"""

import argparse
import functools
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # tests/integration, for harness.py
import harness  # noqa: E402
from testlog import fail, log  # noqa: E402

SEGMENTS = HERE / "inputs" / "segments.json"
GROUND_TRUTH = HERE / "ground_truth"

# The categories the service writes a <category>.json for (llm_extraction_service/model.py);
# Each a list of items, in the service's output and in the ground truth.
CATEGORIES = ("medications", "interventions", "vital_signs", "mechanism_of_injury", "demographics", "medical_history",
              "patient_condition")

# Scoring: quotes whose words overlap at least this much (Jaccard) count as the same mention.
QUOTE_MATCH = 0.5
# The field naming each category's kind of item, compared for matched items and preferred when
# matching: the entity, except where it is free text and another field is the fixed one.
KIND_FIELDS = {"medications": "standard_name", "interventions": "category"}
# The field holding each category's value, compared for matched items (None: no single value).
VALUE_FIELDS = {"medications": ("dose", "dose_unit"), "vital_signs": ("value",), "demographics": ("value",)}


def load_segments(path: Path) -> list[dict]:
    segments = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(segments, list) or not segments:
        fail(f"{path} has no segments")
    return sorted(segments, key=lambda s: s["segment_id"])


def segments_feed(segments: list[dict], interval: float) -> harness.Feed:
    """Writes the segments into <session>/transcript/segments.json one at a time, like the ASR."""
    def feed(session_dir: Path) -> tuple[bool, str]:
        path = session_dir / "transcript" / "segments.json"
        harness.write_json_atomic(path, [])  # the ASR writes an empty list when the session starts
        for n in range(1, len(segments) + 1):
            time.sleep(interval)
            harness.write_json_atomic(path, segments[:n])
            seg = segments[n - 1]
            log.info(f"wrote segment {seg['segment_id']} ({seg['start']:.1f}-{seg['end']:.1f}s, "
                     f"{len(seg.get('text') or [])} words)")
        return True, f"wrote {len(segments)} segments to transcript/segments.json"
    return feed


def read_json(path: Path):
    """The JSON in path; None if it is missing or not JSON."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None


ADMINISTRATIONS = "medication_administrations"


def load_extractions(structured: Path) -> dict:
    """{category: records} from structured/<category>.json, and the medication administrations
    (structured/medication_administrations.json, when there is one); a missing or unreadable
    file counts as empty."""
    out = {c: read_json(structured / f"{c}.json") or [] for c in CATEGORIES}
    if (structured / f"{ADMINISTRATIONS}.json").is_file():
        out[ADMINISTRATIONS] = read_json(structured / f"{ADMINISTRATIONS}.json") or []
    return out


def bad_categories(extractions) -> list[str]:
    """Categories missing or not a list of records, {"segment": int, "extractions": [{...}, ...]}."""
    if not isinstance(extractions, dict):
        return list(CATEGORIES)
    def is_record(r) -> bool:
        return (isinstance(r, dict) and isinstance(r.get("segment"), int)
                and isinstance(r.get("extractions"), list) and all(isinstance(it, dict) for it in r["extractions"]))
    return [c for c in CATEGORIES
            if not (isinstance(extractions.get(c), list) and all(is_record(r) for r in extractions[c]))]


def extracted(extractions: dict, category: str) -> list[dict]:
    """All the items of a category's records, in order. For medications, the administrations
    when there are any (each with all its actions' quotes), since the records hold actions."""
    if category == "medications" and ADMINISTRATIONS in extractions:
        return [a for a in extractions[ADMINISTRATIONS] if isinstance(a, dict)]
    return [it for r in extractions.get(category) or [] for it in r.get("extractions") or [] if isinstance(it, dict)]


def quotes(item: dict) -> list[str]:
    """An item's quotes as text, whether each is text or a {"text", "time"} object."""
    return [q.get("text") if isinstance(q, dict) else q for q in item.get("quotes") or []]


def item_times(item: dict) -> list:
    """An item's times: its "times", or its quotes' times."""
    if "times" in item:
        return list(item["times"] or [])
    return [q.get("time") for q in item.get("quotes") or [] if isinstance(q, dict)]


def value_of(item: dict, fields: tuple[str, ...]) -> frozenset[str]:
    """What item says for fields (a blood pressure's systolic and diastolic as well): its
    numbers when it has any ("140/89 mmHg" and 140, 89 agree), else its words."""
    text = " ".join(str(item.get(f) or "") for f in (*fields, "systolic", "diastolic"))
    nums = frozenset(str(float(n)) for n in re.findall(r"\d+(?:\.\d+)?", text))
    return nums or words(text)


def words(text) -> frozenset[str]:
    """The lowercase alphanumeric words of text, as a set."""
    return frozenset(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def quote_overlap(a: dict, b: dict) -> float:
    """The best word overlap (Jaccard) between a quote of item a and a quote of item b."""
    best = 0.0
    for qa in quotes(a):
        wa = words(qa)
        for qb in quotes(b):
            wb = words(qb)
            if wa and wb:
                best = max(best, len(wa & wb) / len(wa | wb))
    return best


def name(item: dict, category: str = "") -> frozenset[str]:
    """The words naming an item's kind (KIND_FIELDS, else its entity)."""
    return words(item.get(KIND_FIELDS.get(category, "entity")))


def has_quotes(item: dict) -> bool:
    return any(words(q) for q in quotes(item))


def scorable(item: dict, category: str = "") -> bool:
    """False for empty items, e.g. a mechanism_of_injury of {"entity": "", "quotes": []}."""
    return bool(name(item, category) or has_quotes(item))


def best_match(item: dict, pool: list[dict], category: str = "") -> dict | None:
    """The item of pool whose quotes overlap item's (at least QUOTE_MATCH), preferring one of
    the same kind (one utterance often gives several items, e.g. a set of vitals), then the
    largest overlap; where either has no quotes, one of the same kind instead. None if there
    is none."""
    kind = name(item, category)
    candidates = [((kind == name(p, category), overlap), i) for i, p in enumerate(pool)
                  if (overlap := quote_overlap(item, p)) >= QUOTE_MATCH]
    if candidates:
        return pool[max(candidates)[1]]
    return next((p for p in pool if not (has_quotes(item) and has_quotes(p)) and kind and kind == name(p, category)), None)


def same(a: dict, b: dict, fields: tuple[str, ...]) -> bool:
    """Whether a and b agree on fields (see value_of)."""
    return value_of(a, fields) == value_of(b, fields)


def score(extractions: dict, truth_dir: Path) -> None:
    """Logs how the extracted items compare with the ground truth in truth_dir (<category>.json each,
    records like the service's).

    This counts coverage rather than one-to-one matches: a ground-truth item is found when some
    extracted item overlaps it, and an extracted item is supported when it overlaps some
    ground-truth item. Entity and value agreement compare each found item with its best match.
    """
    if not any((truth_dir / f"{c}.json").is_file() for c in CATEGORIES):
        log.info(f"score: no ground truth in {truth_dir}")
        return
    truth = load_extractions(truth_dir)
    log.info(f"score of the extractions against {truth_dir}:")
    log.info(f"  {'category':<20} {'truth':>5} {'found':>5} {'extr':>5} {'suppd':>5} {'prec':>5} {'recall':>6} {'kind':>5} {'value':>6}")
    totals = [0, 0, 0, 0]
    for c in CATEGORIES:
        t = [it for it in extracted(truth, c) if scorable(it, c)]
        g = [it for it in extracted(extractions, c) if scorable(it, c)]
        pairs = [(it, m) for it in t if (m := best_match(it, g, c)) is not None]
        supported = sum(best_match(it, t, c) is not None for it in g)
        same_kind = sum(name(a, c) == name(b, c) for a, b in pairs)
        same_value = sum(same(a, b, VALUE_FIELDS[c]) for a, b in pairs) if c in VALUE_FIELDS else 0
        totals = [totals[0] + len(t), totals[1] + len(pairs), totals[2] + len(g), totals[3] + supported]
        log.info(f"  {c:<20} {len(t):>5} {len(pairs):>5} {len(g):>5} {supported:>5} {ratio(supported, len(g)):>5}"
                 f" {ratio(len(pairs), len(t)):>6}"
                 f" {ratio(same_kind, len(pairs)):>5}"
                 f" {ratio(same_value, len(pairs)) if c in VALUE_FIELDS else '-':>6}")
    log.info(f"  {'all':<20} {totals[0]:>5} {totals[1]:>5} {totals[2]:>5} {totals[3]:>5}"
             f" {ratio(totals[3], totals[2]):>5} {ratio(totals[1], totals[0]):>6}")
    log.info("  (found: truth items some extracted item overlaps; extr: extracted items; suppd: extracted"
             " items that overlap some truth item; prec: suppd / extr, recall: found / truth; kind, value: share of found items"
             " whose best match agrees on the kind (entity; medications: standard name; interventions: category), and on the value)")


def ratio(n: int, d: int) -> str:
    return f"{n / d:.2f}" if d else "-"


def verify(session_dir: Path, session_id: str, checks: harness.Checks,
           segments: list[dict], log_file: Path, truth_dir: Path) -> None:
    structured = session_dir / "structured"

    # --- <category>.json: each category's records ---
    extractions = {c: read_json(structured / f"{c}.json") for c in CATEGORIES}
    administrations = read_json(structured / f"{ADMINISTRATIONS}.json")
    bad = bad_categories(extractions)
    if not checks.check(not bad, "a <category>.json of records for every category"
                        + (f" (missing or malformed: {bad})" if bad else "")):
        return
    total = sum(len(extracted(extractions, c)) for c in CATEGORIES)
    checks.check(total > 0, f"something was extracted ({total} items in all)")

    by_id = {s["segment_id"]: s for s in segments}
    order = [s["segment_id"] for s in segments]
    problems = []
    for c in CATEGORIES:
        ids = [r["segment"] for r in extractions[c]]
        if any(i not in by_id for i in ids):
            problems.append(f"{c}: records for segments that weren't fed: {[i for i in ids if i not in by_id]}")
        elif [i for i in order if i in ids] != ids:
            problems.append(f"{c}: records out of segment order or repeated: {ids}")
        if any(not r["extractions"] for r in extractions[c]):
            problems.append(f"{c}: an empty record")
    checks.check(not problems, "every record is for an input segment, in order, and not empty"
                 + (f" ({'; '.join(problems)})" if problems else ""))

    # Quote times come from matching quotes to their segment's words, so they must fall inside it.
    outside, n = [], 0
    for c in CATEGORIES:
        for r in extractions[c]:
            seg = by_id.get(r["segment"])
            for it in r["extractions"]:
                for t in item_times(it):
                    n += 1
                    if seg is None or not (isinstance(t, (int, float)) and seg["start"] <= t <= seg["end"]):
                        outside.append((c, r["segment"], t))
    checks.check(not outside, f"every quote time ({n}) lies inside its record's segment"
                 + (f" (not: {outside[:5]})" if outside else ""))

    # --- medications: actions on administrations, and their merged state ---
    ids, bad_refs, repeated = set(), [], []
    for r in extractions["medications"]:
        new_ids = set()
        for it in r["extractions"]:
            if it.get("action") in ("given", "started"):
                if it.get("id") in ids | new_ids:
                    repeated.append(it.get("id"))
                new_ids.add(it.get("id"))
            elif it.get("ref") not in ids:  # an update names an administration from an earlier segment
                bad_refs.append((r["segment"], it.get("action"), it.get("ref")))
        ids |= new_ids
    checks.check(not repeated and not bad_refs, f"every new medication administration has its own id ({len(ids)}),"
                 " and every update names an earlier one"
                 + (f" (repeated ids: {repeated}; bad refs: {bad_refs[:5]})" if repeated or bad_refs else ""))
    admin_ids = [a.get("id") for a in administrations] if isinstance(administrations, list) else None
    checks.check(admin_ids is not None and set(admin_ids) == ids and len(admin_ids) == len(ids),
                 f"{ADMINISTRATIONS}.json has each administration once"
                 + ("" if admin_ids is not None else " (missing or not a list)"))
    extractions[ADMINISTRATIONS] = administrations if isinstance(administrations, list) else []

    # --- progress.json: the last segment taken in ---
    final = segments[-1]
    progress = read_json(structured / "progress.json")
    checks.check(progress == {"segment": final["segment_id"], "start": final["start"], "end": final["end"]},
                 f"progress.json names the last segment ({final['segment_id']}); got {progress}")

    checks.check((structured / "extractions.done").is_file(), "extractions.done was created")

    # --- the model's answers, against the ground truth (a report, never a failure) ---
    score(extractions, truth_dir)

    # --- the service's own count of model replies it couldn't parse ---
    for line in harness.session_log(log_file, session_id):
        m = re.search(r"extracted (\d+) segments \((\d+) unparseable replies, (\d+) validation errors\)", line)
        if m:
            log.info(f"service log: extracted {m.group(1)} segments, {m.group(2)} unparseable model replies, "
                     f"{m.group(3)} validation errors (invalid items, quotes not found)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segments-file", type=Path, default=SEGMENTS,
                        help=f"transcript segments to feed, ASR segments.json format (default: {SEGMENTS})")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="seconds between writing one segment and the next (default: 5)")
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH,
                        help=f"ground truth to score against, one <category>.json of records each (default: {GROUND_TRUTH})")
    parser.add_argument("--score", type=Path, metavar="SESSION_DIR",
                        help="only score the extractions (SESSION_DIR/structured/<category>.json) against the ground truth; "
                             "the service isn't contacted")
    harness.add_common_args(parser, service="llm_extraction_service", log_name="llm_extraction", timeout=1800.0, ffmpeg=False)
    args = parser.parse_args()
    harness.setup_log("llm_extraction_service", args)

    if args.score is not None:
        harness.require_file(args.score / "structured" / "progress.json")
        score(load_extractions(args.score / "structured"), args.ground_truth)
        return

    segments = load_segments(harness.require_file(args.segments_file))
    dry_run = (f"write {len(segments)} segments from {args.segments_file} into "
               f"<session>/transcript/segments.json, {args.interval:g}s apart")
    verify_fn = functools.partial(verify, segments=segments, log_file=args.log_file, truth_dir=args.ground_truth)
    harness.run("llm_extraction_service", args, segments_feed(segments, args.interval), dry_run, verify_fn)


if __name__ == "__main__":
    main()
