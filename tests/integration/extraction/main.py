"""Integration test for the LLM extraction service (llm_extraction_service), on its own.

Creates a session, queues it on the extraction service over its bridge, then writes
extraction/inputs/segments.json into <session>/transcript/segments.json one segment
at a time, --interval seconds apart, the way the ASR service writes closed
segments (the whole list so far, replaced atomically). After the last segment
it ends the session, waits for the service to finish (it extracts what is
left and waits LLM_EXTRACTION_DRAIN_IDLE seconds for more), and checks:

    structured/extractions.json  one record per input segment, in order, with the
                                 segment's id and times and every category;
                                 entity times fall inside their segment
    structured/extractions.done  created, so the aggregation service knows the
                                 extractions are final

Then it scores the records against extraction/ground_truth/<segment>.json,
as a report in the log that never fails the test: a model item matches a ground-truth
item of the same segment and category when their quotes overlap (the words of
a quote pair, Jaccard >= 0.5), one to one; an item without quotes matches one
with the same entity instead. Empty items are ignored, and a ground-truth
segment the model has no record for counts as all missed. Per category it
logs the counts, matches, precision and recall, and how often matched items
agree on the entity and on the value (vital signs, demographics) or dose
(medications).
Unparseable model replies are reported from the service's log.

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

# The categories every extraction record carries (llm_extraction_service/model.py);
# mechanism_of_injury is a single object, the rest are lists.
LIST_CATEGORIES = ("medications", "interventions", "vital_signs", "demographics", "general_notes")
OBJECT_CATEGORIES = ("mechanism_of_injury",)
CATEGORIES = ("medications", "interventions", "vital_signs", "mechanism_of_injury", "demographics", "general_notes")

# Scoring: quotes whose words overlap at least this much (Jaccard) count as the same mention.
QUOTE_MATCH = 0.5
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


def bad_categories(record: dict) -> list[str]:
    """Categories missing or of the wrong type (a list, or an object for mechanism_of_injury)."""
    bad = [c for c in LIST_CATEGORIES if not isinstance(record.get(c), list)]
    return bad + [c for c in OBJECT_CATEGORIES if not isinstance(record.get(c), dict)]


def items(record: dict, category: str) -> list[dict]:
    value = record.get(category)
    if isinstance(value, dict):
        return [value] if value else []
    return [it for it in (value or []) if isinstance(it, dict)]


def words(text) -> frozenset[str]:
    """The lowercase alphanumeric words of text, as a set."""
    return frozenset(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def quote_overlap(a: dict, b: dict) -> float:
    """The best word overlap (Jaccard) between a quote of item a and a quote of item b."""
    best = 0.0
    for qa in a.get("quotes") or []:
        wa = words(qa)
        for qb in b.get("quotes") or []:
            wb = words(qb)
            if wa and wb:
                best = max(best, len(wa & wb) / len(wa | wb))
    return best


def name(item: dict) -> frozenset[str]:
    """The words naming an item: its entity, or a general note's text."""
    return words(item.get("entity") or item.get("note"))


def has_quotes(item: dict) -> bool:
    return any(words(q) for q in item.get("quotes") or [])


def scorable(item: dict) -> bool:
    """False for empty items, e.g. a mechanism_of_injury of {"entity": "", "quotes": []}."""
    return bool(name(item) or has_quotes(item))


def match(truth: list[dict], got: list[dict]) -> list[tuple[dict, dict]]:
    """One-to-one (truth, got) pairs: quotes that overlap enough, best overlaps first; then,
    where either item has no quotes to compare, the same entity name."""
    pairs = sorted(((quote_overlap(t, g), i, j) for i, t in enumerate(truth) for j, g in enumerate(got)),
                   key=lambda p: -p[0])
    used_t, used_g, matched = set(), set(), []

    def pair(i: int, j: int) -> None:
        used_t.add(i)
        used_g.add(j)
        matched.append((truth[i], got[j]))

    for overlap, i, j in pairs:
        if overlap < QUOTE_MATCH:
            break
        if i not in used_t and j not in used_g:
            pair(i, j)
    for i, t in enumerate(truth):
        for j, g in enumerate(got):
            if (i not in used_t and j not in used_g and not (has_quotes(t) and has_quotes(g))
                    and name(t) and name(t) == name(g)):
                pair(i, j)
    return matched


def same(a: dict, b: dict, fields: tuple[str, ...]) -> bool:
    """Whether a and b agree on fields, compared as their words."""
    return all(words(a.get(f)) == words(b.get(f)) for f in fields)


def score(records: list[dict], truth_dir: Path) -> None:
    """Logs how the records compare with the ground truth in truth_dir (<segment>.json each)."""
    truth = {}
    for path in truth_dir.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        truth[record.get("segment")] = record
    if not truth:
        log.info(f"score: no ground truth in {truth_dir}")
        return
    model = {r.get("segment"): r for r in records}
    missing = sorted(s for s in truth if s not in model)
    unscored = sorted(s for s in model if s not in truth)
    log.info(f"score against {truth_dir} ({len(truth)} segments"
          + (f"; no model record for segments {missing}, counted as all missed" if missing else "")
          + (f"; no ground truth for segments {unscored}, not scored" if unscored else "") + "):")
    log.info(f"  {'category':<20} {'truth':>5} {'model':>5} {'match':>5} {'prec':>5} {'recall':>6} {'entity':>7} {'value':>6}")
    totals = [0, 0, 0]
    for c in CATEGORIES:
        n_truth = n_got = n_match = same_entity = same_value = 0
        for segment, expected in truth.items():
            t = [it for it in items(expected, c) if scorable(it)]
            g = [it for it in items(model.get(segment, {}), c) if scorable(it)]
            pairs = match(t, g)
            n_truth, n_got, n_match = n_truth + len(t), n_got + len(g), n_match + len(pairs)
            same_entity += sum(name(a) == name(b) for a, b in pairs)
            same_value += sum(same(a, b, VALUE_FIELDS[c]) for a, b in pairs) if c in VALUE_FIELDS else 0
        totals = [totals[0] + n_truth, totals[1] + n_got, totals[2] + n_match]
        log.info(f"  {c:<20} {n_truth:>5} {n_got:>5} {n_match:>5} {ratio(n_match, n_got):>5} {ratio(n_match, n_truth):>6}"
                 f" {ratio(same_entity, n_match) if c != 'general_notes' else '-':>7}"
                 f" {ratio(same_value, n_match) if c in VALUE_FIELDS else '-':>6}")
    log.info(f"  {'all':<20} {totals[0]:>5} {totals[1]:>5} {totals[2]:>5} {ratio(totals[2], totals[1]):>5}"
             f" {ratio(totals[2], totals[0]):>6}")
    log.info("  (prec: matched / model items, recall: matched / truth items; entity, value: share of"
             " matched items that agree)")


def ratio(n: int, d: int) -> str:
    return f"{n / d:.2f}" if d else "-"


def verify(session_dir: Path, session_id: str, checks: harness.Checks,
           segments: list[dict], log_file: Path, truth_dir: Path) -> None:
    structured = session_dir / "structured"

    # --- extractions.json ---
    path = structured / "extractions.json"
    records = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    if not checks.check(isinstance(records, list),
                        f"extractions.json is a list ({len(records) if isinstance(records, list) else 'missing'} records)"):
        return
    checks.check(len(records) == len(segments), f"one record per segment ({len(records)} of {len(segments)})")
    expected = [(s["segment_id"], s["start"], s["end"]) for s in segments]
    got = [(r.get("segment"), r.get("start"), r.get("end")) for r in records]
    checks.check(got == expected[:len(got)], "records carry their segment's id, start and end, in order")
    bad = {r.get("segment"): bad_categories(r) for r in records if bad_categories(r)}
    checks.check(not bad, "every record has every category" + (f" (missing or wrong type: {bad})" if bad else ""))

    total = sum(len(items(r, c)) for r in records for c in LIST_CATEGORIES + OBJECT_CATEGORIES)
    checks.check(total > 0, f"something was extracted ({total} items in all)")

    # Times come from matching an item's quotes to its segment's words, so they must fall inside it.
    outside = [(r.get("segment"), it.get("time")) for r in records
               for c in LIST_CATEGORIES + OBJECT_CATEGORIES for it in items(r, c)
               if isinstance(it.get("time"), (int, float)) and not (r["start"] <= it["time"] <= r["end"])]
    timed = sum(1 for r in records for c in LIST_CATEGORIES + OBJECT_CATEGORIES
                for it in items(r, c) if isinstance(it.get("time"), (int, float)))
    checks.check(not outside, f"every timed item ({timed}) lies inside its segment"
                 + (f" (outside: {outside[:5]})" if outside else ""))

    checks.check((structured / "extractions.done").is_file(), "extractions.done was created")

    # --- the model's answers, against the ground truth (a report, never a failure) ---
    score(records, truth_dir)

    # --- the service's own count of model replies it couldn't parse ---
    for line in harness.session_log(log_file, session_id):
        m = re.search(r"extracted (\d+) segments \((\d+) unparseable replies\)", line)
        if m:
            log.info(f"service log: extracted {m.group(1)} segments, {m.group(2)} unparseable model replies")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--segments-file", type=Path, default=SEGMENTS,
                        help=f"transcript segments to feed, ASR segments.json format (default: {SEGMENTS})")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="seconds between writing one segment and the next (default: 5)")
    parser.add_argument("--ground-truth", type=Path, default=GROUND_TRUTH,
                        help=f"ground-truth records to score against, one <segment>.json each (default: {GROUND_TRUTH})")
    parser.add_argument("--score", type=Path, metavar="SESSION_DIR",
                        help="only score SESSION_DIR/structured/extractions.json against the ground truth; "
                             "the service isn't contacted")
    harness.add_common_args(parser, service="llm_extraction_service", log_name="llm_extraction", timeout=1800.0, ffmpeg=False)
    args = parser.parse_args()
    harness.setup_log("llm_extraction_service", args)

    if args.score is not None:
        path = harness.require_file(args.score / "structured" / "extractions.json")
        score(json.loads(path.read_text(encoding="utf-8")), args.ground_truth)
        return

    segments = load_segments(harness.require_file(args.segments_file))
    dry_run = (f"write {len(segments)} segments from {args.segments_file} into "
               f"<session>/transcript/segments.json, {args.interval:g}s apart")
    verify_fn = functools.partial(verify, segments=segments, log_file=args.log_file, truth_dir=args.ground_truth)
    harness.run("llm_extraction_service", args, segments_feed(segments, args.interval), dry_run, verify_fn)


if __name__ == "__main__":
    main()
