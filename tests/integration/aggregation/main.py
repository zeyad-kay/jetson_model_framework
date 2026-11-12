"""Integration test for the LLM aggregation service (llm_aggregation_service), on its own.

Creates a session, queues it on the aggregation service over its bridge, then writes
the extraction records in aggregation/inputs/<segment>.json into
<session>/structured/extractions.json one record at a time, the way the
extraction service writes them (the whole list so far, replaced atomically). By default it
waits for the aggregate of each list before writing the next record, so every
segment gets its own aggregate; with --no-lockstep it writes them --interval
seconds apart regardless, and the aggregator skips the lists it falls behind
on. After the last record it creates structured/extractions.done (the
extraction service's mark that the list is final), ends the session, waits for the
service to finish, and checks:

    structured/aggregates/<n>.json  one per segment (with lockstep); every
                                    non-medication category equal to
                                    aggregation/ground_truth/<n>.json, as
                                    those are merged without the model;
                                    medication events with quotes and times
                                    inside the segments aggregated
    structured/aggregate.json       every category, the same as the last aggregate

Medications come from the model, so they are only checked for shape; their
counts are logged next to the ground truth's.

    docker compose run --rm llm_aggregation_test
    docker compose run --rm llm_aggregation_test --no-lockstep --interval 5
    python tests/integration/aggregation/main.py --dry-run

Exits non-zero if any check fails. See harness.py for how it runs.
"""

import argparse
import functools
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # tests/integration, for harness.py
import harness  # noqa: E402
from testlog import fail, log  # noqa: E402

# The inputs are the extraction service's ground truth (extraction/ground_truth), copied.
EXTRACTIONS = HERE / "inputs"
AGGREGATES = HERE / "ground_truth"

# The categories of an aggregate (llm_aggregation_service/model.py); mechanism_of_injury is
# a single object, the rest are lists. Only medications go through the model.
LIST_CATEGORIES = ("medications", "interventions", "vital_signs", "demographics", "general_notes")
OBJECT_CATEGORIES = ("mechanism_of_injury",)
MODEL_CATEGORIES = ("medications",)


def load_records(folder: Path) -> list[dict]:
    """The extraction records <segment>.json in folder, in segment order."""
    records = [json.loads(p.read_text(encoding="utf-8")) for p in folder.glob("*.json")]
    if not records:
        fail(f"no extraction records in {folder}")
    return sorted(records, key=lambda r: r["segment"])


def records_feed(records: list[dict], interval: float, lockstep: bool, timeout: float) -> harness.Feed:
    """Writes the records into <session>/structured/extractions.json one at a time, like the extraction service."""
    def feed(session_dir: Path) -> tuple[bool, str]:
        structured = session_dir / "structured"
        for n in range(1, len(records) + 1):
            time.sleep(interval)
            harness.write_json_atomic(structured / "extractions.json", records[:n])
            segment = records[n - 1]["segment"]
            log.info(f"wrote record {n} (segment {segment})")
            if lockstep and not wait_for(structured / "aggregates" / f"{segment}.json", timeout):
                return False, f"no aggregate for segment {segment} within {timeout:g}s"
        (structured / "extractions.done").touch()
        return True, f"wrote {len(records)} records to structured/extractions.json, then extractions.done"
    return feed


def wait_for(path: Path, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while not path.is_file():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.5)
    log.info(f"  {path.parent.name}/{path.name} written")
    return True


def bad_categories(aggregate: dict) -> list[str]:
    """Categories missing or of the wrong type (a list, or an object for mechanism_of_injury)."""
    bad = [c for c in LIST_CATEGORIES if not isinstance(aggregate.get(c), list)]
    return bad + [c for c in OBJECT_CATEGORIES if not isinstance(aggregate.get(c), dict)]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def verify(session_dir: Path, session_id: str, checks: harness.Checks,
           records: list[dict], lockstep: bool) -> None:
    structured = session_dir / "structured"
    spans = {r["segment"]: (r["start"], r["end"]) for r in records}
    order = [r["segment"] for r in records]

    snapshots = {int(p.stem): read_json(p) for p in (structured / "aggregates").glob("*.json") if p.stem.isdigit()}
    if lockstep:
        missing = [s for s in order if s not in snapshots]
        checks.check(not missing, f"an aggregate for every segment ({len(order) - len(missing)} of {len(order)})"
                     + (f" (missing: {missing})" if missing else ""))
    else:
        checks.check(bool(snapshots), f"aggregates written for segments {sorted(snapshots)}")

    for segment in sorted(snapshots):
        aggregate = snapshots[segment]
        label = f"aggregates/{segment}.json"
        bad = bad_categories(aggregate)
        if not checks.check(not bad, f"{label} has every category" + (f" (missing or wrong type: {bad})" if bad else "")):
            continue

        expected = read_json(AGGREGATES / f"{segment}.json")
        if expected is not None:
            differ = [c for c in LIST_CATEGORIES + OBJECT_CATEGORIES
                      if c not in MODEL_CATEGORIES and aggregate.get(c) != expected.get(c)]
            checks.check(not differ, f"{label} matches the ground truth outside medications"
                         + (f" (differs: {differ})" if differ else ""))

        covered = [spans[s] for s in order[:order.index(segment) + 1]] if segment in spans else list(spans.values())
        meds = [m for m in aggregate["medications"] if isinstance(m, dict)]
        times = [t for m in meds for t in [m.get("start_time"), m.get("end_time")]
                 + [rc.get("time") for rc in m.get("rate_changes") or [] if isinstance(rc, dict)]
                 if isinstance(t, (int, float))]
        outside = [t for t in times if not any(a <= t <= b for a, b in covered)]
        checks.check(all(m.get("quotes") for m in meds) and not outside,
                     f"{label}: {len(meds)} medication events (ground truth {len((expected or {}).get('medications') or [])}),"
                     f" all with quotes and times inside the segments aggregated"
                     + (f" (outside: {outside[:5]})" if outside else ""))

    final = read_json(structured / "aggregate.json")
    if checks.check(isinstance(final, dict), "aggregate.json was written"):
        bad = bad_categories(final)
        checks.check(not bad, "aggregate.json has every category" + (f" (missing or wrong type: {bad})" if bad else ""))
        if snapshots:
            last = max(snapshots, key=lambda s: order.index(s) if s in order else -1)
            checks.check(final == snapshots[last], f"aggregate.json is the last aggregate (segment {last})")
        log.info("aggregate: " + ", ".join(
            f"{c} {len(final[c]) if isinstance(final.get(c), list) else int(bool(final.get(c)))}"
            for c in LIST_CATEGORIES + OBJECT_CATEGORIES))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--extractions-dir", type=Path, default=EXTRACTIONS,
                        help=f"extraction records to feed, one <segment>.json each (default: {EXTRACTIONS})")
    parser.add_argument("--interval", type=float, default=0.0,
                        help="seconds between writing one record and the next (default: 0)")
    parser.add_argument("--lockstep", action=argparse.BooleanOptionalAction, default=True,
                        help="wait for each record's aggregate before writing the next (default: on)")
    harness.add_common_args(parser, service="llm_aggregation_service", log_name="llm_aggregation", timeout=1800.0, ffmpeg=False)
    args = parser.parse_args()
    harness.setup_log("llm_aggregation_service", args)

    records = load_records(args.extractions_dir)
    dry_run = (f"write {len(records)} records from {args.extractions_dir} into "
               f"<session>/structured/extractions.json, {args.interval:g}s apart"
               + (", each after the previous one's aggregate" if args.lockstep else "")
               + ", then create extractions.done")
    verify_fn = functools.partial(verify, records=records, lockstep=args.lockstep)
    harness.run("llm_aggregation_service", args, records_feed(records, args.interval, args.lockstep, args.timeout),
                dry_run, verify_fn)


if __name__ == "__main__":
    main()
