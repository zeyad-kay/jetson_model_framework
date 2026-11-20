"""Replays the test transcript through the extraction prompts of chosen categories only, to try a
prompt change in minutes instead of running the whole integration test.

Each segment of inputs/segments.json is extracted the way the service does it
(QwenExtractor.apply_segment, so the previous extractions, validation and medication
resolution are the same), but only for the categories given, and without terminology
linking. The records are written to <out>/<category>.json, every raw model reply to
<out>/replies.jsonl, and the result is scored against ground_truth/ like the test does.

Runs on the GPU, inside the service's image (the code and prompts are mounted from the repo):

    docker compose run --rm --no-deps -v ./tests:/tests --entrypoint python3 \\
        llm_extraction_service /tests/integration/extraction/replay.py interventions \\
        --out /tests/integration/extraction/replays/<name>

Add --segments 1-30 to replay only part of the transcript (the previous extractions then
start from what those segments give).
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # harness, for main
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/app")  # the service's code, in its image

import main as test  # noqa: E402
from llm_extraction_service.config import Config  # noqa: E402
from llm_extraction_service.model import QwenExtractor  # noqa: E402

log = logging.getLogger("replay")


def segment_range(text: str) -> tuple[int, int]:
    first, _, last = text.partition("-")
    return int(first), int(last or first)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("categories", nargs="+", choices=list(QwenExtractor.CATEGORIES))
    parser.add_argument("--out", type=Path, required=True, help="directory for the records, replies and score")
    parser.add_argument("--segments", type=segment_range, help="first-last segment to replay (default: all)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = Config.load()
    model = QwenExtractor(model_id=cfg.extract_model, prompt_path=str(Path(cfg.prompts_dir) / "separated"), link=False)
    model._extracting = {c: c in args.categories for c in model._extracting}

    args.out.mkdir(parents=True, exist_ok=True)
    replies = (args.out / "replies.jsonl").open("w")
    generate = model.generate
    current = {}

    def recording_generate(messages):
        reply = generate(messages)
        category = next(c for c, p in model._prompts.items() if p == messages[0]["content"])
        replies.write(json.dumps({"segment": current.get("segment"), "category": category, "reply": reply},
                                 ensure_ascii=False) + "\n")
        replies.flush()
        return reply

    model.generate = recording_generate

    segments = test.load_segments(test.SEGMENTS)
    if args.segments:
        first, last = args.segments
        segments = [s for s in segments if first <= s["segment_id"] <= last]
    extractions = {c: [] for c in model._extracting}
    started = time.monotonic()
    for s in segments:
        current["segment"] = s["segment_id"]
        extractions = model.apply_segment(s, extractions)
        log.info(f"segment {s['segment_id']}: " + ", ".join(
            f"{c} {sum(len(r['extractions']) for r in extractions[c])}" for c in args.categories))
    replies.close()
    for c in args.categories:
        (args.out / f"{c}.json").write_text(json.dumps(extractions[c], indent=2, ensure_ascii=False))
    log.info(f"replayed {len(segments)} segments in {time.monotonic() - started:.0f} s;"
             f" {model.parse_failures} unparseable replies, {model.validation_errors} validation errors")

    # Score the replayed categories only.
    test.CATEGORIES = tuple(c for c in test.CATEGORIES if c in args.categories)
    test.score(test.load_extractions(args.out), test.GROUND_TRUTH)


if __name__ == "__main__":
    main()
