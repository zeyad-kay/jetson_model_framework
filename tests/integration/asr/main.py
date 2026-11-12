"""Integration test for the ASR service (asr_service), on its own.

Creates a session, queues it on the ASR service over its bridge, then decodes
asr/inputs/audio.mp3 in real time into <session>/audio/audio.pcm (16-bit mono PCM at
--sampling-rate, as the server records it). When the file has been played
out it ends the session, waits for the service to finish, and checks the
transcript it wrote: transcript.txt, transcript.jsonl and segments.json.
Against the ground truth in asr/ground_truth/: transcript.txt is scored by
word error rate (WER), which must be at most --max-wer, and segments.json must
have the same number of segments, each starting and ending within
--segment-tolerance seconds of the reference. Word timestamps aren't compared.

    docker compose run --rm asr_test
    docker compose run --rm asr_test --max-wer 0.1
    docker compose run --rm asr_test --audio-file /tests/integration/asr/inputs/clip.mp3   # no ground truth checks
    python tests/integration/asr/main.py --dry-run

Exits non-zero if any check fails. See harness.py for how it runs.
"""

import argparse
import functools
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # tests/integration, for harness.py
import harness  # noqa: E402
from testlog import log  # noqa: E402

AUDIO = HERE / "inputs" / "audio.mp3"
GROUND_TRUTH = HERE / "ground_truth"  # transcript.txt and segments.json for AUDIO


def normalize(text: str) -> list[str]:
    """Words for scoring: lowercased, hyphens and dashes as spaces, other punctuation dropped.

    Apostrophes stay inside words, so "we've" and "we have" still differ.
    """
    text = re.sub(r"[-\u2010-\u2015]", " ", text.lower().replace("\u2019", "'"))
    text = re.sub(r"[^a-z0-9' ]", " ", text)
    return [w.strip("'") for w in text.split() if w.strip("'")]


def wer(reference: list[str], hypothesis: list[str]) -> tuple[float, int, int, int]:
    """Word error rate of hypothesis against reference, with its (substitutions, deletions, insertions).

    Levenshtein alignment over words; WER = (S + D + I) / len(reference).
    """
    # Each cell holds (cost, S, D, I) for the best alignment of the prefixes.
    prev = [(j, 0, 0, j) for j in range(len(hypothesis) + 1)]
    for i, ref_word in enumerate(reference, 1):
        cur = [(i, 0, i, 0)]
        for j, hyp_word in enumerate(hypothesis, 1):
            if ref_word == hyp_word:
                best = prev[j - 1]
            else:
                c, s, d, n = prev[j - 1]
                best = (c + 1, s + 1, d, n)
            c, s, d, n = prev[j]
            if c + 1 < best[0]:
                best = (c + 1, s, d + 1, n)
            c, s, d, n = cur[j - 1]
            if c + 1 < best[0]:
                best = (c + 1, s, d, n + 1)
            cur.append(best)
        prev = cur
    cost, s, d, n = prev[-1]
    return cost / max(len(reference), 1), s, d, n


def ffmpeg_cmd(ffmpeg: str, audio: Path, sampling_rate: int) -> list[str]:
    """Decodes `audio` at its native rate (-re) to raw PCM on stdout, flushing each packet,
    the same format as the server's AudioListener."""
    return [
        ffmpeg, "-hide_banner", "-loglevel", "warning",
        "-re", "-i", str(audio),
        "-vn", "-f", "s16le", "-ac", "1", "-ar", str(sampling_rate),
        "-flush_packets", "1",
        "pipe:1",
    ]


def compare_segments(segments: list[dict], reference: list[dict], tolerance: float, checks: harness.Checks) -> None:
    """Same number of segments as the reference, each with its start and end within `tolerance` seconds."""
    if not checks.check(len(segments) == len(reference),
                        f"{len(segments)} segment(s), reference has {len(reference)}"):
        return
    for seg, ref in zip(segments, reference):
        d_start = abs(seg["start"] - ref["start"])
        d_end = abs(seg["end"] - ref["end"])
        checks.check(d_start <= tolerance and d_end <= tolerance,
                     f"segment {ref['segment_id']}: {seg['start']:.2f}-{seg['end']:.2f}s vs reference "
                     f"{ref['start']:.2f}-{ref['end']:.2f}s (off by {d_start:.2f}s / {d_end:.2f}s, "
                     f"tolerance {tolerance:g}s)")


def verify(session_dir: Path, session_id: str, checks: harness.Checks, ground_truth: Path | None,
           max_wer: float, segment_tolerance: float) -> None:
    transcript_dir = session_dir / "transcript"

    txt = transcript_dir / "transcript.txt"
    text = txt.read_text(encoding="utf-8").strip() if txt.is_file() else ""
    checks.check(bool(text), f"transcript.txt has text ({len(text.split())} words)")

    jsonl = transcript_dir / "transcript.jsonl"
    entries = []
    if jsonl.is_file():
        for line in jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                entries.append(json.loads(line))
    checks.check(bool(entries), f"transcript.jsonl has timed entries ({len(entries)})")

    segments_path = transcript_dir / "segments.json"
    segments = None
    segments_ok = False
    if segments_path.is_file():
        segments = json.loads(segments_path.read_text(encoding="utf-8"))
    if checks.check(isinstance(segments, list) and bool(segments),
                    f"segments.json has segments ({len(segments) if isinstance(segments, list) else 'missing'})"):
        ids = [s.get("segment_id") for s in segments]
        segments_ok = checks.check(ids == list(range(1, len(ids) + 1)), f"segment ids run 1..{len(ids)}")
        checks.check(all(s.get("text") for s in segments), "every segment has words")

    if ground_truth is not None:
        log.info("against the ground truth in %s:", ground_truth)
        if text:
            reference = normalize((ground_truth / "transcript.txt").read_text(encoding="utf-8"))
            rate, s, d, n = wer(reference, normalize(text))
            checks.check(rate <= max_wer,
                         f"WER {rate:.1%} <= {max_wer:.1%} "
                         f"({len(reference)} words: {s} substituted, {d} deleted, {n} inserted)")
        if segments_ok:
            reference = json.loads((ground_truth / "segments.json").read_text(encoding="utf-8"))
            compare_segments(segments, reference, segment_tolerance, checks)

    if text:
        log.info("transcript (first 300 chars):\n  %s", text[:300])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-file", type=Path, default=AUDIO, help=f"audio source (default: {AUDIO})")
    parser.add_argument("--sampling-rate", type=int, default=16000,
                        help="PCM sample rate; must match the ASR's SAMPLING_RATE (default: 16000)")
    parser.add_argument("--ground-truth", type=Path,
                        help=f"folder with the reference transcript.txt and segments.json (default: "
                             f"{GROUND_TRUTH} for the default audio; none for another --audio-file, "
                             "which skips the ground truth checks)")
    parser.add_argument("--max-wer", type=float, default=0.15,
                        help="highest word error rate that passes, 0-1 (default: 0.15)")
    parser.add_argument("--segment-tolerance", type=float, default=5.0,
                        help="seconds each segment's start and end may differ from the reference (default: 5)")
    harness.add_common_args(parser, service="asr_service", log_name="asr")
    args = parser.parse_args()
    harness.setup_log("asr_service", args)

    audio = harness.require_file(args.audio_file)
    ground_truth = args.ground_truth
    if ground_truth is None and audio.resolve() == AUDIO:
        ground_truth = GROUND_TRUTH
    if ground_truth is not None:
        for name in ("transcript.txt", "segments.json"):
            harness.require_file(ground_truth / name)
    else:
        log.info("no ground truth for this audio; skipping the WER and segment checks")
    verify_fn = functools.partial(verify, ground_truth=ground_truth, max_wer=args.max_wer,
                                  segment_tolerance=args.segment_tolerance)
    feed, dry_run = harness.ffmpeg_feed(ffmpeg_cmd(args.ffmpeg, audio, args.sampling_rate), "audio/audio.pcm")
    harness.run("asr_service", args, feed, dry_run, verify_fn)


if __name__ == "__main__":
    main()
