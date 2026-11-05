#!/usr/bin/env python3
"""ePCR pipeline entry point — extraction only, or extraction + aggregation.

    # extraction only — pick the extraction model + prompt:
    python src/epcr/pipeline/main.py --stage ex --subject <id> --ex_model qwen3_4b --ex_prompt separated

    # extraction + aggregation — also pick the aggregator model + prompt:
    python src/epcr/pipeline/main.py --stage ex_ag --subject <id> --ex_model qwen3_4b --ex_prompt separated \
        --ag_model qwen25_7b --ag_prompt events --cumulative

  --stage ex     extraction only: transcript -> per-segment extractions (with ATC linking)
  --stage ex_ag  extraction then aggregation (default): extractions -> aggregate record

Both stages run the LLM on GPU (fail loud without CUDA). Extraction always runs with ATC linking,
because the aggregator groups medications by their ATC concept name. You choose the extraction model
(--ex_model) + prompt (--ex_prompt) and, for ex_ag, the aggregator model (--ag_model) + prompt
(--ag_prompt). Every default comes from config.yaml; this script only wires the two stages — no logic.
"""
import os
import sys
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))     # src/epcr/pipeline
ROOT = os.path.dirname(HERE)                          # the repo
sys.path.insert(0, ROOT)
import config as cfg


def run(cmd):
    print("\n>>> " + " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"step failed (exit {r.returncode}): {' '.join(cmd)}")


def main():
    D = cfg.DEFAULTS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=["ex", "ex_ag"], default="ex_ag",
                    help="ex = extraction only; ex_ag = extraction then aggregation (default)")
    ap.add_argument("--subject", default=str(D.get("subject", "<id>")),
                    help="subject id (the <id> in dataset/subjectid_<id>.json)")
    ap.add_argument("--ex_model", default=next(iter(cfg.MODELS), "qwen3_4b"), help="extraction model key")
    ap.add_argument("--ex_prompt", default="separated", help="extraction prompt key (combined|separated)")
    ap.add_argument("--ag_model", default=next(iter(cfg.AGG_MODELS), None),
                    help="aggregator model key (only used when stage is ex_ag)")
    ap.add_argument("--ag_prompt", default=next(iter(cfg.AGG_PROMPTS), None),
                    help="aggregator prompt key (only used when stage is ex_ag)")
    ap.add_argument("--cumulative", action="store_true",
                    help="stage 2: also write aggregate_upto_NN.json for each prefix of segments 1..N")
    ap.add_argument("--dry-run", action="store_true", help="print the stage commands without running them")
    a = ap.parse_args()

    results_root = D.get("results_root", "results")
    dataset_dir = D.get("dataset_dir", "dataset")

    ent = cfg.prompt_entry(a.ex_prompt)
    if not ent:
        raise SystemExit(f"unknown extraction prompt '{a.ex_prompt}' (config prompts: {list(cfg.PROMPTS)})")
    mode = ent["mode"]

    ext_out = os.path.join(results_root, f"subjectid_{a.subject}", "extraction", a.ex_prompt, a.ex_model)
    transcript = os.path.join(dataset_dir, f"subjectid_{a.subject}.json")
    py = sys.executable

    steps = []
    # extraction always runs (both stages start with it); ATC linking is always on, because the
    # aggregator groups medications by their ATC concept name.
    steps.append([py, os.path.join(HERE, "extract.py"),
                  "--mode", mode, "--model", a.ex_model,
                  "--out", ext_out, "--transcript", transcript, "--link"])
    if a.stage == "ex_ag":
        cmd = [py, os.path.join(HERE, "aggregate.py"),
               "--subject", a.subject, "--prompt", a.ex_prompt, "--model", a.ex_model]
        if a.ag_model:
            cmd += ["--agg-model", a.ag_model]
        if a.ag_prompt:
            cmd += ["--agg-prompt", a.ag_prompt]
        if a.cumulative:
            cmd.append("--cumulative")
        steps.append(cmd)

    print(f"pipeline: stage={a.stage} subject={a.subject} ex_model={a.ex_model} "
          f"ex_prompt={a.ex_prompt} ag_model={a.ag_model} ag_prompt={a.ag_prompt} link=on", flush=True)
    for cmd in steps:
        if a.dry_run:
            print(">>> " + " ".join(cmd), flush=True)
        else:
            run(cmd)
    print("\n=== pipeline done ===" if not a.dry_run else "\n=== dry run (nothing executed) ===", flush=True)


if __name__ == "__main__":
    main()
