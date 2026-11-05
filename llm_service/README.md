# ePCR extraction

Turn an EMS **transcript** into a structured **electronic Patient Care Report (ePCR)** with an
LLM: a two-stage GPU pipeline (extract → aggregate), plus a local view-only dashboard. The code
lives under `src/epcr/`; transcripts, prompts, and results sit at the repo root.

## Run

Everything goes through `src/epcr/pipeline/main.py` — it runs the LLM on **GPU** (fails loud
without CUDA). Put a transcript at `dataset/subjectid_<id>.json` first.

```bash
# extraction only
python src/epcr/pipeline/main.py --stage ex --subject <id> --ex_model qwen3_4b --ex_prompt separated

# extraction + aggregation (default)
python src/epcr/pipeline/main.py --stage ex_ag --subject <id> \
    --ex_model qwen3_4b --ex_prompt separated --ag_model qwen25_7b --ag_prompt events --cumulative

# dashboard — separate, reads results/ live (just refresh after a run)
python src/epcr/dashboard/server.py            # http://localhost:7860
```

- `--stage ex` = extraction only · `--stage ex_ag` = extraction then aggregation (default)
- `--ex_model` / `--ex_prompt` = extraction model + prompt (`combined` or `separated`)
- `--ag_model` / `--ag_prompt` = aggregator model + prompt (stage 2 only)
- `--cumulative` also writes the running record at each segment · `--dry-run` prints the commands
  without running · `--help` lists all flags

## Output

```
results/subjectid_<id>/
├── extraction/<ex_prompt>/<ex_model>/
│       seg*_extraction.json          stage 1 — one file per segment
└── aggregation/<ex_prompt>/<ex_model>/<ag_model>/<ag_prompt>/
        aggregate.json                stage 2 — the final record (all segments)
        aggregate_upto_NN.json        with --cumulative: the record as of segment N
        aggregate_raw.txt             the raw aggregator-LLM output
        aggregate_prompt.txt          the exact aggregator prompt used
```

The dashboard reads these live — run a pipeline, then refresh the browser to see the new output.
