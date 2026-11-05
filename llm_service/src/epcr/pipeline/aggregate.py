#!/usr/bin/env python3
"""Stage 2 — aggregate the stage-1 per-segment extractions into the final record.

Only **medications** go through the model, PER DRUG. Medication entities are split by their ATC
concept name (one drug at a time); for each drug the aggregator PROMPT (prompts/aggregate.txt) is
given ALL of that drug's extracted entities from the parts seen so far (segments 1..N) and returns
the drug's record of events. The grouping and EVERY field (type single/continuous, dose, start/end
time, rate_changes, quotes, times) are the model's, used exactly as returned — no code rule decides
type (the units can be missing or wrong in the transcript). Code only attaches the drug name.
Everything else — interventions included — needs no grouping, so it is collected deterministically
(no LLM). Same GPU-only policy as extract.py.

Input : <RESULTS_ROOT>/subjectid_<id>/extraction/<prompt>/<model>/seg*_extraction.json   (stage 1)
Output: <RESULTS_ROOT>/subjectid_<id>/aggregation/<prompt>/<model>/aggregate.json
        (--cumulative writes aggregate_upto_NN.json for each prefix of segments 1..N)

Usage (run from the project root, on a GPU node):
  python pipeline/aggregate.py --subject <id> --prompt separated --model qwen3_4b
  python pipeline/aggregate.py --run results/subjectid_<id>/extraction/separated/qwen3_4b --cumulative
"""
import os, re, json, glob, argparse, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.abspath(os.path.join(HERE, ".."))   # src/epcr (holds config.py)

import importlib.util as _il
_spec = _il.spec_from_file_location("pcr_config", os.path.join(CODE_ROOT, "config.py"))
cfg = _il.module_from_spec(_spec); _spec.loader.exec_module(cfg)
_D = cfg.DEFAULTS
ROOT = cfg.ROOT   # data (dataset/results/prompts/terminology) resolves against the repo root

sys.path.insert(0, HERE)                       # reuse the pipeline's model + json helpers
from extract import load_model, make_generate, extract_json, sec   # GPU-only load_model

def rel(p):
    return p if os.path.isabs(p) else os.path.join(ROOT, p)

# The aggregator LLM runs on MEDICATIONS ONLY, and its output is used exactly as returned . Every other category — interventions included — needs no grouping
# reasoning, so it is collected deterministically (no LLM): identical items merged, in time order.
LLM_CATS  = ["medications"]
PASS_CATS = ["interventions", "vital_signs", "mechanism_of_injury", "demographics", "general_notes"]
DET = ["dose", "route", "body_site", "device", "value"]
# gate: keep only interventions whose category is one of the allowed ones (config intervention_vocab),
# dropping anything the extractor invented outside that set. Empty vocab = no gating.
ITV_VOCAB = {str(v).strip().lower() for v in (getattr(cfg, "INTERVENTION_VOCAB", []) or [])}

def _segfiles(run_dir, upto=None):
    files = sorted(glob.glob(os.path.join(run_dir, "seg*_extraction.json")),
                   key=lambda p: int(re.search(r"seg(\d+)", p).group(1)))
    if upto is not None:   # cumulative: only segments 1..upto
        files = [f for f in files if int(re.search(r"seg(\d+)", f).group(1)) <= upto]
    if not files:
        raise SystemExit(f"no seg*_extraction.json in {run_dir} — run stage 1 (pipeline/extract.py) first")
    return files

def n_segments(run_dir):
    return len(_segfiles(run_dir))

def _nrm_q(q):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (q or "").lower())).strip()

def _time_for_quote(q, entities):
    """The transcript time of the entity whose quote matches `q` (never the model's)."""
    qn = _nrm_q(q)
    if not qn:
        return ""
    for e in entities:
        if any(_nrm_q(x) == qn for x in (e.get("quotes") or [])):
            return e.get("time", "")
    return ""

def _derive_times_and_rates(ev, entities):
    """Fill the time-based fields of an LLM event from the entities it groups — found by matching
    the event's (verbatim) quotes back to each entity's quotes. The times come from the extraction's
    transcript-matched `time`/`times` (never the model), so start/end and every rate change carry a
    transcript time, not an invented one.

    Two shapes of `rate_changes` are supported, chosen by the aggregator prompt:
      - the model emits the change list itself ([{direction, rate, quote}, ...], in order): keep it
        and attach a `time` to each item by matching its quote back to an entity;
      - the prompt does not ask for it: build the list from every rate of the matched entities."""
    evq = {_nrm_q(q) for q in (ev.get("quotes") or []) if isinstance(q, str)}
    matched = [e for e in entities if any(_nrm_q(q) in evq for q in (e.get("quotes") or []))]
    if not matched:
        return
    matched.sort(key=lambda e: sec(e["time"]) if sec(e["time"]) is not None else 1e9)
    times = sorted({t for e in matched for t in (e.get("times") or ([e["time"]] if e.get("time") else [])) if t},
                   key=lambda t: sec(t) if sec(t) is not None else 1e9)
    if times:
        ev["start_time"] = times[0]
        ev["end_time"] = times[-1] if times[-1] != times[0] else ""
    rc = ev.get("rate_changes")
    if isinstance(rc, list) and rc and all(isinstance(x, dict) for x in rc):
        for item in rc:                                     # model gave the list; code dates each item
            item["time"] = _time_for_quote(item.get("quote", ""), entities)
    else:
        ev["rate_changes"] = [{"time": e["time"], "rate": r}
                              for e in matched for r in (e.get("rate") or []) if str(r).strip()]
    if not ev.get("rate_unit"):
        ev["rate_unit"] = next((e.get("rate_unit") for e in matched if e.get("rate_unit")), "")

def _t0(seg):
    return str(seg.get("span", "")).split("->")[0].strip()

# ── gather stage-1 medications across segments, in time order, for the aggregator LLM ──
# Each medication's NAME is normalized to its linked ATC concept name (when linked), so the same
# drug reads identically across fragments and the aggregator has a canonical key; falls back to the
# extracted name when a medication is unlinked. Requires the ATC link pass to have run on the
# extraction (it adds the `atc` field to each medication).
def gather_meds(run_dir, upto=None):
    files = _segfiles(run_dir, upto)
    ents = []
    for f in files:
        seg = int(re.search(r"seg(\d+)", f).group(1))
        s = json.load(open(f)); t0 = _t0(s)
        for j, it in enumerate(s.get("medications", [])):
            if not isinstance(it, dict): continue
            atc = it.get("atc") or {}
            name = (atc.get("name")                       # the ATC concept name (canonical, clean)
                    if atc.get("code") else (it.get("standard_name") or it.get("entity") or ""))
            t = it.get("time") or t0
            ents.append({"id": f"m{seg:02d}_{j}", "name": name,
                         "time": t, "dose": it.get("dose", ""), "dose_unit": it.get("dose_unit", ""),
                         "route": it.get("route", ""), "rate": it.get("rate") or [],
                         "rate_unit": it.get("rate_unit", ""),
                         "quotes": it.get("quotes") or [], "times": it.get("times") or ([t] if t else [])})
    ents.sort(key=lambda e: sec(e["time"]) if sec(e["time"]) is not None else 1e9)
    return ents, len(files)

# ── collect the other categories deterministically (no LLM): merge identical items ──
def gather_passthrough(run_dir, upto=None):
    out = {c: {} for c in PASS_CATS}           # c -> key -> merged item
    for f in _segfiles(run_dir, upto):
        s = json.load(open(f)); t0 = _t0(s)
        for c in PASS_CATS:
            raw = s.get(c)
            # MOI is a single object in the new schema; tolerate the old list shape too.
            cat_items = ([raw] if isinstance(raw, dict) and raw else (raw if isinstance(raw, list) else [])) \
                        if c == "mechanism_of_injury" else (raw if isinstance(raw, list) else [])
            for it in cat_items:
                if not isinstance(it, dict): continue
                if c == "interventions" and ITV_VOCAB and str(it.get("category", "")).strip().lower() not in ITV_VOCAB:
                    continue                 # drop interventions whose category is outside the allowed set
                name = it.get("entity") or it.get("note") or ""
                detail = {d: it[d] for d in DET if it.get(d)}
                # schema fields carried through verbatim (interventions: category + details dict)
                passd = {k: it[k] for k in ("standard_name", "category", "details") if it.get(k)}
                key = (name.strip().lower(), json.dumps({**detail, **passd}, sort_keys=True, ensure_ascii=False))
                cur = out[c].get(key)
                t = it.get("time") or t0
                qs = [q for q in (it.get("quotes") or []) if isinstance(q, str) and q.strip()]
                if cur is None:
                    item = {"entity": name, "time": t, "times": list(it.get("times") or ([t] if t else [])),
                            "quotes": list(qs)}
                    item.update(passd)
                    item.update(detail)
                    out[c][key] = item
                else:                           # merge a repeat: union quotes/times, keep earliest time
                    for q in qs:
                        if q not in cur["quotes"]: cur["quotes"].append(q)
                    if t and t not in cur["times"]: cur["times"].append(t)
                    if t and (sec(t) or 1e9) < (sec(cur["time"]) or 1e9): cur["time"] = t
    merged = {}
    for c in PASS_CATS:
        items = list(out[c].values())
        for it in items:
            it["times"].sort(key=lambda x: sec(x) if sec(x) is not None else 1e9)
        items.sort(key=lambda it: sec(it["time"]) if sec(it["time"]) is not None else 1e9)
        # mechanism_of_injury is a single object (one entity), not a list
        merged[c] = (items[0] if items else {}) if c == "mechanism_of_injury" else items
    return merged

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="a stage-1 run dir with seg*_extraction.json (overrides subject/prompt/model)")
    ap.add_argument("--subject", default=str(_D.get("subject", "<id>")))
    ap.add_argument("--prompt", default="separated", help="prompt key used in the results path (combined|separated)")
    ap.add_argument("--model", default=next(iter(cfg.MODELS), "qwen3_4b"), help="extraction model key: which run to read + the results path")
    ap.add_argument("--agg-model", default=next(iter(cfg.AGG_MODELS), None),
                    help="aggregator model key (roster: config aggregate.models)")
    ap.add_argument("--agg-prompt", default=next(iter(cfg.AGG_PROMPTS), None),
                    help="aggregator prompt key (roster: config aggregate.prompts)")
    ap.add_argument("--aggregate-prompt", help="raw aggregator prompt file path (overrides --agg-prompt; for experiments)")
    ap.add_argument("--out", help="output aggregate.json (default: <results>/subjectid_<id>/aggregation/<prompt>/<model>/<agg_model>/<agg_prompt>/aggregate.json)")
    ap.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", _D.get("max_new_tokens", 4096))))
    ap.add_argument("--dtype", default=os.environ.get("DTYPE"))
    ap.add_argument("--cumulative", action="store_true",
                    help="running aggregation: write aggregate_upto_NN.json for each prefix of segments 1..N")
    a = ap.parse_args()

    results_root = _D.get("results_root", "results")
    agg_model_key = a.agg_model or next(iter(cfg.AGG_MODELS), None)
    agg_prompt_key = a.agg_prompt or next(iter(cfg.AGG_PROMPTS), None)
    if not agg_model_key:
        raise SystemExit("no aggregator model in config aggregate.models")
    if not a.aggregate_prompt and not agg_prompt_key:
        raise SystemExit("no aggregator prompt in config aggregate.prompts")

    if a.run:
        run_dir = rel(a.run)
        out = rel(a.out) if a.out else os.path.join(run_dir, "aggregate.json")
    else:
        run_dir = rel(os.path.join(results_root, f"subjectid_{a.subject}", "extraction", a.prompt, a.model))
        # aggregation nests under the extraction run it reads: <ext_prompt>/<ext_model>/<agg_model>/<agg_prompt>/
        out = rel(a.out) if a.out else rel(os.path.join(
            results_root, f"subjectid_{a.subject}", "aggregation",
            a.prompt, a.model, agg_model_key, agg_prompt_key, "aggregate.json"))

    # aggregator prompt: a raw-path override (experiments), else the roster entry for --agg-prompt
    if a.aggregate_prompt:
        ppath = rel(a.aggregate_prompt)
    else:
        ent = cfg.agg_prompt_entry(agg_prompt_key) or {}
        ppath = cfg.rel(ent.get("path", "prompts/aggregate.txt"))
    if not os.path.isfile(ppath):
        raise SystemExit(f"aggregator prompt not found: {ppath}")
    template = cfg.render(open(ppath).read())

    # the aggregator LLM runs on its own model (config aggregate.models, keyed by --agg-model); the
    # extraction run it reads and the results path still key on --model (the extraction model).
    mid = cfg.agg_model_id(agg_model_key)
    tok, model, dtype = load_model(mid, a.dtype)
    print(f"aggregator model: {mid}  |  reading extraction + writing results under: {a.model}", flush=True)
    generate = make_generate(tok, model, a.max_new_tokens)

    outdir = os.path.dirname(out)
    os.makedirs(outdir, exist_ok=True)
    # provenance: save the exact aggregator prompt this run used, so the dashboard shows the
    # prompt that generated these results, not whatever prompts/aggregate.txt later becomes.
    with open(os.path.join(outdir, "aggregate_prompt.txt"), "w") as _fh:
        _fh.write(template)

    def aggregate_prefix(upto, dest):
        """PER DRUG, ONE CALL EACH: group the extracted entities by drug name (ATC concept name) and
        feed ONE drug's entities per call. The model returns that drug's events (same event-schema
        prompt), separating its bolus / infusion / later doses without any cross-drug confusion. The
        grouping and quotes are the model's; rate_changes/times are then rebuilt by code from the
        event's own quotes (matched to the entities, whose rates/times come from the transcript).
        Other categories collected deterministically (no LLM)."""
        ents, nseg = gather_meds(run_dir, upto)
        bydrug = {}
        for e in ents:
            bydrug.setdefault(e["name"], []).append(e)
        meds_out, raws = [], []
        for drug, entities in bydrug.items():
            feed = [{k: e[k] for k in ("name", "time", "dose", "dose_unit", "route", "rate", "rate_unit", "quotes")} for e in entities]
            prompt = template.replace("{entities}", json.dumps(feed, indent=1, ensure_ascii=False))
            raw = generate([{"role": "user", "content": prompt}])
            raws.append(f"===== {drug} =====\n{raw if isinstance(raw, str) else str(raw)}")
            obj = extract_json(raw)
            evs = (obj.get("medications") if isinstance(obj, dict) else None) or []
            for ev in evs:
                if not isinstance(ev, dict): continue
                ev.pop("rate_changes", None)                  # rebuild from the event's own quotes
                _derive_times_and_rates(ev, entities)
                ev.setdefault("entity", drug); ev.setdefault("standard_name", drug)
                meds_out.append(ev)
        # save the model's exact text (before parsing / derivation), one block per drug
        with open(dest[:-5] + "_raw.txt" if dest.endswith(".json") else dest + ".raw.txt", "w") as _rf:
            _rf.write("\n\n".join(raws))
        meds_out.sort(key=lambda r: sec(r.get("start_time")) if sec(r.get("start_time")) is not None else 1e9)
        final = {"medications": meds_out}
        final.update(gather_passthrough(run_dir, upto))         # interventions + the other 4, deterministic
        json.dump(final, open(dest, "w"), indent=1, ensure_ascii=False)
        return nseg, len(ents), final

    total = n_segments(run_dir)
    if a.cumulative:
        for N in range(1, total + 1):
            dest = os.path.join(outdir, f"aggregate_upto_{N:02d}.json")
            _, ne, final = aggregate_prefix(N, dest)
            print(f"upto {N:2d}/{total}: entities={ne:3d} meds={len(final['medications'])} "
                  f"itv={len(final['interventions'])} -> {os.path.basename(dest)}", flush=True)
        print(f"wrote {total} cumulative aggregates in {outdir}")
    else:
        _, ne, final = aggregate_prefix(None, out)
        print(f"read {run_dir}  ({total} segments, {ne} med/intervention entities -> LLM)")
        print(f"wrote {out}  | medications={len(final['medications'])} interventions={len(final['interventions'])}")

if __name__ == "__main__":
    main()
