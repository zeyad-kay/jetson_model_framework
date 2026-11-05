#!/usr/bin/env python3
"""Unified ePCR segment extraction — one entrypoint, two modes.

Given a transcript and prompt(s), split into segments (pause THR, size cap) and, per
segment, call the model one of two ways:

  --mode combined   ONE call per segment: a single prompt with all categories;
                    the model's JSON is saved RAW (entity + quotes + details).
  --mode separated  ONE call PER PROMPT FILE in the prompt dir: each file returns one
                    or more categories (its JSON shape); the returned lists are saved RAW.

Output per segment: <out>/seg<NN>_extraction.json  (consumed by the dashboard builder).

Optional --link attaches codes to the raw entities (keyed on `entity`): ATC (and, with
--rxnav, RxNorm) for medications, ICD-10-PCS for interventions. Needs the code index under
terminology/index (bundled); off by default so a plain run is fast and offline.

Examples (run from the project root):
  python pipeline/extract.py --mode combined --model qwen3_4b \\
      --out results/subjectid_<id>/extraction/combined/qwen3_4b
  python pipeline/extract.py --mode separated --model qwen3_4b \\
      --out results/subjectid_<id>/extraction/separated/qwen3_4b
"""
import os, re, json, glob, argparse, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.abspath(os.path.join(HERE, ".."))   # src/epcr (holds config.py), one up from pipeline/
REPO = CODE_ROOT   # repo root for data paths; reset to cfg.ROOT once config is loaded (below)

CATS = ["medications", "interventions", "vital_signs", "mechanism_of_injury",
        "demographics", "general_notes"]
# mechanism_of_injury is a single object (one entity per case), not a list.
OBJECT_CATS = {"mechanism_of_injury"}

def cat_default(c):
    return {} if c in OBJECT_CATS else []

def cat_items(rec, c):
    """The item dicts for a category, whether it is stored as one object or a list."""
    v = rec.get(c)
    if isinstance(v, dict): return [v] if v else []
    return v if isinstance(v, list) else []

# models / prompts / defaults come from config.yaml (single source of truth)
import importlib.util as _il
_spec = _il.spec_from_file_location("pcr_config", os.path.join(CODE_ROOT, "config.py"))
cfg = _il.module_from_spec(_spec); _spec.loader.exec_module(cfg)
REPO = cfg.ROOT   # data (dataset/results/prompts/terminology) lives at the repo root config found
DEFAULT_PROMPT = {k: v["path"] for k, v in cfg.PROMPTS.items()}
_D = cfg.DEFAULTS
DEFAULT_TRANSCRIPT = f"{_D.get('dataset_dir','dataset')}/subjectid_{_D.get('subject','<id>')}.json"

def rel(p):
    return p if os.path.isabs(p) else os.path.join(REPO, p)

# ── shared text helpers (identical across the original three scripts) ─────────
def sec(ts):
    p = str(ts).split(":")                     # HH:MM:SS(.ms) or MM:SS(.ms)
    try:
        if len(p) == 3: return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])
        if len(p) == 2: return int(p[0]) * 60 + float(p[1])
    except ValueError:
        pass
    return None

def size_split(seg, cap):
    if cap <= 0 or len(seg) <= cap:
        return [seg]
    n = -(-len(seg) // cap); target = len(seg) / n; cuts, used = [], set()
    for p in range(1, n):
        t = round(p * target); bi, bg = t, -1
        for i in range(max(1, t - 15), min(len(seg) - 1, t + 15)):
            if i in used: continue
            g = sec(seg[i]["Start"]) - sec(seg[i - 1]["End"])
            if g > bg: bi, bg = i, g
        cuts.append(bi); used.add(bi)
    idx = [0] + sorted(set(cuts)) + [len(seg)]
    return [seg[idx[k]:idx[k + 1]] for k in range(len(idx) - 1)]

def extract_json(gen):
    s = re.sub(r"<think>.*?</think>", "", gen, flags=re.S)
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s.strip(), flags=re.S)
    i = s.find("{")
    if i < 0: return None
    d = 0
    for j in range(i, len(s)):
        if s[j] == "{": d += 1
        elif s[j] == "}":
            d -= 1
            if d == 0:
                try: return json.loads(s[i:j + 1])
                except Exception: return None
    return None

def segment(transcript_path, thr, max_utts):
    data = json.load(open(transcript_path))
    utts = [s for s in data if (s.get("Utterance") or "").strip()]
    bounds = [0] + [i + 1 for i, (a, b) in enumerate(zip(utts, utts[1:]))
                    if (sec(b["Start"]) - sec(a["End"]) if sec(b["Start"]) is not None and sec(a["End"]) is not None else 0) > thr] + [len(utts)]
    pause = [utts[bounds[k]:bounds[k + 1]] for k in range(len(bounds) - 1)]
    out = []
    for s in pause:
        out.extend(size_split(s, max_utts))
    return pause, out

# ── model ─────────────────────────────────────────────────────────────────────
# ── timestamps: resolve each entity's time from the transcript (not from the model) ──────
def _nrm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())).strip()

def _quote_time(q, seg):
    """The Start time of the utterance that best matches a quote, or None."""
    qn = _nrm(q)
    if not qn: return None
    best, bs = None, 0.0
    for u in seg:
        un = _nrm(u.get("Utterance"))
        if not un: continue
        if qn in un or un in qn:
            sc = min(len(qn), len(un)) / max(len(qn), len(un))
        else:
            a, b = set(qn.split()), set(un.split())
            sc = len(a & b) / len(a | b) if (a | b) else 0
        if sc > bs: bs, best = sc, u
    return best.get("Start") if best and bs >= 0.3 else None

def attach_times(rec, seg):
    """Add 'time' (earliest) and 'times' (all) to each entity, derived from the transcript
    by matching its quotes back to the utterances they came from."""
    for c in CATS:
        for it in cat_items(rec, c):
            if not isinstance(it, dict): continue
            ts = [t for t in (_quote_time(q, seg) for q in (it.get("quotes") or [])) if t]
            if ts:
                it["time"] = min(ts, key=lambda t: sec(t) if sec(t) is not None else 1e9)
                it["times"] = ts

def load_model(mid, dtype):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA GPU available — this pipeline runs the LLM on GPU only. "
                         "Run it on a machine with a CUDA GPU.")
    dtype = dtype or "bfloat16"
    tok = AutoTokenizer.from_pretrained(mid)
    model = AutoModelForCausalLM.from_pretrained(mid, torch_dtype=getattr(torch, dtype), device_map="auto")
    print(f"device: cuda | dtype: {dtype} | model: {mid}", flush=True)
    return tok, model, dtype

def make_generate(tok, model, max_new):
    def generate(messages):
        try:                               # Qwen3 hybrid models: keep them out of thinking mode
            p = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:                  # non-Qwen3 tokenizers don't take the kwarg
            p = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = tok(p, return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False,
                             temperature=None, top_p=None, repetition_penalty=1.1)
        return tok.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True)
    return generate

# ── optional code linking (ATC / ICD-10-PCS), keyed on the raw `entity` ───────
def make_linker(threshold, rxnav):
    sys.path.insert(0, HERE)                       # src/epcr/pipeline/terminology_linker.py
    from terminology_linker import Linker
    idx = cfg.rel(cfg.DEFAULTS.get("code_index_dir", "terminology/index"))
    linker = Linker(threshold=threshold, index_dir=idx)
    print(f"linker vocabularies: {list(linker.vocab)} | index {idx} | threshold {threshold} | rxnav={rxnav}", flush=True)
    rxfn = None
    if rxnav:
        from terminology_linker import rxnav_rxcui
        rxfn = rxnav_rxcui
    def link(items, cat):
        for it in items:
            if not isinstance(it, dict): continue
            name = it.get("entity") or it.get("canonical") or it.get("name") or it.get("text")
            if cat == "medications":
                it["atc"] = linker.link(name, "atc")
                if rxfn: it["rxnorm"] = rxfn(name)
            elif cat == "interventions":
                it["icd10pcs"] = linker.link(name, "icd10pcs")
        return items
    return link

# ── the three extraction modes ────────────────────────────────────────────────
def as_list(obj, key):
    v = (obj or {}).get(key, [])
    return v if isinstance(v, list) else []

def extract(args):
    tpath = rel(args.transcript)
    ppath = rel(args.prompt)
    _pause, segs = segment(tpath, args.thr, args.max_utts)
    print(f"{len(_pause)} pause-segments -> {len(segs)} after size cap (THR={args.thr}, cap={args.max_utts})", flush=True)

    outdir = rel(args.out)
    os.makedirs(outdir, exist_ok=True)

    # load prompt(s) per mode
    if args.mode == "separated":
        if not os.path.isdir(ppath):
            raise SystemExit(f"separated mode needs a prompt DIR of .txt files; got {ppath}")
        pfiles = sorted(glob.glob(os.path.join(ppath, "*.txt")))
        if not pfiles:
            raise SystemExit(f"no .txt prompt files in {ppath}")
        # each file -> one call; a file may return one or several category keys (its JSON shape)
        prompts = {os.path.splitext(os.path.basename(f))[0]: cfg.render(open(f).read()) for f in pfiles}
    else:
        if not os.path.isfile(ppath):
            raise SystemExit(f"{args.mode} mode needs a prompt FILE; got {ppath}")
        prompt = cfg.render(open(ppath).read())

    tok, model, dtype = load_model(args.model, args.dtype)
    generate = make_generate(tok, model, args.max_new_tokens)
    link = make_linker(args.link_threshold, args.rxnav) if args.link else None

    def take(rec, obj, c):   # store one category from a parsed object, by its shape
        if c in OBJECT_CATS:
            rec[c] = obj[c] if isinstance(obj.get(c), dict) else {}
        else:
            rec[c] = as_list(obj, c)

    failn = 0
    for si, seg in enumerate(segs, 1):
        rec = {"segment": si, "span": f"{seg[0]['Start']} -> {seg[-1]['End']}"}
        for c in CATS: rec[c] = cat_default(c)          # defaults; filled below
        plain = "\n".join(f"{s.get('Role','?')}: {(s.get('Utterance') or '').strip()}" for s in seg)

        if args.mode == "combined":
            obj = extract_json(generate([{"role": "system", "content": prompt},
                                         {"role": "user", "content": f"TRANSCRIPT:\n{plain}"}]))
            if not isinstance(obj, dict):
                failn += 1; obj = {}
            for c in CATS: take(rec, obj, c)

        elif args.mode == "separated":
            for name, p in prompts.items():            # one call per prompt file
                obj = extract_json(generate([{"role": "system", "content": p},
                                             {"role": "user", "content": f"TRANSCRIPT:\n{plain}"}]))
                if not isinstance(obj, dict):
                    failn += 1; obj = {}
                for c in CATS:                          # merge whichever categories this file returned
                    if c in obj: take(rec, obj, c)

        if link:
            rec["medications"] = link(rec.get("medications", []), "medications")
            rec["interventions"] = link(rec.get("interventions", []), "interventions")

        attach_times(rec, seg)   # add real timestamps from the transcript

        json.dump(rec, open(f"{outdir}/seg{si:02d}_extraction.json", "w"), indent=1, ensure_ascii=False)
        print(f"== SEG {si} ({rec['span']}) ==  " + " ".join(f"{c[:4]}={len(cat_items(rec, c))}" for c in CATS), flush=True)

    print(f"\nsaved -> {outdir}/  (parse-fails: {failn})", flush=True)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["combined", "separated"], required=True)
    ap.add_argument("--model", default=os.environ.get("MODEL_ID"),
                    help="HF id, or a model key from config.yaml (e.g. qwen3_4b)")
    ap.add_argument("--prompt", help="prompt FILE (combined) or DIR (separated); default per mode")
    ap.add_argument("--transcript", default=os.environ.get("TRANSCRIPT_JSON", DEFAULT_TRANSCRIPT),
                    help="transcript json (default: the config subject's dataset file)")
    ap.add_argument("--out", required=True, help="output directory for seg*_extraction.json")
    ap.add_argument("--thr", type=float, default=float(os.environ.get("THR", _D.get("thr", 180))))
    ap.add_argument("--max-utts", type=int, default=int(os.environ.get("MAX_UTTS", _D.get("max_utts", 80))))
    ap.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("MAX_NEW_TOKENS", _D.get("max_new_tokens", 4096))))
    ap.add_argument("--dtype", default=os.environ.get("DTYPE"))
    ap.add_argument("--link", action="store_true", help="attach ATC/ICD-10-PCS codes to raw entities")
    ap.add_argument("--rxnav", action="store_true", help="also fetch RxNorm via RxNav (network); implies --link use")
    ap.add_argument("--link-threshold", type=float, default=float(os.environ.get("LINK_THRESHOLD", "0.70")))
    ap.add_argument("--repo", help="repo root (default: two levels up from this file)")
    args = ap.parse_args()
    global REPO
    if args.repo: REPO = os.path.abspath(args.repo)
    args.model = cfg.model_id(args.model) if args.model else next(iter(cfg.MODELS.values()))
    if not args.prompt: args.prompt = DEFAULT_PROMPT[args.mode]
    if args.rxnav: args.link = True
    extract(args)

if __name__ == "__main__":
    main()
