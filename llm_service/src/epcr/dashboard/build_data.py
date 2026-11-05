#!/usr/bin/env python3
"""Build the ePCR Segment Bench dashboard data (data.json) from results on disk.

View only. Reads the extraction outputs the pipeline already produced under
  <RESULTS_ROOT>/subjectid_<id>/<prompt_dir>/<model>/seg*.json
together with the gold (from the ground-truth xlsx), and writes data.json for index.html.
Nothing is run here — use pipeline/extract.py to produce results.

The transcript is read from the dataset folder by subject id:
  <DATASET_DIR>/subjectid_<id>.json   (default DATASET_DIR=the repo/dataset, subject=<id>)

Usage:
  python build_data.py                              # all runs on disk for the default subject
  python build_data.py --subject <id>               # choose subject
  python build_data.py --out /path/to/data.json     # choose output location
"""
import os, re, json, glob, zipfile, argparse
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_ROOT = os.path.abspath(os.path.join(HERE, ".."))   # src/epcr (holds config.py), one up from dashboard/

CATS = ["medications", "interventions", "vital_signs", "mechanism_of_injury", "demographics", "general_notes"]
DET  = ["dose", "route", "rate", "body_site", "device", "value"]

# ---- models / prompts / defaults come from config.yaml (single source of truth) ----
import importlib.util as _il
_spec = _il.spec_from_file_location("pcr_config", os.path.join(CODE_ROOT, "config.py"))
cfg = _il.module_from_spec(_spec); _spec.loader.exec_module(cfg)
ROOT = REPO = cfg.ROOT   # data (dataset/results) resolves against the repo root config found

# model key -> (label, HF id)
MODELS = {k: (k, v) for k, v in cfg.MODELS.items()}   # key -> (label=key, hf id)
# prompt key -> {label, results-subdir (= key), extract mode, prompt path, kind}
RUNS = {k: {"label": v.get("label", k), "pdir": k, "mode": v["mode"], "prompt": v["path"],
            "kind": "dir" if v["mode"] == "separated" else "file"} for k, v in cfg.PROMPTS.items()}
# stage-2 aggregator rosters (the aggregator model x prompt grid, mirroring the extraction grid)
AGG_MODELS  = dict(cfg.AGG_MODELS)        # agg model key -> hugging face id
AGG_PROMPTS = dict(cfg.AGG_PROMPTS)       # agg prompt key -> {path, label}

# Nothing is hardcoded to a subject: the subject id is the only variable, every path derives
# from it (defaults from config.yaml, overridable by env). Dataset in, results out.
#   READ  dataset:   <DATASET_DIR>/subjectid_<id>.json
#   SAVE  stage 1:   <RESULTS_ROOT>/subjectid_<id>/extraction/<prompt>/<model>/seg*.json
#   SAVE  stage 2:   <RESULTS_ROOT>/subjectid_<id>/aggregation/<prompt>/<model>/aggregate*.json
#   ground truth:    <GROUND_TRUTH_DIR>/subjectid_<id>.xlsx
DATASET_DIR  = os.environ.get("DATASET_DIR",  cfg.DEFAULTS.get("dataset_dir", "dataset"))
RESULTS_ROOT = os.environ.get("RESULTS_ROOT", cfg.DEFAULTS.get("results_root", "results"))
GOLD_DIR     = os.environ.get("GROUND_TRUTH_DIR", cfg.DEFAULTS.get("ground_truth_dir", "dataset/ground_truth"))
SUBJECT      = os.environ.get("SUBJECT",      str(cfg.DEFAULTS.get("subject", "<id>")))
def subject_paths(subject):
    base = f"{RESULTS_ROOT}/subjectid_{subject}"           # subject -> stage -> prompt/model
    return (f"{DATASET_DIR}/subjectid_{subject}.json",     # transcript (dataset)
            f"{base}/extraction",                          # stage-1 per-segment extractions
            f"{GOLD_DIR}/subjectid_{subject}.xlsx",        # gold (mirrors the dataset name)
            f"{base}/aggregation")                         # stage-2 aggregates
DEFAULT_TRANSCRIPT, RESULTS_BASE, GOLD_XLSX, AGG_BASE = subject_paths(SUBJECT)
THR = float(cfg.DEFAULTS.get("thr", 180))

def sec(ts):
    p = str(ts).split(":")
    try:
        if len(p) == 3: return int(p[0]) * 3600 + int(p[1]) * 60 + float(p[2])
        if len(p) == 2: return int(p[0]) * 60 + float(p[1])
    except Exception: pass
    return None
def hhmmss(s):
    s = int(s); return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

# quote -> transcript time: match each quote back to the utterance it came from (same fuzzy match
# extract.py uses), so the dashboard can show a timestamp beside each quote.
ALL_UTTS = []    # [{Start, Utterance}] for the current subject; set in build_one
def _qnrm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())).strip()
def quote_time(q):
    qn = _qnrm(q)
    if not qn or not ALL_UTTS: return ""
    best, bs = None, 0.0
    for u in ALL_UTTS:
        un = _qnrm(u.get("Utterance"))
        if not un: continue
        if qn in un or un in qn:
            sc = min(len(qn), len(un)) / max(len(qn), len(un))
        else:
            a, b = set(qn.split()), set(un.split())
            sc = len(a & b) / len(a | b) if (a | b) else 0
        if sc > bs: bs, best = sc, u
    return (best.get("Start") or "")[:8] if best and bs >= 0.3 else ""

# ── transcript -> segments (pause split THR, size cap 80) ────────────────────
def build_segments(transcript_path):
    dlg = json.load(open(transcript_path))
    utts = [x for x in dlg if (x.get("Utterance") or "").strip()]
    gaps = [sec(b["Start"]) - sec(a["End"]) for a, b in zip(utts, utts[1:])]
    bounds = [0] + [i + 1 for i, g in enumerate(gaps) if g and g > THR] + [len(utts)]
    def size_split(seg, cap=80):
        if len(seg) <= cap: return [seg]
        n = -(-len(seg) // cap); t = len(seg) / n; cuts, used = [], set()
        for p in range(1, n):
            tt = round(p * t); bi, bg = tt, -1
            for i in range(max(1, tt - 15), min(len(seg) - 1, tt + 15)):
                if i in used: continue
                g = sec(seg[i]["Start"]) - sec(seg[i - 1]["End"])
                if g > bg: bi, bg = i, g
            cuts.append(bi); used.add(bi)
        idx = [0] + sorted(set(cuts)) + [len(seg)]
        return [seg[idx[k]:idx[k + 1]] for k in range(len(idx) - 1)]
    raw = [utts[bounds[k]:bounds[k + 1]] for k in range(len(bounds) - 1)]
    final = []
    for s in raw: final.extend(size_split(s))
    segs = []
    for i, s in enumerate(final, 1):
        lo, hi = sec(s[0]["Start"]), sec(s[-1]["End"])
        tr = [{"t": (u["Start"] or "")[:8], "u": (u.get("Utterance") or "").strip()} for u in s]
        segs.append({"i": i, "lo": lo, "hi": hi, "clock": f"{hhmmss(lo)}–{hhmmss(hi)}", "n": len(s), "transcript": tr})
    return segs

def seg_of(segs, wall):
    for s in segs:
        if s["lo"] - 1 <= wall <= s["hi"] + 1: return s["i"]
    return min(segs, key=lambda s: min(abs(wall - s["lo"]), abs(wall - s["hi"])))["i"]

# ── gold from the ground-truth xlsx ──────────────────────────────────────────
def xlsx_rows(path, sheetidx=0):
    z = zipfile.ZipFile(path); ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    ss = []
    if "xl/sharedStrings.xml" in z.namelist():
        r = ET.fromstring(z.read("xl/sharedStrings.xml"))
        for si in r.findall(ns + "si"): ss.append("".join(t.text or "" for t in si.iter(ns + "t")))
    def cn(ref):
        c = re.match(r"[A-Z]+", ref).group(); n = 0
        for ch in c: n = n * 26 + (ord(ch) - 64)
        return n - 1
    def val(c):
        t = c.get("t"); v = c.find(ns + "v"); isv = c.find(ns + "is")
        if isv is not None: return "".join(x.text or "" for x in isv.iter(ns + "t"))
        if v is None: return ""
        return ss[int(v.text)] if t == "s" else v.text
    sheets = sorted(n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml", n))
    root = ET.fromstring(z.read(sheets[sheetidx])); rows = []
    for row in root.iter(ns + "row"):
        d = {}
        for c in row.findall(ns + "c"): d[cn(c.get("r"))] = val(c)
        mx = max(d) if d else -1; rows.append([d.get(i, "") for i in range(mx + 1)])
    return rows

def build_gold(segs):
    path = os.path.join(REPO, GOLD_XLSX)
    if not os.path.exists(path): return {}
    T0 = 19 * 3600 + 53 * 60 + 36
    ev = xlsx_rows(path, 0); hdr = ev[0]; H = {h: i for i, h in enumerate(hdr)}
    def cat_ent(sheet, desc):
        s = (sheet or "").lower(); d = desc or ""
        if "medication" in s: return "medications", d.split(";")[0].strip()
        if "vital" in s: return "vital_signs", d.split("(")[0].replace("Vitals:", "").strip() or "Vitals"
        if "ultrasound" in s: return "interventions", "eFAST ultrasound"
        if any(k in s for k in ("airway", "ventilator", "breathing", "thoraco", "circulation")):
            return "interventions", d.split(";")[0].strip()
        return "general_notes", d[:60]
    gold = {}
    for r in ev[1:]:
        if not r or not r[0]: continue
        try: wall = T0 + float(r[H["min_from_T0"]]) * 60
        except Exception: continue
        si = seg_of(segs, wall); cat, ent = cat_ent(r[H["sheet"]], r[H["description"]])
        gold.setdefault(str(si), {}).setdefault(cat, []).append(
            {"entity": ent, "quotes": [r[H["description"]]],
             "fields": r[H["all_fields"]] if "all_fields" in H else "", "clock": hhmmss(wall)})
    try:
        pt = xlsx_rows(path, 2); demo = ""
        for r in pt:
            if len(r) >= 2 and "sex" in (r[0] or "").lower(): demo = r[1]
        if demo:
            gold.setdefault("1", {}).setdefault("demographics", []).append(
                {"entity": "age/sex/weight/height", "quotes": [demo], "fields": "", "clock": ""})
    except Exception: pass
    return gold

# ── model-run loader (reads the pipeline's extraction JSONs) ──────────────────
# mechanism_of_injury is a single object in the new schema; tolerate the old list shape too.
def _items_of(raw, c):
    if c == "mechanism_of_injury":
        return [raw] if isinstance(raw, dict) and raw else (raw if isinstance(raw, list) else [])
    return raw if isinstance(raw, list) else []

def _norm_item(it):
    e = it.get("entity") or it.get("note"); e = "" if e is None else str(e)   # general_notes uses 'note'
    qs = [q for q in (it.get("quotes") or []) if isinstance(q, str) and q.strip()]
    o = {"entity": e, "quotes": qs, "qtimes": [quote_time(q) for q in qs]}  # a time beside each quote
    if it.get("standard_name"): o["standard_name"] = it["standard_name"]
    if it.get("category"): o["category"] = it["category"]                 # interventions: which procedure
    if isinstance(it.get("details"), dict) and it["details"]: o["details"] = it["details"]  # its per-category details
    t = it.get("time") or (it.get("times") or [None])[0]  # per-segment has time; aggregates have times
    if t: o["time"] = t
    if it.get("times"): o["times"] = it["times"]
    if it.get("atc"): o["atc"] = it["atc"]                # medication code link {code,name,match,score}
    for extra in ("type", "start_time", "end_time", "dose_unit", "rate_unit"):   # event + unit fields
        if it.get(extra): o[extra] = it[extra]
    if it.get("rate_changes"): o["rate_changes"] = it["rate_changes"]   # [{time, rate}, ...]
    if not o.get("time") and it.get("start_time"): o["time"] = it["start_time"]   # events use start_time
    for dd in DET:
        if it.get(dd): o[dd] = it[dd]
    return o

def _norm_rec(s):
    rec = {}
    for c in CATS:
        out = []
        for it in _items_of(s.get(c), c):
            if not isinstance(it, dict): continue
            o = _norm_item(it)
            if o["entity"] or o["quotes"]: out.append(o)
        rec[c] = out
    return rec

def load_runs():
    runs, present = {}, []
    for pk, meta in RUNS.items():
        for mk in MODELS:
            d = os.path.join(REPO, RESULTS_BASE, meta["pdir"], mk)
            files = sorted(glob.glob(f"{d}/seg*_extraction.json"),
                           key=lambda p: int(re.search(r"seg(\d+)", p).group(1)))
            if not files: continue
            present.append([pk, mk, len(files)])
            runs.setdefault(pk, {})[mk] = {}
            for f in files:
                s = json.load(open(f))
                runs[pk][mk][str(s["segment"])] = _norm_rec(s)
    return runs, present

# ── aggregates: aggregation/<ext_prompt>/<ext_model>/<agg_model>/<agg_prompt>/ ; cumulative
#    aggregate_upto_NN.json keyed by N, and a single full aggregate.json keyed "all" ──
def load_aggs():
    """Returns aggs[ext_prompt][ext_model][agg_model][agg_prompt] = {N: rec, 'all': rec}."""
    aggs = {}
    for pk, meta in RUNS.items():
        for mk in MODELS:
            for am in AGG_MODELS:
                for ap in AGG_PROMPTS:
                    d = os.path.join(REPO, AGG_BASE, meta["pdir"], mk, am, ap)
                    if not os.path.isdir(d): continue
                    byn = {}
                    for f in glob.glob(f"{d}/aggregate_upto_*.json"):
                        m = re.search(r"aggregate_upto_(\d+)\.json$", os.path.basename(f))
                        if not m: continue
                        try: s = json.load(open(f))
                        except Exception: continue
                        byn[str(int(m.group(1)))] = _norm_rec(s)
                    single = os.path.join(d, "aggregate.json")   # single full-run aggregate
                    if os.path.exists(single):
                        try: byn["all"] = _norm_rec(json.load(open(single)))
                        except Exception: pass
                    if byn:
                        aggs.setdefault(pk, {}).setdefault(mk, {}).setdefault(am, {})[ap] = byn
    return aggs

def load_agg_prompts():
    """The aggregator prompt each run actually used: the saved aggregate_prompt.txt in the run dir
    (written by aggregate.py), else the live roster prompt. Keyed the same 4 levels as the aggs,
    so the dashboard shows the prompt that generated the selected results."""
    out = {}
    for pk, meta in RUNS.items():
        for mk in MODELS:
            for am in AGG_MODELS:
                for ap in AGG_PROMPTS:
                    d = os.path.join(REPO, AGG_BASE, meta["pdir"], mk, am, ap)
                    if not os.path.isdir(d): continue
                    saved = os.path.join(d, "aggregate_prompt.txt")
                    txt = open(saved).read() if os.path.exists(saved) else agg_prompt_text(ap)
                    out.setdefault(pk, {}).setdefault(mk, {}).setdefault(am, {})[ap] = txt
    return out

def prompt_text(pk):
    meta = RUNS[pk]; val = meta["prompt"]
    if meta["kind"] == "file":
        p = os.path.join(REPO, val)
        return cfg.render(open(p).read()) if os.path.exists(p) else ""
    parts = []
    for p in sorted(glob.glob(os.path.join(REPO, val, "*.txt"))):   # every prompt file, not just category names
        name = os.path.splitext(os.path.basename(p))[0]
        parts.append(f"━━ {name} ━━\n" + cfg.render(open(p).read()).strip())
    return "SEPARATED — one call per prompt file:\n\n" + "\n\n".join(parts)

def agg_prompt_text(ap_key):
    """The live text of an aggregator prompt (roster key), for the view-prompt link and as the
    fallback when a run saved no aggregate_prompt.txt."""
    ent = AGG_PROMPTS.get(ap_key) or {}
    p = os.path.join(REPO, ent.get("path", "prompts/aggregate.txt"))
    return cfg.render(open(p).read()) if os.path.exists(p) else ""

# ── subjects available in the dataset folder ──────────────────────────────────
def list_subjects():
    ids = []
    for f in glob.glob(os.path.join(REPO, DATASET_DIR, "subjectid_*.json")):
        m = re.match(r"subjectid_(.+)\.json$", os.path.basename(f))
        if m: ids.append(m.group(1))
    return sorted(ids)

# ── per-datapoint payload (segments + gold + the runs on disk for that subject) ──
def build_one(subject):
    """Everything the viewer needs for one datapoint (subject). Reads only; runs nothing."""
    global DEFAULT_TRANSCRIPT, RESULTS_BASE, GOLD_XLSX, AGG_BASE, ALL_UTTS
    DEFAULT_TRANSCRIPT, RESULTS_BASE, GOLD_XLSX, AGG_BASE = subject_paths(subject)
    tpath = os.path.join(ROOT, DEFAULT_TRANSCRIPT)
    if not os.path.exists(tpath):
        raise SystemExit(f"no transcript at {tpath} — put the subject's transcript in "
                         f"{DATASET_DIR}/ named subjectid_<id>.json (e.g. subjectid_{subject}.json)")
    ALL_UTTS = [x for x in json.load(open(tpath)) if (x.get("Utterance") or "").strip()]  # for quote_time
    segs = build_segments(tpath)
    gold = build_gold(segs)
    runs, present = load_runs()
    aggs = load_aggs()   # 4-level: {ext_prompt: {ext_model: {agg_model: {agg_prompt: {N: rec}}}}}
    agg_prompts = load_agg_prompts()   # the prompt each aggregation run actually used (same 4 levels)
    return {"subject": subject, "scenario": f"SubjectID {subject}",
            "segments": segs, "gold": gold, "runs": runs, "present": present,
            "aggs": aggs, "aggPromptsUsed": agg_prompts}

# ── assemble what the viewer shows, read live from the input + output folders ──
# Three variables, all view-only selectors in the dashboard:
#   datapoints = subjects found in the dataset folder (INPUT)   -> bySubject[<id>]
#   prompts    = config.yaml prompts                            -> shared, top level
#   models     = config.yaml models                             -> shared, top level
# The per-datapoint content is the transcript (INPUT) + gold + the results (OUTPUT) on disk.
def build_payload():
    subjects = list_subjects() or [SUBJECT]
    by_subject = {s: build_one(s) for s in subjects}
    default = SUBJECT if SUBJECT in by_subject else subjects[0]
    return {"subjects": subjects, "subject": default, "cats": CATS,
            # extraction grid: model x prompt
            "prompts": [{"key": k, "label": RUNS[k]["label"], "text": prompt_text(k)} for k in RUNS],
            "models": [{"key": k, "label": MODELS[k][0]} for k in MODELS],
            # aggregation grid: model x prompt (each its own dropdown + view-prompt link)
            "aggModels":  [{"key": k, "label": v} for k, v in AGG_MODELS.items()],
            "aggPrompts": [{"key": k, "label": (AGG_PROMPTS[k].get("label", k)), "text": agg_prompt_text(k)} for k in AGG_PROMPTS],
            "bySubject": by_subject}

# Optional: write the payload to a static file (for viewing index.html without the server).
# The server does NOT use this — it serves build_payload() live from input + output per request.
def build(out=None):
    data = build_payload()
    out = out or os.path.join(HERE, "data.json")
    json.dump(data, open(out, "w"), ensure_ascii=False)
    npresent = sum(len(d["present"]) for d in data["bySubject"].values())
    print(f"wrote {out}  ({os.path.getsize(out)} bytes) | datapoints={len(data['subjects'])} "
          f"({', '.join(data['subjects'])}) | runs with data={npresent}")
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", help="write a static data.json here (the server does not need this)")
    a = ap.parse_args()
    build(out=a.out)
