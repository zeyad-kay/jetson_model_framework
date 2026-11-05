"""Single source of truth loader: reads config.yaml.

Both the pipeline (src/epcr/pipeline/extract.py) and the dashboard
(src/epcr/dashboard/build_data.py) import this so models/prompts/defaults are defined
in one place — the YAML — not in code. Override the file with the PCR_CONFIG env var.
"""
import os
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))          # this file's dir (src/epcr)

def _find_root(start):
    """The repo root — the directory that holds config.yaml. The code lives under src/epcr/ while
    config.yaml and the data dirs (dataset, results, prompts, terminology) sit at the repo root, so
    walk up from here to find it. Override with PCR_ROOT."""
    d = start
    for _ in range(6):
        if os.path.isfile(os.path.join(d, "config.yaml")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))   # fallback: two up from src/epcr

ROOT = os.environ.get("PCR_ROOT") or _find_root(HERE)      # every config/data path resolves against it
CONFIG_PATH = os.environ.get("PCR_CONFIG", os.path.join(ROOT, "config.yaml"))

_cfg = yaml.safe_load(open(CONFIG_PATH)) or {}
DEFAULTS  = _cfg.get("defaults", {}) or {}                 # subject, dataset_dir, results_root, thr, ...
MODELS    = _cfg.get("models", {}) or {}                   # key -> hugging face id
PROMPTS   = _cfg.get("prompts", {}) or {}                  # stage-1 extraction: key -> {mode, path, label}
AGGREGATE = _cfg.get("aggregate", {}) or {}                # stage-2 aggregator: {label, models, prompts}
AGG_MODELS  = AGGREGATE.get("models", {}) or {}            # aggregator model roster: key -> hugging face id
AGG_PROMPTS = AGGREGATE.get("prompts", {}) or {}           # aggregator prompt roster: key -> {path, label}
INTERVENTION_VOCAB = _cfg.get("intervention_vocab", []) or []   # gate for aggregator interventions

# shared prompt fragments (prompts/_shared.yaml): {entity} {standard_name} {quotes}
# and {<category>_def}. render() substitutes them into a prompt when it is loaded.
_shared = {}
try:
    _shared = yaml.safe_load(open(os.path.join(ROOT, "prompts", "_shared.yaml"))) or {}
except FileNotFoundError:
    pass
FRAGMENTS = {}
for _k, _v in (_shared.get("fields") or {}).items():       FRAGMENTS[_k] = _v
for _k, _v in (_shared.get("definitions") or {}).items():  FRAGMENTS[_k + "_def"] = _v

def render(text):
    """Substitute the shared {fragment} placeholders into a prompt. Other braces (JSON,
    the runtime {transcript} / {entities} placeholders) are left untouched."""
    for k, v in FRAGMENTS.items():
        text = text.replace("{" + k + "}", str(v))
    return text

def model_id(key):
    """Resolve a model key (e.g. 'qwen3_4b') to its HF id; pass an id/unknown through unchanged."""
    return MODELS.get(key, key)

def prompt_entry(key):
    """Return the {mode, path, label} for a prompt key, or None."""
    return PROMPTS.get(key)

def agg_model_id(key):
    """Resolve an aggregator model key to its HF id; pass an id/unknown through unchanged."""
    return AGG_MODELS.get(key, key)

def agg_prompt_entry(key):
    """Return the {path, label} for an aggregator prompt key, or None."""
    return AGG_PROMPTS.get(key)

def rel(p):
    """Resolve a config path relative to the repo root (absolute paths unchanged)."""
    return p if os.path.isabs(p) else os.path.join(ROOT, p)
