"""Link a clinical name to a code, returning both the code and the concept name.

Local vocabularies (ATC, ICD-10-PCS, SNOMED CT procedures) are matched with a HYBRID
strategy: an exact normalized-string match first (precise), then a SapBERT nearest-neighbour
fallback above a cosine threshold, otherwise uncodeable (fail loud, no forced code). RxNorm is
looked up through the public RxNav API (no license).

Each result carries {code, name, match, score, codeable, system}.
  medications   -> ATC (local) + RxNorm (RxNav)
  interventions -> ICD-10-PCS (local)

The embedding index (<system>.emb.npy + <system>.codes.json) lives under terminology/index in this
folder by default. Override with the CODE_INDEX_DIR env var or the index_dir argument.

Import: from terminology_linker import Linker, rxnav_rxcui
"""
import json, os, re
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

SAP = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
HERE = os.path.dirname(os.path.abspath(__file__))

def _repo_root():
    """The repo root — the dir holding config.yaml (code lives under src/epcr/). Override: PCR_ROOT."""
    d = os.environ.get("PCR_ROOT") or HERE
    for _ in range(6):
        if os.path.isfile(os.path.join(d, "config.yaml")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.abspath(os.path.join(HERE, os.pardir, os.pardir, os.pardir))

def default_index_dir():
    return os.environ.get("CODE_INDEX_DIR") or os.path.join(_repo_root(), "terminology", "index")

# Exact-match normalization: the Whisper English text normalizer (lowercases, maps spelled-out
# numbers to digits, standardizes units/spelling) — more matches land before the SapBERT fallback.
# Both the index keys and the query are normalized through this, so exact matching stays consistent.
_WHISPER_NORM = EnglishTextNormalizer({})
def _nrm(s):
    return _WHISPER_NORM(s or "")


class Linker:
    def __init__(self, threshold=0.70, index_dir=None):
        idx = index_dir or default_index_dir()
        if not os.path.isdir(idx):
            raise SystemExit(f"code index not found at {idx}. Place the embedding index there "
                             f"or set CODE_INDEX_DIR. Needed only for --link.")
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.tok = AutoTokenizer.from_pretrained(SAP)
        self.model = AutoModel.from_pretrained(SAP).to(self.dev).eval()
        self.thr = threshold
        self.vocab = {}       # system -> (emb tensor, codes list)
        self.exact = {}       # system -> {normalized term -> index}
        for f in os.listdir(idx):
            if f.endswith(".emb.npy"):
                sys = f[:-len(".emb.npy")]
                emb = np.load(f"{idx}/{sys}.emb.npy").astype(np.float32)
                codes = json.load(open(f"{idx}/{sys}.codes.json"))
                self.vocab[sys] = (torch.tensor(emb, device=self.dev), codes)
                ex = {}
                for i, rec in enumerate(codes):
                    key = _nrm(rec.get("term") or rec.get("name"))
                    if key and key not in ex:      # first occurrence wins
                        ex[key] = i
                self.exact[sys] = ex

    def _disp(self, rec):
        return rec.get("name") or rec.get("term") or ""

    def _embed(self, name):
        b = self.tok([name], padding=True, truncation=True, max_length=64, return_tensors="pt").to(self.dev)
        with torch.no_grad():
            e = self.model(**b).last_hidden_state[:, 0, :]
        return torch.nn.functional.normalize(e, dim=1)[0]

    def link(self, name, system):
        """Hybrid link of `name` in `system`. None if the system has no index."""
        if not name or system not in self.vocab:
            return None
        emb, codes = self.vocab[system]
        q = _nrm(name)
        i = self.exact[system].get(q)                      # 1) exact normalized match
        if i is not None:
            rec = codes[i]
            return {"code": rec["code"], "name": self._disp(rec), "match": "exact",
                    "score": 1.0, "codeable": True, "system": system}
        sims = emb @ self._embed(name.lower())             # 2) SapBERT nearest neighbour
        i = int(torch.argmax(sims)); score = float(sims[i]); rec = codes[i]
        return {"code": rec["code"], "name": self._disp(rec), "match": "sapbert",
                "score": round(score, 3), "codeable": score >= self.thr, "system": system}


def rxnav_rxcui(name, timeout=10):
    """RxNorm concept via the public RxNav API (no license). Returns {rxcui, name, ...}."""
    import urllib.request, urllib.parse
    try:
        u = "https://rxnav.nlm.nih.gov/REST/rxcui.json?" + urllib.parse.urlencode({"name": name, "search": 2})
        ids = json.load(urllib.request.urlopen(u, timeout=timeout)).get("idGroup", {}).get("rxnormId", [])
        if ids:
            return {"rxcui": ids[0], "name": name, "match": "exact", "approximate": False}
        u2 = "https://rxnav.nlm.nih.gov/REST/approximateTerm.json?" + urllib.parse.urlencode({"term": name, "maxEntries": 1})
        cand = json.load(urllib.request.urlopen(u2, timeout=timeout)).get("approximateGroup", {}).get("candidate", [])
        if cand:
            return {"rxcui": cand[0].get("rxcui"), "name": cand[0].get("name"),
                    "score": cand[0].get("score"), "match": "approximate", "approximate": True}
    except Exception as e:
        return {"error": str(e)}
    return None
