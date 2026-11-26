"""Builds the one ePCR the page draws from a session's extractions/ folder.

The extraction service writes extractions/<category>.json, each a log of per-segment records
[{"segment", "extractions": [...]}], with every quote a {"text", "time"} object, and
extractions/medication_administrations.json, each medication administration's current state
(see llm_extraction_service/schema.py). This reads them into the shape the page expects:

    {"medications": [...], "interventions": [...], "vital_signs": [...], "demographics": [...],
     "mechanism_of_injury": [...], "medical_history": [...], "patient_condition": [...]}

Quotes become their text, and an item gets "time" (its earliest quote, seconds from the start)
and "times" (all of its quotes'). Demographics keep the latest statement of each, and the
mechanism of injury its latest description. A medication is drawn from its administration: an
infusion runs from when it was started to when it was stopped, or to the last segment taken in
while it is still running, with a rate change at each new rate.
"""

import json
from pathlib import Path
from typing import Any

CATEGORIES = ("medications", "interventions", "vital_signs", "mechanism_of_injury",
              "demographics", "medical_history", "patient_condition")
HISTORY_KINDS = {"family_history": "Family history", "patient_history": "Patient history",
                 "comorbidity": "Comorbidity"}
CONDITION_KINDS = {"complaint": "Complaint", "mental_status": "Mental status", "finding": "Finding",
                   "suspected_injury": "Suspected injury"}


def read_json(path: Path):
    """The JSON in path; None if it is missing or not JSON (writes are atomic, so never half-written)."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return None


def _quotes(item: dict) -> tuple[list[str], list[float]]:
    """An item's quote texts, and their times, sorted."""
    texts, times = [], []
    for q in item.get("quotes") or []:
        text = q.get("text") if isinstance(q, dict) else q
        if isinstance(text, str) and text.strip():
            texts.append(text)
        t = q.get("time") if isinstance(q, dict) else None
        if isinstance(t, (int, float)):
            times.append(float(t))
    return texts, sorted(times)


def _timed(item: dict) -> dict:
    """The item with its quotes as text, and "time" and "times" from them."""
    texts, times = _quotes(item)
    return {**item, "quotes": texts, "times": times, "time": times[0] if times else None}


def _items(records: Any) -> list[dict]:
    """Every item of a category's records, in order, timed."""
    if not isinstance(records, list):
        return []
    return [_timed(it) for r in records if isinstance(r, dict)
            for it in r.get("extractions") or [] if isinstance(it, dict)]


def _number(v) -> str:
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def _with_unit(value, unit) -> str:
    if value is None:
        return ""
    unit = (unit or "").strip()
    sep = "" if unit in ("%", "°C", "°F") else " "
    return _number(value) + (sep + unit if unit else "")


def _vital_value(v: dict) -> str:
    """A reading as the page shows it: "146/101", "9 (E2 V3 M4)", "37.2 °C"."""
    if v.get("entity") == "blood_pressure":
        if v.get("systolic") is None or v.get("diastolic") is None:
            return ""
        return f"{_number(v['systolic'])}/{_number(v['diastolic'])}" + (f" {v['unit']}" if v.get("unit") else "")
    if v.get("entity") == "gcs":
        parts = [v.get("eye"), v.get("verbal"), v.get("motor")]
        detail = f" (E{parts[0]} V{parts[1]} M{parts[2]})" if None not in parts else ""
        return _number(v.get("value")) + detail if v.get("value") is not None else ""
    return _with_unit(v.get("value"), v.get("unit"))


def _medications(administrations: Any, now: float | None) -> list[dict]:
    """Each administration as the page draws it: a bolus at the time it was given, an infusion
    from its start to its stop (or to now, while running), with its rates over time."""
    meds = []
    for a in administrations if isinstance(administrations, list) else []:
        if not isinstance(a, dict):
            continue
        events = [e for e in a.get("events") or [] if isinstance(e, dict)]
        start = next((e.get("time") for e in events if e.get("time") is not None), None)
        texts, times = _quotes(a)
        infusion = a.get("type") == "infusion"
        stopped = next((e.get("time") for e in events if e.get("action") == "stopped"), None)
        end = stopped if stopped is not None else (now if infusion and a.get("status") == "running" else None)
        meds.append({
            "id": a.get("id"), "entity": a.get("entity"), "standard_name": a.get("standard_name"),
            "type": "continuous" if infusion else "single", "status": a.get("status"),
            "dose": _number(a["dose"]) if a.get("dose") is not None else "", "dose_unit": a.get("dose_unit") or "",
            "route": a.get("route") or "", "rate_unit": a.get("rate_unit") or "",
            "rate_changes": [{"time": e.get("time"), "rate": _number(e["rate"])}
                             for e in events if e.get("rate") is not None and e.get("time") is not None],
            "start_time": start if start is not None else (times[0] if times else None),
            "end_time": end if infusion else None,
            "quotes": texts, "times": times, **(a.get("codes") or {}),
        })
    return sorted(meds, key=lambda m: m["start_time"] if m["start_time"] is not None else 1e12)


def build(folder: Path) -> dict | None:
    """The session's ePCR, or None until the extraction service has taken in a segment
    (extractions/progress.json)."""
    progress = read_json(folder / "progress.json")
    if not isinstance(progress, dict):
        return None
    now = progress.get("end") if isinstance(progress.get("end"), (int, float)) else None
    raw = {c: read_json(folder / f"{c}.json") for c in CATEGORIES}

    vitals = []
    for v in _items(raw["vital_signs"]):
        vitals.append({**v, "value": _vital_value(v)})

    latest = {}  # demographics: the latest statement of each
    for d in _items(raw["demographics"]):
        latest[d.get("entity")] = {**d, "value": d.get("value") if d.get("entity") == "sex"
                                   else _with_unit(d.get("value"), d.get("unit"))}

    mechanism = _items(raw["mechanism_of_injury"])[-1:]  # the latest description is the fullest

    history = [{**h, "note": (h["quotes"] or [""])[0], "kind": HISTORY_KINDS.get(h.get("entity"), h.get("entity"))}
               for h in _items(raw["medical_history"])]

    condition = [{**c, "kind": CONDITION_KINDS.get(c.get("entity"), c.get("entity"))}
                 for c in _items(raw["patient_condition"])]

    interventions = _items(raw["interventions"])

    return {
        "medications": _medications(read_json(folder / "medication_administrations.json"), now),
        "interventions": interventions,
        "vital_signs": vitals,
        "demographics": list(latest.values()),
        "mechanism_of_injury": mechanism,
        "medical_history": history,
        "patient_condition": condition,
    }


def fingerprint(folder: Path) -> tuple:
    """Changes whenever a file build() reads changes, for the page's ETag."""
    files = [*(f"{c}.json" for c in CATEGORIES), "medication_administrations.json", "progress.json"]
    out = []
    for name in files:
        try:
            st = (folder / name).stat()
            out.append((name, st.st_mtime_ns, st.st_size))
        except OSError:
            out.append((name, None, None))
    return tuple(out)
