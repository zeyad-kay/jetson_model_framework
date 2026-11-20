// Trauma timeline monitor: polls the dashboard server for a session's ePCR (built from its
// structured/ folder by dashboard/epcr.py) and draws it. Times in the ePCR are seconds
// from the start of the recording, so the timeline shows elapsed time.
"use strict";

const POLL_MS = 2000;           // how often the selected session is re-read
const SESSIONS_POLL_MS = 10000; // how often the session list is re-read
const PAD_LEFT = 150;           // px before t = 0, so the sticky lane titles don't cover the first events
const PAD_RIGHT = 260;          // px after the last event, room for its label
const LABEL_MAX = 230;          // px, a label's widest
const LABEL_GAP = 14;           // px between labels on the same row
const ROW_H = 62;               // px per label row
const STEM_MIN = 18;            // px of stem below the nearest row
const MAX_ROWS = 3;             // label rows per lane; events beyond them get a dot only (details on hover)
// Patient condition kinds that go on the injuries panel (mental status doesn't).
const INJURY_KINDS = new Set(["complaint", "finding", "suspected_injury"]);

const LANES = [
  { key: "condition", title: "Patient condition", color: "var(--lane-condition)" },
  { key: "medications", title: "Medications", color: "var(--lane-medications)" },
  { key: "interventions", title: "Interventions", color: "var(--lane-interventions)" },
];

const state = {
  sessions: [],
  sessionId: null,
  chosen: false,       // the viewer picked the session (or the URL named it): don't auto-switch
  etag: null,
  woundName: null,     // the wound image shown
  woundPinned: false,  // the viewer stepped back from the newest image: don't jump to new ones
  detail: null,        // the last /api/sessions/<id> reply
  windowSec: 1200,     // visible span; 0 = everything
  follow: true,        // keep the latest events in view
  pps: 1,              // px per second, from the window and the panel width
  longTimes: false,    // h:mm:ss instead of m:ss
};

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
};

// ── formatting ──────────────────────────────────────────────────────────────

function fmtTime(t) {
  if (typeof t !== "number" || !isFinite(t)) return "–";
  const s = Math.max(0, Math.round(t));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  const mm = String(m).padStart(2, "0"), ss = String(sec).padStart(2, "0");
  return state.longTimes ? `${h}:${mm}:${ss}` : `${String(Math.floor(s / 60)).padStart(2, "0")}:${ss}`;
}

const UNIT_WORDS = [
  [/\bmilligrams?\b/gi, "mg"], [/\bmicrograms?\b/gi, "mcg"], [/\bgrams?\b/gi, "g"],
  [/\bmillilit(re|er)s?\b/gi, "mL"], [/\blit(re|er)s?\b/gi, "L"], [/\bunits?\b/gi, "U"],
  [/\bper kilo(gram)?\b/gi, "/kg"], [/\bper hour\b/gi, "/h"], [/\ban hour\b/gi, "/h"],
  [/\bper minute\b/gi, "/min"], [/\bper\b/gi, "/"],
];
function shortUnit(text) {
  let s = String(text || "");
  for (const [re, short] of UNIT_WORDS) s = s.replace(re, short);
  return s.replace(/\s*\/\s*/g, "/").replace(/\s+/g, " ").trim();
}
const ROUTES = { intravenous: "IV", intramuscular: "IM", subcutaneous: "SC", oral: "PO", intranasal: "IN",
                 intraosseous: "IO", inhaled: "inh", sublingual: "SL", rectal: "PR" };
const shortRoute = (r) => ROUTES[String(r || "").toLowerCase().trim()] || (r || "");
const capitalize = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
const isNum = (t) => typeof t === "number" && isFinite(t);

function clean(value) {
  // Drops the empty fields the extractor fills in ("", null, "not specified").
  if (value === null || value === undefined) return null;
  if (typeof value === "object") {
    const parts = Object.entries(value).map(([k, v]) => [k, clean(v)]).filter(([, v]) => v !== null);
    return parts.length ? parts.map(([k, v]) => `${k} ${v}`).join(", ") : null;
  }
  const s = String(value).trim();
  return s && !/^(not specified|none|null|n\/a)$/i.test(s) ? s : null;
}

// ── ePCR → view model ──────────────────────────────────────────────────

const asList = (v) => (Array.isArray(v) ? v : v && typeof v === "object" && Object.keys(v).length ? [v] : [])
  .filter((it) => it && typeof it === "object");

const VITAL_NAMES = {
  "blood pressure": "BP", "systolic blood pressure": "SBP", "diastolic blood pressure": "DBP",
  "heart rate": "HR", "pulse": "HR", "oxygen saturation": "SpO₂", "respiratory rate": "RR",
  "temperature": "Temp", "glasgow coma scale": "GCS", "eye opening": "E", "verbal response": "V",
  "motor response": "M", "blood glucose": "BGL", "end tidal co2": "EtCO₂", "pain": "Pain",
  "mean arterial pressure": "MAP", "end-tidal carbon dioxide": "EtCO₂", "peak pressure": "Ppeak",
  "body temperature": "Temp", "saturation": "SpO₂", "peak airway pressure": "Ppeak",
  "glucose level": "BGL", "glucose": "BGL", "fraction of inspired oxygen": "FiO₂", "cuff pressure": "Cuff",
};
// The order of the latest vital signs panel; vitals not listed follow, in the order first read.
const VITAL_ORDER = ["BP", "MAP", "HR", "SpO₂", "RR", "EtCO₂", "Temp", "GCS", "BGL", "Ppeak", "FiO₂"];

// Each vital sign reading, for the latest vital signs panel.
function vitalReadings(agg) {
  return asList(agg.vital_signs)
    .map((v) => {
      const name = String(v.standard_name || v.entity || "").toLowerCase().trim();
      return { t: v.time, name, short: VITAL_NAMES[name] || capitalize(v.entity || name), value: clean(v.value), v };
    })
    .filter((r) => isNum(r.t) && r.value)
    .sort((a, b) => a.t - b.t);
}

// The numbers of a reading: [systolic, diastolic] for "157/101", else its first number.
function vitalNumbers(value) {
  const bp = String(value).match(/^\s*(\d+(?:\.\d+)?)\s*\/\s*(\d+(?:\.\d+)?)/);
  if (bp) return [Number(bp[1]), Number(bp[2])];
  const n = String(value).match(/-?\d+(?:\.\d+)?/);
  return n ? [Number(n[0])] : null;
}

// The latest reading of each vital sign, for the side panel, with all its readings over time.
function latestVitals(agg) {
  const latest = new Map(), series = new Map();
  for (const r of vitalReadings(agg)) {
    latest.delete(r.short); // re-insert, so vitals not in VITAL_ORDER keep their first-read order
    latest.set(r.short, r);
    const values = vitalNumbers(r.value);
    if (!values) continue;
    if (!series.has(r.short)) series.set(r.short, []);
    series.get(r.short).push({ t: r.t, values });
  }
  const rank = (s) => (VITAL_ORDER.includes(s) ? VITAL_ORDER.indexOf(s) : VITAL_ORDER.length);
  return [...latest.values()]
    .sort((a, b) => rank(a.short) - rank(b.short))
    .map((r) => ({
      kind: "Vital sign", t: r.t, label: r.short, value: r.value, title: `${r.short} ${r.value}`,
      details: [["Vital", r.v.standard_name || r.v.entity || r.name]],
      chart: { points: series.get(r.short) || [], unit: (String(r.value).match(/^[-\d./]+\s*(.*)$/) || [])[1] || "" },
    }));
}

function buildModel(agg) {
  // Medical history isn't on the timeline: it has its own panel, grouped by kind.
  const history = asList(agg.medical_history).map((n) => ({
    kind: "Medical history", t: n.time, group: n.kind || "Other",
    title: (n.quotes || []).filter(Boolean).join(" ") || n.note || "",
    details: n.kind ? [["Kind", n.kind]] : [], quotes: n.quotes, times: n.times,
  }));
  const condition = asList(agg.patient_condition).map((c) => {
    const site = clean(c.body_site) || "";
    const details = [];
    if (c.kind) details.push(["Kind", c.kind]);
    if (site) details.push(["Body site", site]);
    return {
      kind: "Patient condition", entity: c.entity, t: c.time, title: capitalize(c.standard_name || c.entity || ""),
      sub: [c.kind, site].filter(Boolean).join(" · "), where: [c.standard_name, site].filter(Boolean).join(", "),
      details, quotes: c.quotes, times: c.times,
    };
  });

  const meds = asList(agg.medications).map((m) => {
    const continuous = String(m.type || "").toLowerCase() === "continuous";
    const rates = asList(m.rate_changes).filter((rc) => isNum(rc.time));
    const unit = shortUnit(m.rate_unit);
    const dose = clean(m.dose) ? shortUnit([m.dose, m.dose_unit].filter(Boolean).join(" ")) : "";
    const route = shortRoute(m.route);
    const sub = continuous
      ? ["infusion", rates.length ? `${rates.length} rate change${rates.length > 1 ? "s" : ""}` : "", route].filter(Boolean).join(" · ")
      : [dose, route].filter(Boolean).join(" ");
    const details = [];
    if (dose) details.push(["Dose", dose]);
    if (route) details.push(["Route", m.route]);
    if (continuous) details.push(["Type", "infusion"]);
    if (m.status === "running") details.push(["Status", "running"]);
    else if (isNum(m.end_time)) details.push(["Stopped", fmtTime(m.end_time)]);
    if (rates.length) details.push(["Rates", rates.map((rc) => `${fmtTime(rc.time)} → ${rc.rate}${unit ? " " + unit : ""}`).join("\n")]);
    return {
      kind: "Medication", t: m.start_time, end: isNum(m.end_time) ? m.end_time : null, continuous, rates,
      title: capitalize(m.entity || m.standard_name || ""), sub, details, quotes: m.quotes,
    };
  });

  const interventions = asList(agg.interventions).map((it) => {
    const summary = clean(it.details) || "";
    const category = clean(it.category) ? it.category.replace(/_/g, " ") : "";
    const rows = summary ? [["Details", summary]] : [];
    if (category) rows.unshift(["Category", category]);
    return {
      kind: "Intervention", t: it.time, title: capitalize(String(it.entity || "").replace(/_/g, " ")), sub: summary || category,
      details: rows, quotes: it.quotes, times: it.times,
    };
  });

  const injuries = condition.filter((c) => INJURY_KINDS.has(c.entity));

  return { lanes: { condition, medications: meds, interventions }, injuries, history, latestVitals: latestVitals(agg) };
}

function maxTime(model) {
  let max = 0;
  for (const items of Object.values(model.lanes)) {
    for (const it of items) {
      for (const t of [it.t, it.end, ...(it.rates || []).map((r) => r.time)]) if (isNum(t)) max = Math.max(max, t);
    }
  }
  return max;
}

// ── header ──────────────────────────────────────────────────────────────────

function renderHeader(agg, detail) {
  const demo = asList(agg && agg.demographics);
  const find = (name) => demo.find((d) => String(d.entity || "").toLowerCase() === name);
  const sex = find("sex") || find("gender");
  const age = find("age");
  const weight = find("weight");
  const sexLetter = sex && clean(sex.value) ? clean(sex.value).charAt(0).toUpperCase() : null;
  const ageNum = age && clean(age.value) ? (String(age.value).match(/\d+/) || [clean(age.value)])[0] : null;
  $("patient-text").textContent = [sexLetter, ageNum].filter(Boolean).join(" · ") || "–";
  $("patient").title = ["Sex: " + (sex ? sex.value : "unknown"), "Age: " + (age ? age.value : "unknown"),
                        weight ? "Weight: " + weight.value : null].filter(Boolean).join("\n");

  const moi = asList(agg && agg.mechanism_of_injury)[0];
  $("mechanism-title").textContent = moi && moi.entity ? moi.entity : "No mechanism of injury yet";
  $("mechanism-title").title = moi && moi.entity ? moi.entity : "";

  const status = $("status");
  const st = !detail ? "none" : detail.live ? "live" : "ended";
  status.dataset.state = st;
  $("status-text").textContent = st === "live" ? "Live" : st === "ended" ? "Ended" : "No session";
  status.title = detail && detail.updated ? `Last update ${new Date(detail.updated * 1000).toLocaleTimeString()}` : "";
}

// ── injuries ────────────────────────────────────────────────────────────────

// Body regions an injury note can name. x, y are in the 100 × 222 figure's own units; for a
// lateral region they are the patient's right side, mirrored for the left. Specific names
// come before the general ones they contain (upper arm before arm).
const REGIONS = [
  { label: "Scalp", re: /\b(scalp|skull|cranial|forehead|occipital)\b/, view: "front", x: 50, y: 5 },
  { label: "Head", re: /\bhead\b/, view: "front", x: 50, y: 9 },
  { label: "Face", re: /\b(face|facial|eyes?|nose|nasal|jaw|mandible|mouth|lips?|cheek|ears?)\b/, view: "front", x: 50, y: 19 },
  { label: "Neck", re: /\b(neck|cervical)\b/, view: "front", x: 50, y: 30 },
  { label: "Shoulder", re: /\b(shoulders?|clavicles?|collarbones?)\b/, view: "front", x: 31, y: 41, lateral: true },
  { label: "Chest", re: /\b(chest|thorax|thoracic cavity|ribs?|sternum|lungs?|pneumothorax|hemothorax|haemothorax)\b/, view: "front", x: 50, y: 58, sided: 11 },
  { label: "Upper arm", re: /\b(upper arms?|humerus)\b/, view: "front", x: 27, y: 61, lateral: true },
  { label: "Elbow", re: /\belbows?\b/, view: "front", x: 23, y: 80, lateral: true },
  { label: "Forearm", re: /\b(forearms?|radius|ulna)\b/, view: "front", x: 21, y: 97, lateral: true },
  { label: "Hand", re: /\b(wrists?|hands?|fingers?|thumbs?)\b/, view: "front", x: 18, y: 119, lateral: true },
  { label: "Arm", re: /\barms?\b/, view: "front", x: 25, y: 72, lateral: true },
  { label: "Abdomen", re: /\b(abdomen|abdominal|stomach|belly|liver|spleen)\b/, view: "front", x: 50, y: 88 },
  { label: "Pelvis", re: /\b(pelvis|pelvic|groin)\b/, view: "front", x: 50, y: 112 },
  { label: "Hip", re: /\bhips?\b/, view: "front", x: 36, y: 114, lateral: true },
  { label: "Thigh", re: /\b(thighs?|femurs?|femoral)\b/, view: "front", x: 41, y: 140, lateral: true },
  { label: "Knee", re: /\b(knees?|patella)\b/, view: "front", x: 42, y: 164, lateral: true },
  { label: "Lower leg", re: /\b(lower legs?|shins?|tibia|fibula|calf|calves)\b/, view: "front", x: 42, y: 185, lateral: true },
  { label: "Foot", re: /\b(ankles?|feet|foot|toes?)\b/, view: "front", x: 41, y: 210, lateral: true },
  { label: "Leg", re: /\blegs?\b/, view: "front", x: 42, y: 160, lateral: true },
  { label: "Spine", re: /\b(spine|spinal|vertebrae?|vertebral)\b/, view: "back", x: 50, y: 72 },
  { label: "Upper back", re: /\b(upper back|scapulae?|shoulder blades?)\b/, view: "back", x: 50, y: 52 },
  { label: "Lower back", re: /\b(lower back|lumbar|flanks?)\b/, view: "back", x: 50, y: 96 },
  { label: "Back", re: /\bback\b(?! (to|up|in|on)\b)/, view: "back", x: 50, y: 80 },
  { label: "Buttock", re: /\b(buttocks?|gluteal)\b/, view: "back", x: 42, y: 116, lateral: true },
];
const FIG_X = { front: 14, back: 146 };   // where each figure sits in the body-map svg (100 units wide)
const BADGE_GAP = 7;                      // badge column's distance outside its figure
const BADGE_STEP = 16;                    // min vertical distance between badges in a column
const SIDE_RE = /\b(left|right|bilateral|both)\b(-sided)?|\b([LR])\b/gi;

// The body regions a note names, each with the side(s) it is on: "L", "R", both, or none (midline).
function injuryRegions(text) {
  const found = [];
  let rest = String(text || "");
  for (const region of REGIONS) {
    const re = new RegExp(region.re.source, "gi");
    const sides = new Set();
    let hit = false, plural = false, m;
    while ((m = re.exec(rest))) {
      hit = true;
      if (/(s|feet|calves)$/i.test(m[0]) && !/(thorax|pelvis|humerus|radius|abdomen|patella)$/i.test(m[0])) plural = true;
      // The side is the last "left"/"right" in the few words before the name.
      const before = rest.slice(Math.max(0, m.index - 30), m.index);
      const words = [...before.matchAll(SIDE_RE)].map((s) => s[0].toLowerCase());
      const side = words[words.length - 1];
      if (side === "left" || side === "l") sides.add("L");
      else if (side === "right" || side === "r") sides.add("R");
      else if (side === "bilateral" || side === "both") { sides.add("L"); sides.add("R"); }
    }
    if (!hit) continue;
    // Blank out what matched, so "upper arm" isn't counted again as "arm".
    rest = rest.replace(re, (s) => " ".repeat(s.length));
    if (region.lateral && !sides.size) { sides.add("R"); sides.add("L"); }
    const sideList = [...sides].sort().reverse(); // R before L
    const both = sideList.length === 2;
    const prefix = sideList.length === 1 ? sideList[0] + " " : "";
    const label = prefix + (both && plural ? (region.label === "Foot" ? "Feet" : region.label + "s") : prefix ? region.label.toLowerCase() : region.label);
    found.push({ region, sides: region.lateral || region.sided ? sideList : [], label });
  }
  return found;
}

// A region's points in the svg's units: one per side, the patient's left on the viewer's right
// (front) or the viewer's left (back).
function regionPoints({ region, sides }) {
  const ox = FIG_X[region.view];
  if (!sides.length) return [{ x: ox + region.x, y: region.y }];
  return sides.map((side) => {
    let x = region.lateral ? region.x : region.x - region.sided; // patient's right
    const mirrored = (side === "L") === (region.view === "front");
    if (mirrored) x = 100 - x;
    return { x: ox + x, y: region.y };
  });
}

const SVG_NS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

function highlightInjury(num, on) {
  document.querySelectorAll(`[data-injury~="${num}"]`).forEach((n) => n.classList.toggle("hl", on));
}

function wireInjury(node, note, num, tip = note) {
  attachTooltip(node, tip);
  node.addEventListener("click", () => revealEvent(note));
  node.addEventListener("mouseenter", () => highlightInjury(num, true));
  node.addEventListener("mouseleave", () => highlightInjury(num, false));
  node.addEventListener("focus", () => highlightInjury(num, true));
  node.addEventListener("blur", () => highlightInjury(num, false));
  node.dataset.injury = num;
}

// The tooltip of a location: the injuries there by name, not their quotes.
function locationTip(label, notes) {
  const names = [...new Set(notes.map((n) => n.title || "–"))];
  const times = notes.map((n) => n.t).filter(isNum);
  return {
    kind: names.length > 1 ? `${names.length} injuries` : "Injury", t: times.length ? Math.min(...times) : null,
    title: label, entities: names,
  };
}

function renderInjuries(model) {
  const list = $("injury-list"), chips = $("injury-chips"), marks = $("body-marks");
  list.replaceChildren();
  chips.replaceChildren();
  marks.replaceChildren();
  // Each spot's badge goes in a column beside its figure, on the spot's side, pushed down
  // where badges would overlap; a leader line joins them. A spot is drawn once, for the first
  // injury there; later injuries at the same spot only join its highlight.
  const columns = new Map(); // "front:-1" → [{ p, num, nums, label, note }]
  const spots = new Map();   // "front:x:y" → its mark
  model.injuries.forEach((note, i) => {
    for (const r of injuryRegions(note.where || note.title)) {
      for (const p of regionPoints(r)) {
        const spot = `${r.region.view}:${p.x}:${p.y}`;
        if (spots.has(spot)) {
          const mark = spots.get(spot);
          if (!mark.nums.includes(String(i + 1))) { mark.nums.push(String(i + 1)); mark.notes.push(note); }
          continue;
        }
        const centre = FIG_X[r.region.view] + 50;
        let dir = p.x > centre + 1 ? 1 : p.x < centre - 1 ? -1 : 0;
        if (!dir) { // midline: the emptier column
          const n = (d) => (columns.get(`${r.region.view}:${d}`) || []).length;
          dir = n(-1) <= n(1) ? -1 : 1;
        }
        const key = `${r.region.view}:${dir}`;
        if (!columns.has(key)) columns.set(key, []);
        const mark = { p, num: String(i + 1), nums: [String(i + 1)], notes: [note], label: r.label, note,
                       x: FIG_X[r.region.view] + (dir < 0 ? -BADGE_GAP : 100 + BADGE_GAP) };
        spots.set(spot, mark);
        columns.get(key).push(mark);
      }
    }
  });
  // The locations (a bilateral region's two spots share a label), numbered 1, 2, … in order of
  // their first injury; the map's badges and the chips carry these numbers.
  const locations = new Map(); // label → { id, nums, notes }
  for (const m of [...spots.values()].sort((a, b) => a.num - b.num)) {
    if (!locations.has(m.label)) locations.set(m.label, { id: String(locations.size + 1), nums: [], notes: [] });
    const loc = locations.get(m.label);
    m.nums.forEach((n, j) => { if (!loc.nums.includes(n)) { loc.nums.push(n); loc.notes.push(m.notes[j]); } });
  }
  const injuryLocations = new Map(); // injury number → its location numbers
  for (const loc of locations.values()) {
    for (const n of loc.nums) {
      if (!injuryLocations.has(n)) injuryLocations.set(n, []);
      injuryLocations.get(n).push(loc.id);
    }
  }

  const dots = svgEl("g"), badges = svgEl("g");
  marks.append(dots, badges);
  for (const col of columns.values()) {
    col.sort((a, b) => a.p.y - b.p.y || a.num - b.num);
    let lastY = -Infinity;
    for (const m of col) {
      const by = Math.max(m.p.y, lastY + BADGE_STEP, 8);
      lastY = by;
      const nums = m.nums.join(" ");
      dots.append(svgEl("line", { class: "leader", x1: m.p.x, y1: m.p.y, x2: m.x, y2: by, "data-injury": nums }),
                  svgEl("circle", { class: "spot", cx: m.p.x, cy: m.p.y, r: 2.4, "data-injury": nums }));
      const id = locations.get(m.label).id;
      const label = `Location ${id}: ${m.label}, ${m.nums.length} injur${m.nums.length > 1 ? "ies" : "y"}`;
      const badge = svgEl("g", { class: "badge", tabindex: "0", role: "button", "aria-label": label });
      badge.append(svgEl("circle", { cx: m.x, cy: by, r: 7 }), svgEl("text", { x: m.x, y: by }, id));
      wireInjury(badge, m.note, m.num, locationTip(m.label, m.notes));
      badge.dataset.injury = nums; // highlighted with any injury at this spot
      for (const [on, events] of [[true, ["mouseenter", "focus"]], [false, ["mouseleave", "blur"]]]) {
        for (const e of events) badge.addEventListener(e, () => m.nums.forEach((n) => highlightInjury(n, on)));
      }
      badges.append(badge);
    }
  }

  for (const [label, loc] of locations) {
    const chip = el("button", "chip");
    chip.type = "button";
    chip.append(el("span", "num", loc.id), el("span", null, label));
    wireInjury(chip, loc.notes[0], loc.nums[0], locationTip(label, loc.notes));
    chip.dataset.injury = loc.nums.join(" ");
    for (const [on, events] of [[true, ["mouseenter", "focus"]], [false, ["mouseleave", "blur"]]]) {
      for (const e of events) chip.addEventListener(e, () => loc.nums.forEach((n) => highlightInjury(n, on)));
    }
    const li = el("li");
    li.append(chip);
    chips.append(li);
  }

  model.injuries.forEach((note, i) => {
    const num = String(i + 1);
    const regions = injuryRegions(note.where || note.title);
    const where = regions.map((r, j) => (j && !/^[LR] /.test(r.label) ? r.label.toLowerCase() : r.label)).join(", ");

    const button = el("button", "injury");
    button.type = "button";
    // Numbered by the location(s) it is at, like the map; blank if none was found.
    button.append(el("span", "num", (injuryLocations.get(num) || []).join(",")), el("span", "text", note.title),
                  el("span", "when", [fmtTime(note.t), where].filter(Boolean).join(" · ")));
    wireInjury(button, note, num);
    const li = el("li");
    li.append(button);
    list.append(li);
  });
  chips.hidden = !chips.children.length;
  $("injuries-empty").hidden = model.injuries.length > 0;
}

// ── medical history ─────────────────────────────────────────────────────────

const HISTORY_ORDER = ["Patient history", "Comorbidity", "Family history"];

function renderHistory(model) {
  const box = $("history-groups");
  box.replaceChildren();
  const groups = new Map(HISTORY_ORDER.map((k) => [k, []]));
  for (const item of [...model.history].sort((a, b) => (a.t ?? 1e12) - (b.t ?? 1e12))) {
    if (!groups.has(item.group)) groups.set(item.group, []);
    groups.get(item.group).push(item);
  }
  for (const [kind, items] of groups) {
    if (!items.length) continue;
    const group = el("div", "history-group");
    const head = el("h3", "history-kind", kind);
    head.append(el("span", "count", String(items.length)));
    const list = el("ul", "history-list");
    for (const item of items) {
      const entry = el("div", "history-item");
      entry.tabIndex = 0;
      entry.append(el("span", "text", item.title), el("span", "when", fmtTime(item.t)));
      attachTooltip(entry, item);
      const li = el("li");
      li.append(entry);
      list.append(li);
    }
    group.append(head, list);
    box.append(group);
  }
  $("history-empty").hidden = model.history.length > 0;
}

// ── wound images ────────────────────────────────────────────────────────────

// The images a model predicts into <session>/video/wounds/, oldest first. The viewer shows the
// newest and follows new ones as they arrive, until the arrows step back to an older one.
function renderWounds() {
  const list = asList(state.detail && state.detail.wounds);
  let i = list.findIndex((w) => w.name === state.woundName);
  if (!state.woundPinned || i < 0) i = list.length - 1;
  const w = list[i];
  state.woundName = w ? w.name : null;
  const img = $("wound-img"), link = $("wound-link");
  if (w && img.dataset.url !== w.url) {
    img.src = w.url;
    img.dataset.url = w.url;
  }
  if (w) {
    img.alt = `Wound image ${i + 1} of ${list.length}`;
    link.href = w.url;
  }
  link.hidden = !w;
  $("wound-prev").hidden = $("wound-next").hidden = list.length < 2;
  $("wound-prev").disabled = i <= 0;
  $("wound-next").disabled = i >= list.length - 1;
  $("wound-count").textContent = list.length ? `${i + 1} / ${list.length}` : "";
  $("wound-caption").textContent = w ? `${w.name} · ${new Date(w.time * 1000).toLocaleTimeString()}` : "";
  $("wounds-empty").hidden = !!w;
}

function stepWound(delta) {
  const list = asList(state.detail && state.detail.wounds);
  if (!list.length) return;
  const i = Math.max(0, list.findIndex((w) => w.name === state.woundName));
  const j = Math.min(list.length - 1, Math.max(0, i + delta));
  state.woundName = list[j].name;
  state.woundPinned = j < list.length - 1; // back on the newest: follow new ones again
  renderWounds();
}

// ── latest vital signs ──────────────────────────────────────────────────────

function renderLatestVitals(model) {
  const box = $("vital-tiles");
  box.replaceChildren();
  for (const item of model.latestVitals) {
    const tile = el("div", "vital-tile");
    tile.tabIndex = 0;
    // The number large, its unit small after it: "121 mmHg", "157/101", "97%", "9 (E2 V3 M4)".
    const m = String(item.value).match(/^([-\d./]+%?)\s*(.*)$/);
    const value = el("span", "value", m ? m[1] : item.value);
    if (m && m[2]) value.append(el("span", "unit", m[2]));
    const head = el("span", "head");
    head.append(el("span", "label", item.label), el("span", "when", fmtTime(item.t)));
    tile.append(head, value);
    attachTooltip(tile, item);
    box.append(tile);
  }
  $("vitals-empty").hidden = model.latestVitals.length > 0;
}

// ── timeline ────────────────────────────────────────────────────────────────

const measureCtx = document.createElement("canvas").getContext("2d");
function textWidth(text, font) {
  measureCtx.font = font;
  return measureCtx.measureText(text || "").width;
}
const FONT = getComputedStyle(document.body).fontFamily;

function labelWidth(item) {
  const w = Math.max(textWidth(fmtTime(item.t), `600 13px ${FONT}`), textWidth(item.title, `16px ${FONT}`),
                     textWidth(item.sub, `13px ${FONT}`));
  return Math.min(LABEL_MAX, Math.ceil(w) + 10);
}

// Gives each item the nearest label row where its label doesn't overlap the next one, newest
// first, so the latest events keep their labels. Items that fit no row (row -1) get a dot only.
function layoutRows(items, x) {
  const rowStarts = []; // left edge of the leftmost label placed on each row so far
  for (let i = items.length - 1; i >= 0; i--) {
    const item = items[i];
    const left = x(item.t), right = left + labelWidth(item);
    let row = rowStarts.findIndex((start) => right + LABEL_GAP <= start);
    if (row === -1 && rowStarts.length < MAX_ROWS) row = rowStarts.push(Infinity) - 1;
    if (row !== -1) rowStarts[row] = left;
    item.row = row;
  }
  return Math.max(1, rowStarts.length);
}

function pickTickStep(pps) {
  const steps = [30, 60, 120, 300, 600, 900, 1800, 3600, 7200];
  return steps.find((s) => s * pps >= 110) || steps[steps.length - 1];
}

let eventNodes = new Map(); // view-model item → its label button, for revealEvent()

function renderTimeline(model) {
  const scroll = $("lanes-scroll"), lanesEl = $("lanes"), axis = $("axis");
  const tmax = maxTime(model);
  state.longTimes = tmax >= 3600;
  const viewWidth = scroll.clientWidth;
  const span = state.windowSec || Math.max(tmax, 60);
  // A window spans the whole view; "All" fits everything, with room for the last label.
  state.pps = Math.max(0.01, (state.windowSec ? viewWidth : viewWidth - PAD_LEFT - PAD_RIGHT) / span);
  const x = (t) => PAD_LEFT + t * state.pps;
  const width = Math.max(viewWidth, Math.ceil(x(Math.max(tmax, span)) + PAD_RIGHT));

  const oldLeftTime = (scroll.scrollLeft - PAD_LEFT) / (state.prevPps || state.pps);

  hideTooltip(); // its event is about to be replaced
  lanesEl.replaceChildren();
  lanesEl.style.width = width + "px";
  eventNodes = new Map();

  const step = pickTickStep(state.pps);
  for (let t = 0; x(t) <= width; t += step) {
    const g = el("span", "gridline");
    g.style.left = x(t) + "px";
    lanesEl.append(g);
  }

  for (const lane of LANES) {
    const items = model.lanes[lane.key].filter((it) => isNum(it.t)).sort((a, b) => a.t - b.t);
    const rows = layoutRows(items, x);
    const baseY = rows * ROW_H + STEM_MIN;
    const laneEl = el("div", "lane");
    laneEl.style.setProperty("--lane-color", lane.color);
    const title = el("div", "lane-title");
    title.append(el("span", "bar"), document.createTextNode(lane.title), el("span", "count", String(items.length)));
    const track = el("div", "track");
    track.style.height = baseY + 22 + "px";
    const base = el("div", "baseline");
    base.style.top = baseY + "px";
    track.append(base);

    for (const item of items) {
      const left = x(item.t);
      if (item.continuous) {
        const bar = el("span", "span-bar");
        bar.style.left = left + "px";
        bar.style.top = baseY + "px";
        bar.style.width = Math.max(2, x(item.end ?? item.t) - left) + "px";
        track.append(bar);
        for (const rc of item.rates) {
          const d = el("span", "dot small");
          d.style.left = x(rc.time) + "px";
          d.style.top = baseY + "px";
          track.append(d);
        }
      }
      if (item.row === -1) {
        // No room for a label: a focusable dot carries the details.
        const dot = el("button", "dot unlabeled");
        dot.type = "button";
        dot.setAttribute("aria-label", `${fmtTime(item.t)} ${item.title}`);
        dot.style.left = left + "px";
        dot.style.top = baseY + "px";
        attachTooltip(dot, item);
        track.append(dot);
        eventNodes.set(item, dot);
        continue;
      }
      const top = baseY - STEM_MIN - (item.row + 1) * ROW_H + 4;
      const stem = el("span", "stem");
      stem.style.left = left + "px";
      stem.style.top = top + ROW_H - 8 + "px";
      stem.style.height = baseY - (top + ROW_H - 8) + "px";
      const dot = el("span", "dot");
      dot.style.left = left + "px";
      dot.style.top = baseY + "px";
      const label = el("button", "event");
      label.type = "button";
      label.style.left = left + "px";
      label.style.top = top + "px";
      label.append(el("span", "time", fmtTime(item.t)), el("span", "title", item.title || "–"));
      if (item.sub) label.append(el("span", "sub", item.sub));
      attachTooltip(label, item);
      track.append(stem, dot, label);
      eventNodes.set(item, label);
    }
    laneEl.append(title, track);
    lanesEl.append(laneEl);
  }

  // Axis: same width and scale as the lanes, scrolled with them.
  axis.replaceChildren();
  axis.style.width = width + "px";
  const minor = step / 5 >= 30 ? step / 5 : step / 2;
  for (let t = 0; x(t) <= width; t += minor) {
    const isMajor = Math.abs(t / step - Math.round(t / step)) < 1e-6;
    const tick = el("span", isMajor ? "tick" : "tick minor");
    tick.style.left = x(t) + "px";
    axis.append(tick);
    if (isMajor) {
      const label = el("span", "tick-label", fmtTime(t));
      label.style.left = x(t) + "px";
      axis.append(label);
    }
  }

  $("timeline-empty").hidden = Object.values(model.lanes).some((items) => items.length);

  // Keep the view where it was: at the latest events when following, else at the same time.
  programmaticScroll(() => {
    if (state.follow) scroll.scrollLeft = scroll.scrollWidth;
    else scroll.scrollLeft = Math.max(0, PAD_LEFT + oldLeftTime * state.pps);
  });
  state.prevPps = state.pps;
  syncAxis();
}

let ignoreScroll = false;
function programmaticScroll(fn) {
  ignoreScroll = true;
  fn();
  requestAnimationFrame(() => { ignoreScroll = false; });
}

function syncAxis() {
  const scroll = $("lanes-scroll");
  $("axis").style.transform = `translateX(${-scroll.scrollLeft}px)`;
  const from = (scroll.scrollLeft - PAD_LEFT) / state.pps;
  const to = (scroll.scrollLeft + scroll.clientWidth - PAD_LEFT) / state.pps;
  $("range").textContent = state.detail && state.detail.epcr ? `${fmtTime(Math.max(0, from))} – ${fmtTime(Math.max(0, to))}` : "";
}

function revealEvent(item) {
  const node = eventNodes.get(item);
  if (!node) return;
  setFollow(false);
  const scroll = $("lanes-scroll");
  programmaticScroll(() => {
    scroll.scrollTo({ left: Math.max(0, node.offsetLeft - scroll.clientWidth / 3), behavior: "smooth" });
    node.scrollIntoView({ block: "nearest", behavior: "smooth" });
  });
  node.classList.add("active");
  node.focus({ preventScroll: true });
  setTimeout(() => node.classList.remove("active"), 2500);
}

// ── tooltip ─────────────────────────────────────────────────────────────────

function attachTooltip(node, item) {
  const show = () => showTooltip(node, item);
  node.addEventListener("mouseenter", show);
  node.addEventListener("focus", show);
  node.addEventListener("mouseleave", hideTooltip);
  node.addEventListener("blur", hideTooltip);
}

function showTooltip(node, item) {
  const tip = $("tooltip");
  tip.replaceChildren();
  const head = el("div", "tt-head");
  head.append(el("span", "tt-kind", item.kind), el("span", "tt-time", fmtTime(item.t)));
  tip.append(head, el("div", "tt-title", item.title || "–"));
  const rows = [...(item.details || [])];
  const times = (item.times || []).filter(isNum);
  if (times.length > 1) rows.push(["Mentioned", times.map(fmtTime).join(", ")]);
  if (rows.length) {
    const dl = el("dl");
    for (const [k, v] of rows) {
      const dd = el("dd", null, v);
      dd.style.whiteSpace = "pre-line";
      dl.append(el("dt", null, k), dd);
    }
    tip.append(dl);
  }
  if (item.entities) {
    const ul = el("ul", "tt-entities");
    for (const name of item.entities) ul.append(el("li", null, name));
    tip.append(ul);
  }
  if (item.chart) tip.append(vitalChart(item.chart));
  const quotes = (item.quotes || []).filter((q) => typeof q === "string" && q.trim());
  if (quotes.length) {
    const ul = el("ul", "tt-quotes");
    for (const q of quotes.slice(0, 6)) ul.append(el("li", null, `“${q}”`));
    if (quotes.length > 6) ul.append(el("li", null, `+${quotes.length - 6} more`));
    tip.append(ul);
  }
  tip.hidden = false;
  const r = node.getBoundingClientRect(), tr = tip.getBoundingClientRect();
  let left = Math.min(r.left, window.innerWidth - tr.width - 12);
  let top = r.bottom + 8;
  if (top + tr.height > window.innerHeight - 8) top = Math.max(8, r.top - tr.height - 8);
  tip.style.left = Math.max(8, left) + "px";
  tip.style.top = top + "px";
}
// A vital sign's readings over time, for its tooltip: one line (two for blood pressure,
// systolic over diastolic), the latest reading marked and labelled.
const CHART_W = 300, CHART_H = 130;
const CHART_PAD = { l: 34, r: 56, t: 10, b: 20 };
const CHART_COLORS = ["#d95926", "#3987e5"]; // categorical slots 2 and 1, dark steps
function vitalChart({ points, unit }) {
  const box = el("div", "tt-chart");
  if (!points.length) return box;
  const lines = Math.max(...points.map((p) => p.values.length));
  const all = points.flatMap((p) => p.values);
  let lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo) * 0.1 || Math.max(1, Math.abs(hi) * 0.05);
  lo -= pad; hi += pad;
  const t0 = points[0].t, t1 = points[points.length - 1].t;
  const x = (t) => CHART_PAD.l + (t1 > t0 ? (t - t0) / (t1 - t0) : 0.5) * (CHART_W - CHART_PAD.l - CHART_PAD.r);
  const y = (v) => CHART_PAD.t + (1 - (v - lo) / (hi - lo)) * (CHART_H - CHART_PAD.t - CHART_PAD.b);
  const fmt = (v) => (Math.abs(hi - lo) < 10 ? v.toFixed(1) : String(Math.round(v)));

  const svg = svgEl("svg", { viewBox: `0 0 ${CHART_W} ${CHART_H}`, width: CHART_W, height: CHART_H, role: "img",
                             "aria-label": `${points.length} readings from ${fmtTime(t0)} to ${fmtTime(t1)}` });
  // Recessive grid: the top and bottom of the readings' range, labelled.
  for (const v of [Math.min(...all), Math.max(...all)]) {
    svg.append(svgEl("line", { class: "grid", x1: CHART_PAD.l, x2: CHART_W - CHART_PAD.r, y1: y(v), y2: y(v) }),
               svgEl("text", { class: "axis", x: CHART_PAD.l - 6, y: y(v), "text-anchor": "end", "dominant-baseline": "middle" }, fmt(v)));
  }
  const base = CHART_H - CHART_PAD.b;
  svg.append(svgEl("text", { class: "axis", x: x(t0), y: base + 14, "text-anchor": t1 > t0 ? "start" : "middle" }, fmtTime(t0)));
  if (t1 > t0) svg.append(svgEl("text", { class: "axis", x: x(t1), y: base + 14, "text-anchor": "end" }, fmtTime(t1)));

  const names = lines > 1 ? ["Sys", "Dia"] : [];
  for (let k = 0; k < lines; k++) {
    const pts = points.filter((p) => p.values.length > k);
    const color = CHART_COLORS[k % CHART_COLORS.length];
    if (pts.length > 1) {
      svg.append(svgEl("polyline", { class: "series", stroke: color,
                                     points: pts.map((p) => `${x(p.t).toFixed(1)},${y(p.values[k]).toFixed(1)}`).join(" ") }));
    }
    if (pts.length <= 30) {
      for (const p of pts) svg.append(svgEl("circle", { class: "pt", cx: x(p.t), cy: y(p.values[k]), r: 2, fill: color }));
    }
    const last = pts[pts.length - 1];
    svg.append(svgEl("circle", { class: "last", cx: x(last.t), cy: y(last.values[k]), r: 4, fill: color }));
    // Direct label beside the latest point: the value, and which line it is for blood pressure.
    svg.append(svgEl("text", { class: "value", x: x(last.t) + 8, y: y(last.values[k]), "dominant-baseline": "middle" },
                     [names[k], fmt(last.values[k])].filter(Boolean).join(" ")));
  }
  box.append(svg);
  box.append(el("div", "tt-chart-note", [`${points.length} reading${points.length > 1 ? "s" : ""}`, unit].filter(Boolean).join(" · ")));
  return box;
}

function hideTooltip() { $("tooltip").hidden = true; }

// ── data ────────────────────────────────────────────────────────────────────

function render() {
  const detail = state.detail;
  const agg = detail && detail.epcr;
  // The time format depends on the whole span, and the model formats some times (tooltips), so build twice.
  state.longTimes = maxTime(buildModel(agg || {})) >= 3600;
  const model = buildModel(agg || {});
  renderHeader(agg, detail);
  renderInjuries(model);
  renderHistory(model);
  renderLatestVitals(model);
  renderWounds();
  renderTimeline(model);
  $("timeline-empty").textContent = !detail ? "No session selected." : agg ? "Nothing on the timeline yet." : "Waiting for the first extraction…";
  $("timeline-empty").hidden = !!agg && Object.values(model.lanes).some((items) => items.length);
}

async function loadSessions() {
  try {
    const res = await fetch("/api/sessions", { cache: "no-store" });
    if (!res.ok) return;
    state.sessions = await res.json();
  } catch { return; }
  const select = $("session-select");
  const current = select.value;
  select.replaceChildren();
  for (const s of state.sessions) {
    const date = s.started ? new Date(s.started * 1000) : null;
    const today = date && date.toDateString() === new Date().toDateString();
    const started = date ? date.toLocaleString([], today ? { hour: "2-digit", minute: "2-digit" }
                                                         : { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : "";
    const opt = el("option", null, [s.id, started, s.live ? "live" : ""].filter(Boolean).join(" · "));
    opt.value = s.id;
    select.append(opt);
  }
  let next = state.sessionId;
  if (!state.chosen) {
    // Follow the newest live session; else the newest one with an ePCR.
    const pick = state.sessions.find((s) => s.live && s.has_epcr) || state.sessions.find((s) => s.live)
      || state.sessions.find((s) => s.has_epcr) || state.sessions[0];
    next = pick ? pick.id : null;
  }
  if (next && !state.sessions.some((s) => s.id === next)) next = state.sessions[0] ? state.sessions[0].id : null;
  select.value = next || current || "";
  if (next !== state.sessionId) selectSession(next, state.chosen);
}

function selectSession(id, chosen) {
  state.sessionId = id;
  state.chosen = chosen;
  state.etag = null;
  state.detail = null;
  state.woundName = null;
  state.woundPinned = false;
  setFollow(true);
  const url = new URL(location.href);
  if (chosen && id) url.searchParams.set("session", id); else url.searchParams.delete("session");
  history.replaceState(null, "", url);
  render();
  poll();
}

let pollTimer = null;
async function poll() {
  clearTimeout(pollTimer);
  const id = state.sessionId;
  if (id) {
    try {
      const res = await fetch(`/api/sessions/${encodeURIComponent(id)}`, {
        cache: "no-store", headers: state.etag ? { "If-None-Match": state.etag } : {},
      });
      if (id === state.sessionId && res.status === 200) {
        state.etag = res.headers.get("ETag");
        state.detail = await res.json();
        render();
      } else if (id === state.sessionId && res.status === 404) {
        state.detail = null;
        render();
      }
    } catch { /* the server may be restarting; try again next time */ }
  }
  pollTimer = setTimeout(poll, POLL_MS);
}

// ── controls ────────────────────────────────────────────────────────────────

function setFollow(on) {
  state.follow = on;
  $("follow").setAttribute("aria-pressed", String(on));
}

function init() {
  const fromUrl = new URL(location.href).searchParams.get("session");
  if (fromUrl) { state.sessionId = fromUrl; state.chosen = true; }

  $("session-select").addEventListener("change", (e) => selectSession(e.target.value, true));
  document.querySelectorAll(".zoom button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".zoom button").forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
    state.windowSec = Number(b.dataset.window);
    render();
  }));
  $("follow").addEventListener("click", () => {
    setFollow(!state.follow);
    if (state.follow) programmaticScroll(() => { $("lanes-scroll").scrollLeft = $("lanes-scroll").scrollWidth; });
    syncAxis();
  });
  const scroll = $("lanes-scroll");
  scroll.addEventListener("scroll", () => {
    syncAxis();
    if (ignoreScroll) return; // the page's own scrolling, e.g. following new events
    hideTooltip();
    setFollow(scroll.scrollLeft + scroll.clientWidth >= scroll.scrollWidth - 4);
  });
  $("wound-prev").addEventListener("click", () => stepWound(-1));
  $("wound-next").addEventListener("click", () => stepWound(1));
  $("wound-stage").addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      e.preventDefault();
      stepWound(e.key === "ArrowLeft" ? -1 : 1);
    }
  });
  let resizeTimer = null;
  window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(render, 150); });

  render();
  loadSessions().then(() => { if (state.chosen && state.sessionId) poll(); });
  setInterval(loadSessions, SESSIONS_POLL_MS);
}

init();
