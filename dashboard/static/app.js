// Trauma timeline monitor: polls the dashboard server for a session's aggregate.json
// (llm_aggregation_service's output) and draws it. Times in the aggregate are seconds
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
const INJURY_RE = /\b(injur\w*|trauma\w*|wound\w*|lacerat\w*|bruis\w*|fractur\w*|bleed\w*|blood in|swell\w*|swollen|pain\w*|deform\w*|contusion\w*|abrasion\w*|burn\w*|hematoma\w*|haematoma\w*)\b/i;

const LANES = [
  { key: "history", title: "Patient history", color: "var(--lane-history)" },
  { key: "medications", title: "Medications", color: "var(--lane-medications)" },
  { key: "interventions", title: "Interventions", color: "var(--lane-interventions)" },
];

const state = {
  sessions: [],
  sessionId: null,
  chosen: false,       // the viewer picked the session (or the URL named it): don't auto-switch
  etag: null,
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

// ── aggregate → view model ──────────────────────────────────────────────────

const asList = (v) => (Array.isArray(v) ? v : v && typeof v === "object" && Object.keys(v).length ? [v] : [])
  .filter((it) => it && typeof it === "object");

function buildModel(agg) {
  const notes = asList(agg.general_notes).map((n) => ({
    kind: "Patient history", t: n.time, title: n.entity || n.note || "", sub: "",
    quotes: n.quotes, times: n.times,
  }));

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
    if (isNum(m.end_time)) details.push(["Until", fmtTime(m.end_time)]);
    if (rates.length) details.push(["Rates", rates.map((rc) => `${fmtTime(rc.time)} → ${rc.rate}${unit ? " " + unit : ""}`).join("\n")]);
    return {
      kind: "Medication", t: m.start_time, end: isNum(m.end_time) ? m.end_time : null, continuous, rates,
      title: capitalize(m.entity || m.standard_name || ""), sub, details, quotes: m.quotes,
    };
  });

  const interventions = asList(agg.interventions).map((it) => {
    const details = it.details && typeof it.details === "object" ? it.details : {};
    const summary = clean(details) || "";
    const rows = Object.entries(details).map(([k, v]) => [capitalize(k), clean(v)]).filter(([, v]) => v);
    if (clean(it.category)) rows.unshift(["Category", it.category]);
    return {
      kind: "Intervention", t: it.time, title: capitalize(it.entity || ""), sub: summary || clean(it.category) || "",
      details: rows, quotes: it.quotes, times: it.times,
    };
  });

  const injuries = notes.filter((n) => INJURY_RE.test(n.title) || (n.quotes || []).some((q) => INJURY_RE.test(q)));

  return { lanes: { history: notes, medications: meds, interventions }, injuries };
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
  const quote = moi && (moi.quotes || []).find(Boolean);
  $("mechanism-sub").textContent = moi ? [isNum(moi.time) ? `reported at ${fmtTime(moi.time)}` : "", quote ? `“${quote}”` : ""]
    .filter(Boolean).join(" · ") : "";

  const status = $("status");
  const st = !detail ? "none" : detail.live ? "live" : "ended";
  status.dataset.state = st;
  $("status-text").textContent = st === "live" ? "Live" : st === "ended" ? "Ended" : "No session";
  status.title = detail && detail.updated ? `Last update ${new Date(detail.updated * 1000).toLocaleTimeString()}` : "";
}

// ── injuries ────────────────────────────────────────────────────────────────

function renderInjuries(model) {
  const list = $("injury-list");
  list.replaceChildren();
  model.injuries.forEach((note, i) => {
    const li = el("li");
    const button = el("button", "injury");
    button.type = "button";
    button.append(el("span", "num", String(i + 1)), el("span", "text", note.title), el("span", "when", fmtTime(note.t)));
    attachTooltip(button, note);
    button.addEventListener("click", () => revealEvent(note));
    li.append(button);
    list.append(li);
  });
  $("injuries-empty").hidden = model.injuries.length > 0;
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
  $("range").textContent = state.detail && state.detail.aggregate ? `${fmtTime(Math.max(0, from))} – ${fmtTime(Math.max(0, to))}` : "";
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
function hideTooltip() { $("tooltip").hidden = true; }

// ── data ────────────────────────────────────────────────────────────────────

function render() {
  const detail = state.detail;
  const agg = detail && detail.aggregate;
  // The time format depends on the whole span, and the model formats some times (tooltips), so build twice.
  state.longTimes = maxTime(buildModel(agg || {})) >= 3600;
  const model = buildModel(agg || {});
  renderHeader(agg, detail);
  renderInjuries(model);
  renderTimeline(model);
  $("timeline-empty").textContent = !detail ? "No session selected." : agg ? "Nothing on the timeline yet." : "Waiting for the first aggregate…";
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
    // Follow the newest live session; else the newest one with an aggregate.
    const pick = state.sessions.find((s) => s.live && s.has_aggregate) || state.sessions.find((s) => s.live)
      || state.sessions.find((s) => s.has_aggregate) || state.sessions[0];
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
  let resizeTimer = null;
  window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(render, 150); });

  render();
  loadSessions().then(() => { if (state.chosen && state.sessionId) poll(); });
  setInterval(loadSessions, SESSIONS_POLL_MS);
}

init();
