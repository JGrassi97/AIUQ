/* AIUQ Monitor — single page front-end (vanilla JS + Plotly). The map module lives in map.js. */
"use strict";

function appFromUrl() {
  const q = new URLSearchParams(location.search);
  if (q.get("app") === "launcher") return "launcher";
  return q.get("app") === "monitor" || q.get("exp") ? "monitor" : "home";
}
const S = {
  app: appFromUrl(),
  exp: new URLSearchParams(location.search).get("exp"),
  list: [], meta: null, detail: null,
  refreshSec: 45, timer: null, lastFetch: null, fetching: false,
  gantt: { section: "all", failedOnly: false, mode: "lane" },
  preview: null, job: null,
  open: new Set(), skeletonFor: null, coast: null,
  cache: new Map(),   // expid -> last detail seen: reopening an experiment paints at once, then refreshes
  filter: "",         // text typed in the experiments panel
};

// ------------------------------------------------------------------ vocabulary
const STATUS_VAR = {
  COMPLETED: "--st-completed", RUNNING: "--st-running", QUEUING: "--st-queued", SUBMITTED: "--st-queued",
  READY: "--st-ready", PREPARED: "--st-ready", DELAYED: "--st-held", WAITING: "--st-waiting",
  HELD: "--st-held", SUSPENDED: "--st-held", FAILED: "--st-failed", UNKNOWN: "--st-unknown",
  SKIPPED: "--st-skipped", INTERRUPTED: "--st-held",
};
const STATUS_LABEL = {
  COMPLETED: "Done", RUNNING: "Running", QUEUING: "Queued", SUBMITTED: "Submitted", READY: "Ready",
  PREPARED: "Prepared", DELAYED: "Delayed", WAITING: "Waiting", HELD: "On hold", SUSPENDED: "Suspended",
  FAILED: "Failed", UNKNOWN: "Unknown", SKIPPED: "Skipped", INTERRUPTED: "Interrupted",
};
const STATUS_ORDER = ["COMPLETED", "RUNNING", "QUEUING", "SUBMITTED", "READY", "PREPARED", "DELAYED",
  "WAITING", "HELD", "SUSPENDED", "FAILED", "UNKNOWN", "SKIPPED"];
const SEC_NAME = {
  SYNCHRONIZE: "Sync project", WARMUP_REGRID: "Warm-up regrid", WARMUP_REGRID_ACC: "Warm-up regrid (GPU)",
  PREPARE_ICS: "Prepare ICs", SAVE_ICS: "Save ICs", SIM: "Simulation", POSTPROCESS: "Post-process", CLEAN: "Clean-up",
};
const SEC_SHORT = { PREPARE_ICS: "ICs", SAVE_ICS: "Save", SIM: "Sim", POSTPROCESS: "Post", CLEAN: "Clean" };
const MODEL_NAMES = { aifs: "AIFS", "ms-aurora": "Aurora", neuralgcm: "NeuralGCM", ace: "ACE" };
const ICS_INFO = { eerie: ["EERIE", "MARS · ECMWF Web API"], era5: ["ERA5", "Google Cloud · ARCO-ERA5"], climatology: ["Climatology", "local fallback"] };

// ------------------------------------------------------------------ helpers
const $ = (sel, root = document) => root.querySelector(sel);
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const stColor = (s) => css(STATUS_VAR[s] || "--st-unknown");
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pad = (n) => String(n).padStart(2, "0");
const secName = (s) => SEC_NAME[s] || (s || "").replace(/_/g, " ").toLowerCase().replace(/^\w/, (c) => c.toUpperCase());
const label = (s) => STATUS_LABEL[s] || s;
const info = (text) => `<span class="info" tabindex="0" role="note" aria-label="${esc(text)}" data-tip="${esc(text)}">i</span>`;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

function fmtTime(t, withDate = true) {
  if (!t) return "—";
  const d = new Date(t * 1000);
  const hm = `${pad(d.getHours())}:${pad(d.getMinutes())}`;
  if (!withDate) return `${hm}:${pad(d.getSeconds())}`;
  const sameYear = d.getFullYear() === new Date().getFullYear();
  return `${d.getDate()} ${MONTHS[d.getMonth()]}${sameYear ? "" : " " + d.getFullYear()}, ${hm}`;
}
function fmtDur(sec) {
  if (sec == null || !isFinite(sec)) return "—";
  sec = Math.max(0, Math.round(sec));
  const d = Math.floor(sec / 86400), h = Math.floor((sec % 86400) / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  if (d) return `${d}d ${h}h`;
  if (h) return `${h}h ${pad(m)}m`;
  if (m) return `${m}m ${pad(s)}s`;
  return `${s}s`;
}
const fmtAgo = (t) => (t ? `${fmtDur(Date.now() / 1000 - t)} ago` : "—");
function fmtBytes(b) {
  if (b == null) return "—";
  const u = ["B", "KB", "MB", "GB", "TB"]; let i = 0;
  while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
  return `${b.toFixed(b < 10 && i ? 1 : 0)} ${u[i]}`;
}
function fmtDate(d) {  // 'YYYYMMDD' or 'YYYY-MM-DD' -> '1 Jan 1986'
  if (!d) return "—";
  const m = String(d).match(/^(\d{4})-?(\d{2})-?(\d{2})/);
  return m ? `${+m[3]} ${MONTHS[+m[2] - 1]} ${m[1]}` : d;
}
const isoDate = (d) => (d && d.length === 8 ? `${d.slice(0, 4)}-${d.slice(4, 6)}-${d.slice(6)}` : d);
const shortJob = (name, expid) => (name || "").replace(new RegExp(`^${expid}_`), "");
const pill = (s, n) => `<span class="pill" title="${esc(s)}"><i style="background:${stColor(s)}"></i>${esc(label(s))}${n != null ? ` <b>${n}</b>` : ""}</span>`;
const plural = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;

// Replace the content only when it changed: a refresh then keeps scroll positions, text selections and hover states.
function setHTML(el, html) {
  if (!el) return false;
  if (el._html === html) return false;
  el._html = html; el.innerHTML = html;
  return true;
}
let toastTimer = null;
function toast(msg, isError = false) {
  const t = $("#toast");
  t.textContent = msg; t.className = `toast in${isError ? " err" : ""}`;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.className = "toast"; }, isError ? 6000 : 2600);
}
function fadeMain() {
  const m = $("#main");
  m.classList.remove("fade"); void m.offsetWidth; m.classList.add("fade");
}
// state of an experiment as a whole: running / failed / done / stopped (not finished) / new (no workflow yet)
function expState(e) {
  const s = e.summary;
  if (e.active) return "running";
  if (s.failed) return "failed";
  if (s.total && s.completed === s.total) return "done";
  return s.total ? "stopped" : "new";
}
const STATE_COLOR = { running: "--st-running", failed: "--st-failed", done: "--st-completed", stopped: "--st-unknown", new: "--st-unknown" };
const RING_ICON = {
  done: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
  failed: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 6v8M12 18v.5"/></svg>',
  stopped: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 8v8M15 8v8"/></svg>',
  new: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 8v8M15 8v8"/></svg>',
};
function ring(e, big = false) {
  const st = expState(e), pct = Math.floor(e.summary.pct || 0);
  if (st === "done") big = false;   // a finished experiment needs a tick, not a number
  const inner = big ? `<b>${pct}%</b>` : st === "running" ? "<i></i>" : RING_ICON[st];
  return `<span class="ring${big ? " big" : ""}" style="--p:${st === "done" ? 100 : pct};--c:var(${STATE_COLOR[st]})" aria-hidden="true">${inner}</span>`;
}
const stateBadge = (e) => {
  const st = expState(e);
  return st === "running" ? '<span class="badge active">running</span>' : st === "done" ? '<span class="badge done">completed</span>'
    : st === "failed" ? `<span class="badge bad">${e.summary.failed} failed</span>` : `<span class="badge stopped">${st === "new" ? "not started" : "stopped"}</span>`;
};

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${(await r.text()).slice(0, 300)}`);
  return r.json();
}

function baseLayout(extra = {}) {
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "system-ui, -apple-system, Segoe UI, sans-serif", size: 12, color: css("--text-secondary") },
    margin: { l: 60, r: 16, t: 8, b: 40 },
    hoverlabel: { bgcolor: css("--surface-1"), bordercolor: css("--axis"), font: { color: css("--text-primary") } },
    xaxis: { gridcolor: css("--grid"), linecolor: css("--axis"), zeroline: false },
    yaxis: { gridcolor: css("--grid"), linecolor: css("--axis"), zeroline: false },
  }, extra);
}
const PCFG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"] };
function onClick(el, fn) {  // rebind after every draw (Plotly.purge drops listeners)
  el.removeAllListeners?.("plotly_click");
  el.on("plotly_click", fn);
}

// ------------------------------------------------------------------ theme
const THEMES = ["auto", "light", "dark"];
const THEME_ICON = {
  auto: '<svg class="ico" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 3v18" /><path d="M12 3a9 9 0 0 1 0 18z" fill="currentColor"/></svg>',
  light: '<svg class="ico" viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>',
  dark: '<svg class="ico" viewBox="0 0 24 24"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>',
};
function currentTheme() { try { return localStorage.getItem("aiuq-theme") || "auto"; } catch { return "auto"; } }
function applyTheme(t) {
  if (t === "auto") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = t;
  try { localStorage.setItem("aiuq-theme", t); } catch { /* ignore */ }
  $("#themeBtn").innerHTML = THEME_ICON[t];
  $("#themeBtn").title = `Theme: ${t} (click to change)`;
}
function redrawAll() { S.skeletonFor = null; $("#main")._html = null; if (S.preview) S.preview.drawn = false; render(); }

// ------------------------------------------------------------------ data loop
async function load(force = false) {
  if (S.fetching) return;
  S.fetching = true;
  const want = S.exp;
  const slow = setTimeout(() => $("#progress").classList.add("on"), force ? 0 : 250);  // only show the bar when it is not instant
  if (force) $("#refreshBtn").classList.add("spin");
  try {
    if (force) await api("/api/refresh", { method: "POST" }).catch(() => {});
    const ov = await api("/api/experiments");
    S.list = ov.experiments; S.meta = ov.meta;
    if (S.exp && !S.list.find((e) => e.expid === S.exp)) S.detail = null;
    if (want) {
      let d = null, err = null;
      try { d = (await api(`/api/exp/${encodeURIComponent(want)}`)).exp; S.cache.set(want, d); } catch (e) { err = e.message; }
      if (S.exp === want) { S.detail = d; S.detailError = err; }   // the user may have moved on while this was loading
    }
    S.lastFetch = Date.now() / 1000;
    render();
  } catch (e) {
    const f = $("#freshness");
    f.className = "live down"; f.innerHTML = `<span class="dot"></span>Dashboard server unreachable`; f.title = e.message;
  } finally {
    S.fetching = false;
    clearTimeout(slow);
    $("#progress").classList.remove("on");
    $("#refreshBtn").classList.remove("spin");
    if (S.exp !== want) load();   // another experiment was opened meanwhile: fetch it now
  }
}

// Warm the cache for an experiment the pointer rests on (the server answers from memory in a few ms).
let prefetchTimer = null;
function prefetch(expid) {
  clearTimeout(prefetchTimer);
  if (!expid || expid === S.exp || S.cache.has(expid)) return;
  prefetchTimer = setTimeout(async () => {
    try { S.cache.set(expid, (await api(`/api/exp/${encodeURIComponent(expid)}`)).exp); } catch { /* ignore */ }
  }, 90);
}

function schedule() {
  clearInterval(S.timer);
  if (S.refreshSec > 0) S.timer = setInterval(() => load(), S.refreshSec * 1000);
}

function switchApp(app) {
  S.app = app;
  const url = new URL(location.href);
  for (const k of ["app", "exp", "tab", "view"]) url.searchParams.delete(k);
  if (app !== "home") url.searchParams.set("app", app);
  if (app === "launcher" && L.view !== "new") url.searchParams.set("view", L.view);
  if (app === "monitor" && S.exp) { url.searchParams.delete("app"); url.searchParams.set("exp", S.exp); }
  history.pushState({}, "", url);
  S.skeletonFor = null;
  closeJob();
  render();
  fadeMain();
  window.scrollTo({ top: 0 });
  if (app !== "home") load();
}

function selectExp(expid, push = true) {
  S.app = "monitor";
  S.exp = expid || null;
  S.detail = (expid && S.cache.get(expid)) || null; S.detailError = null; S.skeletonFor = null;
  closeJob();
  if (S.preview?.playing) S.preview.playing = false;
  S.preview = newPreview(); S.job = null; S.tab = null; S.tabFor = null; S.data = null;
  const url = new URL(location.href);
  if (expid) url.searchParams.set("exp", expid); else url.searchParams.delete("exp");
  url.searchParams.delete("tab");
  url.searchParams.delete("app");
  if (!expid) url.searchParams.set("app", "monitor");
  if (push) history.pushState({ exp: expid }, "", url);
  document.title = expid ? `${expid} · AIUQ Monitor` : "AIUQ Monitor";
  render();
  fadeMain();
  window.scrollTo({ top: 0 });
  load();
}

// ------------------------------------------------------------------ freshness
function tick() {
  const m = S.meta, f = $("#freshness");
  if (!m) return;
  const now = Date.now() / 1000;
  const age = m.last_cycle ? now - m.last_cycle : null;
  let cls = "live", text;
  if (m.remote_ok === false) { cls += " down"; text = m.remote_host ? `${m.remote_host.split(".")[0]} unreachable · showing data from ${fmtAgo(m.remote_last_ok)}` : "HPC host not configured"; }
  else if (age == null) { text = "Starting up…"; }
  else if (age > 2.5 * m.refresh) { cls += " stale"; text = `Data delayed · ${fmtDur(age)} old`; }
  else text = `Live · updated ${fmtDur(age)} ago`;
  f.className = cls;
  f.innerHTML = `<span class="dot"></span>${esc(text)}`;
  f.title = `Collected at ${fmtTime(m.last_cycle, false)} · ${m.remote_host || "the HPC"} answered in ${m.remote_elapsed ?? "?"} s\n` +
    `Page updated at ${fmtTime(S.lastFetch, false)}${S.refreshSec ? `, next in ${Math.round(Math.max(0, S.refreshSec - (now - S.lastFetch)))} s` : ", auto-refresh off"}`;
  const bar = $("#refresh-bar");
  bar.style.width = S.refreshSec && S.lastFetch ? `${Math.min(100, (100 * (now - S.lastFetch)) / S.refreshSec)}%` : "0";
}

// ------------------------------------------------------------------ rendering
function listEntry() { return S.list.find((e) => e.expid === S.exp); }
function ensureTab() {
  if (S.tab && S.tabFor === S.exp) return;
  const e = S.detail || listEntry();
  S.tab = defaultTab(e || { active: true });
  if (e) S.tabFor = S.exp;
}

function render() {
  document.body.dataset.app = S.app;
  document.querySelectorAll(".appswitch button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.app === S.app)));
  if (S.app === "monitor" && S.exp) ensureTab();
  renderSidebar();
  renderCrumb();
  renderTabs();
  if (S.app === "home") { S.skeletonFor = null; document.title = "AIUQ"; renderHome(); tick(); return; }
  if (S.app === "launcher") { S.skeletonFor = null; document.title = "AIUQ Launcher"; renderLauncher(); tick(); return; }
  if (!S.exp) { S.skeletonFor = null; renderOverview(); }
  else if (!S.detail) { S.skeletonFor = null; renderExpLoading(); }
  else renderExp(S.detail);
  tick();
}

function renderCrumb() {
  let html = "";
  if (S.app === "launcher") html = `<span class="cr-dim">Create, launch and stop experiments</span>`;
  else if (S.app === "monitor" && !S.exp) html = `<span class="cr-dim">All experiments</span>`;
  else if (S.app === "monitor") {
    const e = S.detail || listEntry();
    html = `<b>${esc(S.exp)}</b>${e ? `<span class="cr-sep">·</span><span>${what(e.conf)}</span>${e.description ? `<span class="cr-sep">·</span><span class="cr-dim">${esc(e.description)}</span>` : ""}` : ""}`;
  }
  setHTML($("#crumb"), html);
}

// ---- sticky view tabs: the three views of an experiment, or the four of the Launcher
const TAB_ICON = {
  new: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 8v8M8 12h8"/></svg>',
  runs: '<svg viewBox="0 0 24 24"><path d="M7 5v14l11-7z"/></svg>',
  delete: '<svg viewBox="0 0 24 24"><path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13M10 11v6M14 11v6"/></svg>',
  activity: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
};
const LN_TABS = [["new", "New experiment"], ["runs", "Launch & stop"], ["delete", "Delete"], ["activity", "Activity"]];
function renderTabs() {
  const nav = $("#viewtabs");
  let html = "";
  if (S.app === "monitor" && S.exp) {
    const s = (S.detail || listEntry())?.summary || {};
    const count = s.failed ? `<span class="count bad" title="${plural(s.failed, "failed job")}">${s.failed}</span>` : s.running ? `<span class="count run" title="${plural(s.running, "job")} running">${s.running}</span>` : "";
    html = TABS.map(([k, t, sub, ico], i) => `<button role="tab" data-tab="${k}" aria-selected="${k === S.tab}" title="${t}: ${sub}  [${i + 1}]">${ico}${t}${k === "live" ? count : ""}</button>`).join("")
      + `<span class="tab-tools"><button type="button" class="small ghost" data-act="collapse">Collapse all</button><button type="button" class="small ghost" data-act="expand">Expand all</button></span>`;
  } else if (S.app === "launcher") {
    const run = S.list.filter((e) => e.active).length;
    html = LN_TABS.map(([k, t], i) => `<button role="tab" class="${k === "delete" ? "danger" : ""}" data-view="${k}" aria-selected="${k === L.view}" title="${t}  [${i + 1}]">${TAB_ICON[k]}${t}${k === "runs" && run ? `<span class="count run" title="${plural(run, "experiment")} running">${run}</span>` : ""}</button>`).join("");
  }
  nav.hidden = !html;
  setHTML(nav, html);
}

// ---- experiments panel
const what = (c) => `${esc(MODEL_NAMES[c.model] || c.model || "?")} · ${esc((ICS_INFO[c.ics] || [c.ics || "?"])[0])}`;
function miniBar(counts, total) {
  if (!total) return "";
  return `<div class="mini-bar" aria-hidden="true">${STATUS_ORDER.filter((s) => counts[s]).map(
    (s) => `<span style="width:${(100 * counts[s]) / total}%;background:${stColor(s)}" title="${label(s)}: ${counts[s]}"></span>`).join("")}</div>`;
}
function statusLine(e) {
  const s = e.summary;
  const parts = [];
  if (s.running) parts.push(`${s.running} running`);
  if (s.failed) parts.push(`<b style="color:var(--st-failed)">${s.failed} failed</b>`);
  if (!parts.length) parts.push(s.completed === s.total ? "all done" : `${s.total - s.completed} to go`);
  return parts.join(" · ");
}
function expWhen(c) {
  const nd = (c.dates || []).length;
  return nd > 1 ? `${nd} dates from ${fmtDate(c.first_date)}` : fmtDate(c.first_date);
}
const GROUPS = [["running", "Running"], ["failed", "Needs attention"], ["stopped", "Not finished"], ["done", "Completed"]];
function matchesFilter(e, text = S.filter) {
  const q = text.trim().toLowerCase();
  if (!q) return true;
  const c = e.conf;
  const hay = [e.expid, c.model, MODEL_NAMES[c.model], c.ics, (ICS_INFO[c.ics] || [])[0], e.description, c.checkpoint, c.branch,
    c.first_date, fmtDate(c.first_date), expState(e)].filter(Boolean).join(" ").toLowerCase();
  return q.split(/\s+/).every((w) => hay.includes(w));
}
function groupedExps(text = S.filter) {  // [[key, title, experiments]] in display order; newest ids first inside a group
  const by = { running: [], failed: [], stopped: [], done: [] };
  for (const e of S.list) if (matchesFilter(e, text)) { const st = expState(e); by[st === "new" ? "stopped" : st].push(e); }
  for (const k of Object.keys(by)) by[k].sort((a, b) => (a.expid < b.expid ? 1 : -1));
  return GROUPS.map(([k, t]) => [k, t, by[k]]).filter((g) => g[2].length);
}
const visibleExps = () => groupedExps().flatMap((g) => g[2]);

function renderSidebar() {
  const side = $("#sidebar");
  if (!side.dataset.built) {
    side.dataset.built = "1";
    side.innerHTML = `<div class="side-top"><div class="side-search">
        <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>
        <input type="search" id="side-q" placeholder="Filter experiments" aria-label="Filter experiments" autocomplete="off" spellcheck="false"><kbd>/</kbd>
      </div></div><div id="side-list"></div>`;
    const q = $("#side-q");
    q.addEventListener("input", () => { S.filter = q.value; renderSideList(); if (S.app === "monitor" && !S.exp) renderOverview(); });
    q.addEventListener("keydown", (e) => {
      if (e.key === "Escape") { q.value = ""; S.filter = ""; renderSideList(); if (S.app === "monitor" && !S.exp) renderOverview(); q.blur(); }
      if (e.key === "Enter") { const first = visibleExps()[0]; if (first) { selectExp(first.expid); q.blur(); } }
    });
    side.addEventListener("pointerover", (e) => prefetch(e.target.closest("a[data-exp]")?.dataset.exp));
  }
  renderSideList();
}
function renderSideList() {
  const row = (e) => {
    const st = expState(e), c = e.conf, nm = (c.members || []).length;
    const meta = st === "done" ? `${expWhen(c)} · ${plural(nm, "member")}` : `${statusLine(e)} · ${expWhen(c)}`;
    return `<a class="exp-row ${e.expid === S.exp ? "sel" : ""}" href="?exp=${e.expid}" data-exp="${e.expid}" title="${esc(e.description || "")}" ${e.expid === S.exp ? 'aria-current="page"' : ""}>
      ${ring(e)}
      <span class="exp-main"><span class="exp-name"><b>${e.expid}</b><span class="exp-model">${what(c)}</span></span><span class="exp-meta">${meta}</span></span>
      ${st === "done" ? "" : `<span class="pct">${Math.floor(e.summary.pct)}%</span>`}</a>`;
  };
  const groups = groupedExps();
  setHTML($("#side-list"), groups.map(([, t, es]) => `<h3 class="side-h">${t}<span>${es.length}</span></h3>${es.map(row).join("")}`).join("")
    || `<div class="side-empty">${S.list.length ? "No experiment matches this filter." : S.meta ? "No experiments yet." : "Loading…"}</div>`);
  if (S.exp && S.sideShown !== S.exp) {   // bring the selected experiment into view once (the panel scrolls sideways on phones)
    const sel = document.querySelector("#side-list .exp-row.sel");
    if (sel) { S.sideShown = S.exp; if (sel.offsetParent) sel.scrollIntoView({ block: "nearest", inline: "nearest" }); }
  }
}

function expCard(e) {
  const c = e.conf, s = e.summary, st = expState(e);
  const nm = (c.members || []).length;
  return `<a class="ov-card" style="--c:var(${STATE_COLOR[st]})" href="?exp=${e.expid}" data-exp="${e.expid}">
    <div class="row1"><span class="id">${e.expid}</span>${stateBadge(e)}${ring(e, true)}</div>
    <div class="what">${what(c)}</div>
    <div class="desc">${esc(e.description || "")}</div>
    ${st === "done" ? "" : `${miniBar(e.counts, s.total)}<div class="facts"><span><b>${s.completed}</b> of ${s.total} jobs</span><span>${statusLine(e)}</span></div>`}
    <div class="facts"><span><b>${expWhen(c)}</b></span><span><b>${nm}</b> member${nm === 1 ? "" : "s"}</span>
      <span>last activity <b>${fmtAgo(s.last_activity)}</b></span></div>
  </a>`;
}

function renderOverview() {
  const main = $("#main");
  main.removeAttribute("data-tab"); main.removeAttribute("data-view");
  if (!S.meta) { setHTML(main, `<div class="page-head"><h2>Experiments</h2></div><div class="kpis">${'<div class="kpi skeleton"></div>'.repeat(4)}</div><div class="skeleton block"></div>`); return; }
  const running = S.list.filter((e) => e.active);
  const failed = S.list.reduce((a, e) => a + (e.summary.failed || 0), 0);
  const jobs = S.list.reduce((a, e) => a + (e.summary.total || 0), 0);
  const last = Math.max(0, ...S.list.map((e) => e.summary.last_activity || 0)) || null;
  const groups = groupedExps();
  setHTML(main, `${errorBanner()}
    <div class="page-head"><h2>Experiments</h2><span class="lede">${S.filter.trim() ? `Filtered by “${esc(S.filter.trim())}”` : "Pick one to follow its jobs, its timing and its forecasts."}</span></div>
    <div class="kpis">
      <div class="kpi"><div class="lbl">Running now</div><div class="val">${running.length}</div><div class="sub">${running.length ? running.map((e) => e.expid).join(", ") : "nothing is running"}</div></div>
      <div class="kpi"><div class="lbl">Experiments</div><div class="val">${S.list.length}</div><div class="sub">${jobs.toLocaleString()} jobs in total</div></div>
      <div class="kpi ${failed ? "alert" : ""}"><div class="lbl">Failed jobs</div><div class="val">${failed}</div><div class="sub">${failed ? "see “Needs attention”" : "no problems"}</div></div>
      <div class="kpi"><div class="lbl">Last activity</div><div class="val">${last ? fmtAgo(last).replace(" ago", "") : "—"}</div><div class="sub">${last ? `ago · ${fmtTime(last)}` : ""}</div></div>
    </div>
    ${groups.map(([, t, es]) => `<h3 class="sec-h">${t}<span>${es.length}</span></h3><div class="ov-grid">${es.map(expCard).join("")}</div>`).join("")
      || `<div class="empty big">${S.list.length ? "No experiment matches this filter." : "No experiments yet. Create the first one in the Launcher."}</div>`}`);
}

// ---- homepage
const HOME_ICON = {
  monitor: '<svg viewBox="0 0 24 24"><path d="M3 12h4l3-8 4 16 3-8h4"/></svg>',
  launcher: '<svg viewBox="0 0 24 24"><path d="M5 19c1.5-4 4-7 7.5-9.5L16 6c2-1.5 3.5-2 5-2-0 1.5-.5 3-2 5l-3.5 3.5C13 16 10 18.5 6 20"/><path d="M9 15l-3-3 2.5-2.5H12M9 15l3 3 2.5-2.5V12"/><circle cx="16" cy="8" r="1.4"/></svg>',
  arrow: '<svg viewBox="0 0 24 24"><path d="M5 12h14M13 6l6 6-6 6"/></svg>',
};
const LOGO = '<svg class="eyebrow-logo" viewBox="0 0 64 64" aria-hidden="true"><rect width="64" height="64" rx="16" fill="#18344e"/><path d="M12 43 27 17h10l15 26H41l-9-17-9 17Z" fill="#5ce0cd"/><circle cx="32" cy="45" r="5" fill="#fff"/></svg>';
function renderHome() {
  const main = $("#main");
  main.removeAttribute("data-tab"); main.removeAttribute("data-view");
  const ready = !!S.meta;
  const running = S.list.filter((e) => e.active);
  const recent = [...S.list].sort((a, b) => (b.summary.last_activity || 0) - (a.summary.last_activity || 0));
  const last = recent[0]?.summary.last_activity || null;
  const failed = S.list.reduce((a, e) => a + (e.summary.failed || 0), 0);
  const models = [...new Set(S.list.map((e) => MODEL_NAMES[e.conf.model] || e.conf.model).filter(Boolean))];
  const liveRow = (e) => `<a class="hl-row" href="?exp=${e.expid}" data-exp="${e.expid}">
      <span class="ring" style="--p:${Math.floor(e.summary.pct)}"><b>${Math.floor(e.summary.pct)}%</b></span>
      <span class="hl-main"><b>${e.expid} · ${what(e.conf)}</b><span>${e.active ? statusLine(e).replace(/<[^>]+>/g, "") : `finished ${fmtAgo(e.summary.last_activity)}`} · ${expWhen(e.conf)}</span></span></a>`;
  const stat = (v, l, cls = "") => `<div class="hstat ${cls}"><b>${ready ? v : "–"}</b><span>${l}</span></div>`;
  setHTML(main, `${errorBanner()}
    <section class="hero2">
      <div class="hero2-text">
        <span class="eyebrow">${LOGO}AIUQ dashboard</span>
        <h1>Run AI weather models and follow every job, live.</h1>
        <p>Run AI weather and climate models with Autosubmit, from the initial conditions to the ensemble forecasts. Follow your experiments on the HPC while they run, or set up and launch new ones from here.</p>
        <div class="hero2-stats">
          ${stat(running.length, "running now")}${stat(S.list.length, "experiments")}${stat(models.length, models.length === 1 ? "model" : "models")}
          ${failed ? stat(failed, "failed jobs", "bad") : ""}${stat(last ? fmtAgo(last).replace(" ago", "") : "—", "since the last activity")}
        </div>
        <div class="hero2-cta">
          <a class="btn-big primary" href="?app=monitor" data-app-link="monitor">Open the Monitor ${HOME_ICON.arrow}</a>
          <a class="btn-big onnavy" href="?app=launcher" data-app-link="launcher">New experiment…</a>
        </div>
      </div>
      <div class="hero2-live">
        ${!ready ? `<h3>Connecting…</h3><div class="hl-idle">Reading the experiments.</div>`
          : running.length ? `<h3><span class="dot"></span>Running now</h3>${running.slice(0, 4).map(liveRow).join("")}${running.length > 4 ? `<div class="hl-idle">and ${running.length - 4} more</div>` : ""}`
          : `<h3>Nothing is running</h3>${recent.slice(0, 3).map(liveRow).join("") || `<div class="hl-idle">No experiments yet.</div>`}`}
      </div>
    </section>
    <div class="home-apps">
      <a class="home-app" href="?app=monitor" data-app-link="monitor">
        <span class="ha-icon">${HOME_ICON.monitor}</span><h2>Autosubmit Monitor</h2><span class="ha-go">${HOME_ICON.arrow}</span>
        <p>Follow every experiment: live progress, timeline, HPC performance and maps of the outputs.</p>
      </a>
      <a class="home-app" href="?app=launcher" data-app-link="launcher">
        <span class="ha-icon">${HOME_ICON.launcher}</span><h2>AIUQ Launcher</h2><span class="ha-go">${HOME_ICON.arrow}</span>
        <p>Create a new experiment in five guided steps, then launch, resume or stop its runs.</p>
      </a>
    </div>
    ${recent.length ? `<h3 class="sec-h">Latest experiments<a href="?app=monitor" data-app-link="monitor">All ${S.list.length} experiments →</a></h3>
      <div class="ov-grid">${recent.slice(0, 6).map(expCard).join("")}</div>` : ""}`);
}

function errorBanner(v) {
  const msgs = [...(S.meta?.errors || [])];
  if (v?.remote?.error) msgs.push(`remote: ${v.remote.error}`);
  if (v?.pkl_error) msgs.push(`job list: ${v.pkl_error} (showing last good state)`);
  if (!msgs.length) return "";
  return `<div class="err-banner"><b>Something needs attention</b><br>${msgs.map((m) => esc(m)).join("<br>")}</div>`;
}

// ---- experiment view: three tabs (live / review / data)
const TABS = [
  ["live", "Live", "what is happening now", '<svg viewBox="0 0 24 24"><path d="M3 12h4l3-8 4 16 3-8h4"/></svg>'],
  ["review", "Run review", "how the run went", '<svg viewBox="0 0 24 24"><path d="M4 19V5M4 19h16M8 16v-5M12 16V8M16 16v-3"/></svg>'],
  ["data", "Data", "explore the forecasts", '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 3.5 3 14.5 0 18M12 3c-3 3.5-3 14.5 0 18"/></svg>'],
];
function defaultTab(v) {
  const t = new URLSearchParams(location.search).get("tab");
  return TABS.some((x) => x[0] === t) ? t : v.active ? "live" : "review";
}
function setTab(t) {
  if (!TABS.some((x) => x[0] === t)) return;
  S.tab = t; S.tabFor = S.exp;
  const url = new URL(location.href);
  url.searchParams.set("tab", t);
  history.replaceState(history.state, "", url);
  renderTabs();
  if (!S.detail) return;
  $("#main").dataset.tab = t;
  if (S.preview) { S.preview.drawn = false; S.preview.psig = null; }
  renderExp(S.detail);
  if (window.scrollY > 320) window.scrollTo({ top: 0 });   // a new view starts from its top, not from where the previous one was left
  requestAnimationFrame(() => document.querySelectorAll(`#main section.card[data-tab="${t}"] .js-plotly-plot`).forEach((p) => Plotly.Plots.resize(p)));
}

// identity of the experiment: id, state, what it is. Works with the full detail or with the short entry of the list.
function expHead(e, v) {
  const c = e?.conf || {};
  const nm = (c.members || []).length, nd = (c.dates || []).length;
  const badge = !e ? "" : v?.active ? `<span class="badge active" title="autosubmit run, pid ${v.pid}">running since ${fmtTime(v.active_since)}</span>` : stateBadge(e);
  return `<div class="exp-head">
    <div class="eh-top"><h2>${esc(S.exp)}</h2>${badge}
      ${e ? `<span class="chip-fact"><b>${what(c)}</b></span><span class="chip-fact"><b>${nm}</b> member${nm === 1 ? "" : "s"}</span>
        <span class="chip-fact">${nd > 1 ? `<b>${nd}</b> start dates from` : "start"} <b>${fmtDate(c.first_date)}</b></span>` : ""}</div>
    <p class="eh-desc">${esc(e?.description || "")}</p>
    <div class="eh-actions"><button type="button" class="small" data-act="to-launcher" title="Launch, resume, stop or reuse this experiment in the Launcher">Manage in Launcher →</button></div>
  </div>`;
}

// shown while the first answer for an experiment is on its way: its identity (already known from the list) and placeholders
function renderExpLoading() {
  const main = $("#main"), e = listEntry();
  main.removeAttribute("data-tab"); main.removeAttribute("data-view");
  if (S.meta && !e) {
    setHTML(main, `<div class="empty big">There is no experiment called <b>${esc(S.exp)}</b>.<br><br><a href="?app=monitor" data-app-link="monitor">See all experiments</a></div>`);
    return;
  }
  setHTML(main, `${S.detailError ? `<div class="err-banner"><b>Could not load ${esc(S.exp)}</b><br>${esc(S.detailError)}</div>` : ""}${expHead(e)}
    <div class="kpis">${'<div class="kpi skeleton"></div>'.repeat(6)}</div>
    <div class="skeleton block" style="margin-bottom:16px"></div><div class="skeleton block"></div>`);
}

function renderExp(v) {
  if (S.skeletonFor !== v.expid) {
    ensureTab();
    const main = $("#main");
    main.removeAttribute("data-view");
    main.dataset.tab = S.tab;
    main._html = null;
    main.innerHTML = `
      <div id="x-errors"></div>
      <div id="x-head"></div>
      <div class="kpis" id="x-kpis"></div>

      <section class="card" data-tab="live" id="s-now"><div class="card-head"><h3>Right now</h3><span class="sub" id="now-sub"></span>
        ${info("Jobs currently running, with how long they have been running compared with the typical duration of their step; what is queued; and what finished or failed most recently.")}</div>
        <div id="x-now"></div></section>
      <section class="card" data-tab="live" id="s-grid"><div class="card-head"><h3>Progress by start date</h3>
        <span class="sub">each square is one job · click one for details</span>
        ${info("Current Autosubmit status of every job, read from the experiment's job list (pkl). Rows are start dates, columns are job steps and ensemble members.")}</div>
        <div class="legend" id="x-legend"></div>
        <div class="plot" id="p-grid"></div><div id="x-once"></div></section>
      <section class="card" data-tab="live" id="s-ics"><div class="card-head"><h3>Initial conditions</h3>
        <span class="sub">preparation and MARS downloads</span>
        ${info(`Parsed from the PREPARE_ICS output on ${v.remote.host}: requests sent to MARS, their state (queued, active, complete), size on tape and errors.`)}</div><div id="x-ics"></div></section>
      <section class="card" data-tab="live" id="s-sim"><div class="card-head"><h3>Simulations & GPUs</h3>
        ${info(`SLURM queue (squeue) and GPU usage (nvidia-smi) on ${v.remote.host}. sacct is disabled there, so finished SLURM jobs are not listed.`)}</div><div id="x-sim"></div></section>
      <section class="card" data-tab="live" id="s-fail"><div class="card-head"><h3>Problems</h3><span class="sub" id="f-note"></span>
        ${info("Failed jobs with the most useful part of their error log (the Python traceback when there is one; shell tracing is removed). Identical errors are grouped.")}</div><div id="x-fail"></div></section>

      <section class="card" data-tab="review" id="s-setup"><div class="card-head"><h3>Setup</h3>
        ${info("Read from conf/main.yml and conf/minimal.yml of the experiment.")}</div><div id="x-setup"></div></section>
      <section class="card" data-tab="review" id="s-gantt"><div class="card-head"><h3>Timeline</h3><span class="sub" id="g-note"></span>
        ${info("Start and end of every job attempt, from the STAT files written by Autosubmit on the HPC. Click a bar to open the job.")}
        <span class="spacer"></span>
        <div class="controls">
          <select id="g-sec" aria-label="Job step"></select>
          <select id="g-mode" aria-label="Rows"><option value="lane" selected>Group by step & member</option><option value="job">One row per job (last 300)</option></select>
          <label class="switch"><input type="checkbox" id="g-failed"><span class="track"></span>Failed only</label>
        </div></div>
        <div class="plot" id="p-gantt"></div></section>
      <section class="card" data-tab="review" id="s-dur"><div class="card-head"><h3>How long jobs take</h3>
        <span class="sub">and how much is left</span>
        ${info("Durations come from completed attempts. Time left is an estimate: jobs left × typical (median) duration ÷ how many jobs of that step usually run at once, summed over steps. Failed jobs are not counted. Steps with no history borrow durations from similar experiments.")}</div>
        <div class="two-col"><div class="plot" id="p-dur"></div><div id="x-eta"></div></div></section>
      <section class="card" data-tab="review" id="s-hpc"><div class="card-head"><h3>HPC performance</h3><span class="sub">where the time goes and what slows the run down</span>
        ${info("Built from the job list (ready and submit times), the STAT files (start and end) and a GPU/queue sample taken at every refresh. Waiting for Autosubmit = ready → submitted; Queued = submitted → started; Running = started → finished.")}</div>
        <div id="x-hpc"></div></section>

      <section class="card" data-tab="data" id="s-out"><div class="card-head"><h3>Maps & time series</h3><span class="sub" id="o-note"></span>
        ${info(`NetCDF outputs on ${v.remote.host}. The map reads one time step and one level on demand; show a single member, the ensemble mean, the spread or a member's departure from the mean. Click a point (or pick an area) for its time series across all members.`)}</div>
        <div id="o-body"></div></section>
      <section class="card" data-tab="data" id="s-spread"><div class="card-head"><h3>How fast the members diverge</h3><span class="sub" id="sp-sub"></span>
        ${info("Ensemble spread = standard deviation across members, area-weighted (cos latitude) over the chosen region, for every day of the forecast. A fast-growing spread means low predictability. Keep curves to compare regions, variables or start dates.")}</div>
        <div id="x-spread"></div></section>
      <section class="card" data-tab="data" id="s-hov"><div class="card-head"><h3>Hovmöller diagram</h3><span class="sub" id="hv-sub"></span>
        ${info("Longitude × time of the field averaged over a latitude band (cos-latitude weighted). Tilted stripes are travelling weather systems: eastward-moving ones tilt to the right going down.")}</div>
        <div id="x-hov"></div></section>
      <section class="card" data-tab="data" id="s-hind"><div class="card-head"><h3>Across start dates</h3><span class="sub" id="hc-sub"></span>
        ${info("For every start date and member, the area mean at a few lead days. The left chart shows how the forecasts follow the seasons; the right one shows the average change with lead time (drift) relative to the first selected day, with the spread across start dates.")}</div>
        <div id="x-hind"></div></section>`;
    S.skeletonFor = v.expid;
    if (S.preview) { S.preview.drawn = false; S.preview.sig = null; S.preview.built = false; }
    S.data = null;
    setupCollapsibles();
    $("#g-sec").addEventListener("change", (e) => { S.gantt.section = e.target.value; drawGantt(S.detail); });
    $("#g-failed").addEventListener("change", (e) => { S.gantt.failedOnly = e.target.checked; drawGantt(S.detail); });
    $("#g-mode").addEventListener("change", (e) => { S.gantt.mode = e.target.value; drawGantt(S.detail); });
  }
  setHTML($("#x-errors"), errorBanner(v));
  setHTML($("#x-head"), expHead(v, v));
  renderKpis(v);
  renderJob(v);
  if (S.tab === "live") { renderNow(v); drawGrid(v); renderIcs(v); renderSim(v); renderFailed(v); }
  if (S.tab === "review") { renderSetup(v); drawGantt(v); drawDurations(v); renderHpc(v); }
  if (S.tab === "data") { renderOutputs(v); renderData(v); }
}

function renderNow(v) {
  const now = Date.now() / 1000;
  const running = v.running.filter((r) => r.status === "RUNNING");
  const queued = v.running.filter((r) => r.status !== "RUNNING");
  const done = v.attempts.filter((a) => a.end).sort((a, b) => b.end - a.end).slice(0, 6);
  $("#now-sub").textContent = running.length ? `${plural(running.length, "job")} running` : v.active ? "nothing running at this moment" : "the run is stopped";
  const bar = (r) => {
    const med = v.dur_stats[r.section]?.median, el = r.start ? now - r.start : null;
    const pct = med && el != null ? Math.min(100, (100 * el) / med) : null;
    const over = med && el > med * 1.5;
    return `<div class="now-job">
      <div class="nj-head"><a href="#" data-job="${esc(r.name)}"><b>${esc(secName(r.section))}</b> <span class="mono note">${esc(shortJob(r.name, v.expid))}</span></a>
        <span class="spacer"></span><span class="nj-time">${el != null ? fmtDur(el) : "starting"}${med ? ` <span class="note">/ typically ${fmtDur(med)}</span>` : ""}</span></div>
      <div class="meter ${over ? "over" : ""}"><span style="width:${pct ?? 100}%;${pct == null ? "opacity:.35" : ""}"></span></div>
      ${over ? `<div class="note" style="color:var(--st-held)">Running ${(el / med).toFixed(1)}× longer than usual${r.section === "PREPARE_ICS" ? " — MARS may be queueing the request on tape" : ""}.</div>` : ""}
    </div>`;
  };
  const REASON = { Priority: "other jobs have higher priority", Resources: "waiting for free resources (GPUs/CPUs)", Dependency: "waiting for another job",
    QOSMaxJobsPerUserLimit: "you reached the per-user job limit", ReqNodeNotAvail: "the requested node is not available", BeginTime: "scheduled for later" };
  const pend = (v.slurm.jobs || []).filter((j) => j.state === "PENDING");
  const since = (j) => { const t = Date.parse(j.submit); return isFinite(t) ? now - t / 1000 : null; };
  const unknown = Object.entries(v.section_counts).reduce((a, [, c]) => a + (c.UNKNOWN || 0), 0);
  if (!running.length && pend.length) $("#now-sub").textContent = `${plural(pend.length, "job")} waiting in the SLURM queue`;
  // jobs waiting for the same reason and asking for the same resources read as one line
  const pgroups = new Map();
  for (const j of pend) {
    const reason = (j.reason || "").replace(/[()]/g, "");
    const key = `${REASON[reason] || reason || "pending"} · asks for ${j.gres && j.gres !== "N/A" ? j.gres.replace("gres:", "").replace("gres/", "") : "no GPU"} · limit ${j.limit}`;
    if (!pgroups.has(key)) pgroups.set(key, []);
    pgroups.get(key).push(j);
  }
  const pendNames = new Set(pend.map((j) => j.name));
  const pendHtml = [...pgroups.entries()].map(([why, js]) => {
    const waits = js.map(since).filter((x) => x != null);
    return `<div class="queue-box"><div class="q-top"><b>${plural(js.length, "job")} waiting in the SLURM queue</b>
        ${waits.length ? `<span class="nj-time note">for ${fmtDur(Math.max(...waits))}${js.length > 1 && Math.max(...waits) - Math.min(...waits) > 60 ? ` (the latest for ${fmtDur(Math.min(...waits))})` : ""}</span>` : ""}</div>
      <div class="note">${esc(why)}</div>
      <div class="queue-jobs">${js.map((j) => `<a href="#" data-job="${esc(j.name)}">${esc(shortJob(j.name, v.expid))}</a>`).join("")}</div></div>`;
  }).join("");
  const unkHtml = unknown ? `<div class="finding f-warn" style="margin-top:10px">${ICONS.warn}<div>Autosubmit reports <b>${plural(unknown, "job")}</b> as <i>Unknown</i>: it could not read their state on the HPC.${pend.length ? " SLURM still has them in the queue, so they are probably fine." : " Check the Autosubmit log if this lasts."}</div></div>` : "";
  setHTML($("#x-now"), `
    <div class="now-grid">
      <div>${running.length ? running.map(bar).join("") : `<div class="empty">${v.active ? "No job is running right now." : "The run is stopped: no job is running."}</div>`}
        ${pendHtml}${unkHtml}
        ${(() => { const q = queued.filter((x) => !pendNames.has(x.name)); return q.length ? `<div class="note" style="margin-top:8px">${plural(q.length, "job")} submitted and waiting: ${q.slice(0, 8).map((x) => `<a href="#" data-job="${esc(x.name)}">${esc(shortJob(x.name, v.expid))}</a>`).join(", ")}${q.length > 8 ? "…" : ""}</div>` : ""; })()}
      </div>
      <div><div class="chart-title">Latest finished</div>
        ${done.length ? `<ul class="recent">${done.map((a) => `<li><i style="background:${stColor(a.result === "FAILED" ? "FAILED" : "COMPLETED")}"></i>
          <a class="grow" href="#" data-job="${esc(a.job)}">${esc(secName(a.section))} <span class="note mono">${esc(shortJob(a.job, v.expid))}</span></a>
          <span class="note">${fmtAgo(a.end)}</span></li>`).join("")}</ul>` : `<div class="empty">Nothing has finished yet.</div>`}
      </div>
    </div>`);
}

function renderKpis(v) {
  const s = v.summary, eta = s.eta;
  let etaVal = "Done", etaSub = "everything finished";
  if (eta) {
    const left = s.total - s.completed;
    const overdue = Object.values(eta.sections).some((x) => x.overdue);
    etaVal = eta.partial || (overdue && eta.total < 60) ? "Unknown" : `~${fmtDur(eta.total)}`;
    if (eta.partial && eta.total) etaVal = `> ${fmtDur(eta.total)}`;
    etaSub = `${plural(left, "job")} left${eta.partial ? " · some steps have no history yet" : ""}`;
    if (!v.active) { etaSub = `if restarted: ${etaVal} · ${etaSub}`; etaVal = "Stopped"; }
    if (!eta.partial && eta.total < 1 && s.failed && left <= s.failed) { etaVal = v.active ? "Blocked" : "Stopped"; etaSub = `${plural(s.failed, "failed job")} need${s.failed === 1 ? "s" : ""} a rerun`; }
  }
  setHTML($("#x-kpis"), `
    <div class="kpi"><div class="lbl">Progress</div><div class="val">${s.pct.toFixed(s.pct < 100 ? 1 : 0)}%</div>
      <div class="sub">${s.completed} of ${s.total} jobs done</div><div class="bar"><span style="width:${s.pct}%"></span></div></div>
    <div class="kpi"><div class="lbl">Running now</div><div class="val">${s.running}</div><div class="sub">${(() => { const q = Math.max(s.queued, (v.slurm.jobs || []).filter((j) => j.state === "PENDING").length); return q ? `${q} in the queue · ` : ""; })()}${s.waiting} waiting</div></div>
    ${s.failed ? `<a class="kpi alert" href="#s-fail" data-gotab="live"><div class="lbl">Failed</div><div class="val">${s.failed}</div><div class="sub">see what went wrong →</div></a>`
      : `<div class="kpi"><div class="lbl">Failed</div><div class="val">0</div><div class="sub">no problems</div></div>`}
    <div class="kpi"><div class="lbl">${v.active ? "Running for" : "Ran for"}</div><div class="val">${fmtDur(s.elapsed)}</div><div class="sub">started ${fmtTime(s.run_start)}${s.runs > 1 ? ` · ${s.runs} runs` : ""}</div></div>
    <div class="kpi"><div class="lbl">Time left ${info("Estimate. See “How long jobs take” for how it is computed.")}</div>
      <div class="val">${etaVal}</div><div class="sub">${etaSub}</div></div>
    <div class="kpi"><div class="lbl">Last activity</div><div class="val">${fmtAgo(s.last_activity).replace(" ago", "")}</div><div class="sub">${s.last_activity ? `ago · ${fmtTime(s.last_activity)}` : ""}</div></div>`);
}

// ---- setup card
const ICON = {
  ics: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3c3 3.5 3 14.5 0 18M12 3c-3 3.5-3 14.5 0 18"/></svg>',
  model: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/></svg>',
  ens: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 18c4-1 6-5 9-9s5-5 9-6M3 18c5 0 8-3 10-6s4-4 8-4M3 18c5 1 9 0 12-2s4-3 6-2"/></svg>',
  out: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 2h9l5 5v15H6z"/><path d="M14 2v6h6M9 13h8M9 17h8"/></svg>',
};

function renderSetup(v) {
  const c = v.conf, nd = (c.dates || []).length, nm = (c.members || []).length, nc = c.num_chunks || 1;
  const unit = c.chunk_unit || "";
  const leadN = (c.chunk_size || 0) * nc;
  const lead = leadN ? plural(leadN, unit) : "—";
  const [icsName, icsSrc] = ICS_INFO[c.ics] || [c.ics || "?", ""];
  const levs = Array.isArray(c.out_levs) ? `${c.out_levs.join(", ")} hPa` : (c.out_levs || "—");
  const res = c.out_res && c.out_res !== "original" ? `${c.out_res}°` : (c.out_res || "—");
  const cmp = v.outputs?.compression;       // what the files say, else the experiment configuration
  const cmpText = cmp ? `${esc(cmp.preset)}${cmp.preset === "bits" && cmp.keepbits ? ` (${cmp.keepbits} bits)` : ""}` : null;
  const stage = (icon, k, big, sub, rows) => `<div class="stage"><div class="stage-k">${ICON[icon]}<span>${k}</span></div>
      <div class="stage-big">${big}</div><div class="stage-sub">${sub}</div>
      <dl>${rows.filter((r) => r[1] != null && r[1] !== "").map(([a, b]) => `<dt>${a}</dt><dd>${b}</dd>`).join("")}</dl></div>`;
  const arrow = '<div class="flow-arrow" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M4 12h15M13 6l6 6-6 6"/></svg></div>';
  const branchUrl = c.origin && c.branch ? `${String(c.origin).replace(/\.git$/, "")}/tree/${encodeURIComponent(c.branch)}` : null;
  const chain = v.sections.map((sec) => {
    const sc = v.section_counts[sec] || {};
    const tot = Object.values(sc).reduce((a, b) => a + b, 0);
    const done = sc.COMPLETED || 0;
    return `<div class="wf-step" title="${esc(STATUS_ORDER.filter((k) => sc[k]).map((k) => `${label(k)}: ${sc[k]}`).join(" · "))}">
      <div class="wf-name">${esc(secName(sec))}</div>
      <div class="wf-count">${done} / ${tot} done${sc.FAILED ? ` · <b style="color:var(--st-failed)">${sc.FAILED} failed</b>` : ""}${sc.RUNNING ? ` · ${sc.RUNNING} running` : ""}</div>
      ${miniBar(sc, tot)}</div>`;
  }).join('<span class="wf-sep" aria-hidden="true">›</span>');

  $("#x-setup").innerHTML = `
    <div class="setup-flow">
      ${stage("ics", "Initial conditions", esc(icsName), esc(icsSrc), [["start dates", `<b>${nd}</b>`], ["first", fmtDate(c.first_date)], ["last", fmtDate(c.last_date)]])}
      ${arrow}
      ${stage("model", "Model", esc(MODEL_NAMES[c.model] || c.model || "?"), `<span class="mono">${esc(c.checkpoint || "")}</span>`,
        [["run type", esc(c.run_type)], ["forcing", c.run_type === "amip" ? esc(c.forcing) : null], ["calendar", esc(c.calendar)]])}
      ${arrow}
      ${stage("ens", "Ensemble", plural(nm, "member"), `× ${plural(nd, "start date")} = <b>${nm * nd}</b> forecasts`, [["forecast length", lead], ["split in", plural(nc, "chunk")]])}
      ${arrow}
      ${stage("out", "Outputs", (c.out_vars || []).map((x) => `<span class="var-tag" title="${esc(varName(x))}">${esc(x)}</span>`).join("") || "—", esc(levs),
        [["resolution", esc(res)], ["frequency", esc(c.out_freq)], ["compression", cmpText],
         ["size on disk", v.outputs?.n ? fmtBytes(v.outputs.size) : null]])}
    </div>
    <div class="setup-meta">
      <span>runs on <b>${esc(c.platform)}</b></span>
      <span>code ${branchUrl ? `<a class="mono" href="${esc(branchUrl)}" target="_blank" rel="noopener">${esc(c.branch)}</a>` : `<b class="mono">${esc(c.branch)}</b>`}</span>
      <span>Autosubmit <b>${esc(c.as_version)}</b></span>
    </div>
    <div class="wf-title">Workflow</div>
    <div class="wf-chain">${chain}</div>
    ${nd > 1 ? `<div class="wf-title">Start dates · click one to open its first job</div><div class="plot" id="p-dates"></div>` : ""}`;
  if (nd > 1) drawDateStrip(v);
}

function dateState(row) {
  const st = row.filter(Boolean).map((c) => c[0]);
  if (!st.length) return "WAITING";
  if (st.includes("FAILED")) return "FAILED";
  if (st.some((x) => ["RUNNING", "QUEUING", "SUBMITTED"].includes(x))) return "RUNNING";
  if (st.every((x) => x === "COMPLETED" || x === "SKIPPED")) return "COMPLETED";
  return "WAITING";
}

function drawDateStrip(v) {
  const g = v.grid, el = $("#p-dates");
  if (!el || !g.dates.length) return;
  const states = g.cells.map(dateState);
  const traces = ["COMPLETED", "RUNNING", "WAITING", "FAILED"].map((st) => {
    const idx = states.map((s, i) => (s === st ? i : -1)).filter((i) => i >= 0);
    if (!idx.length) return null;
    return { type: "scatter", mode: "markers", name: label(st), x: idx.map((i) => isoDate(g.dates[i])), y: idx.map(() => 0),
      marker: { color: stColor(st), size: 12, symbol: "square", line: { color: css("--surface-1"), width: 1.5 } },
      customdata: idx.map((i) => [fmtDate(g.dates[i]), (g.cells[i].find(Boolean) || [])[1] || ""]),
      hovertemplate: `%{customdata[0]} · ${label(st)}<extra></extra>` };
  }).filter(Boolean);
  Plotly.react(el, traces, baseLayout({ height: 64, margin: { l: 10, r: 10, t: 4, b: 24 }, showlegend: false,
    xaxis: { type: "date", gridcolor: css("--grid"), linecolor: css("--axis") },
    yaxis: { visible: false, range: [-1, 1], fixedrange: true } }), { ...PCFG, displayModeBar: false });
  onClick(el, (d) => { const job = d.points[0]?.customdata?.[1]; if (job) openJob(job); });
}

// ---- status grid
function colLabel(section, member, chunk, multiChunk) {
  const base = SEC_SHORT[section] || secName(section);
  return member ? `${base} ${member}${multiChunk && chunk != null ? `.${chunk}` : ""}` : secName(section);
}

function drawGrid(v) {
  const g = v.grid;
  $("#x-legend").innerHTML = STATUS_ORDER.filter((s) => v.counts[s]).map((s) => pill(s, v.counts[s])).join("");
  const el = $("#p-grid");
  if (!g.dates.length || !g.columns.length) { Plotly.purge(el); el.innerHTML = `<div class="empty">No dated jobs.</div>`; }
  else {
    const n = STATUS_ORDER.length, multi = (v.conf.num_chunks || 1) > 1;
    const scale = [];
    STATUS_ORDER.forEach((s, i) => { scale.push([i / n, stColor(s)], [(i + 1) / n, stColor(s)]); });
    const z = g.cells.map((row) => row.map((c) => (c ? STATUS_ORDER.indexOf(c[0]) : null)));
    const text = g.cells.map((row, i) => row.map((c, j) => (c
      ? `<b>${esc(secName(g.columns[j].section))}${g.columns[j].member ? ` · member ${esc(g.columns[j].member)}` : ""}</b><br>start date ${fmtDate(g.dates[i])}<br>${label(c[0])}${c[2] ? ` · ${c[2]} retries` : ""}${c[3] != null ? ` · took ${fmtDur(c[3])}` : ""}`
      : `${fmtDate(g.dates[i])} · ${esc(g.columns[j].label)}<br>no job`)));
    const rowH = g.dates.length > 60 ? 11 : g.dates.length > 20 ? 16 : 30;
    const labels = g.columns.map((c) => colLabel(c.section, c.member, c.chunk, multi));
    const shapes = [];
    g.columns.forEach((c, j) => { if (j && c.section !== g.columns[j - 1].section) shapes.push({ type: "line", xref: "x", yref: "paper", x0: j - 0.5, x1: j - 0.5, y0: 0, y1: 1, line: { color: css("--surface-1"), width: 6 } }); });
    Plotly.react(el, [{
      type: "heatmap", z, text, customdata: g.cells.map((row) => row.map((c) => (c ? c[1] : ""))),
      hovertemplate: "%{text}<extra></extra>", x: labels, y: g.dates.map(fmtDate),
      zmin: -0.5, zmax: n - 0.5, colorscale: scale, showscale: false, xgap: 2, ygap: 2,
    }], baseLayout({
      height: Math.max(150, g.dates.length * rowH + 90),
      margin: { l: 90, r: 8, t: 8, b: 70 },
      xaxis: { tickangle: -40, side: "bottom", showgrid: false, linecolor: "rgba(0,0,0,0)", type: "category", fixedrange: false },
      yaxis: { autorange: "reversed", showgrid: false, type: "category", linecolor: "rgba(0,0,0,0)" },
      shapes,
    }), { ...PCFG, displayModeBar: false });
    onClick(el, (d) => { const name = d.points[0]?.customdata; if (name) openJob(name); });
  }
  $("#x-once").innerHTML = v.once_jobs.length
    ? `<div class="note" style="margin-top:8px">One-off jobs: ${v.once_jobs.map((j) => `<a href="#" data-job="${esc(j.name)}">${pill(j.status)} ${esc(secName(j.section))}</a>`).join(" &nbsp; ")}</div>` : "";
}

// ---- gantt
const RESULT_ORDER = ["COMPLETED", "RUNNING", "FAILED", "INTERRUPTED", "UNKNOWN"];
function drawGantt(v) {
  if (!v) return;
  const secSel = $("#g-sec");
  const secs = ["all", ...v.sections.filter((s) => v.attempts.some((a) => a.section === s))];
  if (secSel.options.length !== secs.length) secSel.innerHTML = secs.map((s) => `<option value="${s}">${s === "all" ? "All steps" : esc(secName(s))}</option>`).join("");
  secSel.value = secs.includes(S.gantt.section) ? S.gantt.section : "all";
  const now = Date.now() / 1000;
  let at = v.attempts.filter((a) => a.start && (S.gantt.section === "all" || a.section === S.gantt.section));
  if (S.gantt.failedOnly) at = at.filter((a) => a.result === "FAILED");
  const total = at.length;
  const laneMode = S.gantt.mode !== "job";
  if (!laneMode) at = at.slice(-300);
  const multi = (v.conf.num_chunks || 1) > 1;
  const rowOf = laneMode ? (a) => colLabel(a.section, a.member, a.chunk, multi) : (a) => `${shortJob(a.job, v.expid)}${a.n ? ` #${a.n}` : ""}`;
  $("#g-note").textContent = `${at.length === total ? total : `${at.length} of ${total}`} attempts`;
  const el = $("#p-gantt");
  if (!at.length) { Plotly.purge(el); el.innerHTML = `<div class="empty">No job has started yet.</div>`; return; }
  const rows = [...new Set(at.map(rowOf))];
  if (laneMode) {
    const order = [...v.once_jobs.map((j) => secName(j.section)), ...v.grid.columns.map((c) => colLabel(c.section, c.member, c.chunk, multi))];
    rows.sort((x, y) => ((order.indexOf(x) + 1 || 1e6) - (order.indexOf(y) + 1 || 1e6)));
  }
  const t0 = Math.min(...at.map((a) => a.start)), t1 = Math.max(...at.map((a) => a.end || now));
  const minW = Math.max(1000, (t1 - t0) * 1000 * 0.0015);
  const traces = RESULT_ORDER.map((res) => {
    const sub = at.filter((a) => (RESULT_ORDER.includes(a.result) ? a.result : "UNKNOWN") === res);
    if (!sub.length) return null;
    return {
      type: "bar", orientation: "h", name: label(res), y: sub.map(rowOf),
      base: sub.map((a) => new Date(a.start * 1000).toISOString()),
      x: sub.map((a) => Math.max(minW, ((a.end || now) - a.start) * 1000)),
      marker: { color: stColor(res), line: { color: res === "FAILED" ? css("--st-failed") : css("--surface-1"), width: res === "FAILED" ? 2 : 0.5 } },
      customdata: sub.map((a) => [a.job, fmtTime(a.start), a.end ? fmtTime(a.end) : "still running", fmtDur((a.end || now) - a.start), label(a.result)]),
      hovertemplate: "<b>%{customdata[0]}</b><br>%{customdata[4]} · %{customdata[3]}<br>%{customdata[1]} → %{customdata[2]}<extra></extra>",
    };
  }).filter(Boolean);
  traces.sort((x, y) => (x.name === label("FAILED")) - (y.name === label("FAILED")));
  Plotly.react(el, traces, baseLayout({
    height: Math.min(2600, Math.max(220, rows.length * (laneMode ? 22 : 13) + 90)),
    barmode: "overlay", bargap: laneMode ? 0.2 : 0.25,
    margin: { l: laneMode ? 130 : 190, r: 16, t: 8, b: 40 },
    legend: { orientation: "h", y: 1.02, yanchor: "bottom", x: 0 },
    xaxis: { type: "date", gridcolor: css("--grid"), linecolor: css("--axis") },
    yaxis: { type: "category", categoryorder: "array", categoryarray: rows, autorange: "reversed", tickfont: { size: 11 }, gridcolor: "rgba(0,0,0,0)" },
  }), PCFG);
  onClick(el, (d) => { const name = d.points[0]?.customdata?.[0]; if (name) openJob(name); });
}

// ---- durations + ETA
function drawDurations(v) {
  const el = $("#p-dur");
  const secs = v.sections.filter((s) => (v.durations[s] || []).length);
  if (!secs.length) { Plotly.purge(el); el.innerHTML = `<div class="empty">No job has finished yet.</div>`; }
  else {
    const unitMin = Math.max(...secs.map((s) => v.dur_stats[s]?.median || 0)) >= 60;
    const traces = secs.map((s) => {
      const d = v.durations[s].map((x) => (unitMin ? x / 60 : x));
      return { type: "box", orientation: "h", name: secName(s), x: d, y: d.map(() => secName(s)),
        boxpoints: d.length <= 400 ? "all" : "suspectedoutliers", jitter: 0.5, pointpos: 0,
        marker: { color: css("--st-running"), size: 4, opacity: 0.5 }, line: { color: css("--st-running"), width: 1.5 },
        fillcolor: css("--accent-soft"), hovertemplate: `${secName(s)}: %{x:.1f} ${unitMin ? "min" : "s"}<extra></extra>` };
    });
    Plotly.react(el, traces, baseLayout({
      height: Math.max(200, secs.length * 50 + 70), showlegend: false,
      margin: { l: 150, r: 16, t: 8, b: 44 },
      xaxis: { title: { text: unitMin ? "minutes" : "seconds" }, gridcolor: css("--grid"), rangemode: "tozero" },
      yaxis: { type: "category", categoryorder: "array", categoryarray: secs.map(secName).reverse(), gridcolor: "rgba(0,0,0,0)" },
    }), PCFG);
  }
  const eta = v.summary.eta;
  const rows = v.sections.filter((s) => v.dur_stats[s] || eta?.sections[s]).map((s) => {
    const d = v.dur_stats[s] || {}, e = eta?.sections[s];
    const etaTxt = e ? (e.seconds != null && !(e.overdue && e.seconds < 60) ? "~" + fmtDur(e.seconds) : "unknown") : "—";
    const notes = [e?.borrowed ? `based on ${e.borrowed.join(", ")}` : "", e?.overdue ? "running longer than usual" : ""].filter(Boolean);
    return `<tr><td>${esc(secName(s))}</td><td class="num">${d.n ?? 0}</td><td class="num">${fmtDur(d.median)}</td>
      <td class="num">${fmtDur(d.max)}</td><td class="num">${e ? e.remaining : 0}</td>
      <td class="num">${etaTxt}${notes.length ? `<div class="note">${esc(notes.join(" · "))}</div>` : ""}</td></tr>`;
  }).join("");
  $("#x-eta").innerHTML = `<div class="tbl-wrap"><table>
      <thead><tr><th>Step</th><th class="num">Done</th><th class="num">Typical</th><th class="num">Longest</th>
      <th class="num">Left</th><th class="num">Time left (est.)</th></tr></thead><tbody>${rows}</tbody></table></div>`;
}

// ---- ICS / MARS
const MARS_STATE = { queued: "waiting in MARS queue", active: "being retrieved", complete: "complete", submitted: "submitted", aborted: "aborted" };
function renderIcs(v) {
  const el = $("#x-ics");
  if (!v.ics.length) { el.innerHTML = `<div class="empty">This experiment has no initial-condition jobs.</div>`; return; }
  const anyMars = v.ics.some((r) => r.mars && r.mars.requests);
  const anyErr = v.ics.some((r) => r.mars?.errors);
  const now = Date.now() / 1000;
  const rows = v.ics.map((r) => {
    const m = r.mars || {};
    const end = r.end || (r.status === "RUNNING" ? now : null);
    return `<tr class="${r.status === "RUNNING" ? "hl" : ""}"><td>${fmtDate(r.date)}</td><td>${pill(r.status)}</td>
      ${anyMars ? `<td class="num">${m.requests ?? "—"}</td><td class="num">${m.fields ?? "—"}</td><td class="num">${m.tape_mb ? fmtBytes(m.tape_mb * 1048576) : "—"}</td>
      <td>${m.last_state ? esc(MARS_STATE[m.last_state] || m.last_state) : "—"}</td>
      ${anyErr ? `<td class="num">${m.errors ? `<b style="color:var(--st-failed)">${m.errors}</b>` : "0"}</td>` : ""}
      <td>${m.last_ts ? esc(m.last_ts.slice(5, 16)) : "—"}</td>` : ""}
      <td class="num">${r.start && end ? fmtDur(end - r.start) : "—"}</td></tr>`;
  }).join("");
  const focus = v.ics.filter((r) => ["RUNNING", "FAILED", "SUBMITTED", "QUEUING"].includes(r.status) && r.mars);
  const logs = focus.slice(0, 6).map((r) => detailsBlock(`ics:${r.job}`,
    `MARS log for ${fmtDate(r.date)} · ${pill(r.status)}`,
    `<pre class="log">${esc((r.mars.events || []).join("\n") || "no MARS messages yet")}</pre>`, r.status === "RUNNING")).join("");
  el.innerHTML = `${anyMars ? "" : `<p class="note">${v.conf.ics === "era5" ? "ERA5 initial conditions come from Google Cloud, so there are no MARS requests to show." : "No MARS messages found in the logs."}</p>`}
    <div class="tbl-wrap scroll-y"><table><thead><tr><th>Start date</th><th>Status</th>
    ${anyMars ? `<th class="num">Requests</th><th class="num">Fields</th><th class="num">Size</th><th>MARS</th>${anyErr ? '<th class="num">Errors</th>' : ""}<th>Last message</th>` : ""}
    <th class="num">Duration</th></tr></thead><tbody>${rows}</tbody></table></div>${logs}`;
  bindDetails(el);
}

// ---- SIM / SLURM / GPU
function renderSim(v) {
  const sl = v.slurm, gpu = v.gpu;
  const ours = new Map(sl.jobs.map((j) => [j.id, j]));
  let html = "";
  if (sl.ok === false) html += `<p class="note">Could not read the SLURM queue: ${esc(sl.error)}</p>`;
  html += sl.jobs.length
    ? `<div class="tbl-wrap"><table><thead><tr><th>Job</th><th>State</th><th class="num">Running for</th><th class="num">Time limit</th><th>Node</th><th>SLURM id</th><th>Note</th></tr></thead><tbody>
       ${sl.jobs.map((j) => `<tr><td><a href="#" data-job="${esc(j.name)}">${esc(shortJob(j.name, v.expid))}</a></td><td>${pill(j.state === "PENDING" ? "QUEUING" : j.state)}</td>
         <td class="num">${esc(j.time)}</td><td class="num">${esc(j.limit)}</td><td>${esc(j.nodes || "—")}</td><td class="mono">${esc(j.id)}</td><td class="note">${esc(j.reason === "None" ? "" : j.reason)}</td></tr>`).join("")}</tbody></table></div>`
    : `<p class="empty">No jobs of ${esc(v.expid)} in the SLURM queue right now${sl.all_jobs.length ? ` (${plural(sl.all_jobs.length, "other job")} of yours)` : ""}.</p>`;
  if (gpu?.ok) {
    html += `<div class="gpu-grid">${gpu.gpus.map((g) => {
      const procs = gpu.procs.filter((p) => p.gpu_uuid === g.uuid);
      const util = +g["utilization.gpu"] || 0, mu = +g["memory.used"] || 0, mt = +g["memory.total"] || 1;
      return `<div class="gpu"><div class="g-head"><b>GPU ${esc(g.index)}</b><span class="note">${esc(g.name)} · ${esc(g["temperature.gpu"])}°C · ${Math.round(+g["power.draw"] || 0)} W</span></div>
        <div class="note" style="margin-top:6px">Busy ${util}%</div><div class="meter"><span style="width:${util}%"></span></div>
        <div class="note">Memory ${fmtBytes(mu * 1048576)} of ${fmtBytes(mt * 1048576)}</div><div class="meter"><span style="width:${(100 * mu) / mt}%"></span></div>
        <div class="note">${procs.length ? procs.map((p) => {
          const j = p.slurm_id && ours.get(p.slurm_id);
          return j ? `Used by <b>${esc(shortJob(j.name, v.expid))}</b>` : `Used by another user (${esc(p.process)})`;
        }).join("<br>") : "Idle"}</div></div>`;
    }).join("")}</div><p class="note">Snapshot of the whole node; GPUs are shared with other users.</p>`;
  } else if (gpu) html += `<p class="note">GPU information unavailable: ${esc(gpu.error)}</p>`;
  const run = v.running.filter((r) => r.tail && ((r.tail.out?.lines || []).length || (r.tail.err?.lines || []).length));
  html += run.map((r) => detailsBlock(`tail:${r.name}`,
    `Latest output of <b>${esc(shortJob(r.name, v.expid))}</b> · running for ${r.start ? fmtDur(Date.now() / 1000 - r.start) : "?"}`,
    ["out", "err"].filter((k) => r.tail[k] && r.tail[k].lines.length).map((k) => `<div class="note">${esc(r.tail[k].file)} · updated ${fmtAgo(r.tail[k].mtime)}</div><pre class="log">${esc(r.tail[k].lines.join("\n"))}</pre>`).join(""),
    true)).join("");
  $("#x-sim").innerHTML = html;
  bindDetails($("#x-sim"));
}

// ---- problems
function lastErrorLine(lines) {
  const l = [...lines].reverse().find((x) => /Error|Exception|error:|CANCELLED|Killed/.test(x) && x !== "as_exit_handler");
  return (l || lines.filter((x) => x !== "as_exit_handler").pop() || "").slice(0, 220);
}
function renderFailed(v) {
  const el = $("#x-fail");
  $("#f-note").textContent = v.failed.length ? plural(v.failed.length, "failed job") : "";
  if (!v.failed.length) { setHTML(el, `<div class="all-good">${ICONS.ok}No failed jobs.</div>`); return; }
  // group identical errors so ten members failing the same way read as one problem
  const groups = new Map();
  for (const f of v.failed) {
    const msg = f.err?.lines?.length ? lastErrorLine(f.err.lines) : "(error log not available yet)";
    if (!groups.has(msg)) groups.set(msg, []);
    groups.get(msg).push(f);
  }
  if (setHTML(el, [...groups.entries()].slice(0, 30).map(([msg, fs], gi) => {
    const f = fs[0], e = f.err;
    const names = fs.map((x) => `<a href="#" data-job="${esc(x.name)}">${esc(shortJob(x.name, v.expid))}</a>`);
    return `<div class="problem">
      <div class="p-head"><b>${esc(secName(f.section))}</b>${fs.length > 1 ? `<span class="badge">${fs.length} jobs</span>` : ""}
        <span class="note">last failure ${fmtTime(Math.max(...fs.map((x) => x.end || 0)))}</span></div>
      <div class="p-msg">${esc(msg)}</div>
      <div class="note" style="margin-top:6px">${names.slice(0, 12).join(", ")}${names.length > 12 ? ` and ${names.length - 12} more` : ""}</div>
      ${e?.lines?.length ? detailsBlock(`fail:${gi}:${f.name}`, `Full ${e.traceback ? "traceback" : "error log"}`, `<div class="note">${esc(e.file || "")}</div><pre class="log">${esc(e.lines.join("\n"))}</pre>`) : ""}
    </div>`;
  }).join(""))) bindDetails(el);
}

// ---- <details> with state preserved across refreshes
function detailsBlock(key, summary, body, defaultOpen = false) {
  const open = S.open.has(key) || (defaultOpen && !S.open.has("closed:" + key));
  return `<details data-key="${esc(key)}" ${open ? "open" : ""}><summary>${summary}</summary>${body}</details>`;
}
function bindDetails(root) {
  root.querySelectorAll("details[data-key]").forEach((d) => d.addEventListener("toggle", () => {
    const k = d.dataset.key;
    if (d.open) { S.open.add(k); S.open.delete("closed:" + k); } else { S.open.delete(k); S.open.add("closed:" + k); }
  }));
}

// ---- collapsible sections (remembered per browser)
function collapsedSet() { try { return new Set(JSON.parse(localStorage.getItem("aiuq-collapsed") || "[]")); } catch { return new Set(); } }
function saveCollapsed() {
  const ids = [...document.querySelectorAll("#main section.card.collapsed")].map((s) => s.id);
  try { localStorage.setItem("aiuq-collapsed", JSON.stringify(ids)); } catch { /* ignore */ }
}
function toggleSection(sec, collapse) {
  const c = collapse ?? !sec.classList.contains("collapsed");
  sec.classList.toggle("collapsed", c);
  sec.querySelector(".collapse-btn")?.setAttribute("aria-expanded", String(!c));
  if (!c) requestAnimationFrame(() => sec.querySelectorAll(".js-plotly-plot").forEach((p) => Plotly.Plots.resize(p)));
}
function setupCollapsibles() {
  const saved = collapsedSet();
  document.querySelectorAll("#main section.card").forEach((sec) => {
    const head = sec.querySelector(":scope > .card-head");
    if (!head || head.querySelector(".collapse-btn")) return;
    const btn = document.createElement("button");
    btn.className = "collapse-btn"; btn.type = "button"; btn.setAttribute("aria-expanded", "true");
    btn.setAttribute("aria-label", "Collapse or expand this section");
    btn.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 9l6 6 6-6"/></svg>';
    head.prepend(btn);
    head.classList.add("toggle");
    head.addEventListener("click", (e) => {
      if (e.target.closest("select, input, label, a, .info") || (e.target.closest("button") && !e.target.closest(".collapse-btn"))) return;
      toggleSection(sec); saveCollapsed();
    });
    if (saved.has(sec.id)) toggleSection(sec, true);
  });
}
function collapseAll(collapse) {
  document.querySelectorAll("#main section.card[data-tab]").forEach((s) => toggleSection(s, collapse));
  saveCollapsed();
}

// ---- job details: a drawer on the right, so the page behind keeps its place
function openJob(name) {
  S.job = name;
  renderJob(S.detail);
}
function closeJob() {
  S.job = null;
  const d = $("#job-drawer");
  d.classList.remove("open"); d.setAttribute("aria-hidden", "true");
}

function findJob(v, name) {
  const g = v.grid;
  for (let i = 0; i < g.cells.length; i++) {
    for (let j = 0; j < g.cells[i].length; j++) {
      const c = g.cells[i][j];
      if (c && c[1] === name) return { status: c[0], fail: c[2], date: g.dates[i], col: g.columns[j] };
    }
  }
  const o = v.once_jobs.find((x) => x.name === name);
  return o ? { status: o.status, fail: 0, date: null, col: { section: o.section } } : null;
}

function renderJob(v) {
  const el = $("#job-drawer");
  if (!S.job || !v) { if (el.classList.contains("open")) closeJob(); return; }
  const j = findJob(v, S.job);
  if (!j) { closeJob(); return; }
  const at = v.attempts.filter((a) => a.job === S.job).sort((a, b) => (a.start || 0) - (b.start || 0));
  const med = v.dur_stats[j.col.section]?.median;
  const now = Date.now() / 1000;
  const last = at[at.length - 1];
  const lastDur = last ? (last.end || now) - last.start : null;
  const fail = v.failed.find((f) => f.name === S.job);
  const run = v.running.find((r) => r.name === S.job);
  const ics = v.ics.find((r) => r.job === S.job);
  const facts = [
    ["step", esc(secName(j.col.section))], ["start date", j.date ? fmtDate(j.date) : null],
    ["member", j.col.member], ["chunk", j.col.chunk], ["attempts", at.length], ["retries", j.fail || 0],
    [last && !last.end ? "running for" : "last run took", lastDur != null ? fmtDur(lastDur) : null],
    ["typical for this step", med != null ? fmtDur(med) : null],
  ].filter((f) => f[1] != null && f[1] !== "");
  let logs = "";
  if (fail?.err?.lines?.length) logs += `<div class="chart-title" style="margin-top:14px">${fail.err.traceback ? "Traceback" : "Last error lines"}</div><div class="note">${esc(fail.err.file || "")}</div><pre class="log">${esc(fail.err.lines.join("\n"))}</pre>`;
  if (run?.tail) logs += ["out", "err"].filter((k) => run.tail[k]?.lines?.length).map((k) => `<div class="chart-title" style="margin-top:14px">Latest ${k === "out" ? "output" : "error output"}</div><div class="note">${esc(run.tail[k].file)}</div><pre class="log">${esc(run.tail[k].lines.join("\n"))}</pre>`).join("");
  if (ics?.mars?.events?.length) logs += `<div class="chart-title" style="margin-top:14px">MARS messages</div><pre class="log">${esc(ics.mars.events.join("\n"))}</pre>`;
  const fresh = setHTML(el, `
    <div class="jd-head"><div style="flex:1;min-width:0"><div class="title">${esc(secName(j.col.section))}${j.col.member ? ` · member ${esc(j.col.member)}` : ""}${j.date ? ` · ${fmtDate(j.date)}` : ""}</div>
        <span class="note mono">${esc(S.job)}</span></div>
      ${pill(j.status)}<button type="button" class="small ghost" id="job-close" title="Close  Esc" aria-label="Close job details">Close ✕</button></div>
    <div class="jd-body">
      <dl class="facts">${facts.map(([k, x]) => `<div><dt>${k}</dt><dd>${x}</dd></div>`).join("")}</dl>
      ${at.length ? `<div class="chart-title">Attempts</div><div class="plot" id="p-job"></div>${med ? `<div class="note">Dotted line = typical duration for this step (${fmtDur(med)}).</div>` : ""}`
        : `<div class="empty">This job has not started yet.</div>`}
      ${logs}</div>`);
  if (fresh) $("#job-close").onclick = closeJob;
  el.classList.add("open"); el.setAttribute("aria-hidden", "false");
  if (!at.length) return;
  const rows = at.map((a, i) => `Attempt ${(a.n ?? i) + 1}`);
  const traces = RESULT_ORDER.map((res) => {
    const idx = at.map((a, i) => ((RESULT_ORDER.includes(a.result) ? a.result : "UNKNOWN") === res ? i : -1)).filter((i) => i >= 0);
    if (!idx.length) return null;
    return { type: "bar", orientation: "h", name: label(res), y: idx.map((i) => rows[i]),
      base: idx.map((i) => new Date(at[i].start * 1000).toISOString()),
      x: idx.map((i) => Math.max(1000, ((at[i].end || now) - at[i].start) * 1000)),
      marker: { color: stColor(res) },
      customdata: idx.map((i) => [fmtTime(at[i].start), at[i].end ? fmtTime(at[i].end) : "still running", fmtDur((at[i].end || now) - at[i].start)]),
      hovertemplate: `${label(res)} · %{customdata[2]}<br>%{customdata[0]} → %{customdata[1]}<extra></extra>` };
  }).filter(Boolean);
  const shapes = med ? at.map((a, i) => ({ type: "line", xref: "x", yref: "y", x0: new Date((a.start + med) * 1000).toISOString(),
    x1: new Date((a.start + med) * 1000).toISOString(), y0: i - 0.45, y1: i + 0.45, line: { color: css("--text-primary"), width: 2, dash: "dot" } })) : [];
  Plotly.react($("#p-job"), traces, baseLayout({ height: Math.max(120, at.length * 36 + 64), barmode: "overlay",
    margin: { l: 80, r: 16, t: 6, b: 36 }, showlegend: false, shapes,
    xaxis: { type: "date", gridcolor: css("--grid") },
    yaxis: { type: "category", categoryorder: "array", categoryarray: rows, autorange: "reversed", gridcolor: "rgba(0,0,0,0)" } }), { ...PCFG, displayModeBar: false });
}

// ------------------------------------------------------------------ wiring
document.addEventListener("click", (e) => {
  const h = e.target.closest("a[data-home], a[data-app-link]");
  if (h && !e.metaKey && !e.ctrlKey) { e.preventDefault(); switchApp(h.dataset.appLink || "home"); return; }
  const a = e.target.closest("a[data-exp]");
  if (a && !e.metaKey && !e.ctrlKey) { e.preventDefault(); selectExp(a.dataset.exp); return; }
  const g = e.target.closest("a[data-gotab]");
  if (g) { e.preventDefault(); setTab(g.dataset.gotab); requestAnimationFrame(() => $(g.getAttribute("href"))?.scrollIntoView({ behavior: "smooth" })); return; }
  const j = e.target.closest("a[data-job]");
  if (j) { e.preventDefault(); openJob(j.dataset.job); return; }
  if (e.target.closest('[data-act="to-launcher"]')) { L.view = "runs"; L.q = S.exp || ""; switchApp("launcher"); }
});
$("#viewtabs").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  if (b.dataset.tab) setTab(b.dataset.tab);
  else if (b.dataset.view) setLauncherView(b.dataset.view);
  else if (b.dataset.act) collapseAll(b.dataset.act === "collapse");
});
$("#refreshBtn").addEventListener("click", () => load(true));
$("#refreshSel").addEventListener("change", (e) => {
  S.refreshSec = +e.target.value;
  try { localStorage.setItem("aiuq-refresh", String(S.refreshSec)); } catch { /* ignore */ }
  schedule(); tick();
});
$("#themeBtn").addEventListener("click", () => {
  const t = THEMES[(THEMES.indexOf(currentTheme()) + 1) % THEMES.length];
  applyTheme(t); redrawAll();
});
$("#helpBtn").addEventListener("click", () => $("#help").showModal());
$("#helpClose").addEventListener("click", () => $("#help").close());
$("#help").addEventListener("click", (e) => { if (e.target === e.currentTarget) e.currentTarget.close(); });  // click on the backdrop
window.addEventListener("popstate", () => {
  const q = new URLSearchParams(location.search);
  S.app = appFromUrl();
  if (S.app === "launcher") L.view = launcherViewFromUrl();
  if (S.app === "monitor") selectExp(q.get("exp"), false); else { S.skeletonFor = null; closeJob(); render(); fadeMain(); }
});
// come back to a tab that was in the background: refresh at once instead of waiting for the next tick
document.addEventListener("visibilitychange", () => { if (!document.hidden && S.refreshSec && S.lastFetch && Date.now() / 1000 - S.lastFetch > S.refreshSec) load(); });

// ---- experiments panel: hidden / shown (remembered per browser; "[" toggles it)
function setSidebar(hidden) {
  document.documentElement.classList.toggle("side-collapsed", hidden);
  const b = $("#sideBtn");
  b.setAttribute("aria-expanded", String(!hidden));
  b.title = `${hidden ? "Show" : "Hide"} the experiments list  [`;
  b.setAttribute("aria-label", `${hidden ? "Show" : "Hide"} the experiments list`);
  try { localStorage.setItem("aiuq-side", hidden ? "hidden" : "shown"); } catch { /* ignore */ }
  requestAnimationFrame(() => {
    document.querySelectorAll("#main .js-plotly-plot").forEach((p) => { if (p.offsetParent) Plotly.Plots.resize(p); });
    if (S.preview && S.app === "monitor" && S.tab === "data") { S.preview.drawn = false; drawMap(); }
  });
}
const sideHidden = () => document.documentElement.classList.contains("side-collapsed");
$("#sideBtn").addEventListener("click", () => setSidebar(!sideHidden()));

// ---- keyboard
function stepExp(dir) {  // next / previous experiment in the order of the panel
  const es = visibleExps();
  if (!es.length) return;
  const i = es.findIndex((e) => e.expid === S.exp);
  const n = i < 0 ? (dir > 0 ? 0 : es.length - 1) : Math.max(0, Math.min(es.length - 1, i + dir));
  if (es[n].expid !== S.exp) selectExp(es[n].expid);
}
document.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === "Escape" && S.job && !document.querySelector("dialog[open]")) { closeJob(); return; }
  if (e.target.closest?.("input, textarea, select, [contenteditable]") || document.querySelector("dialog[open]")) return;
  const k = e.key;
  if (k === "[") setSidebar(!sideHidden());
  else if (k === "?") $("#help").showModal();
  else if (k === "r" || k === "R") load(true);
  else if (k === "/") {
    e.preventDefault();
    if (S.app !== "monitor") switchApp("monitor");
    if (sideHidden()) setSidebar(false);
    $("#side-q")?.focus();
  } else if ((k === "j" || k === "k" || k === "J" || k === "K") && S.app === "monitor") stepExp(k.toLowerCase() === "j" ? 1 : -1);
  else if (k >= "1" && k <= "4") {
    const i = +k - 1;
    if (S.app === "monitor" && S.exp && TABS[i]) setTab(TABS[i][0]);
    else if (S.app === "launcher" && LN_TABS[i]) setLauncherView(LN_TABS[i][0]);
  }
});
setSidebar(sideHidden());
document.querySelector(".appswitch").addEventListener("click", (e) => { const b = e.target.closest("button[data-app]"); if (b && b.dataset.app !== S.app) switchApp(b.dataset.app); });
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (currentTheme() === "auto") redrawAll(); });
try { const r = localStorage.getItem("aiuq-refresh"); if (r !== null) { S.refreshSec = +r; $("#refreshSel").value = r; } } catch { /* ignore */ }
applyTheme(currentTheme());
S.preview = newPreview();
if (S.exp) document.title = `${S.exp} · AIUQ Monitor`;
render();  // first paint (homepage/launcher do not need to wait for data)
setInterval(tick, 1000);
schedule();
load();
