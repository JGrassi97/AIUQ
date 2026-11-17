/* AIUQ Monitor — outputs & maps module: shaded field + contours, time player, point time series.
   Loaded before app.js; uses its helpers ($, css, esc, api, baseLayout, PCFG, onClick, fmtDate, …) at runtime. */
"use strict";

const VAR_INFO = {
  t: { name: "Temperature", tr: "k2c", cmap: "Temperature" },
  "2t": { name: "2 m temperature", tr: "k2c", cmap: "Temperature" },
  t2m: { name: "2 m temperature", tr: "k2c", cmap: "Temperature" },
  sst: { name: "Sea surface temperature", tr: "k2c", cmap: "Temperature" },
  z: { name: "Geopotential", tr: "z2dam", cmap: "Viridis" },
  u: { name: "Zonal wind", cmap: "Blue–Red", sym: true },
  v: { name: "Meridional wind", cmap: "Blue–Red", sym: true },
  "10u": { name: "10 m zonal wind", cmap: "Blue–Red", sym: true },
  "10v": { name: "10 m meridional wind", cmap: "Blue–Red", sym: true },
  w: { name: "Vertical velocity", cmap: "Blue–Red", sym: true },
  q: { name: "Specific humidity", tr: "kg2g", cmap: "Rain" },
  r: { name: "Relative humidity", cmap: "Rain" },
  msl: { name: "Sea-level pressure", tr: "pa2hpa", cmap: "Viridis" },
  sp: { name: "Surface pressure", tr: "pa2hpa", cmap: "Viridis" },
  tp: { name: "Precipitation", tr: "m2mm", cmap: "Rain" },
  tcwv: { name: "Water vapour", cmap: "Rain" },
};
const varName = (v) => VAR_INFO[v]?.name || v;

const TRANSFORMS = {
  none: { label: "As stored", f: (x) => x, units: (u) => u },
  k2c: { label: "°C (from K)", f: (x) => x - 273.15, units: () => "°C" },
  z2dam: { label: "dam (height, from m²/s²)", f: (x) => x / 98.0665, units: () => "dam" },
  z2m: { label: "m (height, from m²/s²)", f: (x) => x / 9.80665, units: () => "m" },
  pa2hpa: { label: "hPa (from Pa)", f: (x) => x / 100, units: () => "hPa" },
  m2mm: { label: "mm (from m)", f: (x) => x * 1000, units: () => "mm" },
  kg2g: { label: "g/kg (from kg/kg)", f: (x) => x * 1000, units: () => "g/kg" },
};

const CMAPS = {
  Temperature: ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"],
  "Blue–Red": ["#053061", "#2166ac", "#4393c3", "#92c5de", "#d1e5f0", "#f7f7f7", "#fddbc7", "#f4a582", "#d6604d", "#b2182b", "#67001f"],
  Viridis: ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725"],
  Plasma: ["#0d0887", "#46039f", "#7201a8", "#9c179e", "#bd3786", "#d8576b", "#ed7953", "#fb9f3a", "#fdca26", "#f0f921"],
  Turbo: ["#30123b", "#4662d7", "#36aaf9", "#1ae4b6", "#72fe5e", "#c8ef34", "#faba39", "#f66b19", "#ca2a04", "#7a0403"],
  Cividis: ["#00224e", "#123570", "#3b496c", "#575d6d", "#707173", "#8a8779", "#a69d75", "#c4b56c", "#e4cf5b", "#fee838"],
  Ocean: ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
  Sunset: ["#ffffcc", "#ffeda0", "#fed976", "#feb24c", "#fd8d3c", "#fc4e2a", "#e31a1c", "#bd0026", "#800026"],
  Rain: ["#ffffd9", "#edf8b1", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#225ea8", "#253494", "#081d58"],
  Spectral: ["#5e4fa2", "#3288bd", "#66c2a5", "#abdda4", "#e6f598", "#ffffbf", "#fee08b", "#fdae61", "#f46d43", "#d53e4f", "#9e0142"],
  Greys: ["#ffffff", "#d9d9d9", "#bdbdbd", "#969696", "#737373", "#525252", "#252525", "#000000"],
};
const cscale = (name, rev) => {
  let st = CMAPS[name] || CMAPS.Ocean;
  if (rev) st = [...st].reverse();
  return st.map((c, i) => [i / (st.length - 1), c]);
};
const grad = (name, rev) => `linear-gradient(90deg, ${(rev ? [...CMAPS[name]].reverse() : CMAPS[name]).join(", ")})`;

const REGIONS = {
  Global: [-180, 180, -90, 90], Europe: [-25, 45, 33, 72], "North Atlantic": [-80, 20, 20, 75],
  Mediterranean: [-10, 40, 29, 48], Italy: [5, 20, 36, 48], "Northern Hemisphere": [-180, 180, 0, 90],
  Tropics: [-180, 180, -30, 30], "Southern Hemisphere": [-180, 180, -90, 0],
};
const LINE_COLORS = [["auto", "Automatic"], ["#111111", "Black"], ["#ffffff", "White"], ["#d03b3b", "Red"], ["#6250d6", "Violet"], ["#eb6834", "Orange"]];
const COAST_COLORS = [["auto", "Automatic"], ["#111111", "Black"], ["#ffffff", "White"], ["#898781", "Grey"]];

const STYLE_KEY = "aiuq-mapstyle-v2";
const STYLE_DEFAULT = {
  vars: {},  // per variable: { cmap, rev, zmin, zmax, tr, sym }
  cont: { mode: "count", n: 12, step: "", from: "", to: "", color: "auto", width: 1.2, labels: true, dashNeg: true, smooth: true, tr: {} },
  map: { region: "Global", coast: true, coastColor: "auto", grid: false, smooth: false, lock: true },
  seriesMode: "members",
};
function loadStyle() {
  const st = JSON.parse(JSON.stringify(STYLE_DEFAULT));
  try {
    const saved = JSON.parse(localStorage.getItem(STYLE_KEY) || "{}");
    Object.assign(st.vars, saved.vars || {}); Object.assign(st.cont, saved.cont || {}); Object.assign(st.map, saved.map || {});
    if (saved.seriesMode) st.seriesMode = saved.seriesMode;
  } catch { /* ignore */ }
  return st;
}
function saveStyle() { try { localStorage.setItem(STYLE_KEY, JSON.stringify(S.preview.st)); } catch { /* ignore */ } }
function vstyle(v) {  // v is a variable, or "var:spread" / "var:anom" for the ensemble views
  const [base, mode] = String(v).split(":");
  const d = VAR_INFO[base] || {};
  const st = S.preview.st;
  const def = mode === "spread" ? { cmap: "Sunset", sym: false } : mode === "anom" ? { cmap: "Blue–Red", sym: true } : { cmap: d.cmap || "Ocean", sym: !!d.sym };
  st.vars[v] = Object.assign({ rev: false, zmin: "", zmax: "", tr: d.tr || "none", ...def }, st.vars[v] || {});
  return st.vars[v];
}
const FIELDS = { member: "One member", mean: "Ensemble mean", spread: "Spread", anom: "Member − mean" };
const fkey = () => { const P = S.preview; return P.field === "spread" || P.field === "anom" ? `${P.fill.var}:${P.field}` : P.fill.var; };
const scaleOnly = () => ["spread", "anom"].includes(S.preview.field);
// Spread and departures are differences: apply only the scale of a unit conversion, not its offset (K -> °C).
const fieldTf = (ft) => (scaleOnly() ? (x) => ft.f(x) - ft.f(0) : ft.f);
function fieldData() {
  const P = S.preview, d = P.dFill, E = P.dEns;
  if (!d) return null;
  if (P.field === "member" || !E) return { z: d.z, lo: d.min, hi: d.max };
  if (P.field === "mean") return { z: E.mean, lo: E.mean_range[0], hi: E.mean_range[1] };
  if (P.field === "spread") return { z: E.std, lo: E.std_range[0], hi: E.std_range[1] };
  let lo = Infinity, hi = -Infinity;
  const z = d.z.map((row, i) => row.map((x, j) => {
    const m = E.mean[i]?.[j];
    if (x == null || m == null) return null;
    const a = x - m; if (a < lo) lo = a; if (a > hi) hi = a; return a;
  }));
  return { z, lo, hi };
}
const ctr = (v) => S.preview.st.cont.tr[v] || VAR_INFO[v]?.tr || "none";

function newPreview() {
  return { member: null, start: null, time: -1, field: "member", dEns: null, dContEns: null, fill: { var: null, level: null }, cont: { var: "", level: null },
    tab: "data", playing: false, dFill: null, dCont: null, loading: false, error: null, autoTried: false,
    sel: null, seriesKind: "point", lastSel: { point: null, area: null }, drawing: false, viewRev: 0, geo: null,
    series: null, seriesErr: null, seriesLoading: false, drawn: false, built: false, sig: null, psig: null,
    locked: {}, st: loadStyle() };
}

function parseOut(fn) {
  const p = fn.startsWith("out-") && fn.endsWith(".nc") ? fn.slice(4, -3).split("-") : [];
  if (p.length < 8) return null;
  return { start: p.slice(0, 3).join("-"), end: p.slice(3, 6).join("-"), member: p.slice(6, -1).join("-"), var: p[p.length - 1] };
}
function fileFor(o, v, member, start) {
  const f = o.files.find((x) => x[0] === v && x[1] === member && parseOut(x[2])?.start === start && parseOut(x[2])?.var === v);
  return f ? f[2] : null;
}
const sortMem = (a, b) => (+a) - (+b) || String(a).localeCompare(String(b));
function niceStep(x) {
  if (!(x > 0)) return 1;
  const p = 10 ** Math.floor(Math.log10(x)), f = x / p;
  return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p;
}
// round step whose number of lines is closest to the requested count
function stepForCount(a, b, n) {
  const raw = (b - a) / Math.max(2, n), p = 10 ** Math.floor(Math.log10(raw));
  let best = niceStep(raw), bestErr = Infinity;
  for (const k of [p / 10, p, p * 10]) for (const m of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8]) {
    const st = k * m, cnt = Math.floor(b / st) - Math.ceil(a / st) + 1, err = Math.abs(cnt - n);
    if (err < bestErr) { best = st; bestErr = err; }
  }
  return best;
}
const decimalsOf = (step) => Math.min(3, Math.max(0, -Math.floor(Math.log10(step) + 1e-9) + (Math.abs(step / 10 ** Math.floor(Math.log10(step)) % 1) > 1e-6 ? 1 : 0)));
const fmtLevel = (x, step) => (x == null || !isFinite(x) ? "—" : (+x.toFixed(decimalsOf(step))).toString());
const fmtNum = (x) => (x == null || !isFinite(x) ? "—" : Math.abs(x) >= 100 ? x.toFixed(0) : Math.abs(x) >= 10 ? x.toFixed(1) : x.toPrecision(3));
const fmtLat = (y) => `${Math.abs(y).toFixed(1)}° ${y >= 0 ? "N" : "S"}`;
const fmtLon = (x) => `${Math.abs(x).toFixed(1)}° ${x >= 0 ? "E" : "W"}`;
function fmtValid(t) {  // '1989-01-26 00:00' -> '26 Jan 1989, 00:00'
  const m = String(t || "").match(/^(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}:\d{2}))?/);
  return m ? `${+m[3]} ${MONTHS[+m[2] - 1]} ${m[1]}${m[4] ? `, ${m[4]}` : ""}` : (t || "");
}
function leadDay(start, t) {
  const a = Date.parse(start), b = Date.parse(String(t).replace(" ", "T"));
  return isFinite(a) && isFinite(b) ? Math.round((b - a) / 86400000) : null;
}

const SVG = {
  first: '<svg viewBox="0 0 24 24"><path d="M6 5h2v14H6zM20 5v14L9 12z"/></svg>',
  prev: '<svg viewBox="0 0 24 24"><path d="M18 5v14L7 12z"/></svg>',
  play: '<svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>',
  pause: '<svg viewBox="0 0 24 24"><path d="M7 5h4v14H7zM13 5h4v14h-4z"/></svg>',
  next: '<svg viewBox="0 0 24 24"><path d="M6 5v14l11-7z"/></svg>',
  last: '<svg viewBox="0 0 24 24"><path d="M16 5h2v14h-2zM4 5v14l11-7z"/></svg>',
  dl: '<svg class="ico" viewBox="0 0 24 24"><path d="M12 3v12M7 10l5 5 5-5M5 21h14"/></svg>',
  pin: '<svg viewBox="0 0 24 24"><path d="M12 21s-7-6.2-7-11a7 7 0 0 1 14 0c0 4.8-7 11-7 11z"/><circle cx="12" cy="10" r="2.5"/></svg>',
};

// ------------------------------------------------------------------ section render
function renderOutputs(v) {
  const P = S.preview, o = v.outputs;
  const body = $("#o-body");
  const vars = Object.keys(o.by_var_member).sort();
  $("#o-note").textContent = o.fetched ? `${o.n} files · ${fmtBytes(o.size)}` : "";
  if (!vars.length) {
    body.innerHTML = `<div class="empty">No outputs yet${o.fetched ? "" : " (not listed yet)"}. Files will appear here once the simulations write them to ${esc(v.remote.root || "?")}/outputs.</div>`;
    P.built = false;
    return;
  }
  if (!P.built || !$("#m-panel")) buildOutputs(body);

  // summary + files table
  const mems = [...new Set(vars.flatMap((x) => Object.keys(o.by_var_member[x])))].sort(sortMem);
  const expected = (v.conf.dates || []).length * (v.conf.num_chunks || 1);
  const latest = Math.max(...vars.flatMap((x) => Object.values(o.by_var_member[x]).map((b) => b.latest)));
  $("#o-sum").innerHTML = `<span class="stat-chip"><b>${o.n}</b> NetCDF files</span><span class="stat-chip"><b>${fmtBytes(o.size)}</b></span>
    <span class="stat-chip">variables <b>${vars.map((x) => esc(varName(x))).join(", ")}</b></span>
    <span class="stat-chip"><b>${mems.length}</b> member${mems.length === 1 ? "" : "s"}</span>
    <span class="stat-chip">newest file <b>${fmtAgo(latest)}</b></span>`;
  $("#x-out-table").innerHTML = `<div class="tbl-wrap"><table><thead><tr><th>Variable</th>${mems.map((m) => `<th class="num">m${esc(m)}</th>`).join("")}<th class="num">Size</th></tr></thead><tbody>
    ${vars.map((x) => { const r = o.by_var_member[x]; const sz = Object.values(r).reduce((a, b) => a + b.size, 0);
      return `<tr><td>${esc(varName(x))} <span class="note mono">${esc(x)}</span></td>${mems.map((m) => `<td class="num">${r[m] ? r[m].n : 0}${expected ? `<span class="note"> / ${expected}</span>` : ""}</td>`).join("")}<td class="num">${fmtBytes(sz)}</td></tr>`; }).join("")}
    </tbody></table></div>`;

  // defaults
  if (!P.fill.var || !vars.includes(P.fill.var)) P.fill.var = (v.conf.out_vars || []).find((x) => vars.includes(x)) || vars[0];
  if (P.cont.var && !vars.includes(P.cont.var)) P.cont.var = "";
  const vm = Object.keys(o.by_var_member[P.fill.var]).sort(sortMem);
  if (!P.member || !vm.includes(P.member)) P.member = vm[0];
  const starts = [...new Set(o.files.filter((f) => f[0] === P.fill.var && f[1] === P.member).map((f) => parseOut(f[2])?.start).filter(Boolean))].sort();
  if (!P.start || !starts.includes(P.start)) P.start = starts[starts.length - 1];
  const levels = P.dFill?.levels?.length ? P.dFill.levels : (Array.isArray(v.conf.out_levs) ? v.conf.out_levs : []);
  if (P.fill.level == null && levels.length) P.fill.level = levels.includes(850) ? 850 : levels[0];
  if (P.cont.level == null && levels.length) P.cont.level = levels.includes(500) ? 500 : levels[levels.length - 1];
  P.opts = { vars, vm, starts, levels };

  renderPanel();
  renderPlayer();
  if (!P.dFill && !P.loading && !P.error && !P.autoTried && P.start) { P.autoTried = true; loadMap(); return; }
  if (!P.drawn) drawMap();
}

function buildOutputs(body) {
  const P = S.preview;
  body.innerHTML = `
    <div class="out-summary" id="o-sum"></div>
    <div class="viz">
      <div class="viz-stage">
        <div class="viz-head">
          <div><div class="viz-title" id="m-title">Map</div><div class="viz-sub" id="m-sub"></div></div>
          <div class="viz-tools">
            <div class="search" id="m-searchbox">
              <svg class="ico" viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>
              <input type="text" id="m-search" placeholder="Find a place or type lat, lon" autocomplete="off" aria-label="Find a place for the time series">
              <div class="search-results" id="m-results" role="listbox"></div>
            </div>
            <select id="m-region" aria-label="Region">${Object.keys(REGIONS).map((r) => `<option ${r === P.st.map.region ? "selected" : ""}>${r}</option>`).join("")}</select>
            <button class="icon" id="m-dl" title="Download as PNG" aria-label="Download map as PNG">${SVG.dl}</button>
          </div>
        </div>
        <div class="map-frame" id="m-frame">
          <div id="p-map"></div>
          <div class="map-hint" id="m-hint">Click anywhere for the time series</div>
          <div class="map-loading"><div class="spinner"></div><span id="m-loadtxt">Loading…</span></div>
        </div>
        <div class="player" id="m-player"></div>
        <div class="series-card" id="m-series"></div>
      </div>
      <aside class="viz-panel" aria-label="Map settings">
        <div class="tabs" role="tablist">
          ${[["data", "Data"], ["colours", "Colours"], ["contours", "Contours"], ["map", "Map"]].map(([k, t]) => `<button role="tab" data-tab="${k}" aria-selected="${P.tab === k}">${t}</button>`).join("")}
        </div>
        <div class="tab-body" id="m-panel"></div>
      </aside>
    </div>
    ${detailsBlock("files", "Files per member", `<div id="x-out-table"></div>`).replace("<details", '<details class="files"')}`;
  bindDetails(body);
  P.built = true; P.psig = null; P.drawn = false;
  body.querySelector(".tabs").addEventListener("click", (e) => {
    const b = e.target.closest("button[data-tab]"); if (!b) return;
    P.tab = b.dataset.tab; P.psig = null;
    body.querySelectorAll(".tabs button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
    renderPanel();
  });
  $("#m-region").onchange = (e) => { P.st.map.region = e.target.value; saveStyle(); drawMap(); };
  $("#m-dl").onclick = downloadMap;
  const panel = $("#m-panel");
  panel.addEventListener("click", onPanelClick);
  panel.addEventListener("change", onPanelChange);
  panel.addEventListener("input", onPanelInput);
  $("#m-player").addEventListener("click", onPlayerClick);
  $("#m-player").addEventListener("input", (e) => {
    if (e.target.id === "m-time" && P.dFill) { const i = +e.target.value; $("#m-when").innerHTML = whenHtml(P.dFill.times[i]); }
  });
  $("#m-player").addEventListener("change", (e) => { if (e.target.id === "m-time") { P.time = +e.target.value; loadMap(); } });
  $("#m-series").addEventListener("click", onSeriesClick);
  const inp = $("#m-search");
  inp.addEventListener("input", onSearchInput);
  inp.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); const r = P.geo?.[0]; if (r) pickPlace(r); }
    if (e.key === "Escape") hideResults();
  });
  inp.addEventListener("focus", () => { if (P.geo?.length) $("#m-results").classList.add("open"); });
  document.addEventListener("click", (e) => { if (!e.target.closest("#m-searchbox")) hideResults(); });
  $("#m-results").addEventListener("click", (e) => { const b = e.target.closest("button[data-i]"); if (b) pickPlace(S.preview.geo[+b.dataset.i]); });
}

// ------------------------------------------------------------------ settings panel
const chips = (k, items, cur) => `<div class="chips-sel" data-k="${k}">${items.map(([v, t, sub]) =>
  `<button type="button" data-v="${esc(v)}" aria-pressed="${String(v) === String(cur)}">${esc(t)}${sub ? `<span class="sub">${esc(sub)}</span>` : ""}</button>`).join("")}</div>`;
const seg = (k, items, cur) => `<div class="seg" data-k="${k}">${items.map(([v, t]) => `<button type="button" data-v="${esc(v)}" aria-pressed="${v === cur}">${esc(t)}</button>`).join("")}</div>`;
const sw = (k, on, text, hint) => `<label class="switch"><input type="checkbox" data-k="${k}" ${on ? "checked" : ""}><span class="track"></span><span>${esc(text)}${hint ? `<span class="hint">${esc(hint)}</span>` : ""}</span></label>`;
const field = (lbl, inner, val) => `<div class="field"><div class="f-lbl"><span>${lbl}</span>${val != null ? `<span class="val">${val}</span>` : ""}</div>${inner}</div>`;
const sel = (k, items, cur) => `<select data-k="${k}">${items.map(([v, t]) => `<option value="${esc(v)}" ${String(v) === String(cur) ? "selected" : ""}>${esc(t)}</option>`).join("")}</select>`;
const colors = (k, items, cur) => `<div class="colors" data-k="${k}">${items.map(([v, t]) => `<button type="button" class="${v === "auto" ? "auto" : ""}" data-v="${v}" title="${t}" aria-label="${t}" aria-pressed="${v === cur}" style="${v === "auto" ? "" : `background:${v}`}"></button>`).join("")}</div>`;

function dataRangeOf(d, tf) { return d ? [tf(d.min), tf(d.max)] : [null, null]; }

function renderPanel() {
  const P = S.preview, panel = $("#m-panel");
  if (!panel || !P.opts) return;
  const { vars, vm, starts, levels } = P.opts;
  const fs = vstyle(fkey()), cs = P.st.cont, ms = P.st.map;
  const sig = JSON.stringify([P.tab, P.opts, P.fill, P.cont, P.member, P.start, P.st, !!P.dFill, !!P.dCont, P.dFill?.min, P.dCont?.min]);
  if (sig === P.psig || (document.activeElement && panel.contains(document.activeElement) && document.activeElement.matches("input[type=number], input[type=text]"))) return;
  P.psig = sig;
  const levChips = (k, cur) => (levels.length ? chips(k, levels.map((l) => [l, `${l}`, "hPa"]), cur) : `<div class="note">single level</div>`);
  const trOpts = Object.entries(TRANSFORMS).map(([k, t]) => [k, t.label]);
  let html = "";
  if (P.tab === "data") {
    html = (vm.length > 1 ? field("Show", seg("field", Object.entries(FIELDS).map(([k, t]) => [k, k === "member" ? "Member" : k === "mean" ? "Mean" : k === "spread" ? "Spread" : "Anomaly"]), P.field))
        + `<div class="data-range">${{ member: "A single ensemble member.", mean: `Average of the ${vm.length} members.`, spread: `Standard deviation across the ${vm.length} members: where they disagree.`, anom: "The selected member minus the ensemble mean." }[P.field]}</div>` : "")
      + field("Shaded variable", chips("fvar", vars.map((x) => [x, varName(x)]), P.fill.var))
      + field("Pressure level", levChips("flev", P.fill.level))
      + (P.field === "mean" || P.field === "spread" ? "" : field("Ensemble member", vm.length <= 16 ? chips("member", vm.map((m) => [m, m]), P.member) : sel("member", vm.map((m) => [m, `member ${m}`]), P.member)))
      + field("Start date", starts.length <= 6 ? chips("start", starts.map((s) => [s, fmtDate(s)]), P.start) : sel("start", starts.map((s) => [s, fmtDate(s)]), P.start), `${starts.length} available`)
      + `<p class="note">Use the player below the map to move through time.</p>`;
  } else if (P.tab === "colours") {
    const tf = fieldTf(TRANSFORMS[fs.tr] || TRANSFORMS.none);
    const fd = fieldData();
    const [lo, hi] = fd ? [tf(fd.lo), tf(fd.hi)] : [null, null];
    html = field("Units", sel("ftr", trOpts, fs.tr))
      + field("Colour scale", `<div class="swatches" data-k="cmap">${Object.keys(CMAPS).map((n) =>
          `<button type="button" class="swatch" data-v="${n}" aria-pressed="${n === fs.cmap}"><span class="grad" style="background:${grad(n, fs.rev)}"></span>${n}</button>`).join("")}</div>`)
      + sw("rev", fs.rev, "Reverse colours")
      + field("Colour range", `<div class="pair"><label>Min<input type="number" step="any" data-k="zmin" value="${esc(fs.zmin)}" placeholder="${fmtNum(lo)}"></label>
          <label>Max<input type="number" step="any" data-k="zmax" value="${esc(fs.zmax)}" placeholder="${fmtNum(hi)}"></label></div>`,
          `<button type="button" class="ghost" data-k="zauto" style="padding:1px 8px;font-size:11px">Auto</button>`)
      + `<div class="data-range">This field goes from ${fmtNum(lo)} to ${fmtNum(hi)}. Leave the boxes empty for automatic limits.</div>`
      + sw("sym", fs.sym, "Centre on zero", "Same distance below and above 0 — for anomalies and winds")
      + sw("lock", ms.lock, "Keep the range while playing", "Colours stay comparable between time steps");
  } else if (P.tab === "contours") {
    const on = !!P.cont.var;
    html = field("Contour variable", chips("cvar", [["", "None"], ...vars.map((x) => [x, varName(x)])], P.cont.var));
    if (on) {
      const ct = TRANSFORMS[ctr(P.cont.var)] || TRANSFORMS.none;
      const [lo, hi] = dataRangeOf(P.dCont, ct.f);
      const cu = ct.units(P.dCont?.units && P.dCont.units !== "unknown" ? P.dCont.units : "");
      const lv = contourLevels(lo, hi);
      html += field("Pressure level", levChips("clev", P.cont.level))
        + field("Units", sel("ctr", trOpts, ctr(P.cont.var)))
        + field("Line spacing", seg("cmode", [["count", "Number of lines"], ["step", "Fixed interval"]], cs.mode))
        + (cs.mode === "count"
          ? field("How many lines", `<div class="range-row"><input type="range" min="3" max="40" step="1" data-k="cn" value="${cs.n}"><output id="cn-out">${cs.n}</output></div>`)
          : field(`Interval${cu ? ` (${esc(cu)})` : ""}`, `<input type="number" step="any" min="0" data-k="cstep" value="${esc(cs.step)}" placeholder="${lv ? fmtNum(lv.step) : ""}">`))
        + `<div class="data-range" id="c-info">${lv ? `A line every <b>${fmtLevel(lv.step, lv.step)}${cu ? " " + esc(cu) : ""}</b>, from ${fmtLevel(lv.start, lv.step)} to ${fmtLevel(lv.end, lv.step)} (${lv.count} lines).` : "Load the map to see the line values."}</div>`
        + field("Only draw between", `<div class="pair"><label>From<input type="number" step="any" data-k="cfrom" value="${esc(cs.from)}" placeholder="${fmtNum(lo)}"></label>
            <label>To<input type="number" step="any" data-k="cto" value="${esc(cs.to)}" placeholder="${fmtNum(hi)}"></label></div>`)
        + field("Line colour", colors("ccol", LINE_COLORS, cs.color))
        + field("Line thickness", `<div class="range-row"><input type="range" min="0.5" max="3" step="0.1" data-k="cw" value="${cs.width}"><output id="cw-out">${cs.width}</output></div>`)
        + sw("clab", cs.labels, "Show values on the lines")
        + sw("cneg", cs.dashNeg, "Dashed lines below zero")
        + sw("csmooth", cs.smooth, "Smooth lines");
    } else {
      html += `<p class="note">Overlay a second field as lines — for example geopotential over temperature.</p>`;
    }
  } else {
    html = sw("coast", ms.coast, "Coastlines")
      + (ms.coast ? field("Coastline colour", colors("coastc", COAST_COLORS, ms.coastColor)) : "")
      + sw("grid", ms.grid, "Latitude / longitude grid")
      + sw("smooth", ms.smooth, "Smooth shading", "Interpolate between grid points")
      + field("Region", chips("region", Object.keys(REGIONS).map((r) => [r, r]), ms.region))
      + `<p class="note">Drag on the map to zoom, double-click to go back to the region.</p>`
      + `<div class="panel-foot"><button type="button" data-k="reset">Reset all map settings</button></div>`;
  }
  panel.innerHTML = html;
}

function onPanelClick(e) {
  const P = S.preview;
  const b = e.target.closest("button[data-v], button[data-k]");
  if (!b) return;
  const k = b.dataset.k || b.closest("[data-k]")?.dataset.k, v = b.dataset.v;
  const fs = vstyle(fkey()), cs = P.st.cont, ms = P.st.map;
  const data = () => { P.psig = null; renderPanel(); loadMap(); };
  const style = () => { saveStyle(); P.psig = null; renderPanel(); drawMap(); };
  switch (k) {
    case "fvar": P.fill.var = v; P.series = null; P.dFill = null; P.start = null; renderOutputs(S.detail); loadMap().then(() => P.sel && loadSeries()); return;
    case "flev": P.fill.level = +v; data(); if (P.sel) loadSeries(); return;
    case "member": P.member = v; data(); drawSeries(); return;
    case "start": P.start = v; P.time = -1; P.series = null; data(); if (P.sel) loadSeries(); return;
    case "cmap": fs.cmap = v; style(); return;
    case "field": P.field = v; data(); return;
    case "zauto": fs.zmin = ""; fs.zmax = ""; delete P.locked[lockKey()]; style(); return;
    case "cvar": P.cont.var = v; P.dCont = null; if (v) data(); else { P.psig = null; renderPanel(); drawMap(); } return;
    case "clev": P.cont.level = +v; data(); return;
    case "cmode": cs.mode = v; style(); return;
    case "ccol": cs.color = v; style(); return;
    case "coastc": ms.coastColor = v; style(); return;
    case "region": ms.region = v; $("#m-region").value = v; style(); return;
    case "reset": P.st = JSON.parse(JSON.stringify(STYLE_DEFAULT)); P.locked = {}; $("#m-region").value = "Global"; style(); return;
    default:
  }
}
function onPanelChange(e) {
  const P = S.preview, t = e.target, k = t.dataset.k;
  if (!k) return;
  const fs = vstyle(fkey()), cs = P.st.cont, ms = P.st.map;
  const val = t.type === "checkbox" ? t.checked : t.value;
  const style = () => { saveStyle(); P.psig = null; renderPanel(); drawMap(); };
  switch (k) {
    case "member": P.member = val; P.psig = null; loadMap(); drawSeries(); return;
    case "start": P.start = val; P.time = -1; P.series = null; P.psig = null; loadMap(); if (P.sel) loadSeries(); return;
    case "ftr": fs.tr = val; fs.zmin = ""; fs.zmax = ""; style(); drawSeries(); return;
    case "rev": fs.rev = val; style(); return;
    case "zmin": fs.zmin = val; style(); return;
    case "zmax": fs.zmax = val; style(); return;
    case "sym": fs.sym = val; style(); return;
    case "lock": ms.lock = val; P.locked = {}; style(); return;
    case "ctr": cs.tr[P.cont.var] = val; cs.from = ""; cs.to = ""; cs.step = ""; style(); return;
    case "cn": cs.n = +val; style(); return;
    case "cstep": cs.step = val; style(); return;
    case "cfrom": cs.from = val; style(); return;
    case "cto": cs.to = val; style(); return;
    case "cw": cs.width = +val; style(); return;
    case "clab": cs.labels = val; style(); return;
    case "cneg": cs.dashNeg = val; style(); return;
    case "csmooth": cs.smooth = val; style(); return;
    case "coast": ms.coast = val; style(); return;
    case "grid": ms.grid = val; style(); return;
    case "smooth": ms.smooth = val; style(); return;
    default:
  }
}
function onPanelInput(e) {  // live feedback while dragging sliders
  const k = e.target.dataset.k;
  if (k === "cn") $("#cn-out").textContent = e.target.value;
  if (k === "cw") $("#cw-out").textContent = e.target.value;
}

// ------------------------------------------------------------------ player
const whenHtml = (t) => `<b>${esc(fmtValid(t))}</b><span>${(() => { const d = leadDay(S.preview.start, t); return d != null ? `day ${d} of the forecast` : ""; })()}</span>`;
function curIdx() { const P = S.preview, n = P.dFill?.times?.length || 0; return P.time < 0 ? n + P.time : P.time; }
function renderPlayer(force) {
  const P = S.preview, el = $("#m-player");
  if (!el) return;
  const n = P.dFill?.times?.length || 0, i = n ? curIdx() : 0;
  const sig = `${n}|${i}|${P.playing}|${P.start}`;
  if (!force && sig === el.dataset.sig) return;
  el.dataset.sig = sig;
  el.innerHTML = `
    <button type="button" data-p="first" title="First time step" aria-label="First time step">${SVG.first}</button>
    <button type="button" data-p="prev" title="Previous time step" aria-label="Previous time step">${SVG.prev}</button>
    <button type="button" class="play" data-p="play" title="${P.playing ? "Pause" : "Play through time"}" aria-label="${P.playing ? "Pause" : "Play"}">${P.playing ? SVG.pause : SVG.play}</button>
    <button type="button" data-p="next" title="Next time step" aria-label="Next time step">${SVG.next}</button>
    <button type="button" data-p="last" title="Last time step" aria-label="Last time step">${SVG.last}</button>
    <input type="range" id="m-time" min="0" max="${Math.max(0, n - 1)}" value="${i}" ${n ? "" : "disabled"} aria-label="Time step">
    <div class="when" id="m-when">${n ? whenHtml(P.dFill.times[i]) : "<span>—</span>"}</div>`;
}
async function onPlayerClick(e) {
  const P = S.preview, b = e.target.closest("button[data-p]");
  if (!b || !P.dFill) return;
  const n = P.dFill.times.length, i = curIdx();
  const go = (j) => { P.time = Math.max(0, Math.min(n - 1, j)); loadMap(); };
  switch (b.dataset.p) {
    case "first": return go(0);
    case "prev": return go(i - 1);
    case "next": return go(i + 1);
    case "last": return go(n - 1);
    case "play":
      P.playing = !P.playing;
      renderPlayer(true);
      while (P.playing && S.preview === P) {
        const nn = P.dFill.times.length;
        P.time = (curIdx() + 1) % nn;
        await loadMap({ quiet: true });
        if (P.error) { P.playing = false; break; }
        await sleep(600);
      }
      renderPlayer(true);
      return;
    default:
  }
}

// ------------------------------------------------------------------ data
async function fetchSlice(v, varName_, level) {
  const P = S.preview;
  const file = fileFor(v.outputs, varName_, P.member, P.start);
  if (!file) throw new Error(`no ${varName_} file for member ${P.member} and start date ${P.start}`);
  const q = new URLSearchParams({ exp: v.expid, var: varName_, member: P.member, file, time: P.time });
  if (level != null) q.set("level", level);
  return api(`/api/preview?${q}`);
}
async function loadMap(opts = {}) {
  const P = S.preview, v = S.detail;
  if (!P.start || !v) return;
  if (P.loading) { P.again = true; return; }
  P.loading = true; P.error = null;
  const frame = $("#m-frame");
  if (!opts.quiet) { frame?.classList.add("loading"); const t = $("#m-loadtxt"); if (t) t.textContent = `Reading ${varName(P.fill.var)} on ${v.remote.host.split(".")[0]}…`; }
  try {
    const ensQ = (vr, lev) => { const q = new URLSearchParams({ exp: v.expid, var: vr, start: P.start, time: P.time }); if (lev != null) q.set("level", lev); return api(`/api/ens?${q}`); };
    const wantEns = P.field !== "member";
    const [f, c, e, ce] = await Promise.all([
      fetchSlice(v, P.fill.var, P.fill.level),
      P.cont.var ? fetchSlice(v, P.cont.var, P.cont.level) : Promise.resolve(null),
      wantEns ? ensQ(P.fill.var, P.fill.level) : Promise.resolve(null),
      wantEns && P.cont.var ? ensQ(P.cont.var, P.cont.level) : Promise.resolve(null)]);
    if (S.preview !== P) return;
    P.dFill = f; P.dCont = c; P.dEns = e; P.dContEns = ce;
  } catch (e) { P.error = e.message; }
  P.loading = false;
  frame?.classList.remove("loading");
  if (S.preview !== P) return;
  P.psig = null;
  renderPanel(); renderPlayer();
  await drawMap();
  if (S.tab === "data" && typeof renderData === "function") renderData(S.detail);
  if (P.again) { P.again = false; loadMap(); }
}

// ------------------------------------------------------------------ drawing
const lockKey = () => { const P = S.preview; return `${fkey()}|${P.fill.level}|${vstyle(fkey()).tr}`; };
function contourLevels(lo, hi) {
  const cs = S.preview.st.cont;
  if (lo == null || hi == null || !isFinite(lo) || !isFinite(hi) || hi <= lo) return null;
  const a = cs.from !== "" && isFinite(+cs.from) ? +cs.from : lo;
  const b = cs.to !== "" && isFinite(+cs.to) ? +cs.to : hi;
  if (b <= a) return null;
  const step = cs.mode === "step" && +cs.step > 0 ? +cs.step : stepForCount(a, b, cs.n);
  const start = Math.ceil(a / step) * step, end = Math.floor(b / step) * step;
  const count = Math.max(0, Math.round((end - start) / step) + 1);
  if (count > 400) return null;
  return { step, start, end, count };
}

async function drawMap() {
  const P = S.preview, el = $("#p-map");
  if (!el) return;
  const fs = vstyle(fkey()), cs = P.st.cont, ms = P.st.map;
  if (P.error) {
    Plotly.purge(el);
    el.innerHTML = `<div class="map-error">Could not read the data: ${esc(P.error)}</div>`;
    return;
  }
  if (!P.dFill) return;
  if (el.querySelector(".map-error")) el.innerHTML = "";
  const d = P.dFill, c = P.dCont;
  const ft = TRANSFORMS[fs.tr] || TRANSFORMS.none, ct = TRANSFORMS[ctr(P.cont.var)] || TRANSFORMS.none;
  const unitOf = (dd, t) => t.units(dd.units && dd.units !== "unknown" ? dd.units : "");
  const tz = (z, f) => z.map((row) => row.map((x) => (x == null ? null : f(x))));
  const fd = fieldData(), ff = fieldTf(ft);
  const zf = tz(fd.z, ff), fu = unitOf(d, ft);
  const cz = c ? (P.field !== "member" && P.dContEns ? P.dContEns.mean : c.z) : null;
  const zc = c ? tz(cz, ct.f) : null, cu = c ? unitOf(c, ct) : "";

  // colour range: manual > locked > this frame; optionally symmetric around zero
  let [lo, hi] = [ff(fd.lo), ff(fd.hi)];
  if (fs.zmin === "" && fs.zmax === "" && ms.lock) {
    const k = lockKey();
    if (P.locked[k]) [lo, hi] = P.locked[k]; else P.locked[k] = [lo, hi];
  }
  if (fs.sym) { const a = Math.max(Math.abs(lo), Math.abs(hi)); lo = -a; hi = a; }
  if (fs.zmin !== "" && isFinite(+fs.zmin)) lo = +fs.zmin;
  if (fs.zmax !== "" && isFinite(+fs.zmax)) hi = +fs.zmax;

  // titles
  const t = d.times[d.time_index];
  const day = leadDay(P.start, t);
  const n = P.dEns?.n;
  const what = { member: "", mean: " · ensemble mean", spread: " · ensemble spread", anom: " · departure from the mean" }[P.dEns ? P.field : "member"];
  $("#m-title").textContent = `${varName(d.var)}${d.level != null ? ` · ${d.level} hPa` : ""}${what}`;
  $("#m-sub").innerHTML = [`${esc(fmtValid(t))}`, day != null ? `day ${day}` : "",
    P.field === "mean" || P.field === "spread" ? `${n} members` : P.field === "anom" ? `member ${esc(d.member)} − mean of ${n}` : `member ${esc(d.member)}`,
    c ? `lines: ${esc(varName(c.var))}${c.level != null ? ` ${c.level} hPa` : ""}${P.field !== "member" && P.dContEns ? " (ensemble mean)" : ""}${cu ? ` (${esc(cu)})` : ""}` : ""]
    .filter(Boolean).join('<span class="dot-sep">·</span>');

  const sameGrid = c && c.lat.length === d.lat.length && c.lon.length === d.lon.length;
  const traces = [{
    type: "heatmap", x: d.lon, y: d.lat, z: zf, colorscale: cscale(fs.cmap, fs.rev), zsmooth: ms.smooth ? "best" : false,
    zauto: false, zmin: lo, zmax: hi,
    colorbar: { thickness: 12, len: 0.92, outlinewidth: 0, ticks: "outside", ticklen: 3, tickfont: { color: css("--text-secondary"), size: 11 },
      title: { text: fu, side: "right", font: { size: 12 } } },
    customdata: sameGrid ? zc : undefined,
    hovertemplate: `<b>%{z:.1f}${fu ? " " + esc(fu) : ""}</b> ${esc(varName(d.var))}${sameGrid ? `<br>%{customdata:.1f}${cu ? " " + esc(cu) : ""} ${esc(varName(c.var))}` : ""}<br>${"%{y:.1f}°, %{x:.1f}°"}<extra></extra>`,
  }];
  if (c) {
    const crng = P.field !== "member" && P.dContEns ? P.dContEns.mean_range : [c.min, c.max];
    const lv = contourLevels(ct.f(crng[0]), ct.f(crng[1]));
    if (lv) {
      const color = cs.color === "auto" ? css("--text-primary") : cs.color;
      const mk = (start, end, dash) => ({
        type: "contour", x: c.lon, y: c.lat, z: zc, showscale: false, hoverinfo: "skip", autocontour: false,
        contours: { start, end, size: lv.step, coloring: "lines", showlabels: !!cs.labels,
          labelfont: { size: 10, color }, labelformat: `.${decimalsOf(lv.step)}f` },
        line: { color, width: cs.width, dash, smoothing: cs.smooth ? 1.1 : 0 },
      });
      if (cs.dashNeg && lv.start < 0 && lv.end > 0) {
        traces.push(mk(lv.start, -lv.step, "dash"));  // levels are multiples of step, so -step is the last negative one
        traces.push(mk(0, lv.end, "solid"));
      } else traces.push(mk(lv.start, lv.end, cs.dashNeg && lv.end < 0 ? "dash" : "solid"));
    }
    const ci = $("#c-info");
    if (ci && lv) ci.innerHTML = `A line every <b>${fmtLevel(lv.step, lv.step)}${cu ? " " + esc(cu) : ""}</b>, from ${fmtLevel(lv.start, lv.step)} to ${fmtLevel(lv.end, lv.step)} (${lv.count} lines).`;
  }
  if (ms.coast) {
    const coast = await coastlines();
    if (coast) traces.push({ type: "scatter", mode: "lines", x: coast.x, y: coast.y, hoverinfo: "skip", showlegend: false,
      line: { color: ms.coastColor === "auto" ? css("--coast") : ms.coastColor, width: 0.9 } });
  }
  const sel = P.sel, shapes = [];
  if (sel?.kind === "point") traces.push({ type: "scatter", mode: "markers", x: [sel.lon], y: [sel.lat], hoverinfo: "skip", showlegend: false,
    marker: { symbol: "circle", size: 13, color: "rgba(0,0,0,0)", line: { width: 3, color: "#ffffff" } } },
    { type: "scatter", mode: "markers", x: [sel.lon], y: [sel.lat], hoverinfo: "skip", showlegend: false,
      marker: { symbol: "circle", size: 7, color: css("--st-failed") } });
  if (sel?.kind === "area") {
    const [a0, a1, b0, b1] = sel.box;
    const rect = (x0r, x1r) => ({ type: "rect", xref: "x", yref: "y", x0: x0r, x1: x1r, y0: b0, y1: b1,
      line: { color: "#ffffff", width: 2.5 }, fillcolor: "rgba(255,255,255,0.12)" });
    if (a0 <= a1) shapes.push(rect(a0, a1)); else shapes.push(rect(a0, 180), rect(-180, a1));  // across the dateline
    shapes.slice().forEach((r) => shapes.push({ ...r, line: { color: "#111111", width: 1, dash: "dot" }, fillcolor: "rgba(0,0,0,0)" }));
  }

  const [x0, x1, y0, y1] = REGIONS[ms.region] || REGIONS.Global;
  const m = { l: 46, r: 80, t: 10, b: 34 };
  const w = el.clientWidth || 900;
  const h = Math.round(Math.min(720, Math.max(260, (w - m.l - m.r) * ((y1 - y0) / (x1 - x0)) + m.t + m.b)));
  const span = Math.max(x1 - x0, y1 - y0), dt = span > 200 ? 30 : span > 60 ? 10 : 5;
  const axis = (range, suffix) => ({ range, showgrid: ms.grid, gridcolor: "rgba(127,127,127,.45)", griddash: "dot", dtick: dt,
    layer: "above traces", ticksuffix: suffix, zeroline: false, showline: false, tickfont: { size: 11 } });
  Plotly.react(el, traces, baseLayout({
    height: h, showlegend: false, margin: m, uirevision: `${ms.region}|${P.viewRev}`, dragmode: "zoom", shapes,
    paper_bgcolor: css("--surface-1"), plot_bgcolor: css("--map-ocean"),
    xaxis: axis([x0, x1], "°"), yaxis: { ...axis([y0, y1], "°"), scaleanchor: "x" },
  }), { ...PCFG, displayModeBar: "hover", modeBarButtonsToRemove: [...PCFG.modeBarButtonsToRemove, "toImage"] });
  P.drawn = true;
  onClick(el, (ev) => {
    if (P.drawing) return;
    const pt = ev.points.find((p) => p.data.type === "heatmap") || ev.points[0];
    if (pt && isFinite(+pt.x) && isFinite(+pt.y)) setPoint(+pt.y, +pt.x, null);
  });
  el.removeAllListeners?.("plotly_relayout");
  el.on("plotly_relayout", (ev) => {  // in "draw a box" mode the zoom rectangle becomes the averaging area
    if (!P.drawing) return;
    const xr = [ev["xaxis.range[0]"], ev["xaxis.range[1]"]], yr = [ev["yaxis.range[0]"], ev["yaxis.range[1]"]];
    if ([...xr, ...yr].some((q) => q == null)) return;
    P.drawing = false; P.viewRev++;
    const cl = (q, lim) => Math.max(-lim, Math.min(lim, q));
    setArea([cl(Math.min(...xr), 180), cl(Math.max(...xr), 180), cl(Math.min(...yr), 90), cl(Math.max(...yr), 90)].map((q) => Math.round(q * 10) / 10), "your box");
  });
  $("#m-hint").textContent = P.drawing ? "Drag a box over the area to average" : "Click anywhere for the time series";
  $("#m-frame").classList.toggle("drawing", P.drawing);
  drawSeries();
}

async function coastlines() {
  if (S.coast !== null) return S.coast;
  S.coast = false;
  try {
    let topo;
    try { topo = await (await fetch("/static/vendor/land-110m.json")).json(); }
    catch { topo = await (await fetch("https://cdn.jsdelivr.net/npm/world-atlas@2.0.2/land-110m.json")).json(); }
    const mesh = topojson.mesh(topo, topo.objects.land);
    const x = [], y = [];
    for (const line of mesh.coordinates) {
      for (let i = 0; i < line.length; i++) {
        if (i && Math.abs(line[i][0] - line[i - 1][0]) > 180) { x.push(null); y.push(null); }
        x.push(line[i][0]); y.push(line[i][1]);
      }
      x.push(null); y.push(null);
    }
    S.coast = { x, y };
  } catch { S.coast = false; }
  return S.coast;
}

function downloadMap() {
  const P = S.preview, el = $("#p-map");
  if (!P.dFill || !el.data) return;
  const t = (P.dFill.times[P.dFill.time_index] || "").slice(0, 10);
  Plotly.downloadImage(el, { format: "png", width: 1800, height: Math.round(1800 * (el.clientHeight / el.clientWidth)),
    filename: `${S.detail.expid}_${P.fill.var}${P.fill.level ?? ""}_${t}_m${P.member}${P.cont.var ? `_${P.cont.var}${P.cont.level ?? ""}` : ""}` });
}

// ------------------------------------------------------------------ time series: a point or an area average
const AREAS = { ...REGIONS, Arctic: [-180, 180, 66.5, 90] };
const fmtBox = ([a0, a1, b0, b1]) => `${fmtLat(b0)} – ${fmtLat(b1)}, ${fmtLon(a0)} – ${fmtLon(a1)}`;

function setPoint(lat, lon, name) {
  const P = S.preview;
  P.seriesKind = "point";
  P.sel = P.lastSel.point = { kind: "point", lat, lon, name };
  loadSeries();
}
function setArea(box, name) {
  const P = S.preview;
  P.seriesKind = "area";
  P.sel = P.lastSel.area = { kind: "area", box, name };
  loadSeries();
}

// Place search: Open-Meteo geocoding, called from the browser; only the typed text is sent.
let geoTimer = null, geoCtl = null;
function hideResults() { $("#m-results")?.classList.remove("open"); }
function showResults(list, msg) {
  const P = S.preview, box = $("#m-results");
  P.geo = list || [];
  box.innerHTML = msg ? `<div class="sr-msg">${esc(msg)}</div>` : P.geo.length
    ? P.geo.map((r, i) => `<button type="button" data-i="${i}" role="option"><b>${esc(r.name)}</b><span>${esc([r.admin1, r.country].filter(Boolean).join(", "))}${r.coords ? "" : ` · ${fmtLat(r.latitude)}, ${fmtLon(r.longitude)}`}</span></button>`).join("")
    : `<div class="sr-msg">No place found.</div>`;
  box.classList.add("open");
}
function onSearchInput(e) {
  const q = e.target.value.trim();
  clearTimeout(geoTimer);
  const m = q.match(/^(-?\d+(?:\.\d+)?)\s*[,;\s]\s*(-?\d+(?:\.\d+)?)$/);
  if (m && Math.abs(+m[1]) <= 90 && Math.abs(+m[2]) <= 180) {
    showResults([{ name: `${fmtLat(+m[1])}, ${fmtLon(+m[2])}`, latitude: +m[1], longitude: +m[2], coords: true, country: "coordinates" }]);
    return;
  }
  if (q.length < 2) { hideResults(); return; }
  geoTimer = setTimeout(async () => {
    geoCtl?.abort(); geoCtl = new AbortController();
    try {
      const r = await fetch(`https://geocoding-api.open-meteo.com/v1/search?name=${encodeURIComponent(q)}&count=6&language=en&format=json`, { signal: geoCtl.signal });
      if (!r.ok) throw new Error(String(r.status));
      showResults((await r.json()).results || []);
    } catch (err) {
      if (err.name !== "AbortError") showResults(null, "Place search is unavailable. Type coordinates (e.g. 45.07, 7.69) or click the map.");
    }
  }, 300);
}
function pickPlace(r) {
  hideResults();
  const name = r.coords ? null : [r.name, r.country].filter(Boolean).join(", ");
  if (!r.coords) $("#m-search").value = r.name;
  setPoint(r.latitude, r.longitude, name);
}

async function loadSeries() {
  const P = S.preview, v = S.detail, sel = P.sel;
  if (!sel || !P.start) return;
  P.seriesLoading = true; P.seriesErr = null;
  drawMap(); drawSeries();
  const q = new URLSearchParams({ exp: v.expid, var: P.fill.var, start: P.start });
  if (P.fill.level != null) q.set("level", P.fill.level);
  if (sel.kind === "point") { q.set("lat", sel.lat); q.set("lon", sel.lon); }
  else { const [a0, a1, b0, b1] = sel.box; q.set("lon0", a0); q.set("lon1", a1); q.set("lat0", b0); q.set("lat1", b1); }
  const key = q.toString(); P.seriesKey = key;
  let res = null, err = null;
  try { res = await api(`/api/series?${q}`); } catch (e) { err = e.message; }
  if (S.preview !== P || P.seriesKey !== key) return;  // a newer request superseded this one
  P.series = res; P.seriesErr = err; P.seriesLoading = false;
  if (res) { res.level = P.fill.level; res.sel = sel; res.var = P.fill.var; }
  drawSeries();
  $("#m-series")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function onSeriesClick(e) {
  const P = S.preview;
  const b = e.target.closest("button");
  if (!b) return;
  if (b.dataset.smode) { P.st.seriesMode = b.dataset.smode; saveStyle(); drawSeries(); return; }
  if (b.dataset.kind) {
    P.seriesKind = b.dataset.kind; P.drawing = false;
    P.sel = P.lastSel[P.seriesKind];
    P.series = null;
    if (P.sel) loadSeries(); else drawMap();
    return;
  }
  if (b.dataset.area) { setArea(AREAS[b.dataset.area], b.dataset.area); return; }
  if (b.dataset.act === "draw") { P.drawing = !P.drawing; drawMap(); return; }
  if (b.dataset.act === "view") {
    const fl = $("#p-map")._fullLayout;
    if (!fl) return;
    const [x0, x1] = fl.xaxis.range, [y0, y1] = fl.yaxis.range;
    const r = (q) => Math.round(q * 10) / 10;
    setArea([r(Math.max(-180, x0)), r(Math.min(180, x1)), r(Math.max(-90, y0)), r(Math.min(90, y1))], "the visible map");
    return;
  }
  if (b.id === "s-close") { P.sel = null; P.lastSel[P.seriesKind] = null; P.series = null; drawMap(); }
}

function seriesTop() {
  const P = S.preview;
  const kinds = seg("kind", [["point", "At a point"], ["area", "Area average"]], P.seriesKind).replace(/data-v=/g, "data-kind=");
  let tools;
  if (P.seriesKind === "point") tools = `<span class="s">Click the map, or use “Find a place” above it.</span>`;
  else {
    const curName = P.sel?.kind === "area" ? P.sel.name : null;
    tools = `<div class="chips-sel">${Object.keys(AREAS).map((a) => `<button type="button" data-area="${esc(a)}" aria-pressed="${a === curName}">${esc(a)}</button>`).join("")}</div>
      <button type="button" data-act="draw" class="${P.drawing ? "primary" : ""}">${P.drawing ? "Drag on the map… (click to cancel)" : "Draw a box"}</button>
      <button type="button" data-act="view">Use the visible map</button>`;
  }
  return `<div class="series-top"><span class="t">Time series</span>${kinds}${tools}</div>`;
}

function drawSeries() {
  const P = S.preview, box = $("#m-series");
  if (!box) return;
  const topSig = JSON.stringify([P.seriesKind, P.drawing, P.sel?.name, P.sel?.box]);
  if (box.dataset.top !== topSig || !$("#s-top")) {
    box.innerHTML = `<div id="s-top"></div><div id="s-body"></div>`;
    box.dataset.top = topSig;
    $("#s-top").innerHTML = seriesTop();
  }
  const body = $("#s-body");
  const sel = P.sel;
  if (!sel) {
    body.dataset.for = "";
    body.innerHTML = `<div class="series-empty">${SVG.pin}<span>${P.seriesKind === "point"
      ? "Pick a point to see how it evolves in every ensemble member."
      : "Pick a region, draw a box on the map or use the visible map area: you get the area-weighted mean of every member."}</span></div>`;
    return;
  }
  const where = sel.kind === "point"
    ? (sel.name ? `in ${esc(sel.name)}` : `at ${fmtLat(P.series?.lat ?? sel.lat)}, ${fmtLon(P.series?.lon ?? sel.lon)}`)
    : `averaged over ${esc(sel.name || "the area")}`;
  const head = (sub) => `<div class="series-head"><span class="t">${esc(varName(P.fill.var))} ${where}</span>
      <span class="s">${sub}</span><span class="spacer"></span>
      ${seg("smode-wrap", [["members", "Members"], ["spread", "Range"], ["boxes", "Boxes"]], P.st.seriesMode).replace(/data-v=/g, "data-smode=")}
      <button type="button" class="ghost" id="s-close" aria-label="Clear the selection">✕</button></div>`;
  if (P.seriesLoading) { body.dataset.for = ""; body.innerHTML = head("loading…") + `<div class="series-empty"><div class="spinner"></div>Reading every member on the HPC…</div>`; return; }
  if (P.seriesErr) { body.dataset.for = ""; body.innerHTML = head("") + `<div class="map-error" style="padding:12px">Could not read the time series: ${esc(P.seriesErr)}</div>`; return; }
  if (!P.series) { body.innerHTML = ""; return; }
  const s = P.series, t = TRANSFORMS[vstyle(s.var).tr] || TRANSFORMS.none;
  const mems = Object.keys(s.members).filter((m) => Array.isArray(s.members[m])).sort(sortMem);
  const vals = mems.map((m) => s.members[m].map((x) => (x == null ? null : t.f(x))));
  const col = (i) => vals.map((r) => r[i]).filter((x) => x != null);
  const mean = s.times.map((_, i) => { const xs = col(i); return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null; });
  const u = t.units(s.units && s.units !== "unknown" ? s.units : "");
  const x = s.times.map((tt) => tt.replace(" ", "T"));
  const key = `${JSON.stringify(sel)}|${s.var}|${s.level}|${P.st.seriesMode}|${u}`;
  if (body.dataset.for !== key || !$("#p-series")) {
    const sub = [s.level != null ? `${s.level} hPa` : "", sel.kind === "area" ? `${fmtBox(s.box || sel.box)} · ${s.npoints} grid points, weighted by cos(latitude)` : "",
      `start ${fmtDate(P.start)}`, `${mems.length} member${mems.length === 1 ? "" : "s"}`, u].filter(Boolean).join(" · ");
    body.innerHTML = head(esc(sub)) + `<div class="plot" id="p-series"></div><div class="note">Click a day on the chart to show it on the map.</div>`;
    body.dataset.for = key;
  }
  const muted = css("--text-muted"), accent = css("--st-running");
  const traces = [];
  if (P.st.seriesMode === "spread" && mems.length > 1) {
    const lo = s.times.map((_, i) => Math.min(...col(i))), hi = s.times.map((_, i) => Math.max(...col(i)));
    traces.push({ type: "scatter", mode: "lines", x, y: hi, line: { width: 0 }, hoverinfo: "skip", showlegend: false },
      { type: "scatter", mode: "lines", x, y: lo, line: { width: 0 }, fill: "tonexty", fillcolor: css("--accent-soft"), name: "range of the members", hovertemplate: "range %{y:.1f}<extra></extra>" });
  } else if (P.st.seriesMode === "boxes" && mems.length > 1) {
    const bx = [], by = [];
    x.forEach((xx, i) => vals.forEach((r) => { if (r[i] != null) { bx.push(xx); by.push(r[i]); } }));
    traces.push({ type: "box", x: bx, y: by, name: "members", boxpoints: false, whiskerwidth: 0.6,
      marker: { color: accent }, line: { color: accent, width: 1.2 }, fillcolor: css("--accent-soft"), hoverinfo: "skip" });
  } else {
    mems.forEach((m, k) => { if (m !== P.member) traces.push({ type: "scatter", mode: "lines", x, y: vals[k], name: !traces.length ? "other members" : `m${m}`,
      legendgroup: "others", showlegend: !traces.length, line: { color: muted, width: 1 }, opacity: 0.65, hovertemplate: `m${m}: %{y:.1f}<extra></extra>` }); });
  }
  if (mems.length > 1) traces.push({ type: "scatter", mode: "lines", x, y: mean, name: "ensemble mean", line: { color: css("--text-primary"), width: 2.5 }, hovertemplate: "mean: %{y:.1f}<extra></extra>" });
  if (mems.includes(P.member) && P.st.seriesMode !== "boxes") traces.push({ type: "scatter", mode: "lines+markers", x, y: vals[mems.indexOf(P.member)], name: `member ${P.member} (on the map)`,
    line: { color: accent, width: 2.5 }, marker: { size: 5, color: accent }, hovertemplate: `m${P.member}: %{y:.1f}<extra></extra>` });
  const cur = P.dFill?.times?.[P.dFill.time_index]?.replace(" ", "T");
  Plotly.react($("#p-series"), traces, baseLayout({
    height: 280, margin: { l: 52, r: 16, t: 6, b: 36 }, hovermode: "x unified",
    legend: { orientation: "h", y: 1.04, yanchor: "bottom", x: 0 },
    xaxis: { type: "date", gridcolor: css("--grid"), linecolor: css("--axis") },
    yaxis: { gridcolor: css("--grid"), title: { text: u } },
    shapes: cur ? [{ type: "line", xref: "x", yref: "paper", x0: cur, x1: cur, y0: 0, y1: 1, line: { color: css("--axis"), width: 2, dash: "dot" } }] : [],
  }), { ...PCFG, displayModeBar: false });
  onClick($("#p-series"), (ev) => {
    const i = ev.points[0]?.pointIndex;
    if (i != null && S.preview.dFill) { S.preview.time = i; loadMap(); }
  });
}
