/* AIUQ Monitor — "Data" tab analyses: spread growth, Hovmöller, across start dates.
   They follow the variable, level and start date chosen in the map panel. Uses helpers from app.js / map.js / hpc.js. */
"use strict";

const BANDS = { "Northern mid-latitudes": [30, 60], Tropics: [-15, 15], "Southern mid-latitudes": [-60, -30], Arctic: [60, 90], Antarctic: [-90, -60] };
const HC_AREAS = ["Global", "Europe", "North Atlantic", "Mediterranean", "Italy", "Northern Hemisphere", "Tropics", "Southern Hemisphere"];
const SEC_PER_FILE = 0.011;  // measured: ~12 s for 1080 files

function dataState() {
  if (!S.data) S.data = {
    spread: { region: "Global", key: null, data: null, loading: false, err: null, pinned: [] },
    hov: { band: "Northern mid-latitudes", member: "mean", key: null, data: null, loading: false, err: null },
    hind: { region: "Europe", leads: null, key: null, data: null, loading: false, err: null, ctx: null },
  };
  return S.data;
}
const ctxOf = () => { const P = S.preview; return P?.fill?.var && P.start ? { var: P.fill.var, level: P.fill.level, start: P.start, member: P.member } : null; };
const tfOf = (v) => TRANSFORMS[vstyle(v).tr] || TRANSFORMS.none;
const unitsOf = (v, raw) => tfOf(v).units(raw && raw !== "unknown" ? raw : "");
const levTxt = (l) => (l != null ? ` ${l} hPa` : "");
const setQ = (q, o) => { for (const [k, x] of Object.entries(o)) if (x != null) q.set(k, x); return q; };

function renderData(v) {
  dataState();
  const c = ctxOf();
  const none = !v.outputs.n;
  for (const id of ["#x-spread", "#x-hov", "#x-hind"]) {
    if (none) $(id).innerHTML = `<div class="empty">No outputs yet.</div>`;
    else if (!c && !$(id).dataset.built) $(id).innerHTML = `<div class="empty">Waiting for the map above to load…</div>`;
  }
  if (none || !c) return;
  renderSpread(v, c);
  renderHov(v, c);
  renderHind(v, c);
}

// ------------------------------------------------------------------ spread growth
function renderSpread(v, c) {
  const D = S.data.spread, box = $("#x-spread");
  if (!box.dataset.built) {
    box.innerHTML = `<div class="ana-bar"><div class="chips-sel" id="sp-areas"></div><span class="spacer"></span>
        <button type="button" id="sp-keep">Keep this curve</button><button type="button" class="ghost" id="sp-clear">Clear kept curves</button></div>
      <div class="plot" id="p-spread"></div><div class="note" id="sp-note"></div>`;
    box.dataset.built = "1";
    $("#sp-areas").addEventListener("click", (e) => { const b = e.target.closest("button[data-v]"); if (b) { D.region = b.dataset.v; renderSpread(S.detail, ctxOf()); } });
    $("#sp-keep").onclick = () => {
      if (!D.data) return;
      const label = `${varName(D.ctx.var)}${levTxt(D.ctx.level)} · ${D.ctx.region} · ${fmtDate(D.ctx.start)}`;
      if (!D.pinned.some((p) => p.label === label)) D.pinned.push({ label, data: D.data, ctx: D.ctx });
      if (D.pinned.length > 6) D.pinned.shift();
      drawSpread();
    };
    $("#sp-clear").onclick = () => { D.pinned = []; drawSpread(); };
  }
  $("#sp-areas").innerHTML = AREA_NAMES().map((a) => `<button type="button" data-v="${esc(a)}" aria-pressed="${a === D.region}">${esc(a)}</button>`).join("");
  $("#sp-sub").textContent = `${varName(c.var)}${levTxt(c.level)} · start ${fmtDate(c.start)} · change these in the map panel`;
  const key = JSON.stringify([v.expid, c.var, c.level, c.start, D.region]);
  if (D.key !== key) {
    D.key = key; D.loading = true; D.err = null;
    drawSpread();
    const [a0, a1, b0, b1] = AREAS[D.region];
    const q = setQ(new URLSearchParams({ exp: v.expid, var: c.var, start: c.start, lon0: a0, lon1: a1, lat0: b0, lat1: b1 }), { level: c.level });
    api(`/api/spread?${q}`).then((d) => { if (D.key !== key) return; D.data = d; D.ctx = { ...c, region: D.region }; })
      .catch((e) => { if (D.key === key) D.err = e.message; })
      .finally(() => { if (D.key === key) { D.loading = false; drawSpread(); } });
    return;
  }
  drawSpread();
}

function drawSpread() {
  const D = S.data?.spread, el = $("#p-spread");   // S.data is null when another experiment was opened while this was loading
  if (!D || !el) return;
  if (D.err) { Plotly.purge(el); el.innerHTML = `<div class="map-error">${esc(D.err)}</div>`; return; }
  if (!D.data) { el.innerHTML = `<div class="series-empty"><div class="spinner"></div>Computing the spread on the HPC…</div>`; return; }
  if (el.querySelector(".map-error, .series-empty")) el.innerHTML = "";
  const curve = (d, ctx) => {
    const t = tfOf(ctx.var), f = (x) => t.f(x) - t.f(0);
    return { x: d.times.map((tt) => leadDay(ctx.start, tt)), y: d.spread.map(f), u: unitsOf(ctx.var, "") };
  };
  const cur = curve(D.data, D.ctx);
  const traces = D.pinned.map((p, i) => {
    const c = curve(p.data, p.ctx);
    return { type: "scatter", mode: "lines", name: p.label, x: c.x, y: c.y, line: { color: catColor(i + 1), width: 1.8, dash: "dot" }, hovertemplate: `${esc(p.label)}: %{y:.2f}<extra></extra>` };
  });
  traces.push({ type: "scatter", mode: "lines+markers", name: `${varName(D.ctx.var)}${levTxt(D.ctx.level)} · ${D.ctx.region} (current)`, x: cur.x, y: cur.y,
    line: { color: css("--st-running"), width: 3 }, marker: { size: 5 }, hovertemplate: "day %{x}: %{y:.2f}<extra>current</extra>" });
  Plotly.react(el, traces, baseLayout({ height: 320, margin: { l: 56, r: 16, t: 8, b: 44 }, hovermode: "x unified",
    legend: { orientation: "h", y: -0.22, x: 0, font: { size: 11 } },
    xaxis: { title: { text: "forecast day" }, gridcolor: css("--grid"), rangemode: "tozero" },
    yaxis: { title: { text: `spread${cur.u ? ` (${cur.u})` : ""}` }, gridcolor: css("--grid"), rangemode: "tozero" } }), { ...PCFG, displayModeBar: false });
  const half = cur.y.findIndex((y) => y >= 0.5 * Math.max(...cur.y));
  $("#sp-note").innerHTML = `${D.data.n} members · ${D.data.npoints} grid points · the spread reaches half of its maximum on <b>day ${cur.x[half] ?? "—"}</b>, and ${fmtNum(cur.y[cur.y.length - 1])}${cur.u ? " " + esc(cur.u) : ""} by the last day.`;
}
const AREA_NAMES = () => Object.keys(AREAS);

// ------------------------------------------------------------------ Hovmöller
function renderHov(v, c) {
  const D = S.data.hov, box = $("#x-hov");
  if (!box.dataset.built) {
    box.innerHTML = `<div class="ana-bar"><div class="chips-sel" id="hv-bands"></div><span class="spacer"></span><div class="seg" id="hv-mem" style="width:260px"></div></div>
      <div class="plot" id="p-hov"></div>`;
    box.dataset.built = "1";
    $("#hv-bands").addEventListener("click", (e) => { const b = e.target.closest("button[data-v]"); if (b) { D.band = b.dataset.v; renderHov(S.detail, ctxOf()); } });
    $("#hv-mem").addEventListener("click", (e) => { const b = e.target.closest("button[data-v]"); if (b) { D.member = b.dataset.v; renderHov(S.detail, ctxOf()); } });
  }
  $("#hv-bands").innerHTML = Object.entries(BANDS).map(([k, [a, b]]) => `<button type="button" data-v="${esc(k)}" aria-pressed="${k === D.band}" title="${fmtLat(a)} – ${fmtLat(b)}">${esc(k)}</button>`).join("");
  const memOpt = D.member === "mean" ? "mean" : c.member;
  if (D.member !== "mean") D.member = c.member;
  $("#hv-mem").innerHTML = [["mean", "Ensemble mean"], [c.member, `Member ${c.member}`]].map(([k, t]) => `<button type="button" data-v="${esc(k)}" aria-pressed="${k === memOpt}">${esc(t)}</button>`).join("");
  const [a, b] = BANDS[D.band];
  $("#hv-sub").textContent = `${varName(c.var)}${levTxt(c.level)} · ${fmtLat(a)} – ${fmtLat(b)} · start ${fmtDate(c.start)}`;
  const key = JSON.stringify([v.expid, c.var, c.level, c.start, D.band, D.member]);
  if (D.key !== key) {
    D.key = key; D.loading = true; D.err = null;
    drawHov(c);
    const q = setQ(new URLSearchParams({ exp: v.expid, var: c.var, start: c.start, lat0: a, lat1: b, member: D.member }), { level: c.level });
    api(`/api/hovmoller?${q}`).then((d) => { if (D.key === key) { D.data = d; D.ctx = c; } })
      .catch((e) => { if (D.key === key) D.err = e.message; })
      .finally(() => { if (D.key === key) { D.loading = false; drawHov(c); } });
    return;
  }
  drawHov(c);
}

function drawHov(c) {
  const D = S.data?.hov, el = $("#p-hov");
  if (!D || !el) return;
  if (D.err) { Plotly.purge(el); el.innerHTML = `<div class="map-error">${esc(D.err)}</div>`; return; }
  if (!D.data || D.loading) { if (!D.data) el.innerHTML = `<div class="series-empty"><div class="spinner"></div>Computing on the HPC…</div>`; if (!D.data) return; }
  if (el.querySelector(".map-error, .series-empty")) el.innerHTML = "";
  const d = D.data, st = vstyle(D.ctx.var), t = tfOf(D.ctx.var);
  const z = d.z.map((r) => r.map((x) => (x == null ? null : t.f(x))));
  const u = unitsOf(D.ctx.var, "");
  Plotly.react(el, [{ type: "heatmap", x: d.lon, y: d.times.map((tt) => tt.replace(" ", "T")), z, colorscale: cscale(st.cmap, st.rev), zsmooth: "best",
    colorbar: { thickness: 12, outlinewidth: 0, title: { text: u, side: "right" } },
    hovertemplate: `%{y|%d %b %Y}<br>lon %{x:.0f}°<br><b>%{z:.1f}${u ? " " + esc(u) : ""}</b><extra></extra>` }],
  baseLayout({ height: Math.max(340, d.times.length * 13 + 80), margin: { l: 80, r: 10, t: 8, b: 40 },
    xaxis: { title: { text: "longitude" }, ticksuffix: "°", dtick: 60, gridcolor: "rgba(0,0,0,0)" },
    yaxis: { type: "date", autorange: "reversed", gridcolor: "rgba(0,0,0,0)" } }), { ...PCFG, displayModeBar: false });
}

// ------------------------------------------------------------------ across start dates
function renderHind(v, c) {
  const D = S.data.hind, box = $("#x-hind");
  const files = v.outputs.files.filter((f) => f[0] === c.var && parseOut(f[2])?.var === c.var);
  const starts = [...new Set(files.map((f) => parseOut(f[2]).start))].sort();
  if (starts.length < 2) {
    box.innerHTML = `<div class="empty">This experiment has a single start date, so there is nothing to compare across start dates.</div>`;
    box.dataset.built = "";
    return;
  }
  const times = S.preview.dFill?.times || [];
  if (!times.length) return;
  if (!D.leads) D.leads = [...new Set([0, Math.floor((times.length - 1) / 2), times.length - 1])];
  if (!box.dataset.built) {
    box.innerHTML = `<div class="ana-bar"><span class="note">Region</span><div class="chips-sel" id="hc-areas"></div></div>
      <div class="ana-bar"><span class="note">Forecast days</span><div class="chips-sel" id="hc-leads"></div></div>
      <div class="ana-bar"><button type="button" class="primary" id="hc-run"></button><span class="note" id="hc-note"></span></div>
      <div class="two-col"><div><div class="chart-title" id="hc-t1"></div><div class="plot" id="p-hc1"></div></div>
        <div><div class="chart-title" id="hc-t2"></div><div class="plot" id="p-hc2"></div></div></div>`;
    box.dataset.built = "1";
    $("#hc-areas").addEventListener("click", (e) => { const b = e.target.closest("button[data-v]"); if (b) { D.region = b.dataset.v; renderHind(S.detail, ctxOf()); } });
    $("#hc-leads").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-v]"); if (!b) return;
      const k = +b.dataset.v;
      D.leads = D.leads.includes(k) ? D.leads.filter((x) => x !== k) : [...D.leads, k].sort((x, y) => x - y).slice(-8);
      if (!D.leads.length) D.leads = [k];
      renderHind(S.detail, ctxOf());
    });
    $("#hc-run").onclick = () => runHind(S.detail, ctxOf());
  }
  $("#hc-areas").innerHTML = HC_AREAS.map((a) => `<button type="button" data-v="${esc(a)}" aria-pressed="${a === D.region}">${esc(a)}</button>`).join("");
  $("#hc-leads").innerHTML = times.map((tt, i) => [i, leadDay(c.start, tt)]).filter(([i]) => i === 0 || i === times.length - 1 || i % 5 === 4)
    .map(([i, d]) => `<button type="button" data-v="${i}" aria-pressed="${D.leads.includes(i)}">day ${d}</button>`).join("");
  const nf = files.length;
  $("#hc-sub").textContent = `${varName(c.var)}${levTxt(c.level)} · ${starts.length} start dates × ${new Set(files.map((f) => f[1])).size} members`;
  $("#hc-run").textContent = D.loading ? "Computing…" : "Compute";
  $("#hc-run").disabled = D.loading;
  const key = JSON.stringify([v.expid, c.var, c.level, D.region, D.leads]);
  $("#hc-note").innerHTML = D.loading ? `<span class="spinner" style="display:inline-block;width:14px;height:14px;vertical-align:middle"></span> reading ${nf} files on the HPC…`
    : D.key === key && D.data ? `Up to date · ${nf} files read` : `Reads ${nf} files on the HPC (about ${fmtDur(Math.max(2, nf * SEC_PER_FILE))}).`;
  drawHind();
}

async function runHind(v, c) {
  const D = S.data.hind;
  if (D.loading) return;
  const [a0, a1, b0, b1] = AREAS[D.region];
  const key = JSON.stringify([v.expid, c.var, c.level, D.region, D.leads]);
  D.loading = true; D.err = null;
  renderHind(v, c);
  const q = setQ(new URLSearchParams({ exp: v.expid, var: c.var, lon0: a0, lon1: a1, lat0: b0, lat1: b1, tidx: D.leads.join(",") }), { level: c.level });
  try { D.data = await api(`/api/hindcast?${q}`); D.key = key; D.ctx = { ...c, region: D.region, times: S.preview.dFill.times }; }
  catch (e) { D.err = e.message; }
  D.loading = false;
  if (S.data?.hind !== D) return;   // another experiment was opened meanwhile
  renderHind(S.detail, ctxOf());
}

function drawHind() {
  const D = S.data.hind, e1 = $("#p-hc1"), e2 = $("#p-hc2");
  if (!e1) return;
  if (D.err) { e1.innerHTML = `<div class="map-error">${esc(D.err)}</div>`; return; }
  if (!D.data) {
    $("#hc-t1").textContent = ""; $("#hc-t2").textContent = "";
    e1.innerHTML = `<div class="series-empty">${SVG.pin}<span>Choose a region and some forecast days, then press <b>Compute</b>.</span></div>`;
    return;
  }
  if (e1.querySelector(".series-empty, .map-error")) e1.innerHTML = "";
  const d = D.data, ctx = D.ctx, t = tfOf(ctx.var), u = unitsOf(ctx.var, "");
  const leadOf = (ti) => leadDay(ctx.start, ctx.times[ti]);
  // ensemble mean (and range) per start date and lead
  const by = new Map();
  for (const [start, , vals] of d.rows) {
    if (!vals) continue;
    if (!by.has(start)) by.set(start, d.tidx.map(() => []));
    vals.forEach((x, k) => { if (x != null) by.get(start)[k].push(t.f(x)); });
  }
  const starts = [...by.keys()].sort();
  const mean = (a) => (a.length ? a.reduce((p, q) => p + q, 0) / a.length : null);
  const ramp = ["--seq-300", "--seq-400", "--seq-500", "--seq-600", "--seq-700"];
  const tr1 = d.tidx.map((ti, k) => {
    const col = css(ramp[Math.min(ramp.length - 1, Math.round((k / Math.max(1, d.tidx.length - 1)) * (ramp.length - 1)))]);
    return { type: "scatter", mode: "lines+markers", name: `day ${leadOf(ti)}`, x: starts, y: starts.map((s) => mean(by.get(s)[k])),
      customdata: starts.map((s) => { const a = by.get(s)[k]; return a.length ? [Math.min(...a), Math.max(...a)] : [null, null]; }),
      line: { color: col, width: 1.8 }, marker: { size: 4, color: col },
      hovertemplate: `day ${leadOf(ti)}: <b>%{y:.1f}</b> (members %{customdata[0]:.1f} … %{customdata[1]:.1f})<extra>%{x|%d %b %Y}</extra>` };
  });
  $("#hc-t1").textContent = `${varName(ctx.var)}${levTxt(ctx.level)} over ${ctx.region}, ensemble mean, by start date`;
  Plotly.react(e1, tr1, baseLayout({ height: 320, margin: { l: 52, r: 10, t: 8, b: 40 }, hovermode: "x unified",
    legend: { orientation: "h", y: -0.2, x: 0 }, xaxis: { type: "date", gridcolor: css("--grid") },
    yaxis: { gridcolor: css("--grid"), title: { text: u } } }), { ...PCFG, displayModeBar: false });
  // drift: change relative to the first selected day, averaged over start dates
  const k0 = 0;
  const diffs = d.tidx.map((_, k) => starts.map((s) => { const a = mean(by.get(s)[k]), b = mean(by.get(s)[k0]); return a != null && b != null ? a - b : null; }).filter((x) => x != null));
  const mu = diffs.map(mean), sd = diffs.map((a, k) => (a.length > 1 ? Math.sqrt(a.reduce((p, q) => p + (q - mu[k]) ** 2, 0) / (a.length - 1)) : 0));
  const xl = d.tidx.map(leadOf);
  $("#hc-t2").textContent = `Average change since day ${xl[0]} (drift), ± 1 standard deviation across start dates`;
  Plotly.react(e2, [
    { type: "scatter", mode: "lines", x: xl, y: mu.map((m, k) => m + sd[k]), line: { width: 0 }, hoverinfo: "skip", showlegend: false },
    { type: "scatter", mode: "lines", x: xl, y: mu.map((m, k) => m - sd[k]), line: { width: 0 }, fill: "tonexty", fillcolor: css("--accent-soft"), hoverinfo: "skip", showlegend: false },
    { type: "scatter", mode: "lines+markers", x: xl, y: mu, line: { color: css("--st-running"), width: 2.5 }, marker: { size: 7 },
      hovertemplate: `day %{x}: <b>%{y:+.2f}${u ? " " + esc(u) : ""}</b><extra>${starts.length} start dates</extra>` }],
  baseLayout({ height: 320, margin: { l: 52, r: 10, t: 8, b: 40 }, showlegend: false,
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: 0, y1: 0, line: { color: css("--axis"), width: 1, dash: "dot" } }],
    xaxis: { title: { text: "forecast day" }, gridcolor: css("--grid") }, yaxis: { gridcolor: css("--grid"), title: { text: u ? `change (${u})` : "change" } } }), { ...PCFG, displayModeBar: false });
}
