/* AIUQ Monitor — HPC performance section: where the time goes and what slows a run down.
   Loaded before app.js; uses its helpers at runtime. */
"use strict";

const CAT = {
  light: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
  dark: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"],
};
const isDark = () => getComputedStyle(document.documentElement).colorScheme === "dark";
const catColor = (i) => (isDark() ? CAT.dark : CAT.light)[i % 8];
const median = (xs) => { if (!xs?.length) return null; const s = [...xs].sort((a, b) => a - b), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const pctl = (xs, q) => { if (!xs?.length) return null; const s = [...xs].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(q * (s.length - 1)))]; };
const hours = (sec) => (sec == null ? "—" : sec >= 36000 ? `${Math.round(sec / 3600)} h` : sec >= 3600 ? `${(sec / 3600).toFixed(1)} h` : fmtDur(sec));
const ICONS = {
  warn: '<svg viewBox="0 0 24 24"><path d="M12 3l10 18H2z"/><path d="M12 10v5M12 18v.5"/></svg>',
  info: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7.5v.5"/></svg>',
  ok: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M8 12.5l3 3 5-6"/></svg>',
};

function gpuCount(v) {
  let n = 0;
  for (const nd of v.slurm?.nodes || []) {
    const m = String(nd.gres || "").match(/gpu(?::[^:(]+)?:(\d+)/);
    if (m) n = Math.max(n, +m[1]);
  }
  return n || (v.gpu?.gpus?.length ?? 0);
}

function findings(v) {
  const p = v.perf, out = [];
  if (!p) return out;
  const lim = p.limits || {}, jl = lim.jobs || {};
  const gpus = gpuCount(v);
  const total = p.compute || 0;
  const secs = v.sections.filter((s) => p.sections[s]?.attempts);
  for (const s of secs) {
    const r = p.sections[s], qm = median(r.queue), sm = median(r.sched), rm = median(r.run);
    const n = r.run.length;
    if (n >= 3 && qm != null && rm != null && qm > Math.max(300, rm)) {
      const plq = jl[s] || {}, par = conc(v, s).avg;
      let why = "The partition is busy or the request (GPU, wall-clock) is hard to schedule.";
      if (plq.gpu && gpus && par >= 0.7 * gpus) why = `They are waiting for one of the ${gpus} GPUs, which this run keeps busy (${par.toFixed(1)} jobs at once on average): <b>the GPUs are the limit</b>.`;
      else if (plq.gpu && gpus) why = `This run itself used only ${par.toFixed(1)} of the ${gpus} GPUs on average, so the GPUs were busy with other jobs: other experiments running at the same time or other users.`;
      else if (rm < 60) why = `They take seconds but sit in the same queue as the long GPU jobs. Running them on a lighter platform (e.g. a <span class="mono">ps</span> platform on the login node) or packing them with Autosubmit wrappers would remove this wait.`;
      out.push(["warn", `<b>${esc(secName(s))}</b> jobs wait longer in the SLURM queue than they run (typically ${fmtDur(qm)} queued vs ${fmtDur(rm)} running${plq.platform ? ` on ${esc(plq.platform)}` : ""}). ${why}`]);
    }
    if (n >= 3 && sm != null && sm > Math.max(300, 0.5 * (rm || 0))) {
      const pl = jl[s] || {};
      const caps = [lim.total_jobs ? `TOTALJOBS = ${lim.total_jobs}` : "", lim.max_waiting ? `MAXWAITINGJOBS = ${lim.max_waiting}` : "",
        pl.platform_total_jobs ? `${pl.platform} allows ${pl.platform_total_jobs} at a time` : ""].filter(Boolean).join(", ");
      out.push(["warn", `<b>${esc(secName(s))}</b> jobs are ready but wait typically ${fmtDur(sm)} before Autosubmit submits them. This usually means a job limit is reached${caps ? ` (${esc(caps)})` : ""} or the Autosubmit loop is slow.`]);
    }
    const pl = jl[s];
    if (pl && +pl.platform_total_jobs === 1 && r.attempts > 3) {
      out.push(["info", `<b>${esc(secName(s))}</b> runs on <b>${esc(pl.platform)}</b>, which allows one job at a time, so its ${r.attempts} jobs run one after another (${hours(r.compute)} in total).`]);
    }
    if (pl?.gpu && gpus && r.run.length >= 3) {
      const par = conc(v, s).avg;
      if (par >= 0.8 * gpus && !out.some((f) => f[1].includes("the GPUs are the limit") && f[1].includes(secName(s)))) out.push(["info", `<b>${esc(secName(s))}</b> keeps the GPUs busy (on average ${par.toFixed(1)} of ${gpus} at once while running). More GPUs would be the way to go faster.`]);
      else if (par < 0.6 * gpus && r.run.length >= 5 && !out.some((f) => f[0] === "warn" && f[1].includes(`<b>${esc(secName(s))}</b> jobs wait longer`))) out.push(["info", `<b>${esc(secName(s))}</b> uses on average ${par.toFixed(1)} of ${gpus} GPUs while it runs: the GPUs are not the limit.`]);
    }
  }
  if (p.span && p.gap_total > 0.1 * p.span) {
    const longest = p.gaps[0];
    out.push(["warn", `Nothing was running for <b>${hours(p.gap_total)}</b> in total (${p.gaps.length} pause${p.gaps.length === 1 ? "" : "s"}${longest ? `, the longest ${hours(longest[1] - longest[0])} from ${fmtTime(longest[0])}` : ""}). Was Autosubmit stopped, or waiting for MARS?`]);
  }
  const att = secs.reduce((a, s) => a + p.sections[s].attempts, 0), fa = secs.reduce((a, s) => a + p.sections[s].failed, 0);
  if (att && fa / att > 0.05) {
    const wasted = v.attempts.filter((a) => a.result === "FAILED" && a.start).reduce((x, a) => x + ((a.end || Date.now() / 1000) - a.start), 0);
    out.push(["warn", `${Math.round((100 * fa) / att)}% of job attempts failed (${fa} of ${att}), using ${hours(wasted)} of compute for nothing. See <a href="#s-fail" data-gotab="live">Problems</a>.`]);
  }
  if (total) {
    const top = secs.map((s) => [s, p.sections[s].compute]).sort((a, b) => b[1] - a[1])[0];
    if (top) out.push(["info", `Most of the compute time goes to <b>${esc(secName(top[0]))}</b>: ${Math.round((100 * top[1]) / total)}% of ${hours(total)}.`]);
  }
  if (!out.some((f) => f[0] === "warn")) out.unshift(["ok", "No obvious bottleneck in the data collected so far."]);
  return out.sort((a, b) => ["warn", "ok", "info"].indexOf(a[0]) - ["warn", "ok", "info"].indexOf(b[0]));
}

// average number of jobs of a step running at once, while at least one is running
const _concCache = new WeakMap();
function conc(v, sec) {
  let c = _concCache.get(v);
  if (!c) { c = {}; _concCache.set(v, c); }
  if (c[sec]) return c[sec];
  const iv = v.attempts.filter((a) => a.section === sec && a.start && a.end && a.end > a.start).map((a) => [a.start, a.end]).sort((a, b) => a[0] - b[0]);
  let covered = 0, cs = null, ce = null, tot = 0;
  for (const [s, e] of iv) {
    tot += e - s;
    if (cs === null || s > ce) { if (cs !== null) covered += ce - cs; cs = s; ce = e; } else ce = Math.max(ce, e);
  }
  if (cs !== null) covered += ce - cs;
  return (c[sec] = { avg: covered ? tot / covered : 0, total: tot });
}

function renderHpc(v) {
  const root = $("#x-hpc");
  if (!root) return;
  const p = v.perf;
  if (!p || !v.attempts.length) { root.innerHTML = `<div class="empty">No job has run yet, so there is nothing to measure.</div>`; return; }
  if (!$("#p-conc")) {
    root.innerHTML = `
      <div class="findings" id="h-find"></div>
      <div class="kpis small" id="h-kpis"></div>
      <div class="two-col">
        <div><div class="chart-title">Jobs running at the same time</div><div class="plot" id="p-conc"></div></div>
        <div><div class="chart-title">Where a typical job spends its time</div><div class="plot" id="p-phase"></div></div>
      </div>
      <div class="two-col" style="margin-top:12px">
        <div><div class="chart-title">Jobs finished over time <span class="note">· shaded: nothing running</span></div><div class="plot" id="p-thru"></div></div>
        <div><div class="chart-title" style="display:flex;align-items:center;gap:10px">GPU use on the node
          <span class="seg" id="h-range" style="margin-left:auto">${[["6", "6 h"], ["24", "24 h"], ["168", "7 days"]].map(([k, t]) => `<button type="button" data-h="${k}" aria-pressed="${String(S.hpcHours || 24) === k}">${t}</button>`).join("")}</span></div>
          <div class="plot" id="p-gpuh"></div></div>
      </div>
      <div class="chart-title" style="margin-top:16px">Per step</div>
      <div class="tbl-wrap" id="h-table"></div>`;
    $("#h-range").addEventListener("click", (e) => {
      const b = e.target.closest("button[data-h]"); if (!b) return;
      S.hpcHours = +b.dataset.h;
      $("#h-range").querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      drawGpuHistory(v);
    });
  }
  const gpus = gpuCount(v);
  $("#h-find").innerHTML = findings(v).map(([lvl, txt]) => `<div class="finding f-${lvl}">${ICONS[lvl]}<div>${txt}</div></div>`).join("");
  const att = v.sections.reduce((a, s) => a + (p.sections[s]?.attempts || 0), 0);
  const fa = v.sections.reduce((a, s) => a + (p.sections[s]?.failed || 0), 0);
  $("#h-kpis").innerHTML = `
    <div class="kpi"><div class="lbl">Jobs at once ${info("Average number of this experiment's jobs running simultaneously, counted only while at least one is running.")}</div><div class="val">${p.parallelism ? p.parallelism.toFixed(1) : "—"}</div><div class="sub">${gpus ? `${gpus} GPUs on the node` : ""}</div></div>
    <div class="kpi"><div class="lbl">Busy ${info("Share of the run's wall-clock time during which at least one job was running.")}</div><div class="val">${p.span ? Math.round((100 * p.busy) / p.span) : 0}%</div><div class="sub">of ${hours(p.span)}</div></div>
    <div class="kpi"><div class="lbl">Compute used</div><div class="val">${hours(p.compute)}</div><div class="sub">sum of all job run times</div></div>
    <div class="kpi"><div class="lbl">Paused</div><div class="val">${hours(p.gap_total)}</div><div class="sub">${p.gaps.length} period${p.gaps.length === 1 ? "" : "s"} with nothing running (&gt; 15 min)</div></div>
    <div class="kpi ${fa ? "alert" : ""}"><div class="lbl">Failed attempts</div><div class="val">${fa}</div><div class="sub">of ${att} attempts</div></div>`;
  drawConcurrency(v, gpus);
  drawPhases(v);
  drawThroughput(v);
  drawGpuHistory(v);
  renderHpcTable(v);
}

function drawConcurrency(v, gpus) {
  const el = $("#p-conc");
  const at = v.attempts.filter((a) => a.start);
  const now = Date.now() / 1000;
  const t0 = Math.min(...at.map((a) => a.start)), t1 = Math.max(...at.map((a) => a.end || now));
  const N = 500, dt = Math.max(1, (t1 - t0) / N);
  const xs = Array.from({ length: N + 1 }, (_, i) => t0 + i * dt);
  const secs = v.sections.filter((s) => at.some((a) => a.section === s));
  const traces = secs.map((s) => {
    const iv = at.filter((a) => a.section === s);
    const y = xs.map((t) => iv.reduce((n, a) => n + (a.start <= t && (a.end || now) > t ? 1 : 0), 0));
    return { type: "scatter", mode: "lines", stackgroup: "one", name: secName(s), x: xs.map((t) => new Date(t * 1000).toISOString()), y,
      line: { width: 0.8, shape: "hv", color: catColor(v.sections.indexOf(s)) }, fillcolor: catColor(v.sections.indexOf(s)),
      hovertemplate: `${secName(s)}: %{y}<extra></extra>` };
  });
  const shapes = gpus ? [{ type: "line", xref: "paper", x0: 0, x1: 1, y0: gpus, y1: gpus, line: { color: css("--text-secondary"), width: 1.5, dash: "dash" } }] : [];
  const annotations = gpus ? [{ xref: "paper", x: 1, y: gpus, yanchor: "bottom", xanchor: "right", text: `${gpus} GPUs on the node`, showarrow: false, font: { size: 11, color: css("--text-secondary") } }] : [];
  Plotly.react(el, traces, baseLayout({ height: 280, margin: { l: 40, r: 10, t: 8, b: 36 }, hovermode: "x unified", shapes, annotations,
    legend: { orientation: "h", y: -0.18, x: 0, font: { size: 11 } },
    xaxis: { type: "date", gridcolor: css("--grid") }, yaxis: { gridcolor: css("--grid"), rangemode: "tozero", title: { text: "jobs" } } }), { ...PCFG, displayModeBar: false });
}

function drawPhases(v) {
  const el = $("#p-phase"), p = v.perf;
  const secs = v.sections.filter((s) => p.sections[s] && (p.sections[s].run.length || p.sections[s].queue.length));
  if (!secs.length) { Plotly.purge(el); el.innerHTML = `<div class="empty">No finished jobs yet.</div>`; return; }
  const med = (k) => secs.map((s) => (median(p.sections[s][k]) || 0) / 60);
  const ys = secs.map(secName);
  const parts = [["sched", "Waiting for Autosubmit", css("--text-muted")], ["queue", "Queued in SLURM", css("--st-queued")], ["run", "Running", css("--st-running")]];
  const traces = parts.map(([k, name, color]) => ({ type: "bar", orientation: "h", name, y: ys, x: med(k), marker: { color, line: { color: css("--surface-1"), width: 1 } },
    hovertemplate: `${name}: %{x:.1f} min<extra>%{y}</extra>` }));
  Plotly.react(el, traces, baseLayout({ height: Math.max(200, secs.length * 34 + 90), barmode: "stack", margin: { l: 140, r: 10, t: 8, b: 40 },
    legend: { orientation: "h", y: -0.2, x: 0, font: { size: 11 } },
    xaxis: { title: { text: "minutes (median per job)" }, gridcolor: css("--grid") },
    yaxis: { type: "category", autorange: "reversed", gridcolor: "rgba(0,0,0,0)" } }), { ...PCFG, displayModeBar: false });
}

function drawThroughput(v) {
  const el = $("#p-thru");
  const done = v.attempts.filter((a) => a.result === "COMPLETED" && a.end).map((a) => a.end).sort((a, b) => a - b);
  if (!done.length) { Plotly.purge(el); el.innerHTML = `<div class="empty">No finished jobs yet.</div>`; return; }
  const x = done.map((t) => new Date(t * 1000).toISOString()), y = done.map((_, i) => i + 1);
  const shapes = (v.perf.gaps || []).map(([a, b]) => ({ type: "rect", xref: "x", yref: "paper", x0: new Date(a * 1000).toISOString(), x1: new Date(b * 1000).toISOString(),
    y0: 0, y1: 1, fillcolor: css("--st-held"), opacity: 0.15, line: { width: 0 } }));
  Plotly.react(el, [{ type: "scatter", mode: "lines", x, y, line: { color: css("--st-completed"), width: 2, shape: "hv" }, name: "finished jobs",
    hovertemplate: "%{y} jobs done<br>%{x|%d %b %H:%M}<extra></extra>" }],
  baseLayout({ height: 260, margin: { l: 50, r: 10, t: 8, b: 36 }, showlegend: false, shapes,
    xaxis: { type: "date", gridcolor: css("--grid") }, yaxis: { gridcolor: css("--grid"), rangemode: "tozero", title: { text: `of ${v.summary.total} jobs` } } }), { ...PCFG, displayModeBar: false });
}

async function drawGpuHistory(v) {
  const el = $("#p-gpuh");
  if (!el) return;
  let h;
  try { h = (await api(`/api/history?hours=${S.hpcHours || 24}`)).history; } catch { h = []; }
  if (!h.length) { Plotly.purge(el); el.innerHTML = `<div class="empty">Collecting… GPU use is sampled every refresh and kept for 7 days.</div>`; return; }
  if (el.querySelector(".empty")) el.innerHTML = "";
  const x = h.map((r) => new Date(r.t * 1000).toISOString());
  const ng = Math.max(...h.map((r) => r.util.length));
  const traces = [];
  for (let g = 0; g < ng; g++) traces.push({ type: "scatter", mode: "lines", name: `GPU ${g} busy`, x, y: h.map((r) => r.util[g] ?? null),
    line: { color: catColor(g), width: 1.5 }, hovertemplate: `GPU ${g}: %{y:.0f}%<extra></extra>` });
  traces.push({ type: "scatter", mode: "lines", name: `${v.expid} jobs running`, x, y: h.map((r) => r.by_exp?.[v.expid] || 0), yaxis: "y2",
    line: { color: css("--st-completed"), width: 1.5, shape: "hv" }, fill: "tozeroy", fillcolor: css("--accent-soft"), hovertemplate: `${v.expid}: %{y} running<extra></extra>` });
  traces.push({ type: "scatter", mode: "lines", name: "your jobs queued", x, y: h.map((r) => r.pending || 0), yaxis: "y2",
    line: { color: css("--st-queued"), width: 1.5, shape: "hv", dash: "dot" }, hovertemplate: "queued: %{y}<extra></extra>" });
  Plotly.react(el, traces, baseLayout({ height: 280, margin: { l: 44, r: 10, t: 8, b: 36 }, hovermode: "x unified",
    legend: { orientation: "h", y: -0.18, x: 0, font: { size: 11 } },
    xaxis: { type: "date", gridcolor: css("--grid"), anchor: "y2" },
    yaxis: { domain: [0.42, 1], range: [0, 100], ticksuffix: "%", gridcolor: css("--grid"), title: { text: "GPU busy" } },
    yaxis2: { domain: [0, 0.32], rangemode: "tozero", gridcolor: css("--grid"), title: { text: "jobs" }, nticks: 3, tickformat: "d" } }), { ...PCFG, displayModeBar: false });
}

function renderHpcTable(v) {
  const p = v.perf, lim = p.limits || {}, jl = lim.jobs || {};
  const total = p.compute || 1;
  const rows = v.sections.filter((s) => p.sections[s]?.attempts).map((s) => {
    const r = p.sections[s], pl = jl[s] || {};
    const limTxt = [pl.platform, pl.platform_total_jobs ? `max ${pl.platform_total_jobs} at once` : "", pl.gpu ? "GPU" : ""].filter(Boolean).join(" · ");
    return `<tr><td>${esc(secName(s))}</td><td class="note">${esc(limTxt || "—")}</td><td class="num">${r.run.length}</td>
      <td class="num">${r.failed ? `<b style="color:var(--st-failed)">${r.failed}</b>` : 0}</td>
      <td class="num">${fmtDur(median(r.sched))}</td><td class="num">${fmtDur(median(r.queue))}</td><td class="num">${fmtDur(pctl(r.queue, 0.9))}</td>
      <td class="num">${fmtDur(median(r.run))}</td><td class="num">${conc(v, s).avg ? conc(v, s).avg.toFixed(1) : "—"}</td>
      <td class="num">${hours(r.compute)}</td>
      <td style="min-width:110px"><div class="mini-bar" style="margin:0"><span style="width:${(100 * r.compute) / total}%;background:${catColor(v.sections.indexOf(s))}"></span></div></td></tr>`;
  }).join("");
  $("#h-table").innerHTML = `<table><thead><tr><th>Step</th><th>Runs on</th><th class="num">Jobs</th><th class="num">Failed</th>
    <th class="num">Waiting for AS</th><th class="num">Queued</th><th class="num">Queued (slowest 10%)</th><th class="num">Running</th>
    <th class="num">At once</th><th class="num">Compute</th><th>Share</th></tr></thead><tbody>${rows}</tbody></table>
    <p class="note">Times are medians per job. “Waiting for AS”: from ready to submitted by Autosubmit (job limits ${lim.total_jobs ? `TOTALJOBS = ${lim.total_jobs}` : ""}${lim.max_waiting ? `, MAXWAITINGJOBS = ${lim.max_waiting}` : ""}). “Queued”: from submitted to started (SLURM queue, or the platform's own limit). Ready/submit times are those of the latest attempt.</p>`;
}
