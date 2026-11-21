/* AIUQ Launcher — create AIUQ experiments and start/stop their runs on the Autosubmit host.
   Every write goes through POST /api/launcher/* with the X-AIUQ-Launcher header and an explicit confirmation. */
"use strict";

function launcherViewFromUrl() {
  const v = new URLSearchParams(location.search).get("view");
  return ["new", "runs", "delete", "activity"].includes(v) ? v : "new";
}
const L = {
  view: launcherViewFromUrl(),   // which of the four views is shown: new | runs | delete | activity
  q: "",                         // filter of the experiment tables
  built: false, opts: null, branches: [], preview: null, pending: null, task: null, poll: null,
  dateMode: "single", reg: { from: "1989-01-01", to: "1989-12-31", every: 10 }, listText: "",
  cfg: {
    branch: "main", model: null, checkpoint: null, ics: "era5", run_type: "hindcast", forcing: "ESA-CCI-v3.0",
    members: 10, dates: ["19890101"], chunk_size: 30, chunk_unit: "day", num_chunks: 1,
    out_vars: ["t", "z"], out_levs: [1000, 850, 500], out_res: "1", out_freq: "daily", description: "",
    output_store: null,
  },
};
const ICS_LABEL = { era5: ["ERA5", "Google Cloud · from 1940"], eerie: ["EERIE", "MARS · ECMWF Web API"],
  "ecmwf-an": ["ECMWF analysis", "Open Data · latest cycle"], "ec-earth3": ["EC-Earth3", "CMIP6 · ESGF · 1850-2100"] };
const SUP = { ok: ["ok", "supported"], wip: ["wip", "work in progress"] };

async function lapi(path, body) {
  const r = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json", "X-AIUQ-Launcher": "1" }, body: JSON.stringify(body ?? {}) });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.detail || `${r.status}`);
  return d;
}

// ------------------------------------------------------------------ dates
const ymd = (d) => d.toISOString().slice(0, 10).replace(/-/g, "");
const iso = (s) => `${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)}`;
function expandDatelist(text) {  // YYYYMMDD, YYYY-MM-DD or Autosubmit 'YYYYMM[DD DD]'
  const out = [];
  const t = String(text || "").replace(/(\d{4})-(\d{2})-(\d{2})/g, "$1$2$3");
  for (const m of t.matchAll(/(\d+)(?:\[([^\]]*)\])?/g)) {
    if (m[2] === undefined) out.push(m[1]); else for (const dd of m[2].split(/\s+/).filter(Boolean)) out.push(m[1] + dd);
  }
  return [...new Set(out)].sort();
}
function regularDates({ from, to, every }) {
  const a = new Date(`${from}T00:00:00Z`), b = new Date(`${to}T00:00:00Z`), step = Math.max(1, +every || 1);
  if (!(a <= b)) return [];
  const out = [];
  for (let d = new Date(a); d <= b && out.length < 2001; d.setUTCDate(d.getUTCDate() + step)) out.push(ymd(d));
  return out;
}
function compress(dates) {
  const g = {};
  for (const d of dates) (g[d.slice(0, 6)] ||= []).push(d.slice(6));
  return Object.keys(g).sort().map((ym) => (g[ym].length === 1 ? ym + g[ym][0] : `${ym}[${g[ym].join(" ")}]`)).join(" ");
}
function syncDates() {
  if (L.dateMode === "regular") L.cfg.dates = regularDates(L.reg);
  else if (L.dateMode === "list") L.cfg.dates = expandDatelist(L.listText);
}

// ------------------------------------------------------------------ page
function setLauncherView(view) {
  L.view = view;
  const url = new URL(location.href);
  if (view === "new") url.searchParams.delete("view"); else url.searchParams.set("view", view);
  history.replaceState(history.state, "", url);
  const main = $("#main");
  if (main.dataset.view) main.dataset.view = view;
  renderTabs();
  if (view === "activity") renderTasks();
  if (view === "delete") renderDeleteList();
  window.scrollTo({ top: 0 });
}

const filterBox = (id) => `<div class="side-search"><svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="M20 20l-3.5-3.5"/></svg>
  <input type="search" id="${id}" data-filter="1" placeholder="Filter experiments" aria-label="Filter experiments" autocomplete="off" spellcheck="false" value="${esc(L.q)}"></div>`;
const lnList = () => groupedExps(L.q).flatMap((g) => g[2]);   // same order as the Monitor panel: running first, newest ids first

function renderLauncher() {
  const main = $("#main");
  if (!L.built || !$("#ln-form")) {
    L.built = true;
    main.removeAttribute("data-tab");
    main._html = null;
    main.innerHTML = `
      <div class="page-head"><h2>AIUQ Launcher</h2><span class="lede">Create AIUQ experiments on mafalda and start or stop their runs. Every action asks for confirmation and is logged.</span></div>
      <div id="ln-banner" data-view="new"></div>
      <div class="ln-grid" data-view="new">
        <div id="ln-form">${'<div class="skeleton block" style="margin-bottom:16px"></div>'.repeat(3)}</div>
        <aside class="ln-side" id="ln-side"><div class="skeleton block"></div></aside>
      </div>
      <section class="card" id="ln-exps" data-view="runs"><div class="card-head"><h3>Experiments</h3><span class="sub">launch, resume, stop or reuse an existing experiment</span></div>
        <div class="ln-toolbar">${filterBox("ln-q1")}<span class="note" id="ln-count1"></span></div><div id="ln-exp-list"></div></section>
      <section class="card danger-zone" id="ln-delete" data-view="delete"><div class="card-head"><h3>Delete experiments</h3>
        <span class="sub">removes the experiment from Autosubmit on mafalda and its working folder on the HPC · cannot be undone</span></div>
        <div class="ln-toolbar">${filterBox("ln-q2")}<span class="note" id="ln-count2"></span></div><div id="ln-del-list"></div></section>
      <section class="card" id="ln-tasks" data-view="activity"><div class="card-head"><h3>Recent actions</h3><span class="sub">this dashboard session · full history in ~/.local/share/aiuq-dashboard/launcher.log</span></div><div id="ln-task-list"></div></section>
      <dialog id="ln-dialog"></dialog>`;
    $("#ln-form").addEventListener("click", onFormClick);
    $("#ln-form").addEventListener("change", onFormChange);
    $("#ln-form").addEventListener("input", onFormInput);
    $("#ln-side").addEventListener("click", onSideClick);
    $("#ln-exp-list").addEventListener("click", onExpClick);
    $("#ln-del-list").addEventListener("click", onDeleteClick);
    main.addEventListener("input", (e) => {
      if (!e.target.dataset?.filter) return;
      L.q = e.target.value;
      main.querySelectorAll("input[data-filter]").forEach((x) => { if (x !== e.target) x.value = L.q; });
      renderExpList(); renderDeleteList(false);
    });
    // coming back to the Launcher: show the form as it was left at once, and re-read the repository behind it
    if (L.opts && model()) { renderForm(); renderSide(); }
    loadBranches(!!L.opts);
  }
  main.dataset.view = L.view;
  main.querySelectorAll("input[data-filter]").forEach((x) => { if (x !== document.activeElement && x.value !== L.q) x.value = L.q; });
  renderExpList();
  renderDeleteList();
  renderTasks();
}

async function loadBranches(quiet = false) {
  try {
    L.branches = (await api("/api/launcher/branches")).branches;
    if (!L.branches.includes(L.cfg.branch)) L.cfg.branch = L.branches.includes("main") ? "main" : L.branches[0];
    await loadOptions(quiet);
  } catch (e) { $("#ln-form").innerHTML = `<div class="err-banner"><b>Cannot read the AIUQ repository</b><br>${esc(e.message)}</div>`; }
}

async function loadOptions(quiet = false) {
  if (!$("#ln-form")) return;
  if (!quiet) $("#ln-form").classList.add("busy");
  try { L.opts = await api(`/api/launcher/options?branch=${encodeURIComponent(L.cfg.branch)}`); }
  catch (e) { if ($("#ln-form")) $("#ln-form").innerHTML = `<div class="err-banner"><b>Cannot read branch ${esc(L.cfg.branch)}</b><br>${esc(e.message)}</div>`; return; }
  if (!$("#ln-form")) return;   // the user left the Launcher while this was loading
  $("#ln-form").classList.remove("busy");
  fixCfg();
  renderForm();
  updatePreview();
}

function model() { return L.opts?.models.find((m) => m.name === L.cfg.model); }
function ckpt() { return model()?.checkpoints.find((c) => c.name === L.cfg.checkpoint); }
// What the chosen checkpoint writes (model card), else what the model writes.
function outVars() {
  const m = model(), k = ckpt();
  if (!k?.variables) return m.variables;
  const by = Object.fromEntries(m.variables.map((v) => [v.name, v]));
  return k.variables.map((n) => by[n]).filter(Boolean);
}
function outLevels() { return ckpt()?.levels || model().levels; }

function fixCfg() {  // keep the form consistent with what the branch offers
  const o = L.opts, c = L.cfg;
  if (!model()) c.model = (o.models.find((m) => m.on_platform) || o.models[0])?.name;
  const m = model();
  if (!ckpt()) c.checkpoint = (m.checkpoints.find((k) => k.present && k.ensemble) || m.checkpoints.find((k) => k.present) || m.checkpoints[0])?.name;
  if (!o.ics.includes(c.ics)) c.ics = o.ics[0];
  if (c.run_type === "amip" && !m.amip) c.run_type = "hindcast";
  const names = new Set(outVars().map((v) => v.name)), levels = outLevels();
  c.out_vars = c.out_vars.filter((v) => names.has(v));
  if (!c.out_vars.length) c.out_vars = ["t", "z"].filter((v) => names.has(v));
  if (c.out_levs !== "original") {
    c.out_levs = c.out_levs.filter((l) => !levels.length || levels.includes(l));
    if (!c.out_levs.length) c.out_levs = [1000, 850, 500].filter((l) => levels.includes(l));
  }
}

// ------------------------------------------------------------------ form
const step = (n, title, sub, body) => `<section class="card ln-step" id="ln-s${n}"><div class="card-head"><span class="step-n">${n}</span><h3>${title}</h3>${sub ? `<span class="sub">${sub}</span>` : ""}</div>${body}</section>`;
const choice = (k, v, on, inner, disabled, title) => `<button type="button" class="choice ${on ? "on" : ""}" data-k="${k}" data-v="${esc(v)}" ${disabled ? "disabled" : ""} ${title ? `title="${esc(title)}"` : ""} aria-pressed="${on}">${inner}</button>`;
const tag = (cls, t) => `<span class="tag ${cls}">${esc(t)}</span>`;

function renderForm() {
  const form = $("#ln-form"), act = document.activeElement;
  const focusKey = form.contains(act) ? [act.dataset.k || act.closest("[data-k]")?.dataset.k, act.dataset.v, act.tagName] : null;
  buildForm();
  if (focusKey?.[0]) {   // the form was rebuilt: put the focus back where it was
    const same = [...form.querySelectorAll(`[data-k="${focusKey[0]}"]`)].flatMap((x) => (x.matches("button, input, select, textarea") ? [x] : [...x.querySelectorAll("button")]))
      .find((x) => x.tagName === focusKey[2] && (focusKey[1] === undefined || x.dataset.v === focusKey[1]));
    same?.focus({ preventScroll: true });
  }
}
function buildForm() {
  const o = L.opts, c = L.cfg, m = model(), k = ckpt();
  const exps = S.list.map((e) => e.expid);
  const icsSup = (ics) => k?.support?.[ics];
  const datesInfo = c.dates.length ? `<b>${c.dates.length}</b> start date${c.dates.length === 1 ? "" : "s"} · ${fmtDate(c.dates[0])}${c.dates.length > 1 ? ` → ${fmtDate(c.dates[c.dates.length - 1])}` : ""}` : `<span style="color:var(--st-failed)">no valid dates</span>`;
  const leadDays = c.chunk_unit === "day" ? c.chunk_size * c.num_chunks : null;
  const byType = {}, levels = outLevels();
  const GROUPS = { pl: "On pressure levels", sfc: "At the surface", soil: "At the surface",
    period: "Totals and means over a period (precipitation, radiation, …)", wave: "Ocean waves (sea points only)" };
  for (const g of Object.values(GROUPS)) byType[g] = [];
  for (const v of outVars()) (byType[GROUPS[v.group || v.type] || "Other"] ||= []).push(v);
  for (const g of Object.keys(byType)) if (!byType[g].length) delete byType[g];

  $("#ln-form").innerHTML =
    step(1, "Start from", "a blank form or the settings of an existing experiment", `
      <div class="chips-sel">${choice("tpl", "", false, "Blank")}${exps.map((x) => choice("tpl", x, false, esc(x), false, S.list.find((e) => e.expid === x)?.description)).join("")}</div>
      <div class="ln-row">
        <label class="ln-field">Code branch <select data-k="branch">${L.branches.map((b) => `<option ${b === c.branch ? "selected" : ""}>${esc(b)}</option>`).join("")}</select></label>
        <label class="ln-field grow">Description <input type="text" data-k="description" maxlength="200" value="${esc(c.description)}" placeholder="e.g. Aurora ensemble, ERA5, winter 1989"></label>
      </div>`)
    + step(2, "Model", `models with a ${esc(o.platform)} configuration on <span class="mono">${esc(c.branch)}</span>`, `
      <div class="choice-grid">${o.models.map((mm) => choice("model", mm.name, mm.name === c.model,
        `<b>${esc(mm.label)}</b><small>${esc(mm.producer || "")} · up to ${mm.variables.length} variables · ${mm.levels.length} levels</small>${mm.on_platform ? "" : tag("no", `not on ${o.platform}`)}`, !mm.on_platform)).join("")}</div>
      <div class="ln-sub">Checkpoint</div>
      <div class="ck-list">${m.checkpoints.map((kk) => choice("checkpoint", kk.name, kk.name === c.checkpoint,
        `<span class="mono">${esc(kk.name)}</span><span class="tags">${kk.ensemble ? tag("ens", "ensemble") : tag("det", "deterministic")}
          ${o.checkpoints_checked ? (kk.present ? tag("ok", "on the HPC") : tag("no", "missing on the HPC")) : ""}
          ${Object.entries(kk.support || {}).map(([ic, s]) => tag(SUP[s]?.[0] || "", `${(ICS_LABEL[ic] || [ic])[0]}: ${SUP[s]?.[1] || s}`)).join("")}</span>`,
        o.checkpoints_checked && !kk.present)).join("")}</div>`)
    + step(3, "Initial conditions & run type", "", `
      <div class="choice-grid">${o.ics.map((ic) => choice("ics", ic, ic === c.ics,
        `<b>${esc((ICS_LABEL[ic] || [ic])[0])}</b><small>${esc((ICS_LABEL[ic] || ["", ""])[1])}</small>${icsSup(ic) ? tag(SUP[icsSup(ic)]?.[0], SUP[icsSup(ic)]?.[1]) : ""}`)).join("")}</div>
      <div class="ln-row">
        <div class="ln-field">Run type ${seg("run_type", [["hindcast", "Hindcast"], ["amip", "AMIP"]], c.run_type)}</div>
        ${c.run_type === "amip" ? `<label class="ln-field">Forcing <select data-k="forcing">${o.forcings.map((f) => `<option ${f === c.forcing ? "selected" : ""}>${esc(f)}</option>`).join("")}</select></label>` : ""}
      </div>${!m.amip ? `<div class="note">AMIP runs are available with NeuralGCM only.</div>` : ""}`)
    + step(4, "Ensemble & start dates", "", `
      <div class="ln-row">
        <div class="ln-field">Members <div class="range-row"><input type="range" min="1" max="50" data-k="members" value="${c.members}"><input type="number" min="1" max="100" data-k="members" value="${c.members}" style="width:70px"></div></div>
        <div class="ln-field">Each forecast <div class="ln-inline"><input type="number" min="1" data-k="chunk_size" value="${c.chunk_size}" style="width:70px">
          <select data-k="chunk_unit">${o.chunk_units.map((u) => `<option ${u === c.chunk_unit ? "selected" : ""}>${u}</option>`).join("")}</select> ×
          <input type="number" min="1" data-k="num_chunks" value="${c.num_chunks}" style="width:60px"> chunk${c.num_chunks === 1 ? "" : "s"}</div>
          <small class="note">${leadDays ? `${leadDays} days of forecast` : `${c.chunk_size * c.num_chunks} ${c.chunk_unit}s of forecast`}</small></div>
      </div>
      <div class="ln-sub">Start dates</div>
      ${seg("dateMode", [["single", "One date"], ["regular", "Regular dates"], ["list", "List"]], L.dateMode)}
      <div class="ln-row" style="margin-top:10px">${
        L.dateMode === "single" ? `<label class="ln-field">Date <input type="date" data-k="single" value="${c.dates[0] ? iso(c.dates[0]) : ""}"></label>`
        : L.dateMode === "regular" ? `<label class="ln-field">From <input type="date" data-k="reg.from" value="${L.reg.from}"></label>
            <label class="ln-field">To <input type="date" data-k="reg.to" value="${L.reg.to}"></label>
            <label class="ln-field">Every <span class="ln-inline"><input type="number" min="1" data-k="reg.every" value="${L.reg.every}" style="width:70px"> days</span></label>`
        : `<label class="ln-field grow">Dates <textarea data-k="list" rows="3" placeholder="19890101 19890201 … or 198901[01 11 21]">${esc(L.listText || compress(c.dates))}</textarea></label>`}</div>
      <div class="note" id="ln-dates-info">${datesInfo}</div>`)
    + step(5, "Outputs", `what the simulations write · ${outVars().length} variables available with <span class="mono">${esc(c.checkpoint || "")}</span>`, `
      ${Object.entries(byType).map(([t, vs]) => `<div class="ln-sub">${t}</div><div class="chips-sel">${vs.map((v) =>
        `<button type="button" data-k="var" data-v="${esc(v.name)}" aria-pressed="${c.out_vars.includes(v.name)}" title="${esc(v.long)}${v.units ? ` [${esc(v.units)}]` : ""}${v.period ? ` · ${esc(v.period)}` : ""}">${esc(v.name)}<span class="sub">${esc(v.long.length > 22 ? v.long.slice(0, 21) + "…" : v.long)}</span></button>`).join("")}</div>`).join("")}
      <div class="ln-sub">Pressure levels <label class="switch" style="display:inline-flex;margin-left:12px"><input type="checkbox" data-k="alllev" ${c.out_levs === "original" ? "checked" : ""}><span class="track"></span>all levels of the model</label></div>
      ${c.out_levs === "original" ? `<div class="note">All ${levels.length} levels: ${levels.join(", ")} hPa.</div>`
        : `<div class="chips-sel">${levels.map((l) => `<button type="button" data-k="lev" data-v="${l}" aria-pressed="${c.out_levs.includes(l)}">${l}</button>`).join("")}</div>`}
      <div class="ln-row" style="margin-top:12px">
        <div class="ln-field">Resolution ${seg("out_res", o.resolutions.map((r) => [r, r === "original" ? "native" : `${r}°`]), c.out_res)}</div>
        <div class="ln-field">Frequency ${seg("out_freq", o.freqs.map((f) => [f, f === "original" ? "every step" : f]), c.out_freq)}</div>
      </div>
      <div class="ln-row" style="margin-top:12px">
        <label class="ln-field grow">Output store <input type="text" class="mono" data-k="output_store" value="${esc(c.output_store || "")}" placeholder="${esc(o.store_by_ics?.[c.ics] || "no output store on this branch")}" spellcheck="false"></label>
      </div>
      <div class="note">${o.store_by_ics?.[c.ics] ? `When every simulation has finished, the ARCHIVE job moves the outputs from the HPC to <span class="mono">&lt;output store&gt;/&lt;expid&gt;/outputs</span> on the Autosubmit host and removes them from the HPC. Choose the folder here, or leave it empty for the default of branch <span class="mono">${esc(c.branch)}</span> (shown in grey).` : `Branch <span class="mono">${esc(c.branch)}</span> has no ARCHIVE job: outputs stay on the HPC.`}</div>`);
}

function onFormClick(e) {
  const b = e.target.closest("button[data-k], .seg[data-k] button[data-v]");
  if (!b || b.disabled) return;
  const k = b.dataset.k || b.closest("[data-k]")?.dataset.k;
  const v = b.dataset.v;
  const c = L.cfg;
  switch (k) {
    case "tpl": if (v) useTemplate(v); else { c.description = ""; renderForm(); updatePreview(); } return;
    case "model": c.model = v; c.checkpoint = null; fixCfg(); break;
    case "checkpoint": c.checkpoint = v; fixCfg(); break;
    case "ics": c.ics = v; break;
    case "var": c.out_vars = c.out_vars.includes(v) ? c.out_vars.filter((x) => x !== v) : [...c.out_vars, v]; break;
    case "lev": { const l = +v; c.out_levs = c.out_levs.includes(l) ? c.out_levs.filter((x) => x !== l) : [...c.out_levs, l].sort((a, b2) => b2 - a); break; }
    default: {
      const sk = b.closest(".seg[data-k]")?.dataset.k;
      if (!sk) return;
      if (sk === "dateMode") { L.dateMode = v; if (v === "list") L.listText = compress(c.dates); syncDates(); }
      else c[sk] = v;
    }
  }
  renderForm(); updatePreview();
}
function onFormChange(e) {
  const t = e.target, k = t.dataset.k, c = L.cfg;
  if (!k) return;
  if (k === "branch") { c.branch = t.value; loadOptions(); return; }
  if (k === "forcing") c.forcing = t.value;
  if (k === "chunk_unit") c.chunk_unit = t.value;
  if (k === "alllev") c.out_levs = t.checked ? "original" : [1000, 850, 500].filter((l) => model().levels.includes(l));
  if (k === "single") c.dates = t.value ? [t.value.replace(/-/g, "")] : [];
  if (["members", "chunk_size", "num_chunks", "reg.every", "reg.from", "reg.to"].includes(k)) return;  // handled on input
  renderForm(); updatePreview();
}
function onFormInput(e) {
  const t = e.target, k = t.dataset.k, c = L.cfg;
  if (k === "description") { c.description = t.value; updatePreview(); return; }
  if (k === "output_store") { c.output_store = t.value.trim() || null; updatePreview(); return; }
  if (k === "members") { c.members = Math.max(1, Math.min(100, +t.value || 1)); document.querySelectorAll('[data-k="members"]').forEach((x) => { if (x !== t) x.value = c.members; }); updatePreview(); return; }
  if (k === "chunk_size" || k === "num_chunks") { c[k] = Math.max(1, +t.value || 1); updatePreview(); return; }
  if (k?.startsWith("reg.")) { L.reg[k.slice(4)] = t.value; syncDates(); updateDatesInfo(); updatePreview(); return; }
  if (k === "list") { L.listText = t.value; syncDates(); updateDatesInfo(); updatePreview(); }
}
function updateDatesInfo() {
  const c = L.cfg, el = $("#ln-dates-info");
  if (el) el.innerHTML = c.dates.length ? `<b>${c.dates.length}</b> start date${c.dates.length === 1 ? "" : "s"} · ${fmtDate(c.dates[0])}${c.dates.length > 1 ? ` → ${fmtDate(c.dates[c.dates.length - 1])}` : ""}` : `<span style="color:var(--st-failed)">no valid dates</span>`;
}

async function useTemplate(expid) {
  try {
    const t = await api(`/api/launcher/template/${encodeURIComponent(expid)}`);
    $("#ln-banner").innerHTML = "";
    if (!L.branches.includes(t.branch)) {
      $("#ln-banner").innerHTML = `<div class="finding f-warn" style="margin-bottom:14px">${ICONS.warn}<div>${esc(expid)} used the branch <span class="mono">${esc(t.branch)}</span>, which no longer exists: the form keeps <span class="mono">${esc(L.cfg.branch)}</span>. Check that the model options below still match.</div></div>`;
      t.branch = L.cfg.branch;
    } else {
      $("#ln-banner").innerHTML = `<div class="finding f-info" style="margin-bottom:14px">${ICONS.info}<div>Form filled with the settings of <b>${esc(expid)}</b>. Change what you need, then create.</div></div>`;
    }
    const branchChanged = t.branch !== L.cfg.branch;
    Object.assign(L.cfg, t);
    L.dateMode = t.dates.length === 1 ? "single" : "list";
    L.listText = compress(t.dates);
    if (branchChanged && L.branches.includes(t.branch)) await loadOptions();
    else { fixCfg(); renderForm(); updatePreview(); }
    if (L.view !== "new") setLauncherView("new");
  } catch (e) { toast(`Cannot read ${expid}: ${e.message}`, true); }
}

// ------------------------------------------------------------------ preview & summary
let previewTimer = null;
function updatePreview() {
  clearTimeout(previewTimer);
  previewTimer = setTimeout(async () => {
    try { L.preview = await lapi("/api/launcher/preview", L.cfg); } catch (e) { L.preview = { ok: false, errors: [e.message], warnings: [] }; }
    renderSide();
  }, 250);
}

function renderSide() {
  const p = L.preview, c = L.cfg, m = model(), side = $("#ln-side");
  if (!p || !m || !side) return;   // no side panel: the user left the Launcher while the preview was computed
  const est = p.estimate;
  const levs = c.out_levs === "original" ? "all levels" : `${c.out_levs.join(", ")} hPa`;
  const lead = c.chunk_size * c.num_chunks;
  const recap = [
    ["Start from", `branch ${c.branch}${c.description ? ` · ${c.description}` : " · no description yet"}`, !c.description.trim()],
    ["Model", `${m.label} · ${c.checkpoint || "no checkpoint"}`, !c.checkpoint],
    ["Initial conditions", `${(ICS_LABEL[c.ics] || [c.ics])[0]} · ${c.run_type === "amip" ? `AMIP, ${c.forcing}` : "hindcast"}`, false],
    ["Ensemble", `${c.members} member${c.members === 1 ? "" : "s"} × ${c.dates.length} start date${c.dates.length === 1 ? "" : "s"} · ${lead} ${c.chunk_unit}${lead === 1 ? "" : "s"}`, !c.dates.length],
    ["Outputs", `${c.out_vars.join(", ") || "no variables"} · ${levs} · ${c.out_res === "original" ? "native" : c.out_res + "°"} · ${c.out_freq === "original" ? "every step" : c.out_freq}`, !c.out_vars.length],
  ];
  const openKeys = new Set([...side.querySelectorAll("details[open]")].map((d) => d.dataset.k));
  side.innerHTML = `
    <div class="ln-summary">
      <div class="note">New experiment</div>
      <div class="ln-what">${esc(m.label)} · ${esc((ICS_LABEL[c.ics] || [c.ics])[0])}</div>
      <div class="note mono">${esc(c.checkpoint || "")}</div>
      ${p.summary ? `<div class="ln-big"><b>${p.summary.sims}</b> simulation${p.summary.sims === 1 ? "" : "s"}</div>
        <div class="note">${p.summary.n_dates} start date${p.summary.n_dates === 1 ? "" : "s"} × ${p.summary.members} member${p.summary.members === 1 ? "" : "s"} × ${p.summary.num_chunks} chunk${p.summary.num_chunks === 1 ? "" : "s"} of ${p.summary.chunk_size} ${p.summary.chunk_unit}${p.summary.chunk_size === 1 ? "" : "s"}</div>` : ""}
      <ol class="ln-recap">${recap.map(([k, v, bad], i) => `<li class="${bad ? "bad" : ""}"><a href="#ln-s${i + 1}" data-scroll="ln-s${i + 1}" title="${esc(v)}"><span class="n">${i + 1}</span><span class="k">${k}</span><span class="v">${esc(v)}</span></a></li>`).join("")}</ol>
      ${est ? `<div class="ln-est"><div><span class="note">GPU time</span><b>~${fmtDur(est.gpu_s)}</b></div><div><span class="note">Prepare ICs</span><b>~${fmtDur(est.prep_s)}</b></div>
        <div><span class="note">Wall time (${est.gpus} GPUs free)</span><b>~${fmtDur(est.wall_s)}</b></div></div>
        <div class="note">Estimate based on ${esc(est.based_on.join(", ") || "—")}. Queue waits are not included.</div>` : `<div class="note">No finished experiment with this model yet, so no time estimate.</div>`}
      ${(p.errors || []).map((x) => `<div class="finding f-err">${ICONS.warn}<div>${esc(x)}</div></div>`).join("")}
      ${(p.warnings || []).map((x) => `<div class="finding f-warn">${ICONS.warn}<div>${esc(x)}</div></div>`).join("")}
      <div class="ln-cta"><button type="button" class="primary ln-create" data-act="create" ${p.ok ? "" : "disabled"}>Create experiment</button>
        <div class="note" style="text-align:center">Registers it in Autosubmit and builds the workflow. It does not start the run.</div></div>
    </div>
    ${p.yaml ? `<details data-k="yaml" ${openKeys.has("yaml") ? "open" : ""}><summary>conf/main.yml that will be written</summary><pre class="log">${esc(p.yaml)}</pre></details>
      <details data-k="cmd" ${openKeys.has("cmd") ? "open" : ""}><summary>Commands that will run</summary><pre class="log">${esc(p.commands.join("\n"))}</pre></details>` : ""}`;
}

function onSideClick(e) {
  const sc = e.target.closest("a[data-scroll]");
  if (sc) { e.preventDefault(); document.getElementById(sc.dataset.scroll)?.scrollIntoView({ behavior: "smooth", block: "start" }); return; }
  const b = e.target.closest("button[data-act]");
  if (b?.dataset.act === "create" && L.preview?.ok) {
    confirmDialog({
      title: "Create this experiment?",
      body: `<p>${esc(model().label)} · ${esc(L.cfg.checkpoint)} · ${esc(L.cfg.ics.toUpperCase())} · <b>${L.preview.summary.sims} simulations</b>.</p>
        <p class="note">This runs <span class="mono">autosubmit expid</span>, writes <span class="mono">conf/main.yml</span> and runs <span class="mono">autosubmit create</span> on mafalda. Nothing is submitted to the HPC yet.</p>`,
      ok: "Create", onOk: async () => startTask(await lapi("/api/launcher/create", L.cfg)),
    });
  }
}

// ------------------------------------------------------------------ experiments & runs
function renderExpList() {
  const el = $("#ln-exp-list");
  if (!el) return;
  const list = lnList();
  setHTML($("#ln-count1"), L.q.trim() ? `${list.length} of ${S.list.length} experiments` : `${S.list.length} experiments`);
  setHTML(el, !S.meta ? `<div class="skeleton block"></div>` : !list.length ? `<div class="empty">${S.list.length ? "No experiment matches this filter." : "No experiments yet."}</div>`
    : `<div class="tbl-wrap"><table><thead><tr><th>Experiment</th><th>Model · ICS</th><th>Description</th><th>Progress</th><th>State</th><th></th></tr></thead><tbody>
    ${list.map((e) => `<tr><td><span style="display:inline-flex;align-items:center;gap:9px">${ring(e)}<b>${e.expid}</b></span></td><td>${what(e.conf)}</td><td class="note wrap">${esc(e.description || "")}</td>
      <td style="min-width:130px">${miniBar(e.counts, e.summary.total)}<span class="note">${e.summary.pct.toFixed(0)}% · ${statusLine(e)}</span></td>
      <td>${stateBadge(e)}</td>
      <td class="ln-actions">${e.active
        ? `<button type="button" class="small danger-ghost" data-act="stop" data-exp="${e.expid}">Stop…</button>`
        : e.summary.total && e.summary.completed === e.summary.total ? ""
          : `<button type="button" class="small primary" data-act="run" data-exp="${e.expid}" ${e.summary.total ? "" : "disabled title=\"no workflow yet\""}>${e.summary.completed ? "Resume…" : "Launch…"}</button>`}
        <button type="button" class="small ghost" data-act="tpl" data-exp="${e.expid}" title="Fill the form with this experiment's settings">Use as template</button>
        <button type="button" class="small ghost" data-act="open" data-exp="${e.expid}">Monitor →</button></td></tr>`).join("")}
    </tbody></table></div>`);
}

function onExpClick(e) {
  const b = e.target.closest("button[data-act]");
  if (!b) return;
  const x = b.dataset.exp, ex = S.list.find((q) => q.expid === x);
  if (b.dataset.act === "open") { switchApp("monitor"); selectExp(x); return; }
  if (b.dataset.act === "tpl") { useTemplate(x); return; }
  if (b.dataset.act === "run") {
    const left = ex.summary.total - ex.summary.completed;
    confirmDialog({
      title: `Launch ${x}?`,
      body: `<p>${what(ex.conf)} · ${left} of ${ex.summary.total} jobs still to do${ex.summary.failed ? ` · <b>${ex.summary.failed} failed</b> (Autosubmit will retry them only if you reset them)` : ""}.</p>
        <p class="note">Runs <span class="mono">autosubmit run ${x}</span> in its own systemd user unit on mafalda; output goes to <span class="mono">~/autosubmit/${x}_run.out</span>. Jobs will be submitted to the HPC.</p>`,
      ok: "Launch", onOk: async () => startTask(await lapi(`/api/launcher/run/${x}`)),
    });
  }
  if (b.dataset.act === "stop") {
    confirmDialog({
      title: `Stop ${x}?`,
      body: `<p>Runs <span class="mono">autosubmit stop ${x}</span>: Autosubmit finishes its current loop and exits. Jobs already submitted keep running on the HPC unless you also cancel them.</p>
        <label class="switch"><input type="checkbox" id="dlg-cancel"><span class="track"></span><span>Also cancel queued and running jobs<span class="hint">autosubmit stop -c</span></span></label>`,
      ok: "Stop", danger: true,
      onOk: async () => startTask(await lapi(`/api/launcher/stop/${x}?cancel=${$("#dlg-cancel").checked}`)),
    });
  }
}

// ------------------------------------------------------------------ dialogs & tasks
function confirmDialog({ title, body, ok, onOk, danger }) {
  const d = $("#ln-dialog");
  d.innerHTML = `<form method="dialog"><h3>${esc(title)}</h3>${body}<div class="dlg-err" id="dlg-err"></div>
    <div class="dlg-actions"><button value="cancel" type="submit">Cancel</button><button type="button" id="dlg-ok" class="${danger ? "danger" : "primary"}">${esc(ok)}</button></div></form>`;
  d.showModal();
  $("#dlg-ok").onclick = async () => {
    $("#dlg-ok").disabled = true;
    try { await onOk(); } catch (e) { $("#dlg-err").textContent = e.message; $("#dlg-ok").disabled = false; }
  };
}

function startTask(t) {
  L.task = t;
  showTask();
  clearInterval(L.poll);
  L.poll = setInterval(async () => {
    try { L.task = await api(`/api/launcher/tasks/${L.task.id}`); } catch { return; }
    showTask();
    if (L.task.status !== "running") { clearInterval(L.poll); load(); renderTasks(); renderDeleteList(); }
  }, 1000);
}

function showTask() {
  const t = L.task, d = $("#ln-dialog");
  const title = { create: "Creating the experiment", run: `Launching ${t.expid}`, stop: `Stopping ${t.expid}`,
    delete: `Deleting ${t.expid}`, cleanup: `Cleaning up the HPC folder of ${t.expid}` }[t.kind];
  const icon = (s) => (s === "done" ? "✓" : s === "failed" ? "✕" : "…");
  const done = t.status === "done", failed = t.status === "failed";
  d.innerHTML = `<div><h3>${esc(title)}${t.expid && t.kind === "create" ? ` · ${esc(t.expid)}` : ""}</h3>
    <ol class="steps">${t.steps.map((s) => `<li class="${s.state}"><span>${icon(s.state)}</span>${esc(s.name)}</li>`).join("")}</ol>
    <pre class="log" id="task-log">${esc(t.lines.join("\n"))}</pre>
    ${failed ? `<div class="dlg-err">${esc(t.error)}</div>` : ""}
    <div class="dlg-actions">
      ${done && t.kind === "create" ? `<button type="button" class="primary" id="t-launch">Launch ${esc(t.expid)} now</button><button type="button" id="t-open">Open in monitor</button>` : ""}
      ${done && (t.kind === "run" || t.kind === "stop") ? `<button type="button" id="t-open">Open in monitor</button>` : ""}
      <button type="button" id="t-close">${t.status === "running" ? "Hide (keeps running)" : "Close"}</button></div></div>`;
  if (!d.open) d.showModal();
  const lg = $("#task-log"); lg.scrollTop = lg.scrollHeight;
  $("#t-close").onclick = () => d.close();
  if ($("#t-open")) $("#t-open").onclick = () => { d.close(); switchApp("monitor"); selectExp(t.expid); };
  if ($("#t-launch")) $("#t-launch").onclick = async () => {
    try { startTask(await lapi(`/api/launcher/run/${t.expid}`)); } catch (e) { toast(e.message, true); }
  };
}

async function renderTasks() {
  const el = $("#ln-task-list");
  if (!el) return;
  let ts = [];
  try { ts = (await api("/api/launcher/tasks")).tasks; } catch { /* ignore */ }
  setHTML(el, ts.length ? `<ul class="recent">${ts.map((t) => `<li><i style="background:${stColor(t.status === "done" ? "COMPLETED" : t.status === "failed" ? "FAILED" : "RUNNING")}"></i>
      <b>${esc(t.kind)}</b> ${esc(t.expid || "")} <span class="note grow" title="${esc(t.error || t.lines[t.lines.length - 1] || "")}">${esc(t.error || t.lines[t.lines.length - 1] || "")}</span><span class="note">${fmtAgo(t.started)}</span></li>`).join("")}</ul>`
    : `<div class="empty">No actions yet in this session.</div>`);
}


// ------------------------------------------------------------------ delete experiments
async function renderDeleteList(refetch = true) {
  let el = $("#ln-del-list");
  if (!el) return;
  if (refetch || !L.leftovers) { try { L.leftovers = (await api("/api/launcher/leftovers")).leftovers; } catch { L.leftovers = L.leftovers || []; } }
  const left = L.leftovers;
  el = $("#ln-del-list");
  if (!el) return;
  const list = lnList();
  setHTML($("#ln-count2"), L.q.trim() ? `${list.length} of ${S.list.length} experiments` : `${S.list.length} experiments`);
  setHTML(el, `
    ${left.length ? `<div class="finding f-warn" style="margin-bottom:12px">${ICONS.warn}<div><b>Leftover folders</b> — these experiments were removed from Autosubmit but a folder could not be deleted:
      <ul class="recent" style="margin-top:6px">${left.map((x) => `<li><b>${esc(x.expid)}</b><span class="note">${x.kind === "store" ? "output store" : "HPC"}</span><span class="mono note">${esc(x.path)}</span>
        <button type="button" class="small" data-act="retry" data-exp="${esc(x.expid)}" style="margin-left:auto">Retry cleanup</button></li>`).join("")}</ul></div></div>` : ""}
    ${!S.meta ? `<div class="skeleton block"></div>` : !list.length ? `<div class="empty">${S.list.length ? "No experiment matches this filter." : "No experiments yet."}</div>`
    : `<div class="tbl-wrap"><table><thead><tr><th>Experiment</th><th>Model · ICS</th><th>Description</th><th>State</th><th></th></tr></thead><tbody>
    ${list.map((e) => `<tr><td><span style="display:inline-flex;align-items:center;gap:9px">${ring(e)}<b>${e.expid}</b></span></td><td>${what(e.conf)}</td><td class="note wrap">${esc(e.description || "")}</td>
      <td>${stateBadge(e)}</td>
      <td class="ln-actions"><button type="button" class="small danger-ghost" data-act="delete" data-exp="${e.expid}" ${e.active ? 'disabled title="Stop the run first"' : ""}>Delete…</button></td></tr>`).join("")}
    </tbody></table></div>`}`);
}

async function onDeleteClick(e) {
  const b = e.target.closest("button[data-act]");
  if (!b || b.disabled) return;
  const x = b.dataset.exp;
  if (b.dataset.act === "retry") {
    confirmDialog({ title: `Retry deleting the leftover folders of ${x}?`, body: `<p class="note">The folders were recorded when the deletion started; each is checked again before removing it.</p>`,
      ok: "Retry", danger: true, onOk: async () => startTask(await lapi(`/api/launcher/leftovers/${x}/retry`)) });
    return;
  }
  const d = $("#ln-dialog");
  d.innerHTML = `<div><h3>Delete ${esc(x)}?</h3><div class="series-empty"><div class="spinner"></div>Checking what would be deleted on mafalda and on the HPC…</div>
    <div class="dlg-actions"><button type="button" id="dl-cancel">Cancel</button></div></div>`;
  d.showModal();
  $("#dl-cancel").onclick = () => d.close();
  let p;
  try { p = await api(`/api/launcher/delete-plan/${encodeURIComponent(x)}`); }
  catch (err) { d.querySelector(".series-empty").outerHTML = `<div class="dlg-err">${esc(err.message)}</div>`; return; }
  if (!d.open) return;
  const r = p.remote;
  const remoteHtml = !r.path ? `<li><span class="note">No HPC folder recorded${r.error ? `: ${esc(r.error)}` : ""}.</span></li>`
    : !r.ok ? `<li><b style="color:var(--st-failed)">Will not be touched:</b> <span class="mono">${esc(r.path)}</span> — ${esc(r.error)}</li>`
    : !r.exists ? `<li><span class="mono">${esc(r.path)}</span> <span class="note">— already gone, nothing to delete</span></li>`
    : `<li><span class="mono">${esc(r.path)}</span> on ${esc(r.host)}<br><b>${fmtBytes(r.bytes)}</b> in ${r.files.toLocaleString()} files${r.partial ? " (count stopped early: there are more)" : ""}
        ${r.nc_files ? `<br><b style="color:var(--st-failed)">including ${r.nc_files.toLocaleString()} NetCDF outputs (${fmtBytes(r.nc_bytes)})</b>` : ""}
        <br><span class="note">contains: ${esc((r.top || []).join(", "))}</span></li>`;
  const s = p.store || {};
  const storeHtml = !s.ok ? `<li><b style="color:var(--st-failed)">Will not be touched:</b> <span class="mono">${esc(s.path)}</span> — ${esc(s.error)}</li>`
    : !s.exists ? `<li><span class="mono">${esc(s.path)}</span> <span class="note">— already gone, nothing to delete</span></li>`
    : `<li><span class="mono">${esc(s.path)}</span><br><b>${fmtBytes(s.bytes)}</b> in ${s.files.toLocaleString()} files
        ${s.nc_files ? `<br><b style="color:var(--st-failed)">including ${s.nc_files.toLocaleString()} NetCDF outputs (${fmtBytes(s.nc_bytes)})</b>` : ""}
        <br><span class="note">contains: ${esc((s.top || []).join(", "))}</span></li>`;
  d.innerHTML = `<form method="dialog"><h3>Delete ${esc(x)} forever?</h3>
    <p>${esc(MODEL_NAMES[p.model] || p.model || "")} · ${esc((p.ics || "").toUpperCase())}${p.description ? ` · ${esc(p.description)}` : ""}</p>
    ${p.blockers.length ? p.blockers.map((m) => `<div class="finding f-warn">${ICONS.warn}<div>${esc(m)}</div></div>`).join("") : ""}
    <div class="del-plan">
      <div><div class="chart-title">On mafalda — <span class="mono">autosubmit delete -f ${esc(x)}</span></div>
        <ul><li><span class="mono">${esc(p.local.path)}</span> · ${fmtBytes(p.local.bytes)} in ${p.local.files.toLocaleString()} files</li>${p.local.extra.map((t) => `<li>${esc(t)}</li>`).join("")}</ul></div>
      <div><div class="chart-title">On the HPC — working folder</div><ul>${remoteHtml}</ul></div>
      ${s.path ? `<div><div class="chart-title">On mafalda — output store (archived experiment)</div><ul>${storeHtml}</ul></div>` : ""}
    </div>
    <p class="note">Deletion runs in this order: Autosubmit first, then the HPC folder, then the output-store folder of an archived experiment. If a folder cannot be removed, it is listed as a leftover you can retry. If other tools (e.g. AIUQ-store) catalogue these outputs, re-sync them afterwards.</p>
    ${p.blockers.length ? "" : `<label class="ln-field" style="margin-top:10px"><span>Type <span class="mono">${esc(x)}</span> to confirm</span>
      <input type="text" id="dl-confirm" autocomplete="off" spellcheck="false" style="max-width:200px"></label>`}
    <div class="dlg-err" id="dlg-err"></div>
    <div class="dlg-actions"><button value="cancel" type="submit">Cancel</button>
      ${p.blockers.length ? "" : `<button type="button" class="danger" id="dl-ok" disabled>Delete forever</button>`}</div></form>`;
  if (p.blockers.length) return;
  const inp = $("#dl-confirm"), ok = $("#dl-ok");
  inp.focus();
  inp.addEventListener("input", () => { ok.disabled = inp.value.trim() !== x; });
  ok.onclick = async () => {
    ok.disabled = true;
    try { startTask(await lapi(`/api/launcher/delete/${encodeURIComponent(x)}`, { confirm: inp.value.trim() })); }
    catch (err) { $("#dlg-err").textContent = err.message; ok.disabled = false; }
  };
}
