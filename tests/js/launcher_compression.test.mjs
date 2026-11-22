// Smoke test of the Launcher's compression controls (tools/dashboard/aiuq_dash/static/launcher.js) without a
// browser: the script is loaded with the few globals it takes from app.js, then the form state is driven through
// the same handlers the page calls. Run: node tests/js/launcher_compression.test.mjs
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const source = fs.readFileSync(path.join(here, "../../tools/dashboard/aiuq_dash/static/launcher.js"), "utf8");
const form = { innerHTML: "", querySelector: () => null, querySelectorAll: () => [], contains: () => false };
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const context = vm.createContext({
  console, URLSearchParams, setTimeout: () => 0, clearTimeout: () => {},
  location: { search: "" }, history: { replaceState() {} },
  document: { activeElement: null, querySelector: () => null, querySelectorAll: () => [] },
  $: (sel) => (sel === "#ln-form" ? form : null),
  S: { list: [] }, ICONS: { warn: "", info: "" }, esc,
  fmtBytes: (b) => `${(b / 1e9).toFixed(1)} GB`, fmtDate: (d) => d, fmtDur: (s) => `${s} s`,
  seg: (k, options, current) => `<div class="seg" data-k="${k}">${options.map(([v, t]) => `<button data-v="${v}" class="${v === current ? "on" : ""}">${t}</button>`).join("")}</div>`,
  toast() {}, api: async () => ({}), groupedExps: () => [],
});
vm.runInContext(source + "\n;globalThis.__L = L; globalThis.__f = { fixCfg, buildForm, compressionBlock, cmpPreset, cmpStorageText, onFormClick, onFormChange, onFormInput, outVars };", context);
const L = context.__L, f = context.__f;

const variable = (name, type, pack, units = "K") => ({ name, long: name.toUpperCase(), units, type, group: type, pack });
const variables = [variable("t2m", "sfc", { mixed: 1, integer: 1, offset: 273.15 }), variable("tp", "sfc", { mixed: 0.0001, integer: 0.001 }, "m"),
  variable("q", "pl", null, "kg kg-1"), variable("z", "pl", { mixed: 9.80665, integer: 9.80665 }, "m2 s-2"), variable("tcc", "sfc", { mixed: 0.01, integer: 0.01 }, "1")];
L.opts = {
  platform: "FELIPE", ics: ["era5", "ecmwf-an"], forcings: ["ESA-CCI-v3.0"], chunk_units: ["day", "month"], resolutions: ["original", "1"], freqs: ["original", "daily"],
  checkpoints_checked: true, store_by_ics: {},
  models: [{ name: "aifs", label: "AIFS", producer: "ecmwf", on_platform: true, amip: false, levels: [1000, 850, 500], variables,
    checkpoints: [{ name: "aifs-ens-crps-1.0.ckpt", present: true, ensemble: true, support: {}, variables: null, levels: null }] }],
  compression: { supported: true, keepbits: { min: 4, max: 23, default: 12 }, chunks: ["point", "map"], precipitation: ["tp"],
    defaults: { era5: "lossless", "ecmwf-an": "mixed" },
    presets: ["none", "lossless", "bits", "integer", "mixed", "custom"].map((name) => ({ name, label: name, description: `${name} preset` })) },
};
L.branches = ["main"];
Object.assign(L.cfg, { model: "aifs", checkpoint: "aifs-ens-crps-1.0.ckpt", out_vars: ["t2m", "tp", "q", "z"], out_levs: [850, 500] });
const click = (k, v) => f.onFormClick({ target: { closest: (sel) => (sel.startsWith("a[") ? null : { dataset: { k, v }, disabled: false, closest: () => null }) } });
const segClick = (k, v) => f.onFormClick({ target: { closest: (sel) => (sel.startsWith("a[") ? null : { dataset: { v }, disabled: false, closest: () => ({ dataset: { k } }) }) } });
const input = (k, value) => f.onFormInput({ target: { dataset: { k }, value } });
const change = (k, value) => f.onFormChange({ target: { dataset: { k }, value } });

// the default follows the initial conditions until a preset is chosen
f.fixCfg(); f.buildForm();
assert.equal(f.cmpPreset(), "lossless");
assert.match(form.innerHTML, /Output compression/);
assert.match(form.innerHTML, /Proposed for ERA5 initial conditions/);
click("ics", "ecmwf-an");
assert.equal(f.cmpPreset(), "mixed");
assert.equal(L.cfg.out_compression, null, "nothing is sent while the default is used: the server applies the same default");
assert.match(form.innerHTML, /t2m<\/span> step 1 °C/);
assert.match(form.innerHTML, /tp<\/span> step 0\.1 mm/);
assert.match(form.innerHTML, /q<\/span> 12 significant bits/);
assert.match(form.innerHTML, /z<\/span> step 1 m/);

// choosing a preset, sizes from the preview
L.preview = { compression: { preset: "integer", sizes: { none: 64e9, lossless: 21e9, bits: 8e9, integer: 2.9e9, mixed: 3.2e9 } } };
click("out_compression", "integer");
assert.equal(L.cfg.out_compression, "integer");
assert.match(form.innerHTML, /tp<\/span> step 1 mm/);
assert.match(form.innerHTML, /~2\.9 GB/);
assert.match(form.innerHTML, /Back to the default \(mixed\)/);

// advanced options: only what differs from the default is kept
input("cmp.keepbits", "9"); input("cmp.complevel", "3"); segClick("cmp.chunks", "map");
assert.deepEqual({ ...L.cfg.out_compression_opts }, { keepbits: 9, complevel: 3, chunks: "map" });
assert.match(form.innerHTML, /q<\/span> 9 significant bits/);
input("cmp.keepbits", "12"); input("cmp.complevel", "1"); segClick("cmp.chunks", "point");
assert.deepEqual({ ...L.cfg.out_compression_opts }, {});

// custom: a rule per selected variable, starting from what 'mixed' does
click("out_compression", "custom");
assert.deepEqual(JSON.parse(JSON.stringify(L.cfg.out_compression_opts.custom)),
  { t2m: { step: 1, offset: 273.15 }, tp: { step: 0.0001 }, q: { keepbits: 12 }, z: { step: 9.80665 } });
assert.match(form.innerHTML, /data-k="cr\.t2m\.step" value="1"/);
input("cr.t2m.step", "0.5"); input("cr.q.keepbits", "10"); input("cr.t2m.offset", "0");
assert.deepEqual(JSON.parse(JSON.stringify(L.cfg.out_compression_opts.custom.t2m)), { step: 0.5 });
assert.equal(L.cfg.out_compression_opts.custom.q.keepbits, 10);
change("cr.z.mode", "lossless");
assert.deepEqual(JSON.parse(JSON.stringify(L.cfg.out_compression_opts.custom.z)), { mode: "lossless" });
change("cr.q.mode", "step");
assert.deepEqual(JSON.parse(JSON.stringify(L.cfg.out_compression_opts.custom.q)), { step: 1 });
// the rules follow the selected variables
click("var", "tcc"); click("var", "tp");
assert.deepEqual(Object.keys(L.cfg.out_compression_opts.custom).sort(), ["q", "t2m", "tcc", "z"]);
assert.deepEqual(JSON.parse(JSON.stringify(L.cfg.out_compression_opts.custom.tcc)), { step: 0.01 });
// leaving custom drops the rules, so they are not sent with another preset
click("out_compression", "mixed");
assert.equal("custom" in L.cfg.out_compression_opts, false);

// a branch without the setting: no controls, a note instead
L.opts.compression.supported = false; f.buildForm();
assert.match(form.innerHTML, /does not have this setting/);
assert.doesNotMatch(form.innerHTML, /data-k="out_compression"/);
console.log("launcher compression controls: all checks passed");
