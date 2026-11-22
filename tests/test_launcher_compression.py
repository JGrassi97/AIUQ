"""The Launcher's side of the output compression: what it accepts, warns about, estimates and writes to main.yml.
The jobs read the same settings with runscripts/AIUQst_lib/compression.py: both must mean the same."""
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runscripts"))
from AIUQst_lib.compression import PRESETS, Scheme  # noqa: E402

STD = {name: spec for file in ("variables", "outputs")
       for spec in yaml.safe_load((ROOT / f"conf/AIUQ-st/v100/{file}.yml").read_text())["data"].values()
       for name in [spec["short_name"]]}
VARIABLES = [{"name": n, "type": s["type"], "pack": s.get("pack")} for n, s in STD.items()]
OUT_VARS = ["t2m", "msl", "tp", "t", "z", "q", "tcc"]


@pytest.fixture()
def launcher(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools" / "dashboard"))
    monkeypatch.setenv("AIUQ_STATE_DIR", str(ROOT / ".pytest_cache" / "aiuq-dash-state"))
    for name in [n for n in sys.modules if n == "aiuq_dash" or n.startswith("aiuq_dash.")]:
        monkeypatch.delitem(sys.modules, name)
    from aiuq_dash import launcher as module
    return module


def clean(launcher, cfg, ics="era5", out_vars=OUT_VARS, supported=True):
    errors, warnings = [], []
    preset, opts = launcher.clean_compression(cfg, ics, out_vars, VARIABLES, supported, errors, warnings)
    return preset, opts, errors, warnings


def test_presets_and_defaults(launcher):
    assert tuple(launcher.COMPRESSION_NAMES) == PRESETS
    assert clean(launcher, {})[:2] == ("lossless", {}) and clean(launcher, {}, ics="ecmwf-an")[:2] == ("mixed", {})
    assert clean(launcher, {"out_compression": "bits", "out_compression_opts": {"keepbits": 12, "chunks": "point", "complevel": 1}})[:2] == ("bits", {})
    preset, opts, errors, _ = clean(launcher, {"out_compression": "bits", "out_compression_opts": {"keepbits": 10, "chunks": "map", "complevel": 4}})
    assert (preset, opts, errors) == ("bits", {"keepbits": 10, "chunks": "map", "complevel": 4}, [])
    # a branch whose code does not know the setting: nothing is written, and the form is told
    preset, opts, errors, warnings = clean(launcher, {"out_compression": "mixed"}, supported=False)
    assert (preset, opts, errors) == (None, {}, []) and "does not read OUT_COMPRESSION" in warnings[0]


@pytest.mark.parametrize("cfg, message", [
    ({"out_compression": "zstd"}, "must be one of"),
    ({"out_compression": "bits", "out_compression_opts": {"keepbits": 3}}, "from 4 to 23"),
    ({"out_compression": "bits", "out_compression_opts": {"keepbits": 11.5}}, "whole number"),
    ({"out_compression": "lossless", "out_compression_opts": {"chunks": "line"}}, "Chunks must be"),
    ({"out_compression": "lossless", "out_compression_opts": {"chunks": [4, 0, 90]}}, "Chunks must be"),
    ({"out_compression": "lossless", "out_compression_opts": {"chunks": [4, 45]}}, "Chunks must be"),
    ({"out_compression": "lossless", "out_compression_opts": {"complevel": 10}}, "from 1 to 9"),
    ({"out_compression": "lossless", "out_compression_opts": {"level": 2}}, "Unknown compression option"),
    ({"out_compression": "custom"}, "needs a rule"),
    ({"out_compression": "custom", "out_compression_opts": {"custom": {"t2m": {"step": 0}}}}, "greater than zero"),
    ({"out_compression": "custom", "out_compression_opts": {"custom": {"t2m": {"step": -0.5}}}}, "greater than zero"),
    ({"out_compression": "custom", "out_compression_opts": {"custom": {"u10": {"step": 1}}}}, "not among the output variables"),
    ({"out_compression": "custom", "out_compression_opts": {"custom": {"t2m": {"digits": 2}}}}, "give a step"),
    ({"out_compression": "mixed", "out_compression_opts": {"custom": {"t2m": {"step": 1}}}}, "only for the custom"),
])
def test_errors(launcher, cfg, message):
    errors = clean(launcher, cfg)[2]
    assert any(message in e for e in errors), errors


def test_warnings(launcher):
    assert any("whole millimetres" in w and "tp" in w for w in clean(launcher, {"out_compression": "integer"})[3])
    assert not clean(launcher, {"out_compression": "integer"}, out_vars=["t2m", "msl"])[3]       # no precipitation asked
    assert not clean(launcher, {"out_compression": "mixed"})[3]
    assert any("7 significant bits" in w for w in clean(launcher, {"out_compression": "bits", "out_compression_opts": {"keepbits": 7}})[3])
    # with 'mixed' the bits only matter for the variables without a step (q here)
    assert any("6 significant bits" in w for w in clean(launcher, {"out_compression": "mixed", "out_compression_opts": {"keepbits": 6}})[3])
    assert not clean(launcher, {"out_compression": "mixed", "out_compression_opts": {"keepbits": 6}}, out_vars=["t2m"])[3]


def render(text):
    """The job configuration Autosubmit makes of a main.yml: single values only, lists as their repr."""
    exp = yaml.safe_load(text)["EXPERIMENT"]
    opts = {str(k).upper(): v for k, v in (exp.get("OUT_COMPRESSION_OPTS") or {}).items()}
    return {"OUT_COMPRESSION": exp.get("OUT_COMPRESSION"),
            **{f"OUT_COMPRESSION_{k}": (str(opts[k]) if k in opts else "") for k in ("KEEPBITS", "CHUNKS", "COMPLEVEL", "CUSTOM")}}


@pytest.mark.parametrize("cfg", [
    {"out_compression": "none"}, {"out_compression": "lossless"}, {"out_compression": "integer"}, {"out_compression": "mixed"},
    {"out_compression": "bits", "out_compression_opts": {"keepbits": 9, "chunks": [8, 30, 60], "complevel": 3}},
    {"out_compression": "mixed", "out_compression_opts": {"keepbits": 16, "chunks": "map"}},
    {"out_compression": "custom", "out_compression_opts": {"custom": {
        "t2m": {"step": 0.5, "offset": 273.15}, "tp": {"step": 0.0001}, "q": {"keepbits": 10}, "z": {"mode": "lossless"},
        "default": {"keepbits": 14}}}},
])
def test_the_jobs_read_what_the_launcher_wrote(launcher, cfg):
    """main.yml as written by the Launcher, through Autosubmit, gives the jobs the storage the form showed."""
    preset, opts, errors, _ = clean(launcher, cfg)
    assert not errors
    text = "EXPERIMENT:\n  RUN_TYPE: hindcast\n" + launcher.compression_yaml(preset, opts)
    scheme = Scheme.from_config(render(text))
    assert scheme.preset == preset and scheme.keepbits == opts.get("keepbits", 12) and scheme.complevel == opts.get("complevel", 1)
    assert scheme.chunks == (tuple(opts["chunks"]) if isinstance(opts.get("chunks"), list) else opts.get("chunks", "point"))
    for var in VARIABLES:
        if var["name"] not in OUT_VARS:
            continue
        plan = scheme.plan(var["name"], STD[var["name"]])
        mode, value = launcher.variable_storage(preset, opts, var)
        assert mode == plan.mode, var["name"]
        assert value == {"integer": plan.step, "bits": plan.keepbits}.get(mode), var["name"]
    # and reading the experiment back (the Monitor, "start from an existing experiment") gives the form again
    monkey_exp = yaml.safe_load(text)["EXPERIMENT"]
    from aiuq_dash import local
    again = local._compression(monkey_exp)
    assert again[0] == preset and clean(launcher, {"out_compression": again[0], "out_compression_opts": again[1]})[:2] == (preset, opts)


def test_size_estimate(launcher):
    """The operational forecast the figures come from: 10 members, 10 days, 0.25 degrees, 6-hourly."""
    forecast = ["t2m", "d2m", "u10", "v10", "u100", "v100", "i10fg", "msl", "avg_tprate", "tcc", "t", "u", "v", "z"]
    sizes = launcher.estimate_size("ms-aurora", "aurora-0.25-v1.5-ensemble.ckpt", "mixed", {}, forecast, VARIABLES, 2, "0.25", "original", 10, 1, 11)
    gb = {k: v / 1e9 for k, v in sizes.items()}
    assert 55 < gb["none"] < 75 and 6 < gb["bits"] < 11 and 2.5 < gb["mixed"] < 5        # measured: bits ~9, mixed ~4
    assert gb["mixed"] <= gb["bits"] < gb["lossless"] < gb["none"] and gb["integer"] <= gb["mixed"] and "custom" not in gb
    ngcm = launcher.estimate_size("neuralgcm", "stochastic_precip_2_8_deg.pkl", "mixed", {}, ["t", "u", "v", "z", "q", "tp"], VARIABLES, 2, "0.25", "original", 10, 1, 11)
    assert 4 < ngcm["bits"] / 1e9 < 7 and 1.5 < ngcm["mixed"] / 1e9 < 3.5                # measured: ~5 and ~2
    # scales with members, days, resolution and frequency; the model grid when nothing is regridded
    half = launcher.estimate_size("ms-aurora", "", "mixed", {}, forecast, VARIABLES, 2, "0.25", "original", 5, 1, 11)
    coarse = launcher.estimate_size("ms-aurora", "", "mixed", {}, forecast, VARIABLES, 2, "1", "daily", 10, 1, 11)
    native = launcher.estimate_size("neuralgcm", "stochastic_1_4_deg.pkl", "lossless", {}, ["t"], VARIABLES, 2, "original", "original", 1, 1, 1)
    assert half["mixed"] == pytest.approx(sizes["mixed"] / 2) and coarse["mixed"] < sizes["mixed"] / 50
    assert native["lossless"] == pytest.approx(2 * 2.7 * 128 * 256 * 4)
    custom = launcher.estimate_size("ms-aurora", "", "custom", {"custom": {"default": {"mode": "lossless"}}}, ["t2m"], VARIABLES, 2, "1", "original", 1, 1, 1)
    assert custom["custom"] == custom["lossless"]


def test_main_yml_lines(launcher):
    assert launcher.compression_yaml(None, {}) == "" and launcher.compression_yaml("lossless", {}) == "  OUT_COMPRESSION: lossless\n"
    text = launcher.compression_yaml("custom", {"keepbits": 10, "chunks": [8, 30, 60], "custom": {"t2m": {"step": 0.5, "offset": 273.15}, "default": {"keepbits": 12}}})
    assert yaml.safe_load("EXPERIMENT:\n" + text)["EXPERIMENT"] == {
        "OUT_COMPRESSION": "custom", "OUT_COMPRESSION_OPTS": {"KEEPBITS": 10, "CHUNKS": [8, 30, 60], "CUSTOM": [
            "t2m: {step: 0.5, offset: 273.15}", "default: {keepbits: 12}"]}}


# ------------------------------------------------------------------------------------------------ Monitor
def test_monitor_says_how_the_outputs_are_stored(launcher):
    from aiuq_dash import collector
    new = Scheme("mixed").tag({"t2m": Scheme("mixed").plan("t2m", STD["t2m"])})
    info = collector.compression_info({"out_compression": "lossless"}, new)        # the files win over the configuration
    assert (info["preset"], info["source"]) == ("mixed", "files") and info["variables"]["t2m"]["step"] == 1.0
    old_mixed = "AIUQ mixed integer packing v1 (t 1 C, wind 1 m/s, pressure 1 hPa, z 1 m, precip 0.1 mm, cloud 1 %)"
    old_bits = "float32, BitRound keepbits=12, zlib level 1 + shuffle, chunks valid_time 4 x lat 45 x lon 90"
    assert collector.compression_info({}, old_mixed)["preset"] == "mixed"
    assert collector.compression_info({}, old_bits) == {**collector.compression_info({}, old_bits), "preset": "bits", "keepbits": 12}
    conf = {"out_compression": "bits", "out_compression_opts": {"keepbits": 10}}
    assert collector.compression_info(conf, None) == {"preset": "bits", "keepbits": 10, "variables": {}, "chunks": None,
                                                      "source": "configuration", "detail": None}
    assert collector.compression_info({}, None) is None and collector.compression_info({}, "unknown text") is None


def test_monitor_helpers_read_packed_files(tmp_path):
    """The preview helper (maps, point series) on a file stored as integers and on one of the earlier scripts'."""
    import base64
    import json
    import subprocess

    import netCDF4
    import numpy as np
    import xarray as xr
    from AIUQst_lib.output import RunInfo, write_output

    helpers = ROOT / "tools" / "dashboard" / "aiuq_dash" / "helpers"
    t0 = np.datetime64("2026-11-21T06:00:00", "ns")
    lat, lon = np.linspace(-90, 90, 19), np.arange(0, 360, 10.0)
    truth = 285 + 10 * np.random.default_rng(0).standard_normal((8, 19, 36))
    data = xr.DataArray(truth, dims=("valid_time", "latitude", "longitude"),
                        coords={"valid_time": t0 + np.arange(1, 9) * np.timedelta64(6, "h"), "latitude": lat, "longitude": lon, "time": t0})
    spec = {**STD["t2m"], "accumulated": False, "window": None}
    packed = write_output(str(tmp_path / "packed.nc"), data, "t2m", spec, RunInfo.from_config({"MODEL_NAME": "aifs", "RNG_KEY": 1, "OUT_COMPRESSION": "mixed"}), step_hours=6)
    legacy = str(tmp_path / "legacy.nc")
    with netCDF4.Dataset(legacy, "w") as nc:
        for name, size in (("valid_time", 8), ("latitude", 19), ("longitude", 36)):
            nc.createDimension(name, size)
        nc.createVariable("valid_time", "i8", ("valid_time",)).units = "hours since 2026-11-21 06:00:00"
        nc["valid_time"][:] = np.arange(1, 9) * 6
        nc.createVariable("latitude", "f8", ("latitude",))[:] = lat
        nc.createVariable("longitude", "f8", ("longitude",))[:] = lon
        var = nc.createVariable("t2m", "i2", ("valid_time", "latitude", "longitude"), zlib=True, fill_value=-32767)
        var.scale_factor, var.add_offset = 1.0, 273.15
        var.set_auto_maskandscale(False)
        var[:] = np.round(truth - 273.15).astype("i2")
        nc.aiuq_compression = "AIUQ mixed integer packing v1 (t 1 C, wind 1 m/s, pressure 1 hPa, z 1 m, precip 0.1 mm, cloud 1 %)"

    def helper(request):
        arg = base64.b64encode(json.dumps(request).encode()).decode()
        out = subprocess.run([sys.executable, str(helpers / "hpc_preview.py"), arg], capture_output=True, text=True, timeout=60)
        assert out.returncode == 0, out.stderr[-400:]
        return json.loads(out.stdout)

    for path in (packed, legacy):
        field = helper({"path": path, "var": "t2m", "time": -1})
        assert field["var"] == "t2m" and field["vars"] == ["t2m"]
        assert abs(field["mean"] - truth[-1].mean()) < 0.5 and abs(field["max"] - truth[-1].max()) <= 0.5 and 230 < field["min"] < 300
        series = helper({"mode": "series", "paths": {"1": path}, "var": "t2m", "lat": 0.0, "lon": 40.0})
        values = np.array(next(iter(series["members"].values())), dtype=float)
        assert values.shape == (8,) and np.all(np.abs(values - truth[:, 9, 4]) <= 0.5)
    # and the probe reads the tag the Monitor shows
    spec_probe = __import__("importlib.util").util.spec_from_file_location("hpc_probe", helpers / "hpc_probe.py")
    probe = __import__("importlib.util").util.module_from_spec(spec_probe)
    spec_probe.loader.exec_module(probe)
    out = tmp_path / "outputs" / "t2m" / "1"
    out.mkdir(parents=True)
    Path(packed).rename(out / "out-2026-11-21-2026-11-23-1-t2m.nc")
    files = probe.list_outputs(str(tmp_path / "outputs"))
    assert json.loads(probe.outputs_tag(str(tmp_path / "outputs"), files))["preset"] == "mixed"
    assert probe.outputs_tag(str(tmp_path / "outputs"), []) is None


def test_launcher_form_in_node():
    """The form's compression controls, driven without a browser (tests/js/launcher_compression.test.mjs)."""
    import shutil
    import subprocess
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    out = subprocess.run(["node", str(ROOT / "tests" / "js" / "launcher_compression.test.mjs")], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-800:]
