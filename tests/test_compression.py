"""Output compression (runscripts/AIUQst_lib/compression.py): every preset, on files shaped as each model writes
them, written by SIM (AIUQst_lib.output.write_output) or compressed afterwards (compress_outputs.py)."""
import json
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pytest
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runscripts"))
import compress_outputs  # noqa: E402
from AIUQst_lib.cards import read_std_version  # noqa: E402
from AIUQst_lib.compression import (PRESETS, CompressionError, Plan, Scheme, check_values, compress_file,  # noqa: E402
                                    parse_custom, read_tag, read_values, stored_plan, xarray_encoding)
from AIUQst_lib.output import RunInfo, write_output  # noqa: E402
from AIUQst_lib.variables import output_specs  # noqa: E402

STD = read_std_version(str(ROOT), "v100")
SPECS = compress_outputs.standard_specs(str(ROOT / "conf/AIUQ-st/v100"))
T0 = np.datetime64("2026-11-21T06:00:00", "ns")
VALID = T0 + np.arange(1, 9) * np.timedelta64(6, "h")
LAT, LON = np.linspace(-90, 90, 91), np.arange(0, 360, 2.0)
RNG = np.random.default_rng(1)


def card_specs(model="aifs"):
    return output_specs(yaml.safe_load((ROOT / f"conf/cards/models/{model}_card.yml").read_text()), STD)


def smooth(shape, base, amplitude, noise):
    """A field with large-scale structure and a little noise, like a model output."""
    lat = np.cos(np.deg2rad(LAT))[:, None] * np.ones((1, LON.size))
    wave = np.sin(np.deg2rad(LON))[None, :] * np.ones((LAT.size, 1))
    field = base + amplitude * (lat + 0.3 * wave)
    return field + noise * RNG.standard_normal(shape + field.shape)


def field(name):
    """(DataArray, AIUQ-st name) as SIM hands them to the writer, with realistic values."""
    coords = {"valid_time": VALID, "latitude": LAT, "longitude": LON, "time": T0}
    if name == "t":
        data = smooth((8, 2), 250.0, 40.0, 0.5)
        return xr.DataArray(data, dims=("valid_time", "level", "latitude", "longitude"), coords={**coords, "level": [850, 500]})
    values = {"t2m": smooth((8,), 255.0, 40.0, 0.5), "msl": smooth((8,), 100000.0, 2000.0, 50.0),
              "tp": np.clip(smooth((8,), -2e-3, 6e-3, 1e-3), 0, None), "q": np.clip(smooth((8,), 1e-3, 8e-3, 5e-4), 1e-6, None),
              "tcc": np.clip(smooth((8,), 0.2, 0.6, 0.2), 0, 1)}[name]
    return xr.DataArray(values, dims=("valid_time", "latitude", "longitude"), coords=coords)


def run(preset, **opts):
    config = {"MODEL_NAME": "aifs", "MODEL_CHECKPOINT": "/x/aifs-ens-crps-1.0.ckpt", "IC_NAME": "era5", "RNG_KEY": 1,
              "EXPID": "a0zz", "STD_VERSION": "v100", "OUT_COMPRESSION": preset}
    config.update({f"OUT_COMPRESSION_{k.upper()}": v for k, v in opts.items()})
    return RunInfo.from_config(config)


def write(tmp_path, name, preset, data=None, **opts):
    data = field(name) if data is None else data
    path = write_output(str(tmp_path / f"{name}-{preset}.nc"), data, name, card_specs()[name], run(preset, **opts), step_hours=6)
    return path, data


# ---------------------------------------------------------------------------------------------- the schemes
def test_plans_of_the_presets():
    spec = card_specs()
    assert Scheme("none").plan("t2m", spec["t2m"]) == Plan("none") and Scheme("lossless").plan("q", spec["q"]) == Plan("lossless")
    assert Scheme("bits", keepbits=10).plan("t2m", spec["t2m"]) == Plan("bits", keepbits=10)
    mixed, integer = Scheme("mixed"), Scheme("integer")
    assert mixed.plan("t2m", spec["t2m"]) == Plan("integer", step=1.0, offset=273.15)          # whole degrees Celsius
    assert mixed.plan("msl", spec["msl"]) == Plan("integer", step=100.0) and mixed.plan("z", spec["z"]).step == 9.80665
    assert mixed.plan("tp", spec["tp"]).step == 1e-4 and integer.plan("tp", spec["tp"]).step == 1e-3   # 0.1 mm and 1 mm
    assert mixed.plan("q", spec["q"]) == Plan("bits", keepbits=12)                                # no step: its bits
    assert Scheme("mixed", keepbits=8).plan("q", spec["q"]).keepbits == 8
    assert Scheme().preset == "lossless" and Scheme("%EXPERIMENT.OUT_COMPRESSION%").preset == "lossless"


def test_the_steps_are_in_the_standard():
    """One table for the code, the Launcher and the documentation: 'pack' of the AIUQ-st definitions."""
    packed = {name: spec["pack"] for name, spec in SPECS.items() if "pack" in spec}
    assert {"t2m", "d2m", "t", "skt", "stl1", "stl2", "sst", "u10", "v10", "u100", "v100", "u", "v", "i10fg", "msl", "sp",
            "z", "tp", "cp", "sf", "ro", "avg_tprate", "avg_tsrwe", "tcc", "lcc", "mcc", "hcc", "ci", "fscov", "tcw",
            "tcwv"} == set(packed)
    assert all({"mixed", "integer"} <= set(rule) <= {"mixed", "integer", "offset"} and rule["mixed"] > 0 for rule in packed.values())
    assert all(rule["mixed"] <= rule["integer"] for rule in packed.values())
    assert packed["avg_tprate"] == {"mixed": 0.1 / 21600, "integer": 1 / 21600}
    assert {name for name in ("q", "clwc", "ciwc", "w", "swvl1", "sd", "ssrd", "strd", "avg_sdswrf") if name in packed} == set()


def test_custom_rules():
    rules = ["t2m: {step: 0.5, offset: 273.15}", "tp: {step: 0.0001}", "q: {keepbits: 10}", "z: {mode: lossless}",
             "default: {keepbits: 14}"]
    as_rendered_by_autosubmit = str(rules)                         # a list reaches the job configuration as its repr
    for given in (rules, as_rendered_by_autosubmit, parse_custom(rules)):
        scheme = Scheme("custom", custom=given)
        assert scheme.plan("t2m") == Plan("integer", step=0.5, offset=273.15) and scheme.plan("tp") == Plan("integer", step=1e-4)
        assert scheme.plan("q") == Plan("bits", keepbits=10) and scheme.plan("z") == Plan("lossless")
        assert scheme.plan("u10") == Plan("bits", keepbits=14)
    assert Scheme("custom", custom=["T2M: {STEP: 2}"]).plan("t2m").step == 2.0     # Autosubmit upper-cases nothing here, but be safe


@pytest.mark.parametrize("kwargs, message", [
    (dict(preset="zstd"), "OUT_COMPRESSION"),
    (dict(preset="bits", keepbits=3), "outside 4-23"), (dict(preset="bits", keepbits=24), "outside 4-23"),
    (dict(preset="bits", keepbits="many"), "whole number"),
    (dict(preset="lossless", chunks="line"), "chunks"), (dict(preset="lossless", chunks=[4, 45]), "chunks"),
    (dict(preset="lossless", chunks=[4, 0, 90]), "chunks"), (dict(preset="lossless", complevel=0), "1 to 9"),
    (dict(preset="custom"), "needs its rules"),                    # a nested mapping in main.yml does not reach the job
    (dict(preset="custom", custom=["t2m: {step: 0}"]), "greater than zero"),
    (dict(preset="custom", custom=["t2m: {step: -1}"]), "greater than zero"),
    (dict(preset="custom", custom=["t2m: {digits: 3}"]), "unknown key"),
    (dict(preset="custom", custom=["t2m: {}"]), "give 'step'"),
    (dict(preset="mixed", custom=["t2m: {step: 1}"]), "only for OUT_COMPRESSION custom"),
])
def test_schemes_that_cannot_be_used(kwargs, message):
    with pytest.raises(CompressionError, match=message):
        Scheme(**kwargs)


def test_chunks():
    sizes = {"valid_time": 44, "level": 2, "latitude": 721, "longitude": 1440, "lat": 721, "lon": 1440, "values": 542080}
    dims = ("valid_time", "level", "latitude", "longitude")
    assert Scheme().chunksizes(dims, sizes) == [4, 1, 45, 90] and Scheme(chunks="map").chunksizes(dims, sizes) == [1, 1, 721, 1440]
    assert Scheme(chunks=[8, 30, 60]).chunksizes(dims, sizes) == [8, 1, 30, 60] and Scheme(chunks="[8, 30, 60]").chunks == (8, 30, 60)
    # by name: NeuralGCM stores longitude before latitude, AIFS after cdo names them lat / lon
    assert Scheme().chunksizes(("valid_time", "level", "longitude", "latitude"), sizes) == [4, 1, 90, 45]
    assert Scheme().chunksizes(("valid_time", "lat", "lon"), sizes) == [4, 45, 90]
    # the AIFS grid without regridding: along its one horizontal dimension
    assert Scheme().chunksizes(("valid_time", "level", "values"), sizes) == [4, 1, 4050]
    assert Scheme(chunks="map").chunksizes(("valid_time", "values"), sizes) == [1, 542080]
    assert Scheme().chunksizes(("valid_time", "latitude", "longitude"), {"valid_time": 2, "latitude": 19, "longitude": 36}) == [2, 19, 36]


# ------------------------------------------------------------------------------------------- written by SIM
@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("name", ["t", "t2m", "msl", "tp", "q", "tcc"])
def test_round_trip_of_every_preset(tmp_path, preset, name):
    custom = ["t2m: {step: 0.5, offset: 273.15}", "tp: {step: 0.0001}", "q: {keepbits: 10}", "t: {mode: lossless}", "default: {keepbits: 12}"]
    path, data = write(tmp_path, name, preset, **({"custom": custom} if preset == "custom" else {}))
    scheme = Scheme(preset, custom=custom if preset == "custom" else None)
    plan = scheme.plan(name, card_specs()[name])
    truth = data.transpose("valid_time", ...).values
    if "level" in data.dims:
        truth = truth[:, ::-1] if list(data["level"].values) != sorted(data["level"].values, reverse=True) else truth
    with netCDF4.Dataset(path) as nc:
        var = nc[name]
        assert stored_plan(var) == plan
        check_values(truth, read_values(var), plan)                         # the bound of this plan
        tag = json.loads(nc.aiuq_compression)
        assert tag["schema"] == 1 and tag["preset"] == preset and tag["variables"] == {name: plan.describe()}
        assert var.units == card_specs()[name]["units"] and var.long_name == card_specs()[name]["long_name"]
        assert np.array_equal(nc["latitude"][:], LAT) and np.array_equal(nc["longitude"][:], LON)
        if plan.mode == "integer":
            assert var.dtype in (np.int16, np.int32) and var.getncattr("_FillValue") in (-32767, -2147483647)
            assert var.scale_factor == plan.step and var.add_offset == plan.offset
        elif plan.mode == "none":
            assert var.dtype == data.dtype and var.chunking() == "contiguous" and not var.filters()["zlib"]
        else:
            assert var.dtype == np.float32 and var.filters()["zlib"] and var.filters()["shuffle"]
    # what a reader gets: floating point in the units of the variable, whatever is on disk
    decoded = xr.open_dataset(path)[name]
    assert decoded.dtype.kind == "f" and decoded.dims == (("valid_time", "level") if name == "t" else ("valid_time",)) + ("latitude", "longitude")
    limit = {"integer": plan.step / 2 * 1.001, "bits": 2.0 ** -plan.keepbits * np.abs(truth).max() * 1.001 if plan.mode == "bits" else 0}.get(plan.mode, 1e-3)
    assert float(np.abs(decoded.values - truth).max()) <= max(limit, np.abs(truth).max() * 1e-6)


def test_missing_values_and_integer_width(tmp_path):
    data = field("t2m")
    data.values[:, :10, :20] = np.nan                                       # e.g. the wave fields over land
    path, _ = write(tmp_path, "t2m", "mixed", data)
    with netCDF4.Dataset(path) as nc:
        raw = nc["t2m"]
        raw.set_auto_maskandscale(False)
        assert raw.dtype == np.int16 and np.all(raw[:, :10, :20] == -32767) and not np.any(raw[:, 10:, :] == -32767)
    assert np.array_equal(np.isnan(xr.open_dataset(path)["t2m"].values), np.isnan(data.values))
    # the geopotential in whole metres does not fit in 16 bits above 32 km: 32-bit integers then
    high = xr.DataArray(np.full((8, 91, 180), 33000 * 9.80665), dims=("valid_time", "latitude", "longitude"),
                        coords={"valid_time": VALID, "latitude": LAT, "longitude": LON, "time": T0})
    path = write_output(str(tmp_path / "z.nc"), high, "z", card_specs()["z"], run("mixed"), step_hours=6)
    with netCDF4.Dataset(path) as nc:
        assert nc["z"].dtype == np.int32
    with pytest.raises(CompressionError, match="too small"):
        write(tmp_path, "msl", "custom", custom=["msl: {step: 1e-9}"])
    assert not list(tmp_path.glob("*.writing"))


def test_sizes_decrease_with_the_presets(tmp_path):
    """On fields like the models' (structure plus noise): none > lossless > bits > mixed."""
    total = {}
    for preset in ("none", "lossless", "bits", "mixed"):
        total[preset] = sum(Path(write(tmp_path, name, preset)[0]).stat().st_size for name in ("t", "t2m", "msl", "tp", "tcc"))
    assert total["mixed"] < total["bits"] < total["lossless"] < total["none"], total
    assert total["none"] > 4 * total["mixed"]


def test_xarray_reads_packed_files_and_can_write_them(tmp_path):
    paths = [write_output(str(tmp_path / f"t2m-{member}.nc"), field("t2m"), "t2m", card_specs()["t2m"],
                          RunInfo.from_config({"MODEL_NAME": "aifs", "RNG_KEY": member, "OUT_COMPRESSION": "mixed"}), step_hours=6)
             for member in (1, 2)]
    members = xr.open_mfdataset(paths, combine="nested", concat_dim="number")
    assert members["t2m"].dims[0] == "number" and members["t2m"].dtype.kind == "f" and list(members["number"].values) == [1, 2]
    assert 200 < float(members["t2m"].mean()) < 320 and members["t2m"].attrs["units"] == "K"
    # the same storage through xarray's encoding
    scheme, data = Scheme("mixed"), field("msl")
    plan = scheme.plan("msl", card_specs()["msl"])
    data.to_dataset(name="msl").to_netcdf(tmp_path / "xr.nc", encoding={"msl": xarray_encoding("msl", data.dims, data.sizes, plan, scheme)})
    with netCDF4.Dataset(tmp_path / "xr.nc") as nc:
        assert stored_plan(nc["msl"]) == plan and list(nc["msl"].chunking()) == scheme.chunksizes(data.dims, data.sizes)
        check_values(data.values, read_values(nc["msl"]), plan)


# ------------------------------------------------------------------------------- compressing existing files
def legacy_file(path, model, tag=None, packed=None, keepbits=None):
    """A file as SIM wrote it before the output format, optionally as the earlier scripts left it."""
    dims = {"neuralgcm": ("valid_time", "level", "longitude", "latitude"), "ms-aurora": ("valid_time", "level", "latitude", "longitude"),
            "aifs": ("valid_time", "level", "lat", "lon"), "aifs-native": ("valid_time", "level", "values")}[model]
    sizes = {"valid_time": 8, "level": 2, "latitude": 91, "longitude": 180, "lat": 91, "lon": 180, "values": 500}
    values = 250.0 + 40 * RNG.random([sizes[d] for d in dims]) + np.arange(8).reshape([8] + [1] * (len(dims) - 1))
    values[0, 0].flat[:7] = np.nan
    with netCDF4.Dataset(path, "w") as nc:
        for dim in dims:
            nc.createDimension(dim, sizes[dim])
        nc.createVariable("valid_time", "i8", ("valid_time",)).units = "hours since 2026-11-21 06:00:00"
        nc["valid_time"][:] = np.arange(8) * 6
        nc.createVariable("level", "i8", ("level",))[:] = [1000, 850]
        for dim in dims[2:]:
            nc.createVariable("latitude" if dim == "values" else dim, "f8", (dim,))[:] = np.linspace(-80, 80, sizes[dim])
        kind = "f4" if model.startswith("aifs") else "f8"
        values = values.astype(kind).astype("f8")             # what the file holds
        if packed:
            var = nc.createVariable("t", "i2", dims, zlib=True, complevel=1, shuffle=True, fill_value=-32767,
                                    chunksizes=Scheme().chunksizes(dims, sizes))
            var.scale_factor, var.add_offset = packed
            var.set_auto_maskandscale(False)
            ints = np.round((values - packed[1]) / packed[0])
            var[:] = np.where(np.isnan(ints), -32767, ints).astype("i2")
        elif keepbits:
            var = nc.createVariable("t", "f4", dims, zlib=True, complevel=1, shuffle=True, significant_digits=keepbits,
                                    quantize_mode="BitRound", chunksizes=Scheme().chunksizes(dims, sizes), fill_value=np.float32(np.nan))
            var[:] = values.astype("f4")
        else:
            var = nc.createVariable("t", kind, dims, fill_value=np.nan)
            var[:] = values
        var.units, var.long_name = "K", "Temperature"
        if tag:
            nc.aiuq_compression = tag
    return values


@pytest.mark.parametrize("model", ["neuralgcm", "ms-aurora", "aifs", "aifs-native"])
@pytest.mark.parametrize("preset", ["lossless", "bits", "integer", "mixed"])
def test_compress_existing_files(tmp_path, model, preset):
    path = str(tmp_path / "out-2026-11-21-2026-11-23-1-t.nc")
    values = legacy_file(path, model)
    scheme = Scheme(preset)
    plan = scheme.plan("t", SPECS["t"])
    status, before, after = compress_file(path, scheme, SPECS)
    assert status == "ok" and after < before
    with netCDF4.Dataset(path) as nc:
        var = nc["t"]
        assert stored_plan(var) == plan and var.units == "K" and var.long_name == "Temperature"
        assert var.dimensions == {"neuralgcm": ("valid_time", "level", "longitude", "latitude"), "ms-aurora": ("valid_time", "level", "latitude", "longitude"),
                                  "aifs": ("valid_time", "level", "lat", "lon"), "aifs-native": ("valid_time", "level", "values")}[model]
        assert list(var.chunking()) == scheme.chunksizes(var.dimensions, {d: len(nc.dimensions[d]) for d in var.dimensions})
        check_values(values, read_values(var), plan)
        assert list(nc["level"][:]) == [1000, 850] and nc["valid_time"].units == "hours since 2026-11-21 06:00:00"
        assert json.loads(nc.aiuq_compression)["variables"]["t"] == plan.describe()
    # again: nothing to do
    assert compress_file(path, scheme, SPECS)[0] == "skip"
    assert not list(tmp_path.glob("*.compressing"))


def test_files_of_the_earlier_scripts_are_recognised(tmp_path):
    mixed_tag = "AIUQ mixed integer packing v1 (t 1 C, wind 1 m/s, pressure 1 hPa, z 1 m, precip 0.1 mm, cloud 1 %)"
    bits_tag = "float32, BitRound keepbits=12, zlib level 1 + shuffle, chunks valid_time 4 x lat 45 x lon 90"
    assert read_tag(mixed_tag)["preset"] == "mixed" and read_tag(bits_tag) == {**read_tag(bits_tag), "preset": "bits", "keepbits": 12}
    assert read_tag(Scheme("mixed").tag())["preset"] == "mixed" and read_tag("") == {} and read_tag("something else") == {}
    packed, rounded = str(tmp_path / "packed.nc"), str(tmp_path / "rounded.nc")
    legacy_file(packed, "ms-aurora", tag=mixed_tag, packed=(1.0, 273.15))
    legacy_file(rounded, "neuralgcm", tag=bits_tag, keepbits=12)
    stamp = Path(packed).stat().st_mtime_ns
    # already what the scheme asks: untouched, the old tag stays
    assert compress_file(packed, Scheme("mixed"), SPECS)[0] == "skip" and compress_file(packed, Scheme("integer"), SPECS)[0] == "skip"
    assert compress_file(rounded, Scheme("bits"), SPECS)[0] == "skip" and Path(packed).stat().st_mtime_ns == stamp
    with netCDF4.Dataset(packed) as nc:
        assert nc.aiuq_compression == mixed_tag
    # asking for more precision than the file has cannot bring it back: left alone unless forced
    assert compress_file(packed, Scheme("lossless"), SPECS)[0] == "kept" and compress_file(rounded, Scheme("lossless"), SPECS)[0] == "kept"
    assert compress_file(packed, Scheme("bits"), SPECS)[0] == "kept"
    assert compress_file(rounded, Scheme("bits", keepbits=10), SPECS)[0] == "ok"            # fewer bits: a real change
    assert compress_file(rounded, Scheme("mixed"), SPECS)[0] == "ok" and compress_file(packed, Scheme("lossless"), SPECS, force=True)[0] == "ok"


def test_command_line_tool(tmp_path, capsys):
    outputs = tmp_path / "a0zz" / "outputs"
    for member in ("1", "2"):
        (outputs / "t" / member).mkdir(parents=True)
        legacy_file(str(outputs / "t" / member / f"out-2026-11-21-2026-11-23-{member}-t.nc"), "neuralgcm")
    (outputs / "checkpoints").mkdir()
    legacy_file(str(outputs / "checkpoints" / "out-restart.nc"), "neuralgcm")
    legacy_file(str(outputs / "t" / "1" / "out-2026-11-21-2026-11-23-1-t_temp.nc"), "aifs")
    files = compress_outputs.output_files(str(tmp_path / "a0zz"))
    assert [Path(f).parent.name for f in files] == ["1", "2"]                    # no checkpoint, no intermediate file
    untouched = (outputs / "checkpoints" / "out-restart.nc").stat().st_size
    assert compress_outputs.main([str(tmp_path / "a0zz"), "--scheme", "mixed", "--dry", "--procs", "1"]) == 0
    assert "2 dry" in capsys.readouterr().out and stored_plan_of(files[0]) == Plan("none")
    assert compress_outputs.main([str(tmp_path / "a0zz"), "--scheme", "mixed", "--procs", "2"]) == 0
    assert "2 ok" in capsys.readouterr().out and stored_plan_of(files[0]) == Plan("integer", step=1.0, offset=273.15)
    assert compress_outputs.main([str(tmp_path / "a0zz"), "--scheme", "mixed", "--procs", "1"]) == 0
    assert "2 skip" in capsys.readouterr().out
    assert (outputs / "checkpoints" / "out-restart.nc").stat().st_size == untouched
    assert compress_outputs.main([files[0], "--scheme", "custom", "--custom", "t: {step: 0.1}", "--chunks", "2,30,60", "--force", "--procs", "1"]) == 0
    with netCDF4.Dataset(files[0]) as nc:
        assert nc["t"].scale_factor == 0.1 and list(nc["t"].chunking()) == [2, 1, 60, 30]
    with pytest.raises(SystemExit, match="needs its rules"):
        compress_outputs.main([files[0], "--scheme", "custom"])


def stored_plan_of(path):
    with netCDF4.Dataset(path) as nc:
        return stored_plan(nc["t"])


def test_a_file_that_does_not_hold_what_was_written_is_refused():
    plan = Plan("integer", step=1.0)
    with pytest.raises(CompressionError, match="half the step"):
        check_values(np.array([1.0, 2.0]), np.array([1.0, 2.6]), plan)
    with pytest.raises(CompressionError, match="missing values"):
        check_values(np.array([1.0, np.nan]), np.array([1.0, 2.0]), plan)
    with pytest.raises(CompressionError, match="2\\*\\*-12"):
        check_values(np.array([280.0]), np.array([280.2]), Plan("bits", keepbits=12))
    with pytest.raises(CompressionError, match="float32"):
        check_values(np.array([280.0]), np.array([280.00001]), Plan("lossless"))
