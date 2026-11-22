"""The AIUQ output convention (runscripts/AIUQst_lib/output.py): one CF-compliant file format for every model.

The files are written from arrays shaped as each model produces them (dimension order, names, directions and
types all differ) and read back with netCDF4: the result must be the same layout for all. When the IOOS
compliance checker is installed (``pip install compliance-checker``) the files are also checked against CF-1.8.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import netCDF4
import numpy as np
import pytest
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runscripts"))
from AIUQst_lib.cards import read_std_version  # noqa: E402
from AIUQst_lib.compression import CompressionError  # noqa: E402
from AIUQst_lib.output import CONVENTIONS, FORMAT_VERSION, RunInfo, time_cell, window_hours, write_output  # noqa: E402
from AIUQst_lib.variables import output_specs  # noqa: E402

STD = read_std_version(str(ROOT), "v100")
MODELS = ("neuralgcm", "ms-aurora", "aifs")
T0 = np.datetime64("2026-11-21T06:00:00", "ns")
VALID = T0 + np.arange(1, 9) * np.timedelta64(6, "h")
LAT = np.linspace(-90, 90, 19)
LON = np.arange(0, 360, 10.0)
RNG = np.random.default_rng(0)
CONFIG = {"MODEL_NAME": "neuralgcm", "MODEL_CHECKPOINT": "/support/models/neuralgcm/stochastic_1_4_deg.pkl",
          "IC_NAME": "ecmwf-an", "RNG_KEY": 3, "RUN_TYPE": "hindcast", "EXPID": "a0zz", "STD_VERSION": "v100",
          "GIT_COMMIT": "abc123", "GIT_BRANCH": "main", "GIT_ORIGIN": "https://github.com/JGrassi97/AIUQ.git",
          "INSTITUTION": "%OUTPUT_FORMAT.INSTITUTION%", "OUT_COMPRESSION": "lossless",
          "OUT_COMPRESSION_KEEPBITS": "", "OUT_COMPRESSION_CHUNKS": "", "OUT_COMPRESSION_CUSTOM": ""}


def specs(model):
    card = yaml.safe_load((ROOT / f"conf/cards/models/{model}_card.yml").read_text())
    return output_specs(card, STD)


def run(model="neuralgcm", **over):
    return RunInfo.from_config({**CONFIG, "MODEL_NAME": model, **over})


def level_field(dims, base=280.0, dtype="f8", levels=(1000, 850, 500), lat=LAT):
    """A pressure-level field with its dimensions in the given order."""
    sizes = {"valid_time": 8, "level": len(levels), "latitude": 19, "longitude": 36, "lat": 19, "lon": 36}
    coords = {"valid_time": VALID, "level": list(levels), "time": T0}
    coords["lat" if "lat" in dims else "latitude"] = lat
    coords["lon" if "lon" in dims else "longitude"] = LON
    values = base + RNG.standard_normal([sizes[d] for d in dims])
    return xr.DataArray(values.astype(dtype), dims=dims, coords=coords)


def surface_field(base=285.0, times=VALID, lat=LAT):
    return xr.DataArray(base + RNG.standard_normal((len(times), 19, 36)), dims=("valid_time", "latitude", "longitude"),
                        coords={"valid_time": times, "latitude": lat, "longitude": LON, "time": T0})


def check_convention(path, name, levels=None, daily=False, period=None, height=None, member=3):
    """Every rule of the convention that does not depend on the variable."""
    with netCDF4.Dataset(path) as nc:
        nc.set_auto_mask(False)
        assert nc.data_model == "NETCDF4" and nc.Conventions == CONVENTIONS and nc.aiuq_output_format == FORMAT_VERSION
        for attr in ("title", "source", "history", "references", "aiuq_model", "aiuq_checkpoint",
                     "aiuq_initial_conditions", "aiuq_experiment", "aiuq_member", "aiuq_commit", "aiuq_compression"):
            assert getattr(nc, attr), attr
        assert not any(d.isunlimited() for d in nc.dimensions.values())
        var = nc[name]
        expected = ("valid_time",) + (("level",) if levels else ()) + ("latitude", "longitude")
        assert var.dimensions == expected and var.dtype == np.float32
        assert var.filters()["zlib"] and var.filters()["shuffle"] and np.isnan(var._FillValue)
        assert var.long_name and var.units and "missing_value" not in var.ncattrs()
        # coordinates: CF attributes, no fill value, no 64-bit integers, fixed directions
        for coord in nc.variables.values():
            assert coord.dtype in (np.float32, np.float64, np.int32), (coord.name, coord.dtype)
            assert coord.name == name or "_FillValue" not in coord.ncattrs(), coord.name
        vt = nc["valid_time"]
        assert vt.standard_name == "time" and vt.axis == "T" and vt.units == "hours since 1970-01-01 00:00:00"
        assert vt.calendar == "proleptic_gregorian" and np.all(np.diff(vt[:]) > 0)
        ref = nc["time"]
        assert ref.dimensions == () and ref.standard_name == "forecast_reference_time" and ref.units == vt.units
        lat, lon = nc["latitude"], nc["longitude"]
        assert (lat.standard_name, lat.units, lat.axis) == ("latitude", "degrees_north", "Y") and np.all(np.diff(lat[:]) > 0)
        assert (lon.standard_name, lon.units, lon.axis) == ("longitude", "degrees_east", "X") and np.all(np.diff(lon[:]) > 0)
        scalars = ["time"]
        if daily:
            assert "step" not in nc.variables
        else:
            step = nc["step"]
            assert step.standard_name == "forecast_period" and step.units == "hours"
            assert np.array_equal(step[:], vt[:] - ref[...])
            scalars.append("step")
        if member is not None:
            assert nc["number"].standard_name == "realization" and int(nc["number"][...]) == member
            scalars.append("number")
        if levels:
            lev = nc["level"]
            assert (lev.standard_name, lev.units, lev.positive, lev.axis) == ("air_pressure", "hPa", "down", "Z")
            assert list(lev[:]) == sorted(levels, reverse=True)
        if height is not None:
            assert nc["height"].standard_name == "height" and nc["height"].units == "m" and float(nc["height"][...]) == height
            scalars.append("height")
        else:
            assert "height" not in nc.variables
        assert var.coordinates.split() == scalars
        if period or daily:
            start, end = (0, 24) if daily else period
            bounds = nc[vt.bounds]
            # the data variable comes before the bounds: the first variable with two dimensions is the data
            assert [k for k, v in nc.variables.items() if v.ndim >= 2][0] == name
            assert bounds.dimensions == ("valid_time", "bnds") and not bounds.ncattrs()
            assert np.array_equal(bounds[:, 0], vt[:] + start) and np.array_equal(bounds[:, 1], vt[:] + end)
            assert var.cell_methods.startswith("valid_time: ")
        else:
            assert "bounds" not in vt.ncattrs() and "cell_methods" not in var.ncattrs()
        return {k: var.getncattr(k) for k in var.ncattrs()}, var[:]


def test_the_three_models_give_the_same_layout(tmp_path):
    """NeuralGCM (longitude before latitude, float64), Aurora (north to south) and AIFS after cdo (lat/lon,
    levels upside down, float32, no initial time in the file) end up identical in layout."""
    reference = level_field(("valid_time", "level", "latitude", "longitude"))
    as_written = {
        "neuralgcm": reference.transpose("valid_time", "level", "longitude", "latitude"),
        "ms-aurora": reference.isel(latitude=slice(None, None, -1)),
        "aifs": reference.isel(level=slice(None, None, -1)).astype("f4").rename(latitude="lat", longitude="lon").drop_vars("time"),
    }
    values = {}
    for model, data in as_written.items():
        path = write_output(str(tmp_path / f"{model}.nc"), data, "t", specs(model)["t"], run(model), step_hours=6,
                            reference_time=T0)
        attrs, values[model] = check_convention(path, "t", levels=[1000, 850, 500])
        assert attrs["standard_name"] == "air_temperature" and attrs["units"] == "K" and attrs["GRIB_paramId"] == 130
    assert np.array_equal(values["neuralgcm"], values["ms-aurora"]) and np.array_equal(values["neuralgcm"], values["aifs"])
    # the values given, at the precision the models compute in (the default, lossless, rounds nothing else)
    truth = reference.transpose("valid_time", "level", "latitude", "longitude").values
    assert np.array_equal(values["aifs"], truth.astype("f4"))


def test_totals_means_daily_values_and_fixed_heights(tmp_path):
    aurora, aifs, ngcm = specs("ms-aurora"), specs("aifs"), specs("neuralgcm")
    days = np.array(["2026-11-21", "2026-11-22"], dtype="datetime64[ns]")
    cases = [  # file, data, variable, spec, daily, period (hours from valid_time), height
        ("t2m", surface_field(), "t2m", aurora["t2m"], False, None, 2.0),
        ("u100", surface_field(5.0), "u100", aifs["u100"], False, None, 100.0),
        ("tp_6h", surface_field(1e-3), "tp", aifs["tp"], False, (-6, 0), None),
        ("tp_step", surface_field(1e-3), "tp", ngcm["tp"], False, (-6, 0), None),
        ("rate", surface_field(1e-5), "avg_tprate", aurora["avg_tprate"], False, (-1, 0), None),
        ("t2m_daily", surface_field(times=days), "t2m", aurora["t2m"], True, None, 2.0),
        ("tp_daily", surface_field(1e-3, times=days), "tp", aifs["tp"], True, None, None),
    ]
    for file, data, name, spec, daily, period, height in cases:
        path = write_output(str(tmp_path / f"{file}.nc"), data, name, spec, run(), daily=daily, step_hours=6)
        attrs, _ = check_convention(path, name, daily=daily, period=period, height=height)
        if name == "tp":
            assert attrs["cell_methods"] == "valid_time: sum" and attrs["standard_name"] == "lwe_thickness_of_precipitation_amount"
        if file == "t2m_daily":
            assert attrs["cell_methods"] == "valid_time: mean (interval: 6 hours)"
        if file == "rate":
            assert attrs["cell_methods"] == "valid_time: mean" and attrs["units"] == "kg m-2 s-1"


def test_model_grid_without_regridding(tmp_path):
    """OUT_RES original for AIFS: one horizontal dimension with latitude and longitude along it."""
    data = xr.DataArray((280 + RNG.standard_normal((8, 2, 50))).astype("f4"), dims=("valid_time", "level", "ncells"),
                        coords={"valid_time": VALID, "level": [500.0, 850.0], "lat": ("ncells", RNG.uniform(-90, 90, 50)),
                                "lon": ("ncells", RNG.uniform(0, 360, 50))})
    path = write_output(str(tmp_path / "native.nc"), data, "t", specs("aifs")["t"], run("aifs"), step_hours=6,
                        reference_time=T0)
    with netCDF4.Dataset(path) as nc:
        assert nc["t"].dimensions == ("valid_time", "level", "values") and list(nc["level"][:]) == [850, 500]
        assert nc["latitude"].dimensions == ("values",) and "axis" not in nc["latitude"].ncattrs()
        assert nc["t"].coordinates.split() == ["time", "step", "number", "latitude", "longitude"]


def test_every_output_of_every_model_card_can_be_written(tmp_path):
    """Each variable a card can write has what the convention needs: CF units, and a period for totals."""
    for model in MODELS:
        for name, spec in specs(model).items():
            data = level_field(("valid_time", "level", "latitude", "longitude")) if spec["type"] == "pl" else surface_field()
            path = write_output(str(tmp_path / f"{model}-{name}.nc"), data, name, spec, run(model), step_hours=6)
            with netCDF4.Dataset(path) as nc:
                units = nc[name].units
                assert "**" not in units and "(" not in units and " of " not in units, (model, name, units)
                assert ("cell_methods" in nc[name].ncattrs()) == bool(spec["accumulated"] or spec["window"]), (model, name)


def test_global_attributes_and_settings(tmp_path):
    info = run("aifs", MODEL_CHECKPOINT="/x/aifs-ens-crps-2.0.ckpt", IC_NAME="ec-earth3", IC_EXPERIMENT="ssp585",
               IC_MEMBER="r4i1p1f1", INSTITUTION="Example institute")
    path = write_output(str(tmp_path / "t2m.nc"), surface_field(), "t2m", specs("aifs")["t2m"], info, step_hours=6)
    with netCDF4.Dataset(path) as nc:
        assert nc.title == "AIFS forecast from EC-Earth3 initial conditions: 2 metre temperature"
        assert nc.institution == "Example institute" and nc.aiuq_ics_experiment == "ssp585" and nc.aiuq_member == "3"
        assert nc.source.startswith("AIFS (aifs-ens-crps-2.0.ckpt)")
        assert json.loads(nc.aiuq_compression)["preset"] == "lossless"                    # the default: nothing rounded
        assert "_QuantizeBitRoundNumberOfSignificantBits" not in nc["t2m"].ncattrs()
    # an unresolved Autosubmit placeholder is "not set": the institution is left out, the compression is the default
    plain = run("aifs", OUT_COMPRESSION="%EXPERIMENT.OUT_COMPRESSION%")
    path = write_output(str(tmp_path / "plain.nc"), surface_field(), "t2m", specs("aifs")["t2m"], plain, step_hours=6)
    with netCDF4.Dataset(path) as nc:
        assert "institution" not in nc.ncattrs() and json.loads(nc.aiuq_compression)["preset"] == "lossless"
    with pytest.raises(CompressionError, match="OUT_COMPRESSION"):
        run(OUT_COMPRESSION="smallest")


def test_periods_from_the_cards():
    assert window_hours("6 h ending at valid_time") == 6 and window_hours("1 h model step ending at valid_time") == 1
    assert window_hours("output step ending at valid_time", 6) == 6 and window_hours(None) is None
    assert time_cell({"accumulated": False, "window": None}, step_hours=6) == (None, None, None)
    assert time_cell({"accumulated": False, "window": None}, daily=True, step_hours=6)[:2] == ("valid_time: mean (interval: 6 hours)", (0.0, 24.0))
    with pytest.raises(ValueError, match="period"):
        time_cell({"accumulated": True, "window": None, "long_name": "x"})


def test_a_failed_write_leaves_no_file(tmp_path):
    bad = surface_field().isel(valid_time=[1, 0, 2])                       # valid times must increase
    with pytest.raises(ValueError, match="increase"):
        write_output(str(tmp_path / "bad.nc"), bad, "t2m", specs("aifs")["t2m"], run(), step_hours=6)
    assert not list(tmp_path.iterdir())


@pytest.mark.skipif(shutil.which("compliance-checker") is None, reason="IOOS compliance-checker is not installed")
def test_cf_compliance(tmp_path):
    """No CF-1.8 error in any kind of file; the only warnings are the two that are known and accepted."""
    aurora, aifs = specs("ms-aurora"), specs("aifs")
    days = np.array(["2026-11-21", "2026-11-22"], dtype="datetime64[ns]")
    native = xr.DataArray((280 + RNG.standard_normal((8, 2, 50))).astype("f4"), dims=("valid_time", "level", "ncells"),
                          coords={"valid_time": VALID, "level": [500.0, 850.0], "time": T0,
                                  "lat": ("ncells", RNG.uniform(-90, 90, 50)), "lon": ("ncells", RNG.uniform(0, 360, 50))})
    files = [
        write_output(str(tmp_path / "level.nc"), level_field(("valid_time", "level", "longitude", "latitude")), "t", aifs["t"], run(), step_hours=6),
        write_output(str(tmp_path / "height.nc"), surface_field(), "t2m", aifs["t2m"], run(), step_hours=6),
        write_output(str(tmp_path / "total.nc"), surface_field(1e-3), "tp", aifs["tp"], run(), step_hours=6),
        write_output(str(tmp_path / "rate.nc"), surface_field(1e-5), "avg_tprate", aurora["avg_tprate"], run(), step_hours=6),
        write_output(str(tmp_path / "daily.nc"), surface_field(times=days), "t2m", aifs["t2m"], run(), daily=True, step_hours=6),
        write_output(str(tmp_path / "native.nc"), native, "t", aifs["t"], run(), step_hours=6),
        # the same kinds of variable stored as integers (scale_factor / add_offset) and with rounded bits
        write_output(str(tmp_path / "packed.nc"), surface_field(), "t2m", aifs["t2m"], run(OUT_COMPRESSION="mixed"), step_hours=6),
        write_output(str(tmp_path / "packed_total.nc"), surface_field(1e-3), "tp", aifs["tp"], run(OUT_COMPRESSION="mixed"), step_hours=6),
        write_output(str(tmp_path / "bits.nc"), level_field(("valid_time", "level", "latitude", "longitude")), "q", aifs["q"], run(OUT_COMPRESSION="bits"), step_hours=6),
    ]
    accepted = ("_QuantizeBitRoundNumberOfSignificantBits",        # written by the netCDF library for rounded data
                "recommended order T, Z, Y, X")                    # a grid with one horizontal dimension has no Y, X
    for path in files:
        report = tmp_path / "report.json"
        subprocess.run(["compliance-checker", "-t", "cf:1.8", "-c", "strict", "-f", "json", "-o", str(report), path],
                       capture_output=True, text=True, timeout=300)
        result = json.loads(report.read_text())["cf:1.8"]
        problems = [msg for group in ("high_priorities", "medium_priorities", "low_priorities")
                    for item in result[group] if item["value"][0] != item["value"][1] for msg in item["msgs"]]
        assert result["high_count"] == 0, (path, problems)
        assert all(any(text in msg for text in accepted) for msg in problems), (path, problems)
