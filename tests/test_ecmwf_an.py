"""ECMWF Open Data analysis ICs: cycle selection, field selection and standardisation."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import sys

import numpy as np
import pytest
import xarray as xr
import yaml

pytest.importorskip("cfgrib")
pytest.importorskip("scipy")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runscripts"))
import download_ecmwf_an as an  # noqa: E402
from AIUQst_lib.support_data import static_fields_flipped  # noqa: E402

UTC = timezone.utc


def card(name="ecmwf-an"):
    return yaml.safe_load((ROOT / f"conf/cards/ics/{name}_card.yml").read_text())


def standard():
    return {
        "variables": yaml.safe_load((ROOT / "conf/AIUQ-st/v010/variables.yml").read_text()),
        "pressure_levels": yaml.safe_load((ROOT / "conf/AIUQ-st/v010/pressure_levels.yml").read_text()),
    }


def cycle(text):
    return datetime.strptime(text, "%Y-%m-%d %H").replace(tzinfo=UTC)


# ---------------------------------------------------------------- cycle selection

def test_candidates_today_start_from_latest_published_slot():
    cycles = an.candidate_cycles("2026-11-16", 24, now=cycle("2026-11-16 15") + timedelta(minutes=40))
    assert cycles[0] == cycle("2026-11-16 12")
    assert cycles[-1] == cycle("2026-11-15 00")


def test_candidates_past_date_stop_at_18z_of_that_day():
    cycles = an.candidate_cycles("2026-11-10", 0, now=cycle("2026-11-16 12"))
    assert cycles == [cycle(f"2026-11-10 {h:02d}") for h in (18, 12, 6, 0)]


def test_select_cycle_needs_both_states():
    published = {cycle("2026-11-16 06"), cycle("2026-11-16 00"), cycle("2026-11-15 18")}
    # 12Z is not out yet; 06Z and its predecessor 00Z are.
    previous, current = an.select_cycle(
        an.candidate_cycles("2026-11-16", 24, now=cycle("2026-11-16 13")),
        lambda c: {"cycle": c} if c in published else None)
    assert (previous["cycle"], current["cycle"]) == (cycle("2026-11-16 00"), cycle("2026-11-16 06"))


def test_select_cycle_skips_cycle_without_predecessor():
    published = {cycle("2026-11-16 00")}
    with pytest.raises(RuntimeError, match="No ECMWF analysis pair"):
        an.select_cycle([cycle("2026-11-16 00")], lambda c: {"cycle": c} if c in published else None)


def index_text(entries):
    return "\n".join(json.dumps(e) for e in entries).encode()


def test_locate_tries_streams_and_sources_in_order():
    od = card()["open_data"]
    target = an.file_url(od["sources"][1], cycle("2024-03-01 06"), "scda", od, "index")
    calls = []

    def fetch(url):
        calls.append(url)
        if url == target:
            return index_text([{"step": "0", "param": "2t", "levtype": "sfc", "_offset": 0, "_length": 1},
                               {"step": "3", "param": "2t", "levtype": "sfc", "_offset": 1, "_length": 1}])
        return None

    found = an.locate_cycle(cycle("2024-03-01 06"), od, fetch)
    assert found["stream"] == "scda" and found["source"] == od["sources"][1]
    assert [e["step"] for e in found["index"]] == ["0"]
    assert found["grib"].endswith("/20240301/06z/ifs/0p25/scda/20240301060000-0h-scda-fc.grib2")
    assert len(calls) == 4


def test_locate_skips_a_failing_mirror():
    od = card()["open_data"]

    def fetch(url):
        if url.startswith(od["sources"][0]):
            raise OSError("503 SlowDown")
        return index_text([{"step": "0", "param": "2t", "levtype": "sfc", "_offset": 0, "_length": 1}])

    assert an.locate_cycle(cycle("2026-11-16 06"), od, fetch)["source"] == od["sources"][1]


# ---------------------------------------------------------------- field selection

def fake_index(levels=(10, 50, 100, 500, 1000), geopotential="z", soil=(1, 2, 3, 4)):
    entries, offset = [], 0

    def add(levtype, param, levelist=None):
        nonlocal offset
        entry = {"step": "0", "levtype": levtype, "param": param, "_offset": offset, "_length": 10}
        if levelist is not None:
            entry["levelist"] = str(levelist)
        entries.append(entry)
        offset += 10

    for param in [geopotential, "t", "u", "v", "q", "w", "r"]:
        for level in levels:
            add("pl", param, level)
    for param in ["2t", "2d", "10u", "10v", "100u", "100v", "msl", "sp", "skt", "tcw", "tcwv",
                  "sd", "tcc", "lcc", "mcc", "hcc", "lsm", "sithick", "tp", "z"]:
        add("sfc", param)
    for param in ["vsw", "sot"]:
        for layer in soil:
            add("sol", param, layer)
    return entries


def test_selection_takes_only_needed_messages():
    fields = an.required_fields(card(), standard()["variables"]["data"])
    entries = an.select_entries(fake_index(), fields)
    keys = {(e["levtype"], e["param"], e.get("levelist")) for e in entries}
    assert ("pl", "r", "500") not in keys and ("sfc", "tp", None) not in keys
    assert ("sfc", "z", None) not in keys          # orography comes from the static file
    assert ("sfc", "skt", None) in keys and ("sfc", "sithick", None) in keys
    assert {k[2] for k in keys if k[1] in {"vsw", "sot"}} == {"1", "2"}
    assert {k[2] for k in keys if k[1] == "t"} == {"10", "50", "100", "500", "1000"}


def test_geopotential_height_replaces_missing_geopotential():
    fields = an.required_fields(card(), standard()["variables"]["data"])
    entries = an.select_entries(fake_index(geopotential="gh"), fields)
    assert {e["param"] for e in entries if e["levtype"] == "pl"} >= {"gh", "t"}


def test_missing_required_field_or_level_fails():
    fields = an.required_fields(card(), standard()["variables"]["data"])
    index = [e for e in fake_index() if e["param"] != "sot"]
    with pytest.raises(RuntimeError, match="sot"):
        an.select_entries(index, fields)
    with pytest.raises(RuntimeError, match="50 hPa"):
        an.select_entries(fake_index(levels=(100, 500, 1000)), fields)


def test_ranges_are_merged_and_download_is_verified(tmp_path):
    entries = [{"_offset": 0, "_length": 4}, {"_offset": 4, "_length": 4}, {"_offset": 20, "_length": 4}]
    assert an.merge_ranges(entries) == [(0, 7), (20, 23)]
    blob = b"GRIBGRIB" + b"x" * 12 + b"GRIB"
    target = tmp_path / "out.grib2"
    an.download_fields("url", entries, str(target), fetch=lambda url, r: blob[r[0]:r[1] + 1])
    assert target.read_bytes() == b"GRIBGRIBGRIB"
    with pytest.raises(RuntimeError, match="Incomplete download"):
        an.download_fields("url", entries, str(target), fetch=lambda url, r: b"GRIB")


# ---------------------------------------------------------------- vertical levels

def column(levels, values):
    return xr.DataArray(np.asarray(values, dtype="float32")[None, :, None, None],
                        dims=("time", "level", "latitude", "longitude"),
                        coords={"time": [0], "level": levels, "latitude": [0.], "longitude": [0.]})


def test_levels_interpolated_in_log_pressure_and_extrapolated_above_top():
    source = [10, 50, 100, 1000]
    ds = xr.Dataset({"t": column(source, [230., 210., 200., 290.]),
                     "z": column(source, [300000., 200000., 160000., 1000.]),
                     "u": column(source, [30., 20., 10., 0.])})
    target = standard()["pressure_levels"]["values"]
    out = an.to_standard_levels(ds, target)
    assert out.level.values.tolist() == target and out.level.dtype == np.int64
    assert out.t.dims == ("time", "level", "latitude", "longitude") and out.t.dtype == np.float32
    for level, value in zip(source, [230., 210., 200., 290.]):
        assert out.t.sel(level=level).item() == value
    weight = np.log(20 / 10) / np.log(50 / 10)
    np.testing.assert_allclose(out.t.sel(level=20).item(), 230. + weight * (210. - 230.), rtol=1e-6)
    assert out.u.sel(level=1).item() == 30.
    t_std = an.standard_atmosphere_temperature
    np.testing.assert_allclose(out.t.sel(level=1).item(), 230. + t_std(1.) - t_std(10.), rtol=1e-6)
    np.testing.assert_allclose(out.z.sel(level=7).item(),
                               300000. + an.RD * 0.5 * (230. + out.t.sel(level=7).item()) * np.log(10 / 7),
                               rtol=1e-6)
    assert np.all(np.diff(out.z.sel(level=[1, 2, 3, 5, 7, 10]).values.ravel()) < 0)
    assert out.attrs["levels_extrapolated_hPa"] == [1., 2., 3., 5., 7.]


def test_standard_atmosphere_reference_values():
    t_std = an.standard_atmosphere_temperature
    np.testing.assert_allclose(t_std(1.), 270.65)                    # stratopause
    np.testing.assert_allclose(t_std(54.74889), 216.65, rtol=1e-6)   # 20 km
    np.testing.assert_allclose(t_std(8.680187), 228.65, rtol=1e-6)   # 32 km
    assert 227. < t_std(10.) < 228.5
    assert t_std(np.array([10., 5.])).shape == (2,)


# ---------------------------------------------------------------- sea surface

def test_sea_ice_and_sst_derivation():
    grid = {"latitude": [0.], "longitude": [0., 1., 2., 3., 4.]}
    field = lambda v: xr.DataArray(np.array([v], dtype="float32"), dims=("latitude", "longitude"), coords=grid)
    ds = xr.Dataset({
        #                  land    open sea  ocean ice  lake ice  open lake
        "lsm": field([1., 0., 0., 0., 0.]),
        "skt": field([300., 290., 250., 260., 280.]),
        "sithick": field([np.nan, 0., 1.5, np.nan, np.nan]),
    })
    out = an.derive_sea_surface(ds)
    np.testing.assert_array_equal(out.ci.values[0], [np.nan, 0., 1., 1., 0.])
    np.testing.assert_allclose(out.sst.values[0], [np.nan, 290., an.SST_UNDER_ICE, an.SST_UNDER_ICE, 280.])


# ---------------------------------------------------------------- end to end (synthetic)

def synthetic_open_data():
    times = np.array(["2026-11-16T00", "2026-11-16T06"], dtype="datetime64[ns]")
    lat, lon = [60., 0., -60.], [-10., 0., 10.]
    levels = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50, 10]
    rng = np.random.default_rng(0)
    sfc = lambda base: (("time", "latitude", "longitude"), base + rng.random((2, 3, 3)).astype("float32"))
    pl = lambda base: (("time", "isobaricInhPa", "latitude", "longitude"),
                       base + rng.random((2, len(levels), 3, 3)).astype("float32"))
    soil = lambda base: (("time", "soilLayer", "latitude", "longitude"),
                         base + rng.random((2, 4, 3, 3)).astype("float32"))
    data = {name: sfc(base) for name, base in {
        "t2m": 280, "d2m": 275, "u10": 1, "v10": 1, "u100": 2, "v100": 2, "sp": 1e5, "msl": 1e5,
        "tcw": 20, "tcwv": 19, "sd": 0, "tcc": 0, "lcc": 0, "mcc": 0, "hcc": 0, "skt": 280,
        "sithick": 0}.items()}
    data["lsm"] = (("time", "latitude", "longitude"),
                   np.tile(np.array([[1., 0., 0.]] * 3, dtype="float32"), (2, 1, 1)))
    data.update({name: pl(base) for name, base in {"z": 1e4, "t": 250, "u": 5, "v": 5, "q": 1e-3, "w": 0}.items()})
    data.update({"vsw": soil(0.2), "sot": soil(280)})
    return xr.Dataset(data, coords={"time": times, "latitude": lat, "longitude": lon,
                                    "isobaricInhPa": levels, "soilLayer": [1, 2, 3, 4]})


def support_files(tmp_path):
    lat, lon = np.arange(-90., 91., 30.), np.arange(0., 360., 10.)
    shape = (lat.size, lon.size)
    static = xr.Dataset({name: (("latitude", "longitude"), np.full(shape, value))
                         for name, value in {"lsm": 0.5, "sdor": 1., "slor": .1, "z": 10., "slt": 1.}.items()},
                        coords={"latitude": lat, "longitude": lon})
    levels = standard()["pressure_levels"]["values"]
    climate = xr.Dataset({name: (("hour", "level", "latitude", "longitude"),
                                 np.full((4, len(levels)) + shape, 1e-6, dtype="float32"))
                          for name in ("clwc", "ciwc")},
                         coords={"hour": [0, 6, 12, 18], "level": levels, "latitude": lat, "longitude": lon})
    static.to_netcdf(tmp_path / "static.nc")
    climate.to_netcdf(tmp_path / "climatology.nc")
    return {"HPCROOTDIR": str(ROOT), "STATIC_DATA": str(tmp_path / "static.nc"),
            "CLIMATOLOGY_DATA": str(tmp_path / "climatology.nc")}


def test_prepared_ics_contain_the_whole_standard(tmp_path):
    std = standard()
    out = an.prepare_initial_conditions(synthetic_open_data(), support_files(tmp_path), card(), std)
    expected = {v["short_name"] for v in std["variables"]["data"].values()}
    assert expected <= set(out.data_vars)
    assert out.level.values.tolist() == std["pressure_levels"]["values"]
    assert out.longitude.values.tolist() == [0., 10., 350.]
    assert out.sot.dims == ("time", "SoilLevel", "latitude", "longitude") and out.SoilLevel.size == 2
    assert out.t.dims == ("time", "level", "latitude", "longitude")
    assert out.ci.isel(time=0, latitude=0).isnull().values.tolist() == [False, False, True]
    assert out.clwc.attrs["source"] == "ERA5 climatology"
    assert out.orog.attrs["source"] == "static support data"
    assert out.sst.attrs["units"] == "K" and out.sst.attrs["long_name"] == "Sea surface temperature"


# ---------------------------------------------------------------- AIFS orientation

def mask(latitudes, south_land):
    values = np.zeros((len(latitudes), 2))
    values[np.argmin(latitudes) if south_land else np.argmax(latitudes)] = 1
    return xr.Dataset({"lsm": (("latitude", "longitude"), values)},
                      coords={"latitude": latitudes, "longitude": [0., 1.]})


def test_static_orientation_detection():
    assert not static_fields_flipped(mask([90., 0., -90.], south_land=True))
    assert static_fields_flipped(mask([90., 0., -90.], south_land=False))
    assert not static_fields_flipped(mask([-90., 0., 90.], south_land=True))
    with pytest.raises(ValueError):
        static_fields_flipped(xr.Dataset({"lsm": (("latitude", "longitude"), np.zeros((2, 2)))},
                                         coords={"latitude": [90., -90.], "longitude": [0., 1.]}))
