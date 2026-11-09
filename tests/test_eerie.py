"""MARS mapping and IC fallback contracts, without model/GPU dependencies."""
from pathlib import Path
import sys

import numpy as np
import pytest
import xarray as xr
import yaml

pytest.importorskip("ecmwfapi")
pytest.importorskip("cfgrib")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runscripts"))
from download_eerie import (
    add_fallback, build_base_request, build_request, mars_variables, normalize_soil,
)


def card(name):
    return yaml.safe_load((ROOT / f"conf/cards/ics/{name}_card.yml").read_text())


def test_requests_match_historical_archive():
    eerie = card("eerie")
    standard = yaml.safe_load((ROOT / "conf/AIUQ-st/v010/variables.yml").read_text())["data"]
    params = mars_variables(eerie, standard)
    base = build_base_request("1980-01-01", ".25/.25", eerie["expver"])
    assert base["expver"] == "0001"
    assert base["time"] == "00:00:00/06:00:00"
    assert params["sol"] == [260199, 260360]
    assert not {228246, 228247}.intersection(params["sfc"])
    assert build_request(base, "sol", params["sol"])["levelist"] == "1/2"
    assert "levelist" not in build_request(base, "sfc", params["sfc"])


def test_soil_moisture_uses_actual_layers():
    ds = xr.Dataset({"vsw": (("soilLayer", "latitude", "longitude"),
                             np.array([[[0.1]], [[0.3]]]))},
                    coords={"soilLayer": [1., 2.], "latitude": [45.], "longitude": [7.]})
    out = normalize_soil(ds, card("eerie"))
    assert out.swvl1.item() == 0.1
    assert out.swvl2.item() == 0.3
    assert "soilLayer" not in out.swvl1.coords


def target():
    return xr.Dataset(coords={"time": np.array(["1980-01-01T00", "1980-01-01T06"],
                                               dtype="datetime64[ns]"),
                              "latitude": [46., 45.], "longitude": [7., 8.]})


def winds():
    return {228246: {"short_name": "u100", "long_name": "U wind", "units": "m s**-1"},
            228247: {"short_name": "v100", "long_name": "V wind", "units": "m s**-1"}}


def test_climatology_selects_utc_06_and_aligns_latitude(tmp_path):
    values = np.arange(24)[:, None, None] + np.array([45., 46.])[None, :, None]
    values = np.broadcast_to(values, (24, 2, 2))
    climate = xr.Dataset({"u100": (("hour", "latitude", "longitude"), values),
                          "v100": (("hour", "latitude", "longitude"), -values)},
                         coords={"hour": np.arange(24), "latitude": [45., 46.],
                                 "longitude": [7., 8.]})
    path = tmp_path / "climate.nc"
    climate.to_netcdf(path)
    out, missing = add_fallback(target(), winds(), card("climatology"), str(path), True)
    assert missing is None
    np.testing.assert_array_equal(out.u100[:, 0, 0], [46., 52.])
    assert out.u100.attrs["source"] == "ERA5 climatology"
    assert out.u100.dims == ("time", "latitude", "longitude")


def test_missing_climatology_winds_fail_explicitly(tmp_path):
    path = tmp_path / "climate.nc"
    xr.Dataset().to_netcdf(path)
    with pytest.raises(ValueError, match="Extend the ERA5 climatology"):
        add_fallback(target(), winds(), card("climatology"), str(path), True)


def test_fallback_does_not_hide_uncovered_grid(tmp_path):
    path = tmp_path / "climate.nc"
    xr.Dataset({v: (("hour", "latitude", "longitude"), np.ones((2, 2, 2)))
                for v in ["u100", "v100"]},
               coords={"hour": [0, 6], "latitude": [0., 1.], "longitude": [7., 8.]}).to_netcdf(path)
    with pytest.raises(ValueError, match="Non-finite u100"):
        add_fallback(target(), winds(), card("climatology"), str(path), True)
