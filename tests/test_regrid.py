"""Output regridding (OUT_RES) leaves no NaN at the poles or across 360 deg."""
from pathlib import Path
import sys

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("scipy")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runscripts"))
from AIUQst_lib.regrid import regrid_latlon, target_grid  # noqa: E402


def _field(lat, lon, times=3, levels=(850, 500)):
    """Smooth field with time and level dims, plus coords/attrs as written by the SIM scripts."""
    la, lo = np.meshgrid(np.deg2rad(lat), np.deg2rad(lon), indexing="ij")
    base = np.cos(la) * np.sin(lo) + 0.01 * np.rad2deg(la)
    data = base[None, None] + np.arange(times)[:, None, None, None] + 10 * np.arange(len(levels))[None, :, None, None]
    valid_time = np.datetime64("2010-01-01T06") + np.arange(times) * np.timedelta64(6, "h")
    ds = xr.Dataset(
        {"t": (("valid_time", "level", "latitude", "longitude"), data.astype("float32"), {"units": "K"})},
        coords={
            "valid_time": valid_time,
            "step": ("valid_time", np.arange(1, times + 1) * np.timedelta64(6, "h")),
            "time": np.datetime64("2010-01-01"),
            "level": list(levels),
            "latitude": ("latitude", lat, {"units": "degrees_north"}),
            "longitude": ("longitude", lon, {"units": "degrees_east"}),
        },
    )
    return ds


GAUSSIAN_2P8 = np.linspace(-87.86, 87.86, 64)        # NeuralGCM 2.8 deg: no poles
LON_2P8 = np.arange(128) * 2.8125                    # 0 .. 357.19: no column at 358-359
AURORA_LAT = np.linspace(90, -89.75, 720)            # Aurora: descending, no -90 row
AURORA_LON = np.arange(1440) * 0.25


@pytest.mark.parametrize("lat, lon", [(GAUSSIAN_2P8, LON_2P8), (AURORA_LAT, AURORA_LON)])
@pytest.mark.parametrize("res", ["1", "2", "0.25"])
def test_no_nan_and_same_values_where_plain_interp_was_defined(lat, lon, res):
    ds = _field(lat, lon)
    lats, lons = target_grid(res)
    out = regrid_latlon(ds, lats, lons)
    assert not np.isnan(out["t"].values).any()
    np.testing.assert_array_equal(out.latitude, lats)
    np.testing.assert_array_equal(out.longitude, lons)
    plain = ds.sortby("latitude").interp(latitude=lats, longitude=lons, method="linear")
    defined = ~np.isnan(plain["t"].values)
    assert defined.any() and not defined.all()       # the old behaviour had NaN edges
    np.testing.assert_array_equal(out["t"].values[defined], plain["t"].values[defined])


def test_edges_follow_outermost_row_and_wrap_across_360():
    ds = _field(GAUSSIAN_2P8, LON_2P8, times=1, levels=(500,))
    out = regrid_latlon(ds, *target_grid("1"))
    t = out["t"].isel(valid_time=0, level=0)
    src = ds["t"].isel(valid_time=0, level=0)
    # Poles take the outermost model row (interpolated in longitude).
    north = src.isel(latitude=-1).interp(longitude=[10.0]).item()
    assert t.sel(latitude=90, longitude=10).item() == pytest.approx(north, rel=1e-6)
    # 359 deg lies between the last column (357.19) and the first one + 360 deg.
    w = (359 - LON_2P8[-1]) / (360 - LON_2P8[-1])
    row = src.sel(latitude=GAUSSIAN_2P8[32])
    expected = (1 - w) * row.isel(longitude=-1).item() + w * row.isel(longitude=0).item()
    got = regrid_latlon(ds, [GAUSSIAN_2P8[32]], [359.0])["t"].isel(valid_time=0, level=0).item()
    assert got == pytest.approx(expected, rel=1e-6)


def test_coords_attrs_and_dims_are_kept():
    ds = _field(AURORA_LAT, AURORA_LON)
    out = regrid_latlon(ds, *target_grid("2"))
    assert out["t"].dims == ("valid_time", "level", "latitude", "longitude")
    assert out["t"].attrs == {"units": "K"}
    assert out.latitude.attrs == {"units": "degrees_north"} and out.longitude.attrs == {"units": "degrees_east"}
    assert "longitude" not in out["step"].dims and "longitude" not in out["time"].dims
    np.testing.assert_array_equal(out["step"].values, ds["step"].values)
