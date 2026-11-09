"""Climatology averaging, safe file update and restart validation."""
from pathlib import Path
import sys

import numpy as np
import pytest
import xarray as xr

pytest.importorskip("dask")
pytest.importorskip("netCDF4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runscripts"))
import update_wind_climatology as wind


def source():
    time = np.arange(np.datetime64("1980-01-01"), np.datetime64("1982-01-01"),
                     np.timedelta64(6, "h"))
    values = np.where(time < np.datetime64("1981-01-01"), 2., 5.)[:, None, None]
    return xr.Dataset({name: (("time", "latitude", "longitude"), values * sign)
                       for name, sign in zip(wind.WIND_NAMES, [1, -1])},
                      coords={"time": time, "latitude": [45.], "longitude": [7.]})


def test_leap_year_weighting_and_preserved_existing_fields(tmp_path, monkeypatch):
    path = tmp_path / "climatology.nc"
    cloud = xr.Dataset({"clwc": (("hour", "latitude", "longitude"), np.ones((4, 1, 1)))},
                       coords={"hour": [0, 6, 12, 18], "latitude": [45.], "longitude": [7.]},
                       attrs={"original": "preserve"})
    cloud.to_netcdf(path)
    monkeypatch.setattr(wind.xr, "open_zarr", lambda *a, **kw: source())
    wind.update_climatology(path, 1980, 1981, workers=1)
    with xr.open_dataset(path) as result:
        xr.testing.assert_equal(result[["clwc"]], cloud)
        np.testing.assert_allclose(result.u100, (366 * 2. + 365 * 5.) / 731)
        np.testing.assert_allclose(result.v100, -result.u100)
        assert result.u100.attrs["samples_per_hour"] == 731
        assert result.u100.attrs["climatology_period"] == "1980-01-01/1981-12-31"
    backup, = tmp_path.glob("*.bak")
    with xr.open_dataset(backup) as original:
        xr.testing.assert_identical(original, cloud)
    assert len(list(tmp_path.glob("*-checkpoints/*.nc"))) == 2


def test_incomplete_year_is_rejected():
    with pytest.raises(ValueError, match="Incomplete ARCO timestamps"):
        wind.yearly_sums(source().isel(time=slice(1, None)), 1980, [0, 6, 12, 18])
