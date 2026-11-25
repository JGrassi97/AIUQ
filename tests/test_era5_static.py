"""Static fields merged into ERA5 initial conditions keep their geography (they were stored upside down)."""
from pathlib import Path
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "runscripts"))
from AIUQst_lib.support_data import align_support_grid, static_fields_flipped  # noqa: E402


def test_static_file_south_to_north_on_era5_grid_north_to_south():
    lat_static = np.array([-90.0, -45.0, 0.0, 45.0, 90.0])               # the static file: south to north
    lon = np.array([0.0, 90.0, 180.0, 270.0])
    lsm = np.repeat(np.array([1.0, 0.0, 0.0, 1.0, 0.0])[:, None], lon.size, axis=1)   # Antarctica land, Arctic sea
    orog = np.repeat(lat_static[:, None] * 10.0, lon.size, axis=1)
    static = xr.Dataset({"lsm": (("latitude", "longitude"), lsm), "orog": (("latitude", "longitude"), orog)},
                        coords={"latitude": lat_static, "longitude": lon})
    era5 = xr.Dataset(coords={"latitude": lat_static[::-1], "longitude": lon})   # ERA5: north to south

    aligned = align_support_grid(static, era5)

    assert list(aligned.latitude.values) == list(era5.latitude.values)
    assert not static_fields_flipped(aligned)
    for la in lat_static:
        assert np.allclose(aligned.orog.sel(latitude=la).values, la * 10.0)
