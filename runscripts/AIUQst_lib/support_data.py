"""
Helpers shared by the IC retrieval scripts to merge static/climatological support data
and to reason about the latitude orientation of the stored fields.
"""

# Third party
import numpy as np
import xarray as xr

# Local
from AIUQst_lib.variables import define_ics_mappers, reassign_long_names_units


def align_support_grid(dataset: xr.Dataset, selected: xr.Dataset) -> xr.Dataset:
    """Align coordinates by value; never attach a new grid to unchanged data."""
    dataset = dataset.assign_coords(longitude=dataset.longitude % 360).sortby("longitude")
    # Periodic extension avoids NaNs around the Greenwich meridian.
    if dataset.longitude.size > 1 and float(dataset.longitude[-1] - dataset.longitude[0]) > 350:
        left = dataset.isel(longitude=[-1]).assign_coords(longitude=[float(dataset.longitude[-1]) - 360])
        right = dataset.isel(longitude=[0]).assign_coords(longitude=[float(dataset.longitude[0]) + 360])
        dataset = xr.concat([left, dataset, right], dim="longitude")
    return dataset.interp(latitude=selected.latitude, longitude=selected.longitude)


def add_fallback(selected: xr.Dataset, missing: dict | None, card: dict,
                 path: str, climatology: bool = False,
                 reason: str | None = None) -> tuple:
    """Fill the variables in ``missing`` from the static or climatology support file.

    Returns the merged dataset and the variables that are still missing (or None).
    """
    if not missing:
        return selected, None
    names, rename, long_names, units, remaining = define_ics_mappers(card["variables"], missing)
    if not names:
        return selected, remaining
    with xr.open_dataset(path) as source:
        absent = set(names.values()) - set(source.data_vars)
        if absent:
            raise ValueError(f"Missing fallback variables {sorted(absent)} in {path}. "
                             "Extend the ERA5 climatology with u100/v100 for Aurora 1.5.")
        fallback = source[list(names.values())].rename(rename).load()
    if climatology:
        # Select actual UTC hours, not the first two entries (which may be 00/01).
        if "month" in fallback.dims:
            fallback = fallback.sel(month=selected.time.dt.month)
        if "dayofyear" in fallback.dims:
            fallback = fallback.sel(dayofyear=selected.time.dt.dayofyear)
        if "hour" not in fallback.dims:
            raise ValueError(f"ERA5 climatology {path} must have an 'hour' dimension (UTC).")
        fallback = fallback.sel(hour=selected.time.dt.hour)
        fallback = fallback.drop_vars([c for c in ["hour", "month", "dayofyear"] if c in fallback.coords])
    fallback = align_support_grid(fallback, selected)
    fallback = reassign_long_names_units(fallback, long_names, units)
    for var in fallback.data_vars:
        fallback[var].attrs["source"] = "ERA5 climatology" if climatology else "static support data"
        if climatology and reason:
            fallback[var].attrs["fallback_reason"] = reason
        if var in {"u100", "v100"} and not bool(np.isfinite(fallback[var]).all()):
            raise ValueError(f"Non-finite {var} in ERA5 climatology after grid alignment: {path}")
    return xr.merge([selected, fallback], join="exact", compat="no_conflicts"), remaining


def static_fields_flipped(dataset: xr.Dataset, mask: str = "lsm") -> bool:
    """True when the static fields are stored upside down with respect to their latitudes.

    The ERA5 retrieval attaches the (south-to-north) static file to the (north-to-south)
    IC grid by overwriting the coordinates, which flips the data; EERIE and the ECMWF
    analysis align by value. The land-sea mask tells them apart: the southernmost row
    (Antarctica) must be land and the northernmost (Arctic Ocean) sea.
    """
    lsm = dataset[mask]
    south = float(lsm.sel(latitude=lsm.latitude.min()).mean())
    north = float(lsm.sel(latitude=lsm.latitude.max()).mean())
    if south > 0.5 and north < 0.5:
        return False
    if south < 0.5 and north > 0.5:
        return True
    raise ValueError(f"Cannot infer latitude orientation from '{mask}' "
                     f"(mean at south pole {south:.2f}, north pole {north:.2f}).")
