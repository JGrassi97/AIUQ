"""
Regridding of model outputs to the regular latitude-longitude grids selected with OUT_RES.

``xarray.Dataset.interp`` returns NaN wherever a target point is outside the source grid. Model
grids rarely cover the whole target grid: Gaussian grids (NeuralGCM) stop short of both poles,
Aurora's 0.25 deg grid has no -90 row, and no source grid repeats 0 deg at 360 deg. Plain
``interp`` therefore left all-NaN rows at the poles and all-NaN columns just below 360 deg in the
outputs. ``regrid_latlon`` interpolates periodically in longitude and gives target latitudes beyond
the outermost model row the value of that row.
"""

# Third party
import numpy as np
import xarray as xr

OUT_RESOLUTIONS = ("0.25", "0.5", "1", "1.5", "2")


def target_grid(out_res: str):
    """Regular grid for an OUT_RES value: latitudes -90..90 and longitudes 0..360-step (degrees)."""
    step = float(out_res)
    latitudes = np.arange(-90, 90 + step / 2, step)
    longitudes = np.arange(0, 360, step)
    return latitudes, longitudes


def regrid_latlon(ds, latitudes, longitudes, method: str = "linear"):
    """Interpolate ``ds`` (dims ``latitude``/``longitude``, degrees, longitudes in [0, 360)) to the
    given latitudes/longitudes without NaN at the grid edges.

    Longitude is periodic: targets between the last source column and the first one + 360 deg are
    interpolated across the date line. Target latitudes beyond the outermost source row take the
    value of that row. Only the output-sized arrays are created, never a padded copy of the input.
    """
    ds = ds.sortby("latitude").sortby("longitude")
    lat_src = ds["latitude"].values
    lon_src = ds["longitude"].values
    latitudes = np.asarray(latitudes, dtype=float)
    longitudes = np.asarray(longitudes, dtype=float)

    # Latitude: clamp to the source range (nearest row beyond it), relabel afterwards.
    lat_query = np.clip(latitudes, lat_src[0], lat_src[-1])

    inside = (longitudes >= lon_src[0]) & (longitudes <= lon_src[-1])
    parts = []
    if inside.any():
        parts.append(ds.interp(latitude=lat_query, longitude=longitudes[inside], method=method))
    if (~inside).any():
        # Wrap-around segment [last column, first column + 360].
        edge = ds.isel(longitude=[-1, 0]).assign_coords(longitude=[lon_src[-1], lon_src[0] + 360.0])
        query = longitudes[~inside]
        query = np.where(query < lon_src[0], query + 360.0, query)
        wrapped = edge.interp(latitude=lat_query, longitude=query, method=method)
        parts.append(wrapped.assign_coords(longitude=longitudes[~inside]))

    out = parts[0] if len(parts) == 1 else xr.concat(
        parts, dim="longitude", data_vars="minimal", coords="minimal", compat="override",
        combine_attrs="override",
    )
    out = out.sortby("longitude").assign_coords(latitude=latitudes)
    for name in ("latitude", "longitude"):
        out[name].attrs = ds[name].attrs
    return out
