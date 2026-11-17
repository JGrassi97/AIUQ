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


# Target points interpolated at once: scipy's temporaries are several times this size (in float64),
# so a block stays around 1-2 GB however long the run or fine the output grid.
MAX_BLOCK_POINTS = 2**25


def regrid_latlon(ds, latitudes, longitudes, method: str = "linear", max_block_points: int = MAX_BLOCK_POINTS):
    """Interpolate ``ds`` (dims ``latitude``/``longitude``, degrees, longitudes in [0, 360)) to the
    given latitudes/longitudes without NaN at the grid edges.

    Longitude is periodic: targets between the last source column and the first one + 360 deg are
    interpolated across the date line. Target latitudes beyond the outermost source row take the
    value of that row.

    Memory: the output is filled in blocks along the first non-horizontal dimension (time), so the
    peak is the output itself plus one block of interpolation temporaries. Select pressure levels
    *before* calling this function: regridding every model level and discarding most of them
    afterwards costs (model levels / output levels) times more memory.
    """
    lead = [d for d in ds.dims if d not in ("latitude", "longitude")]
    if not lead:
        return _regrid_block(ds, latitudes, longitudes, method)
    dim, n = lead[0], ds.sizes[lead[0]]
    per_step = len(latitudes) * len(longitudes) * int(np.prod([ds.sizes[d] for d in lead[1:]]))
    block = max(1, int(max_block_points // max(per_step, 1)))
    if block >= n:
        return _regrid_block(ds, latitudes, longitudes, method)

    first = _regrid_block(ds.isel({dim: slice(0, block)}), latitudes, longitudes, method)
    blocked = {}
    for name, var in first.data_vars.items():
        if dim in var.dims:
            axis = var.dims.index(dim)
            shape = list(var.shape)
            shape[axis] = n
            blocked[name] = (axis, np.empty(shape, dtype=var.dtype))

    def store(part, start):
        for name, (axis, array) in blocked.items():
            index = [slice(None)] * array.ndim
            index[axis] = slice(start, start + part.sizes[dim])
            array[tuple(index)] = part[name].values

    store(first, 0)
    for start in range(block, n, block):
        store(_regrid_block(ds.isel({dim: slice(start, start + block)}), latitudes, longitudes, method), start)

    data_vars = {name: (first[name].dims, blocked[name][1], first[name].attrs) if name in blocked else first[name]
                 for name in first.data_vars}
    coords = {name: coord for name, coord in first.coords.items() if dim not in coord.dims}
    coords.update({name: coord for name, coord in ds.coords.items() if dim in coord.dims})
    return xr.Dataset(data_vars, coords=coords, attrs=first.attrs)


def _regrid_block(ds, latitudes, longitudes, method):
    """``regrid_latlon`` on one block (no blocking)."""
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
