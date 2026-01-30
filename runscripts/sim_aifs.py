"""

"""

# Built-in/Generics
import os
import yaml
import re
import ast
import json
import pickle
import logging
from datetime import datetime, timedelta, timezone

# Third party
import numpy as np
import xarray as xr
import earthkit.regrid as ekr
from anemoi.inference.checkpoint import Checkpoint
from anemoi.inference.runners.simple import SimpleRunner

# Local
from AIUQst_lib.functions import parse_arguments, read_config, normalize_out_vars
from AIUQst_lib.pressure_levels import check_pressure_levels
from AIUQst_lib.cards import read_model_card, read_ic_card, read_std_version
from AIUQst_lib.variables import name_mapper_for_model

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Set the earthkit regrid cache directory
from earthkit.regrid.utils.config import CONFIG
EARTHKIT_REGRID_CACHE = os.environ.get('EARTHKIT_REGRID_CACHE', '')
CONFIG.set("cache-policy", "user")
CONFIG.set("user-cache-directory", EARTHKIT_REGRID_CACHE)


LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
SOIL_LEVELS = [1, 2]

LEVEL_VAR_RE = re.compile(r"^(?P<var>[a-zA-Z0-9]+)_(?P<level>\d+)$")

# Vars che AIFS usa come PL (base names)
PL_BASE_VARS = {"t", "u", "v", "w", "q", "z"}   # (z è geopotential)

# Vars di superficie / constant che AIFS usa tipicamente
# NOTE: use 10u/10v as expected pipeline names (not u10/v10)
SFC_VARS = {
    "2t", "2d", "10u", "10v", "msl", "sp", "tcw", "skt",
    "lsm", "sdor", "slor", "z"   # z qui è *surface geopotential* (static)
}

SOIL_VARS = {"stl1", "stl2"}


def fix_lon_to_180(lons: np.ndarray) -> np.ndarray:
    """Convert longitudes to [-180, 180]."""
    lons = np.asarray(lons)
    return np.where(lons > 180, lons - 360, lons)

def state_to_xarray_points(state: dict, shift_lon_180: bool = True) -> xr.Dataset:
    fields = state["fields"]

    lat = np.asarray(state["latitudes"], dtype=np.float32)
    lon = np.asarray(state["longitudes"], dtype=np.float32)
    if shift_lon_180:
        lon = fix_lon_to_180(lon)

    npts = lat.size
    time = np.asarray([np.datetime64(state["date"])], dtype="datetime64[ns]")

    level_buckets = {}
    surface_vars = {}

    for name, arr in fields.items():
        arr = np.asarray(arr)
        if arr.size != npts:
            continue

        m = LEVEL_VAR_RE.match(name)
        if m:
            var = m.group("var")
            level = int(m.group("level"))
            level_buckets.setdefault(var, {})[level] = arr.astype(np.float32, copy=False)
        else:
            surface_vars[name] = arr.astype(np.float32, copy=False)

    data_vars = {}
    for name, v in surface_vars.items():
        data_vars[name] = (("time", "point"), v[None, :])

    # Canonical desired level order:
    # You said "occhio che sono al contrario, il 12 è 1000" -> we want 50..1000
    desired_levels = list(reversed(LEVELS))  # [50, 100, ..., 1000]

    level_coord = None  # will be set if we have at least one PL var

    for var, lev_dict in level_buckets.items():
        # keep only levels that exist, in desired order
        levels = [lev for lev in desired_levels if lev in lev_dict]
        if not levels:
            continue

        stack = np.stack([lev_dict[lev] for lev in levels], axis=0).astype(np.float32, copy=False)
        data_vars[var] = (("time", "level", "point"), stack[None, :, :])

        # ensure a single shared level coord for the dataset
        if level_coord is None:
            level_coord = np.array(levels, dtype=np.int32)
        else:
            # if some var is missing a level, you can either:
            # - allow different level sets (messy) or
            # - enforce the intersection
            # Here we enforce consistency: keep intersection in desired order
            common = [lev for lev in level_coord.tolist() if lev in levels]
            level_coord = np.array(common, dtype=np.int32)

    coords = {
        "time": time,
        "point": np.arange(npts, dtype=np.int32),
        "latitude": ("point", lat),
        "longitude": ("point", lon),
    }
    if level_coord is not None:
        coords["level"] = ("level", level_coord)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={
            "step_seconds": state.get("step").total_seconds() if state.get("step") is not None else None,
            "previous_step_seconds": state.get("previous_step").total_seconds()
            if state.get("previous_step") is not None
            else None,
        },
    )

    if "level" in ds.coords:
        ds["level"].attrs["units"] = "hPa"
    ds["latitude"].attrs["units"] = "degrees_north"
    ds["longitude"].attrs["units"] = "degrees_east"
    return ds

def regrid_n320_to_regular025(ds_points: xr.Dataset) -> xr.Dataset:
    """
    Regrid from N320 (dim: point) to regular_ll 0.25° (dim: latitude, longitude).
    Assumes the point order is ECMWF N320.
    """
    in_grid = {"grid": "N320"}
    out_grid = {"grid": [0.25, 0.25]}

    lat = np.linspace(90.0, -90.0, int(180 / 0.25) + 1, dtype=np.float32)     # 721
    lon = np.linspace(0.0, 360.0 - 0.25, int(360 / 0.25), dtype=np.float32)  # 1440

    data_vars = {}
    time = ds_points["time"].values
    has_level = "level" in ds_points.dims

    for v in ds_points.data_vars:
        da = ds_points[v]
        if "point" not in da.dims:
            continue

        if has_level and "level" in da.dims:
            out = np.empty((da.sizes["time"], da.sizes["level"], lat.size, lon.size), dtype=np.float32)
            for ti in range(da.sizes["time"]):
                for li in range(da.sizes["level"]):
                    out[ti, li, :, :] = ekr.interpolate(
                        da.values[ti, li, :],
                        in_grid,
                        out_grid,
                        method="linear",
                    ).astype(np.float32, copy=False)
            data_vars[v] = (("time", "level", "latitude", "longitude"), out)
        else:
            out = np.empty((da.sizes["time"], lat.size, lon.size), dtype=np.float32)
            for ti in range(da.sizes["time"]):
                out[ti, :, :] = ekr.interpolate(
                    da.values[ti, :],
                    in_grid,
                    out_grid,
                ).astype(np.float32, copy=False)
            data_vars[v] = (("time", "latitude", "longitude"), out)

    coords = {"time": time, "latitude": lat, "longitude": lon}
    if has_level:
        coords["level"] = ds_points["level"].values

    ds_out = xr.Dataset(data_vars=data_vars, coords=coords)
    ds_out["latitude"].attrs["units"] = "degrees_north"
    ds_out["longitude"].attrs["units"] = "degrees_east"
    if has_level:
        ds_out["level"].attrs["units"] = "hPa"
    return ds_out


def main() -> None:

    # Read config
    args = parse_arguments()
    config = read_config(args.config)

    _INI_DATA_PATH      = config.get('INI_DATA_PATH', "")
    _START_TIME         = config.get("START_TIME", "")
    _END_TIME           = config.get("END_TIME", "")
    _STATIC_DATA        = config.get("STATIC_DATA", "")
    _HPCROOTDIR         = config.get("HPCROOTDIR", "")
    _MODEL_NAME         = config.get("MODEL_NAME", "")
    _IC                 = config.get("IC_NAME", "")
    _STD_VERSION        = config.get("STD_VERSION", "")
    _MODEL_CHECKPOINT   = config.get("MODEL_CHECKPOINT", "")
    _INNER_STEPS        = config.get("INNER_STEPS", 1)
    _RNG_KEY            = config.get("RNG_KEY", 1)
    _OUTPUT_PATH        = config.get("OUTPUT_PATH", "")
    _ICS_TEMP_DIR       = config.get("ICS_TEMP_DIR", "")
    _OUT_VARS           = config.get("OUT_VARS", [])
    _OUT_FREQ           = config.get("OUT_FREQ", "")
    _OUT_RES            = config.get("OUT_RES", "")
    _OUT_LEVS           = config.get("OUT_LEVS", "")

    # IC settings
    model_card = read_model_card(_HPCROOTDIR, _MODEL_NAME)
    standard_dict = read_std_version(_HPCROOTDIR, _STD_VERSION)

    # Format time settings
    start_date = datetime.strptime(_START_TIME, '%Y-%m-%d')
    input_end_date = datetime.strptime(_START_TIME, '%Y-%m-%d') + timedelta(days=1)
    end_date = datetime.strptime(_END_TIME, '%Y-%m-%d')

    days_to_run = (end_date - start_date).days + 1
    outer_steps = days_to_run * 24 // _INNER_STEPS
    delta_t = np.timedelta64(_INNER_STEPS, 'h')
    times = np.arange(outer_steps) * _INNER_STEPS  # in hours

    # Load model
    runner = SimpleRunner(str(_MODEL_CHECKPOINT), device="cuda")

    # Load also the zarr to have the starting time
    initial_conditions = xr.open_zarr(_INI_DATA_PATH, chunks=None).load()

    # Real ICS are stored in npz files
    ics_basename = f"ics_aifs_{start_date.strftime('%Y%m%d')}"
    ics_npz = os.path.join(_ICS_TEMP_DIR, f"{ics_basename}.npz")
    npz = np.load(ics_npz, allow_pickle=False)
    timestamp = int(npz["timestamp"][0]) if "timestamp" in npz.files else None
    if timestamp is None:
        raise RuntimeError(f"No 'timestamp' found in {ics_npz}")
    input_state = {"date": datetime.fromtimestamp(timestamp, tz=timezone.utc), "fields": {}}
    for key in npz.files:
        if key.startswith("f__"):
            fname = key[len("f__"):]
            input_state["fields"][fname] = npz[key]

    # Map names for model specifics
    mapper = name_mapper_for_model(model_card['variables'], standard_dict['variables'])
    initial_conditions = initial_conditions.rename(mapper)

    states = []
    for state in runner.run(input_state=input_state, lead_time=outer_steps):
        states.append(state_to_xarray_points(state))

    states = xr.concat(states, dim="time")

    # Regrid back N320 -> regular 0.25
    final_025 = regrid_n320_to_regular025(states)

    # --- Build valid_time/step coords consistent with produced outputs ---
    delta_t = np.timedelta64(int(_INNER_STEPS), "h")

    initial_time = initial_conditions.time.isel(time=0).values

    # how many forecast steps we actually have
    n_out = final_025.sizes["time"]

    valid_time = initial_time + np.arange(n_out) * delta_t
    steps = np.arange(n_out) * delta_t

    # rename time -> valid_time and attach coords with matching length
    final_025 = final_025.rename({"time": "valid_time"})
    final_025 = final_025.assign_coords(valid_time=("valid_time", valid_time))
    final_025 = final_025.assign_coords(time=initial_time)
    final_025 = final_025.assign_coords(step=("valid_time", steps))

    # Drop sim_time if present
    if "sim_time" in final_025.variables:
        final_025 = final_025.drop_vars("sim_time")

    # Format output variables and select
    output_vars = normalize_out_vars(_OUT_VARS)
    if 'all' not in output_vars:
        final_025 = final_025[output_vars]
    
    # Format output frequency
    if _OUT_FREQ == "daily":
        final_025 = final_025.resample(valid_time="1D").mean()
    
    # Format output resolution
    if _OUT_RES == "0.25":
        latitudes = np.arange(-90, 90.25, 0.25)
        longitudes = np.arange(0, 360, 0.25)
    elif _OUT_RES == "0.5":
        latitudes = np.arange(-90, 90.5, 0.5)
        longitudes = np.arange(0, 360, 0.5)
    elif _OUT_RES == "1":
        latitudes = np.arange(-90, 91, 1.0)
        longitudes = np.arange(0, 360, 1.0)
    elif _OUT_RES == "1.5":
        latitudes = np.arange(-90, 91.5, 1.5)
        longitudes = np.arange(0, 360, 1.5)
    elif _OUT_RES == "2":
        latitudes = np.arange(-90, 92, 2.0)
        longitudes = np.arange(0, 360, 2.0)
    else:
        latitudes = final_025.latitude.values
        longitudes = final_025.longitude.values
    
    if _OUT_RES in ["0.25", "0.5", "1", "1.5", "2"]:
        final_025 = final_025.interp(latitude=latitudes, longitude=longitudes, method="linear")

    # Format output pressure levels
    if _OUT_LEVS != 'original':
        desired_levels = [
            int(plev)
            for plev in _OUT_LEVS.strip('[]').split(',')
        ]
        final_025 = final_025.interp(level=desired_levels)
    
    # Create output directory
    os.makedirs(_OUTPUT_PATH, exist_ok=True)

    # Write output
    final_file = f"{_OUTPUT_PATH}/model_state-{_START_TIME}-{_END_TIME}-{_RNG_KEY}_regular025.nc"
    final_025.to_netcdf(final_file)
    logging.info("Wrote: %s", final_file)


if __name__ == "__main__":
    main()