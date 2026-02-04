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
from copy import deepcopy
from datetime import datetime, timedelta, timezone

# Third party
import netCDF4 as nc
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

from ics_aifs import ics_aifs
from postprocess_aifs import post_process_aifs

# Configure logging
logging.basicConfig(level=logging.DEBUG)


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




def main() -> None:

    # Read config
    args = parse_arguments()
    config = read_config(args.config)

    _INI_DATA_PATH      = config.get('INI_DATA_PATH', "")
    _START_TIME         = config.get("START_TIME", "")
    _END_TIME           = config.get("END_TIME", "")
    _HPCROOTDIR         = config.get("HPCROOTDIR", "")
    _MODEL_NAME         = config.get("MODEL_NAME", "")
    _MODEL_CHECKPOINT   = config.get("MODEL_CHECKPOINT", "")
    _INNER_STEPS        = config.get("INNER_STEPS", 1)
    _OUTPUT_PATH        = config.get("OUTPUT_PATH", "")
    _OUT_VARS           = config.get("OUT_VARS", [])
    _OUT_FREQ           = config.get("OUT_FREQ", "")
    _OUT_RES            = config.get("OUT_RES", "")
    _RNG_KEY            = config.get("RNG_KEY", "")
    _OUT_LEVS           = config.get("OUT_LEVS", "")

    output_vars = normalize_out_vars(_OUT_VARS)


    # Format time settings
    start_date = datetime.strptime(_START_TIME, '%Y-%m-%d')
    end_date = datetime.strptime(_END_TIME, '%Y-%m-%d')

    days_to_run = (end_date - start_date).days + 1
    outer_steps = days_to_run * 24

    print(f"Running from {_START_TIME} to {_END_TIME} ({days_to_run} days)")
    print(f"Total outer steps: {outer_steps} (inner steps: {_INNER_STEPS}h)")

    # Load model
    runner = SimpleRunner(str(_MODEL_CHECKPOINT), device="cuda")
    
    input_state = ics_aifs(_INI_DATA_PATH, _START_TIME, _HPCROOTDIR, _MODEL_NAME)
    
    os.makedirs(_OUTPUT_PATH, exist_ok=True)
    
    dataset = []
    for state in runner.run(input_state=input_state, lead_time=outer_steps):
        state_name = state['date'].strftime('%Y%m%d%H')
        print(f"Generated state for {state_name}")

        for var_name, data in state['fields'].items():
            if var_name in output_vars:
                dataset.append(post_process_aifs(var_name, data, state['date'], _OUT_LEVS))
    
    dataset = xr.concat(dataset, dim='time').sortby('time')


    # --- Build valid_time/step coords consistent with produced outputs ---
    delta_t = np.timedelta64(int(_INNER_STEPS), "h")

    initial_conditions = xr.open_zarr(_INI_DATA_PATH).load()
    initial_time = initial_conditions.time.isel(time=0).values

    # how many forecast steps we actually have
    n_out = dataset.sizes["time"]

    valid_time = initial_time + np.arange(n_out) * delta_t
    steps = np.arange(n_out) * delta_t

    # rename time -> valid_time and attach coords with matching length
    dataset = dataset.rename({"time": "valid_time"})
    dataset = dataset.assign_coords(valid_time=("valid_time", valid_time))
    dataset = dataset.assign_coords(time=initial_time)
    dataset = dataset.assign_coords(step=("valid_time", steps))

    # Drop sim_time if present
    if "sim_time" in dataset.variables:
        dataset = dataset.drop_vars("sim_time")

    # Format output frequency
    if _OUT_FREQ == "daily":
        dataset = dataset.resample(valid_time="1D").mean()

    # Format output resolution
    if _OUT_RES == "0.5":
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
        latitudes = dataset.latitude.values
        longitudes = dataset.longitude.values
    
    if _OUT_RES in ["0.5", "1", "1.5", "2"]:
        dataset = dataset.interp(latitude=latitudes, longitude=longitudes, method="linear")
    

    for var in output_vars:
        predictions_datarray = dataset[var]
        OUTPUT_BASE_PATH = f"{_OUTPUT_PATH}/{var}/{str(_RNG_KEY)}"
        os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
        OUTPUT_FILE = f"{OUTPUT_BASE_PATH}/ngcm-{_START_TIME}-{_END_TIME}-{_RNG_KEY}-{var}.nc"
        predictions_datarray.to_netcdf(OUTPUT_FILE)
            
    

if __name__ == "__main__":
    main()