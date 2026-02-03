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

# Configure logging
logging.basicConfig(level=logging.DEBUG)


def main() -> None:

    # Read config
    args = parse_arguments()
    config = read_config(args.config)

    _INI_DATA_PATH      = config.get('INI_DATA_PATH', "")
    _START_TIME         = config.get("START_TIME", "")
    _END_TIME           = config.get("END_TIME", "")
    _INNER_STEPS        = config.get("INNER_STEPS", 1)
    _OUTPUT_PATH        = config.get("OUTPUT_PATH", "")
    _OUT_VARS           = config.get("OUT_VARS", [])
    _OUT_FREQ           = config.get("OUT_FREQ", "")
    _OUT_RES            = config.get("OUT_RES", "")
    _OUT_LEVS           = config.get("OUT_LEVS", "")
    _RNG_KEY            = config.get("RNG_KEY", "")
    _OUTPUT_TEMP_PATH   = config.get("OUTPUT_TEMP_PATH", "")

    files = os.listdir(_OUTPUT_TEMP_PATH)
    files = [os.path.join(_OUTPUT_TEMP_PATH, f) for f in files]

    grid_out = [0.25, 0.25]

    dataset_final = []

    for file in files:
        data = np.load(file, allow_pickle=True)

        fields = [key for key in data.keys() if key.startswith('f__')]
        fields = [x.replace('f__', '') for x in fields]

        pl_fields = [x for x in fields if '_' in x]
        sl_fields = [x for x in fields if '_' not in x]

        date = data['timestamp']
        dt_object = datetime.fromtimestamp(date)

        dataarrays = []
        for field in sl_fields:
            var_name = f'f__{field}'
            var = data[var_name]

            var = ekr.interpolate(
                        var,
                        {"grid": "N320"},
                        {"grid": grid_out},
                        "linear"
                    )

            dataarray = xr.DataArray(
                var,
                dims=['latitude', 'longitude'],
                attrs={'long_name': field, 'units': 'unknown'}
            ).rename(field)
            dataarrays.append(dataarray)

        dataset_sl = xr.merge(dataarrays)

        for field in pl_fields:
            var_name = field.split('_')[0]

            for level in _OUT_LEVS:
                field_name = f'f__{var_name}_{level}'

                if field_name in data:
                    var = data[field_name]

                    var = ekr.interpolate(
                        var,
                        {"grid": "N320"},
                        {"grid": grid_out},
                        "linear"
                    )

                    dataarray = xr.DataArray(
                        var,
                        dims=['latitude', 'longitude'],
                        attrs={'long_name': f'{var_name} at {level} hPa', 'units': 'unknown'}
                    ).rename(f'{var_name}').expand_dims('level').assign_coords(level=[level])
                    dataarrays.append(dataarray)

            
        dataset_pl = xr.merge(dataarrays, join='outer')

        dataset_complete = xr.merge([dataset_sl, dataset_pl], join='outer')
        dataset_complete = dataset_complete.assign_coords(time=[dt_object])

        dataset_final.append(dataset_complete)

    dataset = xr.concat(dataset_final, dim='time').sortby('time')

    
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
        latitudes = dataset.latitude.values
        longitudes = dataset.longitude.values
    
    if _OUT_RES in ["0.5", "1", "1.5", "2"]:
        dataset = dataset.interp(latitude=latitudes, longitude=longitudes, method="linear")
    
    # Create output directory
    os.makedirs(_OUTPUT_PATH, exist_ok=True)

    # Write output
    # Format output variables and select
    output_vars = normalize_out_vars(_OUT_VARS)
    for var in output_vars:
        predictions_datarray = dataset[var]
        OUTPUT_BASE_PATH = f"{_OUTPUT_PATH}/{var}/{str(_RNG_KEY)}"
        os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
        OUTPUT_FILE = f"{OUTPUT_BASE_PATH}/ngcm-{_START_TIME}-{_END_TIME}-{_RNG_KEY}-{var}.nc"
        predictions_datarray.to_netcdf(OUTPUT_FILE)


if __name__ == "__main__":
    main()