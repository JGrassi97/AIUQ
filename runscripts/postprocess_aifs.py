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
    _OUTPUT_TEMP_PATH   = config.get("OUTPUT_TEMP_PATH", "")

    files = os.listdir(_OUTPUT_TEMP_PATH)
    files = [os.path.join(_OUTPUT_TEMP_PATH, f) for f in files]

    if _OUT_RES == "0.25":
        lat = np.linspace(90.0, -90.0, int(180 / 0.25) + 1, dtype=np.float32)
        lon = np.linspace(0.0, 360.0 - 0.25, int(360 / 0.25), dtype=np.float32)
        grid_out = [0.25, 0.25] 
    
    if _OUT_RES == "1":
        lat = np.linspace(90.0, -90.0, int(180 / 1) + 1, dtype=np.float32)  
        lon = np.linspace(0.0, 360.0 - 1, int(360 / 1), dtype=np.float32)
        grid_out = [1.0, 1.0]

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
                        'linear'
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
                        'linear'
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

    dataset_complete = xr.concat(dataset_final, dim='time')



    # --- Build valid_time/step coords consistent with produced outputs ---
    delta_t = np.timedelta64(int(_INNER_STEPS), "h")

    initial_conditions = xr.open_zarr(_INI_DATA_PATH).load()
    initial_time = initial_conditions.time.isel(time=0).values

    # how many forecast steps we actually have
    n_out = dataset_complete.sizes["time"]

    valid_time = initial_time + np.arange(n_out) * delta_t
    steps = np.arange(n_out) * delta_t

    # rename time -> valid_time and attach coords with matching length
    dataset_complete = dataset_complete.rename({"time": "valid_time"})
    dataset_complete = dataset_complete.assign_coords(valid_time=("valid_time", valid_time))
    dataset_complete = dataset_complete.assign_coords(time=initial_time)
    dataset_complete = dataset_complete.assign_coords(step=("valid_time", steps))

    # Drop sim_time if present
    if "sim_time" in dataset_complete.variables:
        dataset_complete = dataset_complete.drop_vars("sim_time")

    # Format output variables and select
    output_vars = normalize_out_vars(_OUT_VARS)
    if 'all' not in output_vars:
        dataset_complete = dataset_complete[output_vars]
    
    # Format output frequency
    if _OUT_FREQ == "daily":
        dataset_complete = dataset_complete.resample(valid_time="1D").mean()
    
    # Create output directory
    os.makedirs(_OUTPUT_PATH, exist_ok=True)

    # Write output
    final_file = f"{_OUTPUT_PATH}/model_state-{_START_TIME}-{_END_TIME}-{_RNG_KEY}_regular025.nc"
    dataset_complete.to_netcdf(final_file)
    logging.info("Wrote: %s", final_file)


if __name__ == "__main__":
    main()