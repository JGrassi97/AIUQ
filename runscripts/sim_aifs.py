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
    _OUTPUT_TEMP_PATH   = config.get("OUTPUT_TEMP_PATH", "")

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
    

    os.makedirs(_OUTPUT_TEMP_PATH, exist_ok=True)
    
    for state in runner.run(input_state=input_state, lead_time=outer_steps):
        state_name = state['date'].strftime('%Y%m%d%H')
        print(f"Generated state for {state_name}")

        # Save state dict to npz
        output_fields = {}
        for var_name, data in state['fields'].items():
            output_fields[f"f__{var_name}"] = data.astype(np.float32)
        
        output_npz = os.path.join(_OUTPUT_TEMP_PATH, f"sim_aifs_{state_name}.npz")
        np.savez_compressed(output_npz, timestamp=int(state['date'].timestamp()), **output_fields)
        print(f"Saved output to {output_npz}")



    

if __name__ == "__main__":
    main()