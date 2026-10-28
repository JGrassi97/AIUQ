"""

"""

# Built-in/Generics
import os
import ast
import logging
from datetime import datetime, timezone

# Third party
import numpy as np
import xarray as xr
from anemoi.inference.runners.simple import SimpleRunner

# Local
from AIUQst_lib.functions import parse_arguments, read_config, normalize_out_vars
from AIUQst_lib.cards import read_model_card, read_std_version
from AIUQst_lib.output import (
    resolve_output_variables, output_metadata, standardize_output, write_forecast,
)

from postprocess_aifs import build_dataset_for_state

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main() -> None:

    # Read config
    args = parse_arguments()
    config = read_config(args.config)

    _START_TIME         = config.get("START_TIME", "")
    _END_TIME           = config.get("END_TIME", "")
    _HPCROOTDIR         = config.get("HPCROOTDIR", "")
    _MODEL_NAME         = config.get("MODEL_NAME", "")
    _MODEL_CHECKPOINT   = config.get("MODEL_CHECKPOINT", "")
    _INNER_STEPS        = config.get("INNER_STEPS", 1)
    _OUTPUT_PATH        = config.get("OUTPUT_PATH", "")
    _OUT_VARS           = config.get("OUT_VARS", [])
    _OUT_FREQ           = config.get("OUT_FREQ", "")
    _RNG_KEY            = config.get("RNG_KEY", "")
    _OUT_LEVS           = config.get("OUT_LEVS", "")
    _ICS_TEMP_DIR       = config.get("ICS_TEMP_DIR", "")
    _STD_VERSION        = config.get("STD_VERSION", "")
    
      
    #IC settings
    model_card = read_model_card(_HPCROOTDIR, _MODEL_NAME)
    standard_dict = read_std_version(_HPCROOTDIR, _STD_VERSION)
   
    out_vars = normalize_out_vars(_OUT_VARS)
    output_vars, out_vars = resolve_output_variables(
        out_vars, model_card, standard_dict["variables"],
    )
    metadata = output_metadata(config)

    desired_levels = None
    if _OUT_LEVS != 'original':
        raw_levels = _OUT_LEVS if isinstance(_OUT_LEVS, list) else ast.literal_eval(_OUT_LEVS)
        desired_levels = [int(plev) for plev in raw_levels]

    # Format time settings
    start_date = datetime.strptime(_START_TIME, '%Y-%m-%d')
    end_date = datetime.strptime(_END_TIME, '%Y-%m-%d')

    days_to_run = (end_date - start_date).days + 1
    outer_steps = days_to_run * 24

    print(f"Running from {_START_TIME} to {_END_TIME} ({days_to_run} days)")
    print(f"Total outer steps: {outer_steps} (inner steps: {_INNER_STEPS}h)")

    # Load model
    runner = SimpleRunner(str(_MODEL_CHECKPOINT), device="cuda")
    
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
    
    os.makedirs(_OUTPUT_PATH, exist_ok=True)
    
    datasets_per_time = []

    for state in runner.run(input_state=input_state, lead_time=outer_steps):
        state_name = state["date"].strftime("%Y%m%d%H")
        print(f"Generated state for {state_name}")


        ds_t = build_dataset_for_state(state, output_vars, desired_levels)
        datasets_per_time.append(ds_t)

    dataset = xr.concat(datasets_per_time, dim="time", join="exact").sortby("time")
    # Preserve the dates returned by Anemoi and use the actual NPZ initialization.
    initial_time = np.datetime64(input_state["date"].replace(tzinfo=None), "ns")
    dataset = dataset.rename({"time": "valid_time"})

    # # Format output frequency
    if _OUT_FREQ == "daily":
        dataset = dataset.resample(valid_time="1D").mean()

    
    dataset = standardize_output(
        dataset, dict(zip(output_vars, out_vars)), standard_dict["variables"],
        initial_time, metadata, unstructured=True,
    )
    for var in out_vars:
        output_file = (
            f"{_OUTPUT_PATH}/{var}/{_RNG_KEY}/"
            f"out-{_START_TIME}-{_END_TIME}-{_RNG_KEY}-{var}_temp.nc"
        )
        write_forecast(dataset[[var]], output_file)

if __name__ == "__main__":
    main()
