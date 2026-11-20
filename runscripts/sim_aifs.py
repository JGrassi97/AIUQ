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
from AIUQst_lib.variables import name_mapper_for_model, output_translator

from postprocess_aifs import build_dataset_for_state
from AIUQst_lib.streaming import StepBuffer
from AIUQst_lib.waves import cos_sin_to_direction

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def grid_file(file_path, lats, lons):
    grid_file = os.path.join(file_path)
    with open(grid_file, "w") as f:
        f.write("gridtype = unstructured\n")
        f.write(f"gridsize = {lats.size}\n")
        f.write("xname = lon\n")
        f.write("yname = lat\n")
        f.write("xunits = degrees_east\n")
        f.write("yunits = degrees_north\n")
        f.write("xvals = " + " ".join(map(str, lons)) + "\n")
        f.write("yvals = " + " ".join(map(str, lats)) + "\n")


def _as_datetime64(date) -> np.datetime64:
    """UTC datetime (aware or naive) -> naive numpy datetime64[ns]."""
    if date.tzinfo is not None:
        date = date.astimezone(timezone.utc).replace(tzinfo=None)
    return np.datetime64(date, "ns")


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
    _ICS_TEMP_DIR       = config.get("ICS_TEMP_DIR", "")
    _STD_VERSION        = config.get("STD_VERSION", "")
    
      
    #IC settings
    model_card = read_model_card(_HPCROOTDIR, _MODEL_NAME)
    standard_dict = read_std_version(_HPCROOTDIR, _STD_VERSION)
   
    out_vars = normalize_out_vars(_OUT_VARS)
    translate = output_translator(model_card['variables'], standard_dict['variables'])
    output_vars = [translate.get(item, item) for item in out_vars]

    
    desired_levels = None                    # 'original': keep every model level
    if _OUT_LEVS != 'original':
        desired_levels = [
            int(plev)
            for plev in _OUT_LEVS.strip('[]').split(',')
        ]

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

    # Each state is reduced to the requested variables/levels and written to per-variable
    # buffers on disk, so memory does not grow with the length of the forecast.
    _TEMP_DIR = config.get("TEMP_DIR") or os.path.join(_OUTPUT_PATH, "tmp")
    stream_dir = os.path.join(_TEMP_DIR, "aifs-stream")
    capacity = outer_steps // 6 + 2          # AIFS steps are 6 h; small margin

    # Times come from the model: the initial time is the date of the input state (the second,
    # most recent IC state, T) and each valid time is the date of the predicted state (T+6h, ...).
    # They used to be rebuilt as ICs.time[0] + n*6h, i.e. from T-6h, so every AIFS output was
    # labelled 12 h early.
    initial_time = _as_datetime64(input_state["date"])
    valid_times = []

    with StepBuffer(stream_dir, capacity) as buffer:
        template = None                      # first state: dims, coords and attributes
        for state in runner.run(input_state=input_state, lead_time=outer_steps):
            state_name = state["date"].strftime("%Y%m%d%H")
            print(f"Generated state for {state_name}")
            valid_times.append(_as_datetime64(state["date"]))

            # AIFS ENS 2.0 carries the mean wave direction as cos/sin components.
            if "mwd" in output_vars and "mwd" not in state["fields"] and "cos_mwd" in state["fields"]:
                state["fields"]["mwd"] = cos_sin_to_direction(
                    state["fields"]["cos_mwd"], state["fields"]["sin_mwd"]).astype(np.float32)

            ds_t = build_dataset_for_state(state, output_vars, desired_levels)
            buffer.append({name: ds_t[name].values[0] for name in ds_t.data_vars})
            if template is None:
                template = ds_t
            del ds_t

        n_out = buffer.count
        print(f"Collected {n_out} states for {buffer.names}")

        # # # --- valid_time/step coords: the dates of the produced states ---
        valid_time = np.array(valid_times, dtype="datetime64[ns]")
        steps = valid_time - initial_time
        print(f"Initial time {initial_time}, first valid time {valid_time[0]}, last {valid_time[-1]}")

        # Drop requested variables AIFS does not produce, as Aurora does.
        keep = [(v, n) for v, n in zip(output_vars, out_vars) if template is not None and v in template.data_vars]
        for v, n in zip(output_vars, out_vars):
            if template is None or v not in template.data_vars:
                print(f"WARNING: requested output variable '{n}' not found in AIFS predictions; skipped.")
        if not keep:
            raise ValueError(f"None of the requested output variables {out_vars} is produced by AIFS.")

        # Outputs are written under the AIUQ-st names (OUT_VARS), like NeuralGCM and Aurora:
        # templates/postprocess_aifs.sh looks them up by those names.
        for var, var_name in keep:
            reference = template[var]
            predictions = xr.DataArray(
                buffer.read(var),
                dims=reference.dims,
                coords={name: coord for name, coord in reference.coords.items() if name != "time"},
                name=reference.name,
                attrs=reference.attrs,
            )
            dataset = predictions.to_dataset(name=var)
            dataset.attrs["Conventions"] = "CF-1.8"

            # # rename time -> valid_time and attach coords with matching length
            dataset = dataset.rename({"time": "valid_time"})
            dataset = dataset.assign_coords(valid_time=("valid_time", valid_time))
            dataset = dataset.assign_coords(time=initial_time)
            dataset = dataset.assign_coords(step=("valid_time", steps))

            # # Drop sim_time if present
            if "sim_time" in dataset.variables:
                dataset = dataset.drop_vars("sim_time")

            # # Format output frequency
            if _OUT_FREQ == "daily":
                dataset = dataset.resample(valid_time="1D").mean()

            predictions_datarray = dataset[var].rename(var_name)
            OUTPUT_BASE_PATH = f"{_OUTPUT_PATH}/{var_name}/{str(_RNG_KEY)}"
            os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
            OUTPUT_FILE = f"{OUTPUT_BASE_PATH}/out-{_START_TIME}-{_END_TIME}-{_RNG_KEY}-{var_name}_temp.nc"
            predictions_datarray.to_netcdf(OUTPUT_FILE)
            print(f"Saved {OUTPUT_FILE}")
            del dataset, predictions, predictions_datarray

if __name__ == "__main__":
    main()