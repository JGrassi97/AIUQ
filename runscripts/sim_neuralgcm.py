"""

"""

# Built-in/Generics
import os
import pickle
import logging
from datetime import datetime, timedelta

# Third party
import numpy as np
import xarray as xr
import neuralgcm
from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils

# Local
from AIUQst_lib.functions import parse_arguments, read_config, normalize_out_vars
from AIUQst_lib.cards import read_model_card, read_std_version
from AIUQst_lib.variables import name_mapper_for_model, output_translator

# Configure logging
logging.basicConfig(level=logging.DEBUG)


def main() -> None:
    # Read config
    args = parse_arguments()
    config = read_config(args.config)

    _INI_DATA_PATH = config.get("INI_DATA_PATH", "")
    _START_TIME = config.get("START_TIME", "")
    _END_TIME = config.get("END_TIME", "")
    _HPCROOTDIR = config.get("HPCROOTDIR", "")
    _MODEL_NAME = config.get("MODEL_NAME", "")
    _STD_VERSION = config.get("STD_VERSION", "")
    _MODEL_CHECKPOINT = config.get("MODEL_CHECKPOINT", "")
    _INNER_STEPS = config.get("INNER_STEPS", 1)
    _RNG_KEY = config.get("RNG_KEY", 1)
    _OUTPUT_PATH = config.get("OUTPUT_PATH", "")
    _OUT_VARS = config.get("OUT_VARS", [])
    _OUT_FREQ = config.get("OUT_FREQ", "")
    _OUT_RES = config.get("OUT_RES", "")
    _OUT_LEVS = config.get("OUT_LEVS", "")
    _RUN_TYPE = config.get("RUN_TYPE", "hindcast").lower()
    _AMIP_FORCING_PATH = config.get("AMIP_FORCING_PATH", "")

    # Experimental settings for long simulations
    _CHECKPOINT_DIR = config.get("CHECKPOINT_DIR", "")
    _CHUNK_STEPS = config.get("CHUNK_STEPS", 48)

    if _CHUNK_STEPS <= 0:
        raise ValueError("CHUNK_STEPS must be greater than 0")

    checkpoint_dir = os.path.join(_CHECKPOINT_DIR, str(_RNG_KEY))
    os.makedirs(checkpoint_dir, exist_ok=True)

    state_checkpoint_file = os.path.join(
        checkpoint_dir,
        f"state-{_START_TIME}-{_END_TIME}-{_RNG_KEY}.pkl",
    )

    # IC settings
    model_card = read_model_card(_HPCROOTDIR, _MODEL_NAME)
    standard_dict = read_std_version(_HPCROOTDIR, _STD_VERSION)

    # Format time settings
    start_date = datetime.strptime(_START_TIME, "%Y-%m-%d")
    input_end_date = datetime.strptime(_START_TIME, "%Y-%m-%d") + timedelta(days=1)
    end_date = datetime.strptime(_END_TIME, "%Y-%m-%d")

    days_to_run = (end_date - start_date).days + 1
    outer_steps = days_to_run * 24 // _INNER_STEPS
    delta_t = np.timedelta64(_INNER_STEPS, "h")
    times = np.arange(outer_steps) * _INNER_STEPS  # in hours

    # Auto-set CHUNK_STEPS to outer_steps for hindcast to process in single chunk
    if _RUN_TYPE == "hindcast":
        _CHUNK_STEPS = outer_steps
        logging.info(f"Hindcast mode: auto-setting CHUNK_STEPS to {_CHUNK_STEPS} (entire run in single chunk)")

    # Load model
    with open(_MODEL_CHECKPOINT, "rb") as f:
        ckpt = pickle.load(f)
    model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)

    # Load data
    initial_conditions = xr.open_zarr(_INI_DATA_PATH, chunks=None).load()

    # Map names for model specifics
    mapper = name_mapper_for_model(
        model_card["variables"],
        standard_dict["variables"],
    )
    initial_conditions = initial_conditions.rename(mapper)

    # Setup grid and regridder
    data_grid = spherical_harmonic.Grid(
        latitude_nodes=initial_conditions.sizes["latitude"],
        longitude_nodes=initial_conditions.sizes["longitude"],
        latitude_spacing=xarray_utils.infer_latitude_spacing(
            initial_conditions.latitude
        ),
        longitude_offset=xarray_utils.infer_longitude_offset(
            initial_conditions.longitude
        ),
    )
    regridder = horizontal_interpolation.ConservativeRegridder(
        data_grid,
        model.data_coords.horizontal,
        skipna=True,
    )

    # Regrid and fill NaNs
    regridded = xarray_utils.regrid(initial_conditions, regridder)
    data = xarray_utils.fill_nan_with_nearest(regridded)

    # Prepare inputs and forcings
    inputs = model.inputs_from_xarray(data.isel(time=0))
    input_forcings = model.forcings_from_xarray(data.isel(time=1))
    initial_state = model.encode(inputs, input_forcings, int(_RNG_KEY))

    # Recognize different types of forcings - hindcast/amip
    if _RUN_TYPE == "hindcast":
        all_forcings = model.forcings_from_xarray(data.head(time=1))
    elif _RUN_TYPE == "amip":
        if not _AMIP_FORCING_PATH:
            _AMIP_FORCING_PATH = os.path.join(_HPCROOTDIR, "forcing", "amip")

        if not os.path.exists(_AMIP_FORCING_PATH):
            raise FileNotFoundError(
                "AMIP forcing path not found: "
                f"{_AMIP_FORCING_PATH}. Provide a pre-built monthly NetCDF forcing file."
            )

        # Load AMIP monthly forcing NetCDF (SST, sea-ice, land-sea mask)
        amip_forcing = xr.open_dataset(_AMIP_FORCING_PATH, chunks=None)

        # Standardize longitude to [0, 360]
        amip_forcing["longitude"] = amip_forcing["longitude"] % 360
        amip_forcing = amip_forcing.sortby("longitude")

        # Apply model variable mapping for forcing variables present in AMIP dataset
        forcing_mapper = {
            k: v
            for k, v in mapper.items()
            if k in amip_forcing.data_vars
        }
        if forcing_mapper:
            amip_forcing = amip_forcing.rename(forcing_mapper)

        # Regrid forcing to model grid and fill NaNs
        amip_regridded = xarray_utils.regrid(amip_forcing, regridder)
        amip_data = xarray_utils.fill_nan_with_nearest(amip_regridded)

        all_forcings = model.forcings_from_xarray(amip_data)
    else:
        raise ValueError(f"Unsupported RUN_TYPE '{_RUN_TYPE}'. Use 'hindcast' or 'amip'.")

    # Prepare output time coordinates
    initial_time = initial_conditions.time.isel(time=0).values

    # Resume from checkpoint if available
    if os.path.exists(state_checkpoint_file):
        logging.info("Checkpoint found, resuming from saved state")
        with open(state_checkpoint_file, "rb") as f:
            checkpoint_data = pickle.load(f)
        current_state = checkpoint_data["state"]
        start_step = int(checkpoint_data.get("step_done", 0))
    else:
        logging.info("No checkpoint found, starting from scratch")
        current_state = initial_state
        start_step = 0

    if start_step >= outer_steps:
        raise RuntimeError(
            "Checkpoint indicates the simulation is already complete "
            f"(step_done={start_step}, total_steps={outer_steps}). "
            "No chunks to process. Remove the checkpoint file and rerun."
        )

    # Pre-compute output settings (invariant across chunks)
    out_vars = normalize_out_vars(_OUT_VARS)
    translate = output_translator(
        model_card["variables"],
        standard_dict["variables"],
    )
    output_vars = [translate.get(item, item) for item in out_vars]

    if _OUT_RES == "0.25":
        out_latitudes = np.arange(-90, 90.25, 0.25)
        out_longitudes = np.arange(0, 360, 0.25)
    elif _OUT_RES == "0.5":
        out_latitudes = np.arange(-90, 90.5, 0.5)
        out_longitudes = np.arange(0, 360, 0.5)
    elif _OUT_RES == "1":
        out_latitudes = np.arange(-90, 91, 1.0)
        out_longitudes = np.arange(0, 360, 1.0)
    elif _OUT_RES == "1.5":
        out_latitudes = np.arange(-90, 91.5, 1.5)
        out_longitudes = np.arange(0, 360, 1.5)
    elif _OUT_RES == "2":
        out_latitudes = np.arange(-90, 92, 2.0)
        out_longitudes = np.arange(0, 360, 2.0)
    else:
        out_latitudes = None
        out_longitudes = None

    if _OUT_LEVS != "original":
        desired_levels = [
            int(plev)
            for plev in _OUT_LEVS.strip("[]").split(",")
        ]
    else:
        desired_levels = None

    for chunk_start in range(start_step, outer_steps, _CHUNK_STEPS):
        chunk_end = min(chunk_start + _CHUNK_STEPS, outer_steps)
        current_chunk_steps = chunk_end - chunk_start

        logging.info(f"Running inference {chunk_start} -> {chunk_end}")

        final_state, predictions = model.unroll(
            current_state,
            all_forcings,
            steps=current_chunk_steps,
            timedelta=delta_t,
            start_with_input=(chunk_start == 0),
        )

        current_state = final_state

        # Save state checkpoint after each chunk
        with open(state_checkpoint_file, "wb") as f:
            pickle.dump(
                {
                    "state": current_state,
                    "step_done": chunk_end,
                },
                f,
            )

        # Convert current chunk predictions to xarray
        chunk_valid_time = initial_time + (
            np.arange(chunk_start, chunk_end) * delta_t
        )
        chunk_steps = np.arange(chunk_start, chunk_end) * delta_t

        predictions_ds_chunk = model.data_to_xarray(
            predictions,
            times=chunk_valid_time,
        )
        predictions_ds_chunk = predictions_ds_chunk.rename({"time": "valid_time"})
        predictions_ds_chunk = predictions_ds_chunk.assign_coords(time=initial_time)
        predictions_ds_chunk = predictions_ds_chunk.assign_coords(
            step=("valid_time", chunk_steps)
        )

        if "sim_time" in predictions_ds_chunk.variables:
            predictions_ds_chunk = predictions_ds_chunk.drop_vars("sim_time")

        # Apply post-processing per chunk
        if _OUT_FREQ == "daily":
            predictions_ds_chunk = predictions_ds_chunk.resample(valid_time="1D").mean()

        if _OUT_RES in ["0.25", "0.5", "1", "1.5", "2"]:
            predictions_ds_chunk = predictions_ds_chunk.interp(
                latitude=out_latitudes,
                longitude=out_longitudes,
                method="linear",
            )

        if desired_levels is not None:
            predictions_ds_chunk = predictions_ds_chunk.interp(level=desired_levels)

        # Save chunk netCDF per variable
        chunk_t_start = str(chunk_valid_time[0])[:10]
        chunk_t_end = str(chunk_valid_time[-1])[:10]

        for var, var_name in zip(output_vars, out_vars):
            predictions_datarray = predictions_ds_chunk[var].rename(var_name)
            output_base_path = f"{_OUTPUT_PATH}/{var_name}/{str(_RNG_KEY)}"
            os.makedirs(output_base_path, exist_ok=True)
            output_file = (
                f"{output_base_path}/out-{chunk_t_start}-{chunk_t_end}-{_RNG_KEY}-{var_name}.nc"
            )
            predictions_datarray.to_netcdf(output_file)
            logging.info(f"Saved {output_file}")

    # Remove checkpoint at successful completion
    if os.path.exists(state_checkpoint_file):
        os.remove(state_checkpoint_file)


if __name__ == "__main__":
    main()