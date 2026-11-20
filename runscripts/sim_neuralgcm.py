"""

"""

# Built-in/Generics
import os
import pickle
import logging
from glob import glob
from datetime import datetime, timedelta

# Third party
import numpy as np
import xarray as xr
import neuralgcm
from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils

# Local
from AIUQst_lib.functions import parse_arguments, read_config, normalize_out_vars
from AIUQst_lib.cards import read_model_card, read_std_version
from AIUQst_lib.variables import (name_mapper_for_model, output_attrs, resample_daily, resolve_outputs,
                                  deaccumulate)
from AIUQst_lib.streaming import StepBuffer
from AIUQst_lib.regrid import regrid_latlon

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

    # Hindcast: unroll in blocks of CHUNK_STEPS (bounded GPU memory) and stream the requested
    # variables to disk, then write one file per variable for the whole run, as before.
    # (Unrolling the whole run at once kept every prediction on the GPU and in host memory:
    # 20 days at 0.7 deg exhausted a 46 GB L40S and used 23 GB of RAM.)
    stream_outputs = _RUN_TYPE == "hindcast"
    if stream_outputs:
        logging.info(f"Hindcast mode: unrolling in blocks of {_CHUNK_STEPS} steps, single output file per variable")

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

        amip_forcing = xr.open_mfdataset(os.path.join(_AMIP_FORCING_PATH, "*.nc")).load()

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

        print(f"amip_data dims: {amip_data.dims}")
        print(f"amip_data.time size: {amip_data.time.size if 'time' in amip_data.dims else 'NO TIME DIM'}")
        print(f"all_forcings['sim_time'] shape: {all_forcings['sim_time'].shape}")
        
    else:
        raise ValueError(f"Unsupported RUN_TYPE '{_RUN_TYPE}'. Use 'hindcast' or 'amip'.")

    # Prepare output time coordinates
    initial_time = initial_conditions.time.isel(time=0).values

    # Resume from checkpoint if available (AMIP only: hindcast outputs are assembled at the end
    # of the run, so a hindcast always restarts from the initial conditions)
    if stream_outputs and os.path.exists(state_checkpoint_file):
        logging.info("Hindcast mode: ignoring the stale state checkpoint, starting from scratch")
        os.remove(state_checkpoint_file)
    if os.path.exists(state_checkpoint_file):
        logging.info("Checkpoint found, resuming from saved state")
        with open(state_checkpoint_file, "rb") as f:
            checkpoint_data = pickle.load(f)
        current_state = checkpoint_data["state"]
        start_step = int(checkpoint_data.get("step_done", 0))
        resumed_totals = checkpoint_data.get("last_total", {})
    else:
        logging.info("No checkpoint found, starting from scratch")
        current_state = initial_state
        start_step = 0
        resumed_totals = {}

    if start_step >= outer_steps:
        raise RuntimeError(
            "Checkpoint indicates the simulation is already complete "
            f"(step_done={start_step}, total_steps={outer_steps}). "
            "No chunks to process. Remove the checkpoint file and rerun."
        )

    # Pre-compute output settings (invariant across chunks)
    # Requested outputs (AIUQ-st names) -> NeuralGCM names; names this checkpoint cannot write are skipped.
    selected, unknown = resolve_outputs(normalize_out_vars(_OUT_VARS), model_card, standard_dict, _MODEL_CHECKPOINT)
    for name in unknown:
        logging.warning(f"'{name}' is not an output of {os.path.basename(_MODEL_CHECKPOINT)} (model card); skipped.")
    out_vars = [name for name, _ in selected]
    specs = dict(selected)
    output_vars = [spec["model"] for _, spec in selected]
    if not output_vars:
        raise ValueError(f"None of the requested output variables {_OUT_VARS} is written by {os.path.basename(_MODEL_CHECKPOINT)}.")
    # Running totals (precipitation_cumulative_mean) at the end of the previous chunk, to write increments.
    last_total = dict(resumed_totals)

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

    buffer = None
    template = None
    if stream_outputs:
        stream_dir = os.path.join(config.get("TEMP_DIR") or os.path.join(_OUTPUT_PATH, "tmp"), "neuralgcm-stream")
        buffer = StepBuffer(stream_dir, outer_steps)

    for chunk_start in range(start_step, outer_steps, _CHUNK_STEPS):
        chunk_end = min(chunk_start + _CHUNK_STEPS, outer_steps)
        current_chunk_steps = chunk_end - chunk_start

        logging.info(f"Running inference {chunk_start} -> {chunk_end}")

        final_state, predictions = model.unroll(
            current_state,
            all_forcings,
            steps=current_chunk_steps,
            timedelta=delta_t,
            # The input state of every chunk is the previous final state (step chunk_start), so it
            # is the first output of the chunk: chunks then concatenate exactly like a single
            # unroll. (With start_with_input=False after the first chunk, step chunk_start was
            # skipped and every later output was labelled one step early.)
            start_with_input=True,
        )

        current_state = final_state

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
        # Surface diagnostics (precipitation, evaporation) carry a length-1 'surface' dimension.
        if "surface" in predictions_ds_chunk.dims:
            predictions_ds_chunk = predictions_ds_chunk.squeeze("surface", drop=True)

        if chunk_start == start_step:
            # Drop requested variables the model does not produce (e.g. t2m), as Aurora does.
            keep = [(v, n) for v, n in zip(output_vars, out_vars) if v in predictions_ds_chunk.data_vars]
            for v, n in zip(output_vars, out_vars):
                if v not in predictions_ds_chunk.data_vars:
                    logging.warning(f"Requested output variable '{n}' not found in NeuralGCM predictions; skipped.")
            if not keep:
                raise ValueError(f"None of the requested output variables {out_vars} is produced by NeuralGCM.")
            output_vars, out_vars = [v for v, _ in keep], [n for _, n in keep]

        if stream_outputs:
            # Keep only the requested variables, one step at a time, on disk.
            selected = predictions_ds_chunk[output_vars].transpose("valid_time", ...)
            if template is None:
                template = selected.isel(valid_time=slice(0, 1)).load()
            for i in range(selected.sizes["valid_time"]):
                buffer.append({var: selected[var].isel(valid_time=i).values for var in output_vars})
            del predictions, predictions_ds_chunk, selected
            continue

        # Apply post-processing per chunk: increments of running totals, units, frequency
        for var, var_name in zip(output_vars, out_vars):
            spec = specs[var_name]
            if spec["deaccumulate"]:
                da = predictions_ds_chunk[var].transpose("valid_time", ...)
                total = da.values
                da = da.copy(data=deaccumulate(total, last_total.get(var)))
                last_total[var] = total[-1]
                predictions_ds_chunk[var] = da
            if spec["scale"] != 1.0:
                predictions_ds_chunk[var] = predictions_ds_chunk[var] * spec["scale"]
        if _OUT_FREQ == "daily":
            predictions_ds_chunk = xr.merge([
                resample_daily(predictions_ds_chunk[[var]], var, specs[var_name]["accumulated"])
                for var, var_name in zip(output_vars, out_vars)])

        # Levels first: regridding only the requested levels needs far less memory.
        if desired_levels is not None:
            predictions_ds_chunk = predictions_ds_chunk.interp(level=desired_levels)

        if _OUT_RES in ["0.25", "0.5", "1", "1.5", "2"]:
            predictions_ds_chunk = regrid_latlon(predictions_ds_chunk, out_latitudes, out_longitudes)

        # Save chunk netCDF per variable
        chunk_t_start = str(chunk_valid_time[0])[:10]
        chunk_t_end = str(chunk_valid_time[-1])[:10]

        for var, var_name in zip(output_vars, out_vars):
            predictions_datarray = predictions_ds_chunk[var].rename(var_name)
            predictions_datarray.attrs.update(output_attrs(specs[var_name], _OUT_FREQ == "daily"))
            output_base_path = f"{_OUTPUT_PATH}/{var_name}/{str(_RNG_KEY)}"
            os.makedirs(output_base_path, exist_ok=True)
            output_file = (
                f"{output_base_path}/out-{chunk_t_start}-{chunk_t_end}-{_RNG_KEY}-{var_name}.nc"
            )
            predictions_datarray.to_netcdf(output_file)
            logging.info(f"Saved {output_file}")

        # Save the state checkpoint once the chunk's outputs are written (with the running totals of the
        # accumulated diagnostics, so a resumed run keeps writing increments).
        with open(state_checkpoint_file, "wb") as f:
            pickle.dump({"state": current_state, "step_done": chunk_end, "last_total": last_total}, f)

    if stream_outputs:
        # Whole-run post-processing, one variable at a time (same steps and file names as the
        # former single-chunk hindcast).
        run_valid_time = initial_time + np.arange(outer_steps) * delta_t
        run_steps = np.arange(outer_steps) * delta_t
        run_t_start = str(run_valid_time[0])[:10]
        run_t_end = str(run_valid_time[-1])[:10]
        try:
            for var, var_name in zip(output_vars, out_vars):
                spec = specs[var_name]
                reference = template[var]
                coords = {name: coord for name, coord in reference.coords.items()
                          if "valid_time" not in coord.dims}
                coords["valid_time"] = run_valid_time
                coords["step"] = ("valid_time", run_steps)
                values = buffer.read(var)
                if spec["deaccumulate"]:
                    values = deaccumulate(values)
                if spec["scale"] != 1.0:
                    values = values * spec["scale"]
                dataset = xr.DataArray(
                    values, dims=reference.dims, coords=coords,
                    name=reference.name, attrs=reference.attrs,
                ).to_dataset(name=var)

                if _OUT_FREQ == "daily":
                    dataset = resample_daily(dataset, var, spec["accumulated"])

                # Levels first: regridding all 37 model levels to 0.25 deg and discarding most of
                # them afterwards ran out of memory (a00d: 16 days, 2 levels, > 48 GB).
                if desired_levels is not None and "level" in dataset.dims:
                    dataset = dataset.interp(level=desired_levels)

                if _OUT_RES in ["0.25", "0.5", "1", "1.5", "2"]:
                    dataset = regrid_latlon(dataset, out_latitudes, out_longitudes)

                predictions_datarray = dataset[var].rename(var_name)
                predictions_datarray.attrs.update(output_attrs(spec, _OUT_FREQ == "daily"))
                output_base_path = f"{_OUTPUT_PATH}/{var_name}/{str(_RNG_KEY)}"
                os.makedirs(output_base_path, exist_ok=True)
                output_file = (
                    f"{output_base_path}/out-{run_t_start}-{run_t_end}-{_RNG_KEY}-{var_name}.nc"
                )
                predictions_datarray.to_netcdf(output_file)
                logging.info(f"Saved {output_file}")
                del dataset, predictions_datarray
        finally:
            buffer.close()

    # Remove checkpoint at successful completion
    if os.path.exists(state_checkpoint_file):
        os.remove(state_checkpoint_file)


if __name__ == "__main__":
    main()