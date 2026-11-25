"""
Runscript for Google DeepMind WeatherNext 2 (FGN) simulations.

WeatherNext 2 is an ensemble model: an ensemble is made of independently trained models ('seeds'), each
generating members from a 32-dimensional noise vector drawn anew at every 6-hour step. AIUQ follows the
per-job member convention of the other models: one Autosubmit MEMBER == one RNG_KEY == one trajectory.
Member m runs seed ((m - 1) mod n_seeds) + 1 with JAX random key fold_in(PRNGKey(0), m), so an ensemble
of 4k members takes k members from each of 4 seeds, as in the paper (arXiv:2506.10772), and every member
is reproducible on the same hardware.

MODEL_CHECKPOINT is the folder of the checkpoint (conf/cards/models/weathernext_card.yml, 'checkpoints'
block) with the seeds as model1.npz, model2.npz, ...
"""

# Built-in/Generics
import dataclasses
import glob
import logging
import os
import re
from datetime import datetime

# Third party
import dask.array as da
import haiku as hk
import jax
import numpy as np
import xarray as xr
import xarray_jax
from weathernext.utils import checkpoint, data_utils, fiddle_config_io, rollout
from weathernext.weathernext2 import fgn

# Local
from AIUQst_lib.functions import parse_arguments, read_config, normalize_out_vars
from AIUQst_lib.cards import read_model_card, read_std_version
from AIUQst_lib.variables import checkpoint_entry, name_mapper_for_model, resample_daily, resolve_outputs
from AIUQst_lib.streaming import StepBuffer
from AIUQst_lib.regrid import OUT_RESOLUTIONS, regrid_latlon, target_grid
from AIUQst_lib.output import RunInfo, write_output

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_STEP_HOURS = 6
STATIC_VARIABLES = ("geopotential_at_surface", "land_sea_mask")


# Checkpoint ===========================================================================================

def seed_files(checkpoint_dir: str) -> list[str]:
    """Weights of the seeds of a checkpoint folder, ordered model1, model2, ..."""
    files = glob.glob(os.path.join(checkpoint_dir, "model*.npz"))
    number = lambda path: int(re.search(r"model(\d+)\.npz$", path).group(1))
    files = sorted((f for f in files if re.search(r"model(\d+)\.npz$", f)), key=number)
    if not files:
        raise FileNotFoundError(f"No model<k>.npz weights in '{checkpoint_dir}' (see remote_set/get_models.txt).")
    return files


def member_seed(member: int, n_seeds: int) -> int:
    """Seed (0-based) that runs member m (1-based): equal numbers of members per seed."""
    return (int(member) - 1) % n_seeds


def load_model(checkpoint_dir: str, config_name: str, member: int):
    """(config, params, seed file) of the seed that runs this member."""
    files = seed_files(checkpoint_dir)
    weights = files[member_seed(member, len(files))]
    config = fiddle_config_io.get_fiddle_config_by_name(f"weathernext2/configs/{config_name}")
    with open(weights, "rb") as f:
        ckpt = checkpoint.load(f, fgn.CheckPoint)
    return config, ckpt.params, weights


def inference_config(config, backend: str):
    """Predictor configuration for sampling: without the training-time ensemble wrapper, with an attention
    implementation that runs on the backend (the splash kernel of the configs is TPU only)."""
    transformer_kwargs = config.predictor_kwargs["noisy_function_kwargs"]["mesh_model_ctor"].keywords[
        "transformer_kwargs"]
    if backend != "tpu":
        transformer_kwargs["attention_type"] = "triblockdiag_mha"
    return fgn.PredictorConfig(
        task=config.task,
        predictor_constructor=config.predictor_constructor,
        predictor_kwargs=config.predictor_kwargs,
        predictor_wrappers=config.predictor_wrappers[:-1],      # the last wrapper is the training ensemble
    )


# Initial conditions ===================================================================================

def model_grid(ds: xr.Dataset, resolution: float) -> xr.Dataset:
    """Latitudes south to north, longitudes [0, 360), subsampled to the model resolution."""
    ds = ds.sortby("latitude")
    ds = ds.assign_coords(longitude=ds.longitude % 360).sortby("longitude")
    spacing = float(np.diff(ds.latitude.values[:2])[0])
    stride = int(round(resolution / spacing))
    if stride < 1 or abs(stride * spacing - resolution) > 1e-6:
        raise ValueError(f"Initial conditions at {spacing} deg cannot be subsampled to {resolution} deg.")
    if stride > 1:
        ds = ds.isel(latitude=slice(None, None, stride), longitude=slice(None, None, stride))
    expected = int(round(180 / resolution)) + 1
    if ds.sizes["latitude"] != expected or abs(float(ds.latitude[0]) + 90) > 1e-6:
        raise ValueError(f"The model needs a global {resolution} deg grid from pole to pole "
                         f"(got {ds.sizes['latitude']} latitudes from {float(ds.latitude[0])}).")
    return ds


def initial_state(path: str, mapper: dict, task, resolution: float) -> tuple[xr.Dataset, np.datetime64]:
    """The input times (T-6h, T) of the AIUQ initial conditions in the layout of the WeatherNext datasets:
    variables (batch, time, [level,] lat, lon), static fields (lat, lon), time as lead time (-6h, 0) and the
    'datetime' coordinate. Returns (dataset, T)."""
    ic = xr.open_zarr(path, chunks=None)
    wanted = set(task.input_variables) | set(task.target_variables)
    rename = {short: name for short, name in mapper.items() if short in ic and name in wanted}
    ic = ic[list(rename)].rename(rename)
    missing = [v for v in task.input_variables if v not in ic and v not in data_utils._DERIVED_VARS]
    if missing:
        raise KeyError(f"Initial conditions lack the model inputs {missing}.")

    ic = ic.sel(level=list(task.pressure_levels))
    ic = model_grid(ic, resolution).rename({"latitude": "lat", "longitude": "lon"})
    ic = ic.assign_coords(level=ic.level.values.astype(np.int32))

    times = ic.time.values[-2:]
    step = np.timedelta64(MODEL_STEP_HOURS, "h")
    if len(times) != 2 or times[1] - times[0] != step:
        raise ValueError(f"The model needs the initial conditions at T-6h and T (got {ic.time.values}).")

    state = xr.Dataset(coords={"lat": ic.lat.values.astype(np.float32), "lon": ic.lon.values.astype(np.float32),
                               "level": ic.level.values})
    for name in ic.data_vars:
        field = ic[name]
        if name in STATIC_VARIABLES:
            field = field.isel(time=-1, drop=True) if "time" in field.dims else field
            state[name] = (field.dims, field.astype(np.float32).values)
        else:
            field = field.sel(time=times).expand_dims(batch=1)
            state[name] = (field.dims, field.astype(np.float32).values)
    state = state.assign_coords(time=times - times[-1])
    state = state.assign_coords(datetime=(("batch", "time"), times[None]))
    return state, times[-1]


def forecast_inputs(state: xr.Dataset, task, n_steps: int):
    """(inputs, lazy targets template, forcings) for n_steps 6-hour steps after T.

    The targets template is a dask array of NaNs: the rollout computes one step of it at a time, so its size
    does not depend on the length of the run."""
    step = np.timedelta64(MODEL_STEP_HOURS, "h")
    lead = (np.arange(1, n_steps + 1) * step).astype("timedelta64[ns]")
    reference = state.datetime.values[0, -1]

    inputs = state.copy()
    data_utils.add_derived_vars(inputs)
    inputs = inputs.drop_vars("datetime")[list(task.input_variables)]

    nlat, nlon, nlev = state.sizes["lat"], state.sizes["lon"], state.sizes["level"]
    template = xr.Dataset(coords={"lat": state.lat, "lon": state.lon, "level": state.level, "time": lead})
    for name in task.target_variables:
        upper = name in state and "level" in state[name].dims
        dims = ("batch", "time", "level", "lat", "lon") if upper else ("batch", "time", "lat", "lon")
        shape = (1, n_steps, nlev, nlat, nlon) if upper else (1, n_steps, nlat, nlon)
        chunks = (1, 1, nlev, nlat, nlon) if upper else (1, 1, nlat, nlon)
        template[name] = (dims, da.full(shape, np.nan, dtype=np.float32, chunks=chunks))

    forcings = xr.Dataset(coords={"lon": state.lon, "lat": state.lat, "time": lead})
    forcings = forcings.assign_coords(datetime=(("batch", "time"), (reference + lead)[None]))
    data_utils.add_derived_vars(forcings)
    forcings = forcings.drop_vars("datetime")[list(task.forcing_variables)]
    return inputs, template, forcings


# Outputs ==============================================================================================

def _output_level_selection(model_levels, out_levs) -> list[int] | None:
    """Indices of the requested output levels when all of them are model levels, else None (all levels are
    then kept and interpolated after the rollout)."""
    if out_levs in (None, "", "original"):
        return None
    desired = [int(plev) for plev in str(out_levs).strip("[]").split(",") if str(plev).strip()]
    levels = list(model_levels)
    if desired and all(p in levels for p in desired):
        return [levels.index(p) for p in desired]
    return None


def _host_array(field) -> np.ndarray:
    """A step of a prediction as a host numpy array (the rollout yields JAX-backed xarray objects)."""
    try:
        return np.asarray(jax.device_get(xarray_jax.unwrap_data(field)))
    except Exception:                                    # noqa: BLE001 - plain numpy-backed variable
        return np.asarray(field.values)


def _variable_dataset(var, data, lats, lons, levels, initial_time, delta_t) -> xr.Dataset:
    """Standardised xarray.Dataset for one output variable (data: steps x [level] x lat x lon)."""
    n_out = data.shape[0]
    valid_time = initial_time + np.arange(1, n_out + 1) * delta_t
    dims = ("valid_time", "latitude", "longitude") if data.ndim == 3 else ("valid_time", "level", "latitude", "longitude")
    coords = {"valid_time": valid_time, "time": initial_time, "step": ("valid_time", np.arange(1, n_out + 1) * delta_t),
              "latitude": lats, "longitude": lons}
    if data.ndim == 4:
        coords["level"] = levels
    return xr.Dataset(data_vars={var: (dims, data)}, coords=coords)


# Main =================================================================================================

def main() -> None:
    args = parse_arguments()
    config = read_config(args.config)

    _INI_DATA_PATH = config.get("INI_DATA_PATH", "")
    _START_TIME = config.get("START_TIME", "")
    _END_TIME = config.get("END_TIME", "")
    _HPCROOTDIR = config.get("HPCROOTDIR", "")
    _MODEL_NAME = config.get("MODEL_NAME", "")
    _STD_VERSION = config.get("STD_VERSION", "")
    _MODEL_CHECKPOINT = str(config.get("MODEL_CHECKPOINT", "")).rstrip("/")
    _INNER_STEPS = int(config.get("INNER_STEPS", MODEL_STEP_HOURS))
    _RNG_KEY = int(config.get("RNG_KEY", 1) or 1)
    _OUTPUT_PATH = config.get("OUTPUT_PATH", "")
    _OUT_VARS = config.get("OUT_VARS", [])
    _OUT_FREQ = config.get("OUT_FREQ", "")
    _OUT_RES = config.get("OUT_RES", "")
    _OUT_LEVS = config.get("OUT_LEVS", "")

    if _INNER_STEPS != MODEL_STEP_HOURS:
        logging.warning(f"WeatherNext 2 steps 6 hours; INNER_STEPS={_INNER_STEPS} ignored.")

    model_card = read_model_card(_HPCROOTDIR, _MODEL_NAME)
    standard_dict = read_std_version(_HPCROOTDIR, _STD_VERSION)
    entry = checkpoint_entry(model_card, _MODEL_CHECKPOINT)
    if "config" not in entry:
        raise ValueError(f"Checkpoint '{os.path.basename(_MODEL_CHECKPOINT)}' is not in the 'checkpoints' "
                         f"block of {_MODEL_NAME}_card.yml.")
    resolution = float(entry.get("resolution", 0.25))

    mapper = name_mapper_for_model(model_card["variables"], standard_dict["variables"])

    selected, unknown = resolve_outputs(normalize_out_vars(_OUT_VARS), model_card, standard_dict, _MODEL_CHECKPOINT)
    for name in unknown:
        logging.warning(f"'{name}' is not an output of {os.path.basename(_MODEL_CHECKPOINT)} (model card); skipped.")
    if not selected:
        raise ValueError(f"None of the requested output variables {_OUT_VARS} is written by "
                         f"{os.path.basename(_MODEL_CHECKPOINT)}.")
    out_vars = [name for name, _ in selected]
    specs = dict(selected)
    output_vars = [spec["model"] for _, spec in selected]

    start_date = datetime.strptime(_START_TIME, "%Y-%m-%d")
    end_date = datetime.strptime(_END_TIME, "%Y-%m-%d")
    days_to_run = (end_date - start_date).days + 1
    n_steps = days_to_run * 24 // MODEL_STEP_HOURS
    delta_t = np.timedelta64(MODEL_STEP_HOURS, "h")
    run_info = RunInfo.from_config(config)

    backend = jax.default_backend()
    model_config, params, weights = load_model(_MODEL_CHECKPOINT, entry["config"], _RNG_KEY)
    task = model_config.task
    logging.info(f"Running WeatherNext {os.path.basename(_MODEL_CHECKPOINT)} member {_RNG_KEY} with seed "
                 f"{os.path.basename(weights)} on {backend} ({jax.devices()[0].device_kind}): "
                 f"{_START_TIME} to {_END_TIME}, {n_steps} steps at {resolution} deg")

    try:
        gpu_bytes = (jax.devices()[0].memory_stats() or {}).get("bytes_limit", 0) if backend == "gpu" else 0
    except Exception:                                    # noqa: BLE001 - memory stats are informative only
        gpu_bytes = 0
    if resolution < 1.0 and 0 < gpu_bytes < 70 * 2**30:
        logging.warning(f"The 0.25 deg model needs an 80 GB GPU; this one has {gpu_bytes / 2**30:.0f} GB. "
                        "It does not fit on FELIPE's L40S (46 GB): expect out of GPU memory at the first step.")

    state, initial_time = initial_state(_INI_DATA_PATH, mapper, task, resolution)
    inputs, template, forcings = forecast_inputs(state, task, n_steps)
    lats, lons = state.lat.values, state.lon.values
    model_levels = [int(level) for level in state.level.values]
    del state

    missing = [v for v in output_vars if v not in task.target_variables]
    for v in missing:
        logging.warning(f"'{v}' is not predicted by {entry['config']}; skipped.")

    predictor_config = inference_config(model_config, backend)

    @hk.transform
    def run_forward(inputs, targets_template, forcings):
        predictor = fgn.construct_predictor(predictor_config)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    apply = jax.jit(run_forward.apply)

    def predictor_fn(rng, inputs, targets_template, forcings):
        return apply(params, rng, inputs, targets_template, forcings)

    rng = jax.random.fold_in(jax.random.PRNGKey(0), _RNG_KEY)

    # Only the requested variables and levels of each step leave the device, streamed to per-variable
    # buffers on disk: memory does not grow with run length.
    level_idx = _output_level_selection(model_levels, _OUT_LEVS)
    out_levels = [model_levels[i] for i in level_idx] if level_idx is not None else model_levels
    stream_dir = os.path.join(config.get("TEMP_DIR") or os.path.join(_OUTPUT_PATH, "tmp"), "weathernext-stream")
    wanted = [v for v in output_vars if v in task.target_variables]

    with StepBuffer(stream_dir, n_steps) as buffer:
        for i, chunk in enumerate(rollout.chunked_prediction_generator(
                predictor_fn=predictor_fn, rng=rng, inputs=inputs, targets_template=template,
                forcings=forcings, num_steps_per_chunk=1)):
            step = {}
            for var in wanted:
                field = chunk[var].isel(batch=0, time=0)
                field = field.transpose(*[d for d in ("level", "lat", "lon") if d in field.dims])
                arr = _host_array(field)
                if arr.ndim == 3 and level_idx is not None:
                    arr = arr[level_idx]
                step[var] = arr
            buffer.append(step)
            del chunk, step
            if (i + 1) % 4 == 0 or i + 1 == n_steps:
                logging.info(f"Step {i + 1}/{n_steps} (+{(i + 1) * MODEL_STEP_HOURS} h)")

        for var, var_name in zip(output_vars, out_vars):
            if var not in buffer.names:
                continue
            dataset = _variable_dataset(var, buffer.read(var), lats, lons, out_levels, initial_time, delta_t)
            spec = specs[var_name]
            if spec["scale"] != 1.0:
                dataset[var] = dataset[var] * spec["scale"]
            if _OUT_FREQ == "daily":
                dataset = resample_daily(dataset, var, spec["accumulated"])
            if level_idx is None and _OUT_LEVS not in ("", "original", None) and "level" in dataset.dims:
                desired_levels = [int(plev) for plev in str(_OUT_LEVS).strip("[]").split(",")]
                dataset = dataset.interp(level=desired_levels)
            if _OUT_RES in OUT_RESOLUTIONS:
                dataset = regrid_latlon(dataset, *target_grid(_OUT_RES))

            output_file = (f"{_OUTPUT_PATH}/{var_name}/{_RNG_KEY}/"
                           f"out-{_START_TIME}-{_END_TIME}-{_RNG_KEY}-{var_name}.nc")
            write_output(output_file, dataset[var], var_name, spec, run_info, daily=_OUT_FREQ == "daily",
                         step_hours=float(MODEL_STEP_HOURS), reference_time=initial_time)
            logging.info(f"Saved {output_file}")
            del dataset


if __name__ == "__main__":
    main()
