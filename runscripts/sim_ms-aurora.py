"""
Runscript for MS-Aurora simulations: v1.0 (deterministic), v1.5 (deterministic) and
v1.5 Ensemble (stochastic). Ensemble members follow the same per-job convention as
AIFS/NeuralGCM: one Autosubmit MEMBER == one RNG_KEY == one reproducible noise seed.
"""

# Built-in/Generics
import os
import pickle
import logging
from datetime import datetime, timedelta

# Third party
import numpy as np
import torch
import xarray as xr
try:
    from aurora import AuroraV1p5
except ImportError:
    AuroraV1p5 = None

try:
    from aurora import AuroraV1p5Ensemble
except ImportError:
    AuroraV1p5Ensemble = None

from aurora import Aurora, AuroraPretrained, Batch, Metadata, rollout

try:
    from aurora.insolation import insolation
except ImportError:
    insolation = None

# Local
from AIUQst_lib.functions import parse_arguments, read_config, normalize_out_vars
from AIUQst_lib.cards import read_model_card, read_std_version
from AIUQst_lib.variables import name_mapper_for_model, output_translator
from AIUQst_lib.streaming import StepBuffer
from AIUQst_lib.regrid import OUT_RESOLUTIONS, regrid_latlon, target_grid

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

AURORA_V1_SURF_VARS = ("2t", "10u", "10v", "msl")
AURORA_V1P5_SURF_VARS = (
    "2t", "10u", "10v", "msl", "2d", "tcwv", "tcc", "100u", "100v",
    "sp", "lcc", "mcc", "hcc", "skt", "stl1", "swvl1", "ci", "scaled_sd",
)
AURORA_ATMOS_VARS = ("t", "u", "v", "q", "z")


def _extract_datetime_list(time_coord: xr.DataArray) -> list[datetime]:
    """Convert xarray time coordinate to list of Python datetime objects."""
    t_sec = time_coord.values.astype("datetime64[s]").astype(np.int64)
    return [datetime.utcfromtimestamp(int(ts)) for ts in t_sec]


def _load_static_vars(
    is_v1p5: bool,
    static_folder: str,
    ic_ds: xr.Dataset,
) -> dict[str, torch.Tensor]:
    """Load static variables from pickle in STATIC_FOLDER (or fallback to IC dataset for v1.0)."""
    pickle_name = "aurora-0.25-v1.5-static.pickle" if is_v1p5 else "aurora-0.25-static.pickle"
    pickle_path = os.path.join(static_folder, pickle_name) if static_folder else ""

    if pickle_path and os.path.exists(pickle_path):
        logging.info(f"Loading static variables from {pickle_path}")
        with open(pickle_path, "rb") as f:
            static_raw = pickle.load(f)
        return {k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in static_raw.items()}

    if is_v1p5:
        raise FileNotFoundError(
            f"Aurora 1.5 requires '{pickle_name}' in STATIC_FOLDER ('{static_folder}')."
        )

    logging.info("Static pickle not found for Aurora v1.0; reading static variables from IC dataset")
    static_vars = {}
    for s_var, ic_candidates in [("z", ["orog", "z"]), ("lsm", ["lsm"]), ("slt", ["slt"])]:
        found = None
        for cand in ic_candidates:
            if cand in ic_ds:
                da = ic_ds[cand]
                if "level" in da.dims and s_var == "z":
                    continue
                vals = da.isel(time=0).values if "time" in da.dims else da.values
                found = vals
                break
        if found is None:
            raise KeyError(f"Static variable '{s_var}' not found in IC dataset or {pickle_path}")
        static_vars[s_var] = torch.from_numpy(np.nan_to_num(found, nan=0.0).astype(np.float32))
    return static_vars


def _build_batch(
    ic_ds: xr.Dataset,
    levels: tuple[int, ...],
    is_v1p5: bool,
    static_folder: str,
) -> Batch:
    """Build aurora.Batch from standardized AIUQ initial conditions Dataset."""
    # Ensure latitude is strictly decreasing (90 -> -90) as required by Aurora
    if ic_ds.latitude.values[0] < ic_ds.latitude.values[-1]:
        ic_ds = ic_ds.sortby("latitude", ascending=False)

    # Ensure longitude is in [0, 360) and increasing
    ic_ds = ic_ds.assign_coords(longitude=(ic_ds.longitude % 360)).sortby("longitude")

    # Handle SoilLevel for sot -> stl1 if still present as 3D soil variable
    if "sot" in ic_ds and "stl1" not in ic_ds:
        if "SoilLevel" in ic_ds["sot"].dims:
            ic_ds["stl1"] = ic_ds["sot"].isel(SoilLevel=0)
        else:
            ic_ds["stl1"] = ic_ds["sot"]

    surf_var_names = AURORA_V1P5_SURF_VARS if is_v1p5 else AURORA_V1_SURF_VARS
    surf_vars = {}
    for v_name in surf_var_names:
        if v_name not in ic_ds:
            raise KeyError(f"Required surface variable '{v_name}' missing from initial conditions.")
        da = ic_ds[v_name]
        if "SoilLevel" in da.dims:
            da = da.isel(SoilLevel=0)
        data = da.isel(time=slice(0, 2)).values[None]  # (1, 2, H, W)
        surf_vars[v_name] = torch.from_numpy(np.nan_to_num(data, nan=0.0).astype(np.float32))

    lat_vals = ic_ds.latitude.values.astype(np.float32)
    lon_vals = ic_ds.longitude.values.astype(np.float32)
    input_times = _extract_datetime_list(ic_ds.time.isel(time=slice(0, 2)))

    if is_v1p5:
        if insolation is None:
            raise ImportError("aurora.insolation is required for Aurora 1.5 but could not be imported.")
        sol = np.stack(
            [insolation([t], lat_vals, lon_vals, enforce_2d=True)[0] for t in input_times],
            axis=0,
        )
        surf_vars["insolation"] = torch.from_numpy(sol[None].astype(np.float32))

    atmos_vars = {}
    for v_name in AURORA_ATMOS_VARS:
        if v_name not in ic_ds:
            raise KeyError(f"Required atmospheric variable '{v_name}' missing from initial conditions.")
        da = ic_ds[v_name].sel(level=list(levels)).isel(time=slice(0, 2))
        da = da.transpose("time", "level", "latitude", "longitude")
        data = da.values[None]  # (1, 2, C, H, W)
        atmos_vars[v_name] = torch.from_numpy(np.nan_to_num(data, nan=0.0).astype(np.float32))

    static_vars = _load_static_vars(is_v1p5=is_v1p5, static_folder=static_folder, ic_ds=ic_ds)

    return Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=Metadata(
            lat=torch.from_numpy(ic_ds.latitude.values),
            lon=torch.from_numpy(ic_ds.longitude.values.astype(np.float64)),
            time=(input_times[1],),
            atmos_levels=levels,
        ),
    )


def _load_aurora_model(checkpoint_path: str, is_v1p5: bool, is_ensemble: bool):
    """Instantiate the appropriate Aurora model class and load local checkpoint."""
    ckpt_name = os.path.basename(checkpoint_path)

    if is_ensemble:
        if AuroraV1p5Ensemble is None:
            raise ImportError(
                "Checkpoint name suggests Aurora 1.5 Ensemble but the installed 'microsoft-aurora' "
                "package does not provide 'AuroraV1p5Ensemble'. Rebuild the Singularity image "
                "installing aurora from GitHub main "
                "('pip install git+https://github.com/microsoft/aurora.git') to use the ensemble."
            )
        model = AuroraV1p5Ensemble()
    elif is_v1p5:
        if AuroraV1p5 is None:
            raise ImportError(
                "Checkpoint name suggests Aurora 1.5 but the installed 'microsoft-aurora' package "
                "does not provide 'AuroraV1p5'. Rebuild the Singularity image installing aurora from "
                "GitHub main ('pip install git+https://github.com/microsoft/aurora.git') to use v1.5."
            )
        model = AuroraV1p5()
    elif "pretrained" in ckpt_name:
        model = AuroraPretrained()
    else:
        model = Aurora(use_lora="lora" in ckpt_name or "finetuned" in ckpt_name)

    if os.path.isfile(checkpoint_path):
        logging.info(f"Loading local Aurora checkpoint from {checkpoint_path}")
        model.load_checkpoint_local(checkpoint_path, strict=False)
    else:
        logging.info(f"Local checkpoint '{checkpoint_path}' not found; loading '{ckpt_name}' from microsoft/aurora")
        model.load_checkpoint("microsoft/aurora", ckpt_name, strict=False)

    return model


def _output_level_selection(model_levels: tuple[int, ...], out_levs) -> list[int] | None:
    """Indices of the requested output levels when all of them are model levels, else None
    (all levels are then kept and interpolated after the rollout)."""
    if out_levs in (None, "", "original"):
        return None
    desired = [int(plev) for plev in str(out_levs).strip("[]").split(",") if str(plev).strip()]
    levels = list(model_levels)
    if desired and all(p in levels for p in desired):
        return [levels.index(p) for p in desired]
    return None


def _extract_outputs(pred: Batch, output_vars: list[str], level_idx: list[int] | None) -> dict:
    """Copy to host memory only the requested variables (and levels) of one rollout step.

    Keeping whole predicted Batches (all surface and atmospheric variables on 13 levels at
    0.25 deg, ~345 MB per step for Aurora 1.5) exhausts the job memory on multi-week runs.
    """
    out = {}
    for var in output_vars:
        if var in pred.surf_vars:
            out[var] = pred.surf_vars[var][:, 0].float().cpu().numpy()          # (b, lat, lon)
        elif var in pred.atmos_vars:
            arr = pred.atmos_vars[var][:, 0]                                     # (b, level, lat, lon)
            if level_idx is not None:
                arr = arr[:, level_idx]
            out[var] = arr.float().cpu().numpy()
    return out


def _variable_dataset(
    var: str,
    data: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    levels: list[int],
    initial_time: np.datetime64,
    delta_t: np.timedelta64,
) -> xr.Dataset:
    """Standardized xarray.Dataset for one output variable (data: steps x [level] x lat x lon)."""
    n_out = data.shape[0]
    valid_time = initial_time + np.arange(1, n_out + 1) * delta_t
    step_coord = np.arange(1, n_out + 1) * delta_t

    dims = ("valid_time", "latitude", "longitude") if data.ndim == 3 else ("valid_time", "level", "latitude", "longitude")
    coords = {
        "valid_time": valid_time,
        "time": initial_time,
        "step": ("valid_time", step_coord),
        "latitude": lats,
        "longitude": lons,
    }
    if data.ndim == 4:
        coords["level"] = levels
    return xr.Dataset(data_vars={var: (dims, data)}, coords=coords)


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
    _STATIC_FOLDER = config.get("STATIC_FOLDER", "")
    _INNER_STEPS = int(config.get("INNER_STEPS", 6))
    _RNG_KEY = config.get("RNG_KEY", 1)
    _OUTPUT_PATH = config.get("OUTPUT_PATH", "")
    _OUT_VARS = config.get("OUT_VARS", [])
    _OUT_FREQ = config.get("OUT_FREQ", "")
    _OUT_RES = config.get("OUT_RES", "")
    _OUT_LEVS = config.get("OUT_LEVS", "")

    # IC & model card settings
    model_card = read_model_card(_HPCROOTDIR, _MODEL_NAME)
    standard_dict = read_std_version(_HPCROOTDIR, _STD_VERSION)

    mapper = name_mapper_for_model(
        model_card["variables"],
        standard_dict["variables"],
    )
    # Keep atmospheric geopotential 'z' distinct from static 'orog' before static extraction
    if "orog" in mapper:
        mapper.pop("orog")

    out_vars = normalize_out_vars(_OUT_VARS)
    translate = output_translator(
        model_card["variables"],
        standard_dict["variables"],
    )
    output_vars = [translate.get(item, item) for item in out_vars]

    # Format time settings
    start_date = datetime.strptime(_START_TIME, "%Y-%m-%d")
    end_date = datetime.strptime(_END_TIME, "%Y-%m-%d")

    days_to_run = (end_date - start_date).days + 1
    ckpt_basename = os.path.basename(_MODEL_CHECKPOINT)
    is_ensemble = "ensemble" in ckpt_basename
    is_v1p5 = "v1.5" in ckpt_basename or "v1p5" in ckpt_basename

    # Aurora base autoregressive step is 6h; v1.5 supports fine_lead_times down to 1h
    base_step_hours = 6
    ar_steps = (days_to_run * 24) // base_step_hours
    delta_t = np.timedelta64(_INNER_STEPS, "h")

    logging.info(
        f"Running {_MODEL_NAME} "
        f"({'v1.5-ensemble' if is_ensemble else ('v1.5' if is_v1p5 else 'v1.0')}) "
        f"from {_START_TIME} to {_END_TIME} ({days_to_run} days, {ar_steps} AR steps)"
    )

    # Load initial conditions and map variable names to Aurora canonical names
    initial_conditions = xr.open_zarr(_INI_DATA_PATH, chunks=None).load()
    if "sot" in initial_conditions and "SoilLevel" in initial_conditions["sot"].dims:
        initial_conditions["stl1"] = initial_conditions["sot"].isel(SoilLevel=0)
        initial_conditions = initial_conditions.drop_vars("sot")
        if "SoilLevel" in initial_conditions.dims:
            initial_conditions = initial_conditions.drop_dims("SoilLevel")

    rename_present = {k: v for k, v in mapper.items() if k in initial_conditions}
    initial_conditions = initial_conditions.rename(rename_present)

    levels = tuple(int(lev) for lev in model_card["pressure_levels"]["values"])

    if not _STATIC_FOLDER and config.get("STATIC_DATA"):
        _STATIC_FOLDER = os.path.dirname(config["STATIC_DATA"])

    batch = _build_batch(
        ic_ds=initial_conditions,
        levels=levels,
        is_v1p5=is_v1p5,
        static_folder=_STATIC_FOLDER,
    )

    # Load model and run inference
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _load_aurora_model(_MODEL_CHECKPOINT, is_v1p5=is_v1p5, is_ensemble=is_ensemble)
    model.eval()
    model = model.to(device)

    # Aurora 1.5 Ensemble injects stochastic noise through the backbone; a reproducible
    # torch.Generator is derived from RNG_KEY/MEMBER so each ensemble member (one Autosubmit
    # job/member per RNG_KEY, same convention as AIFS/NeuralGCM) gets a distinct but
    # reproducible realization. Deterministic checkpoints (Aurora, AuroraPretrained, AuroraV1p5)
    # ignore the generator: running several MEMBERS with them yields identical output, as expected
    # for a deterministic model.
    is_stochastic = bool(getattr(getattr(model, "backbone", None), "stochastic", False))
    generator = None
    if is_stochastic:
        generator = torch.Generator(device=device).manual_seed(int(_RNG_KEY))
        logging.info(f"Stochastic Aurora model detected: seeding noise generator with RNG_KEY={_RNG_KEY}")
    elif str(_RNG_KEY) not in ("1", "", "0"):
        logging.warning(
            f"MEMBER/RNG_KEY={_RNG_KEY} was requested but checkpoint '{ckpt_basename}' is "
            "deterministic (no stochastic noise injection). All members will produce identical "
            "output; use 'aurora-0.25-v1.5-ensemble.ckpt' for a real ensemble."
        )

    rollout_kwargs = {}
    if is_v1p5 and _INNER_STEPS < base_step_hours:
        rollout_kwargs["fine_lead_times"] = [
            float(h) for h in range(_INNER_STEPS, base_step_hours + 1, _INNER_STEPS)
        ]
    if generator is not None:
        rollout_kwargs["generator"] = generator

    # Only the requested variables/levels of each step leave the GPU, and they are written to
    # per-variable buffers on disk: memory no longer grows with run length or output size.
    level_idx = _output_level_selection(levels, _OUT_LEVS)
    out_levels = [levels[i] for i in level_idx] if level_idx is not None else list(levels)
    stream_dir = os.path.join(config.get("TEMP_DIR") or os.path.join(_OUTPUT_PATH, "tmp"), "aurora-stream")
    initial_time = initial_conditions.time.isel(time=1).values

    with StepBuffer(stream_dir, ar_steps) as buffer:
        lats = lons = None
        with torch.inference_mode():
            for pred in rollout(model, batch, steps=ar_steps, **rollout_kwargs):
                if lats is None:
                    lats = pred.metadata.lat.cpu().numpy()
                    lons = pred.metadata.lon.cpu().numpy()
                outputs = _extract_outputs(pred, output_vars, level_idx)
                buffer.append({name: arr[0] for name, arr in outputs.items()})   # batch size 1
                del pred, outputs

        if not any(var in buffer.names for var in output_vars):
            raise ValueError(f"None of the requested output variables {out_vars} is produced by Aurora.")

        # Post-processing (frequency, horizontal resolution, vertical levels), one variable at a time
        for var, var_name in zip(output_vars, out_vars):
            if var not in buffer.names:
                logging.warning(f"Requested output variable '{var}' not found in Aurora predictions.")
                continue
            dataset = _variable_dataset(
                var, buffer.read(var), lats, lons, out_levels, initial_time, delta_t,
            )

            if _OUT_FREQ == "daily":
                dataset = dataset.resample(valid_time="1D").mean()

            if _OUT_RES in OUT_RESOLUTIONS:
                dataset = regrid_latlon(dataset, *target_grid(_OUT_RES))
            else:
                dataset = dataset.sortby("latitude")

            if level_idx is None and _OUT_LEVS != "original" and "level" in dataset.dims:
                desired_levels = [int(plev) for plev in _OUT_LEVS.strip("[]").split(",")]
                dataset = dataset.interp(level=desired_levels)

            # Save NetCDF per output variable
            predictions_datarray = dataset[var].rename(var_name)
            output_base_path = f"{_OUTPUT_PATH}/{var_name}/{str(_RNG_KEY)}"
            os.makedirs(output_base_path, exist_ok=True)
            output_file = f"{output_base_path}/out-{_START_TIME}-{_END_TIME}-{_RNG_KEY}-{var_name}.nc"
            predictions_datarray.to_netcdf(output_file)
            logging.info(f"Saved {output_file}")
            del dataset, predictions_datarray


if __name__ == "__main__":
    main()

