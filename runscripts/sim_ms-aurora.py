"""
Runscript for MS-Aurora (v1.0 and v1.5) simulations.
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

from aurora import Aurora, AuroraPretrained, Batch, Metadata, rollout

try:
    from aurora.insolation import insolation
except ImportError:
    insolation = None

# Local
from AIUQst_lib.functions import parse_arguments, read_config, normalize_out_vars
from AIUQst_lib.cards import read_model_card, read_std_version
from AIUQst_lib.variables import name_mapper_for_model, output_translator

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


def _load_aurora_model(checkpoint_path: str, is_v1p5: bool):
    """Instantiate the appropriate Aurora model class and load local checkpoint."""
    ckpt_name = os.path.basename(checkpoint_path)

    if is_v1p5:
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


def _preds_to_dataset(
    preds: list[Batch],
    output_vars: list[str],
    initial_time: np.datetime64,
    delta_t: np.timedelta64,
) -> xr.Dataset:
    """Convert list of predicted Aurora Batches into a standardized xarray.Dataset."""
    lats = preds[0].metadata.lat.cpu().numpy()
    lons = preds[0].metadata.lon.cpu().numpy()
    levels = list(preds[0].metadata.atmos_levels)
    n_out = len(preds)

    valid_time = initial_time + np.arange(1, n_out + 1) * delta_t
    steps = np.arange(1, n_out + 1) * delta_t

    data_vars = {}
    for var in output_vars:
        if var in preds[0].surf_vars:
            arr = np.concatenate([p.surf_vars[var][:, 0].cpu().numpy() for p in preds], axis=0)
            data_vars[var] = (("valid_time", "latitude", "longitude"), arr)
        elif var in preds[0].atmos_vars:
            arr = np.concatenate([p.atmos_vars[var][:, 0].cpu().numpy() for p in preds], axis=0)
            data_vars[var] = (("valid_time", "level", "latitude", "longitude"), arr)
        else:
            logging.warning(f"Requested output variable '{var}' not found in Aurora predictions.")

    coords = {
        "valid_time": valid_time,
        "time": initial_time,
        "step": ("valid_time", steps),
        "latitude": lats,
        "longitude": lons,
    }
    if any(len(v[0]) == 4 for v in data_vars.values()):
        coords["level"] = levels

    return xr.Dataset(data_vars=data_vars, coords=coords)


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
    is_v1p5 = "v1.5" in os.path.basename(_MODEL_CHECKPOINT) or "v1p5" in os.path.basename(_MODEL_CHECKPOINT)

    # Aurora base autoregressive step is 6h; v1.5 supports fine_lead_times down to 1h
    base_step_hours = 6
    ar_steps = (days_to_run * 24) // base_step_hours
    delta_t = np.timedelta64(_INNER_STEPS, "h")

    logging.info(
        f"Running {_MODEL_NAME} ({'v1.5' if is_v1p5 else 'v1.0'}) from {_START_TIME} to {_END_TIME} "
        f"({days_to_run} days, {ar_steps} AR steps)"
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
    model = _load_aurora_model(_MODEL_CHECKPOINT, is_v1p5=is_v1p5)
    model.eval()
    model = model.to(device)

    rollout_kwargs = {}
    if is_v1p5 and _INNER_STEPS < base_step_hours:
        rollout_kwargs["fine_lead_times"] = [
            float(h) for h in range(_INNER_STEPS, base_step_hours + 1, _INNER_STEPS)
        ]

    with torch.inference_mode():
        preds = [
            pred.to("cpu")
            for pred in rollout(model, batch, steps=ar_steps, **rollout_kwargs)
        ]

    initial_time = initial_conditions.time.isel(time=1).values
    dataset = _preds_to_dataset(
        preds=preds,
        output_vars=output_vars,
        initial_time=initial_time,
        delta_t=delta_t,
    )

    # Post-processing (frequency, horizontal resolution, vertical levels)
    if _OUT_FREQ == "daily":
        dataset = dataset.resample(valid_time="1D").mean()

    if _OUT_RES in ["0.25", "0.5", "1", "1.5", "2"]:
        step_deg = float(_OUT_RES)
        out_latitudes = np.arange(-90, 90 + step_deg / 2, step_deg)
        out_longitudes = np.arange(0, 360, step_deg)
        dataset = dataset.interp(
            latitude=out_latitudes,
            longitude=out_longitudes,
            method="linear",
        )
    else:
        dataset = dataset.sortby("latitude")

    if _OUT_LEVS != "original" and "level" in dataset.dims:
        desired_levels = [int(plev) for plev in _OUT_LEVS.strip("[]").split(",")]
        dataset = dataset.interp(level=desired_levels)

    # Save NetCDF per output variable
    for var, var_name in zip(output_vars, out_vars):
        if var not in dataset:
            continue
        predictions_datarray = dataset[var].rename(var_name)
        output_base_path = f"{_OUTPUT_PATH}/{var_name}/{str(_RNG_KEY)}"
        os.makedirs(output_base_path, exist_ok=True)
        output_file = f"{output_base_path}/out-{_START_TIME}-{_END_TIME}-{_RNG_KEY}-{var_name}.nc"
        predictions_datarray.to_netcdf(output_file)
        logging.info(f"Saved {output_file}")


if __name__ == "__main__":
    main()

