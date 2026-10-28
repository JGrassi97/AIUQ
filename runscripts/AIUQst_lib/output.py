"""Shared forecast output contract (independent of model inference packages)."""

from datetime import datetime, timezone
from importlib import metadata as package_metadata
import json
from pathlib import Path
import subprocess

import numpy as np
import xarray as xr

AIUQ_VERSION = "0.2.0"
OUTPUT_SCHEMA_VERSION = "1.0"


def resolve_output_variables(requested, model_card, standard):
    """Return model names and canonical short names, in the requested order.

    Selection uses parameter IDs, so shared native names (e.g. Aurora's z)
    cannot accidentally overwrite the atmospheric/static distinction.
    """
    native, canonical = [], []
    for name in requested:
        matches = [(key, spec) for key, spec in standard["data"].items()
                   if name in (spec["short_name"], spec["long_name"])]
        if len(matches) != 1 or matches[0][0] not in model_card["variables"]:
            raise ValueError(f"Unsupported AIUQ-st output variable {name!r} for {model_card['model']}")
        key, spec = matches[0]
        model_name = model_card["variables"][key]
        # Anemoi emits atmospheric geopotential as z, although its input card
        # calls it gh. Surface z is geopotential too, not elevation in metres.
        if spec["short_name"] == "orog" and model_card["model"] in ("aifs", "ms-aurora"):
            raise ValueError("Static orography is not a supported forecast output")
        if model_card["model"] == "aifs" and key == 129:
            model_name = "z"
        if spec["short_name"] in canonical:
            raise ValueError(f"Duplicate output variable: {name}")
        native.append(model_name)
        canonical.append(spec["short_name"])
    return native, canonical


def source_revision(root):
    """Use the revision exported by SYNCHRONIZE on HPC, or the local checkout."""
    revision_file = Path(root) / "conf" / "source_revision.txt"
    if revision_file.exists():
        return revision_file.read_text().strip() or "unknown"
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            text=True, timeout=5).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"],
            stderr=subprocess.DEVNULL, text=True, timeout=5).strip()
        return revision + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def output_metadata(config):
    """NetCDF-safe, explicit provenance. No model/GRIB attributes leak through."""
    model = str(config["MODEL_NAME"])
    package = {"aifs": "anemoi-inference", "neuralgcm": "neuralgcm",
               "ms-aurora": "microsoft-aurora"}.get(model, model)
    library_version, library_revision = "unknown", "unknown"
    try:
        distribution = package_metadata.distribution(package)
        library_version = distribution.version
        direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
        library_revision = direct_url.get("vcs_info", {}).get("commit_id", "unknown")
    except package_metadata.PackageNotFoundError:
        pass
    return {
        "Conventions": "CF-1.8",
        "title": "AIUQ forecast",
        "model": model,
        "model_library": package,
        "model_library_version": library_version,
        "model_library_revision": library_revision,
        "model_checkpoint": Path(config["MODEL_CHECKPOINT"]).name,
        "model_checkpoint_path": str(config["MODEL_CHECKPOINT"]),
        "initial_conditions": str(config.get("IC_NAME", "unknown")),
        "initial_conditions_path": str(config.get("INI_DATA_PATH", "unknown")),
        "aiuq_version": AIUQ_VERSION,
        "aiuq_st_version": str(config["STD_VERSION"]),
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "source_revision": source_revision(config["HPCROOTDIR"]),
        "date_created": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "experiment_id": str(config.get("EXPERIMENT_ID", "unknown")),
        "member": str(config.get("RNG_KEY", 1)),
        "run_type": str(config.get("RUN_TYPE", "hindcast")),
        "output_frequency": str(config.get("OUT_FREQ", "original")),
        "requested_output_resolution": str(config.get("OUT_RES", "original")),
        "requested_pressure_levels": str(config.get("OUT_LEVS", "original")),
        "forcing_version": str(config.get("FORCING_VERSION", "unknown"))
        if str(config.get("RUN_TYPE", "hindcast")).lower() == "amip" else "not_applicable",
        "chunk_start": str(config.get("START_TIME", "unknown")),
        "chunk_end": str(config.get("END_TIME", "unknown")),
    }


def standardize_output(dataset, variables, standard, initial_time, metadata, *, unstructured=False):
    """Normalize a forecast and discard backend attributes/encoding.

    ``variables`` maps native names to AIUQ-st short names. Call after all
    averaging/interpolation; daily valid_time denotes the start of the day.
    """
    missing = set(variables) - set(dataset.data_vars)
    if missing:
        raise ValueError(f"Requested variables missing from predictions: {sorted(missing)}")
    ds = dataset[list(variables)].rename({k: v for k, v in variables.items() if k != v})
    ds = ds.rename({k: v for k, v in {"lat": "latitude", "lon": "longitude"}.items() if k in ds})
    if "valid_time" not in ds.dims:
        raise ValueError("Forecast output must have a valid_time dimension")
    init = np.datetime64(initial_time, "ns")
    valid = np.asarray(ds.valid_time.values, dtype="datetime64[ns]")
    if not valid.size or np.isnat(init) or np.isnat(valid).any() or np.any(np.diff(valid) <= np.timedelta64(0, "ns")):
        raise ValueError("Forecast times must be finite, unique and increasing")
    if np.any(valid < init):
        raise ValueError("Forecast valid_time precedes initialization")
    allowed = {"latitude", "longitude", "valid_time", "level", "points"}
    ds = ds.drop_vars([c for c in ds.coords if c not in allowed])
    ds = ds.assign_coords(time=init, step=("valid_time", valid - init))
    if not unstructured:
        for coord in ("latitude", "longitude"):
            if coord not in ds.coords or ds[coord].dims != (coord,):
                raise ValueError(f"Expected a one-dimensional {coord} coordinate")
        ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180)
        ds = ds.sortby("longitude").sortby("latitude")
        for coord in ("latitude", "longitude"):
            values = ds[coord].values
            if not values.size or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
                raise ValueError(f"{coord} must be finite and unique")
        if np.any(np.abs(ds.latitude.values) > 90):
            raise ValueError("latitude outside [-90, 90]")
    if "level" in ds.dims:
        ds = ds.sortby("level", ascending=False)
    order = [d for d in ("valid_time", "level", "latitude", "longitude", "points") if d in ds.dims]
    if set(ds.dims) - set(order):
        raise ValueError(f"Unexpected forecast dimensions: {set(ds.dims) - set(order)}")
    ds = ds.transpose(*order, missing_dims="ignore").copy(deep=False)
    for name in ds.variables:
        ds[name].attrs = {}
        ds[name].encoding = {}
    specs = {v["short_name"]: v for v in standard["data"].values()}
    for name in ds.data_vars:
        ds[name].attrs = {k: specs[name][k] for k in ("long_name", "units")}
        if metadata["output_frequency"] == "daily":
            ds[name].attrs["cell_methods"] = "valid_time: mean"
    coord_attrs = {
        "latitude": {"standard_name": "latitude", "units": "degrees_north", "axis": "Y"},
        "longitude": {"standard_name": "longitude", "units": "degrees_east", "axis": "X"},
        "time": {"standard_name": "forecast_reference_time", "long_name": "Forecast initialization time"},
        "valid_time": {"standard_name": "time", "long_name": "Forecast valid time", "axis": "T"},
        "step": {"standard_name": "forecast_period", "long_name": "Time since forecast initialization"},
        "level": {"standard_name": "air_pressure", "units": "hPa", "positive": "down", "axis": "Z"},
    }
    for name, attrs in coord_attrs.items():
        if name in ds.coords:
            ds[name].attrs = attrs
    ds.attrs = dict(
        metadata,
        initialization_time=np.datetime_as_string(init, unit="s") + "Z",
        time_coverage_start=np.datetime_as_string(valid[0], unit="s") + "Z",
        time_coverage_end=np.datetime_as_string(valid[-1], unit="s") + "Z",
        output_grid="unstructured" if unstructured else "regular_latitude_longitude",
    )
    if not unstructured:
        for name in ("latitude", "longitude"):
            values = ds[name].values
            ds.attrs[f"geospatial_{name}_min"] = float(values.min())
            ds.attrs[f"geospatial_{name}_max"] = float(values.max())
            if len(values) > 1 and np.allclose(np.diff(values), np.diff(values)[0]):
                ds.attrs[f"geospatial_{name}_resolution"] = float(np.diff(values)[0])
    return ds


def write_forecast(dataset, path):
    """Write a Dataset so provenance is stored as global NetCDF attributes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    try:
        dataset.to_netcdf(temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
