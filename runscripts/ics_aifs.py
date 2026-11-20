"""
Author: Jacopo Grassi
Institution: Politecnico di Torino
Email: jacopo.grassi@polito.it

Created: 2025-01-12
Last modified: 2026-11-20

Description:
    SAVE_ICS job of AIFS: turns the AIUQ-st initial conditions (zarr, 0.25 deg regular grid)
    into the input state of an anemoi checkpoint (N320 reduced Gaussian grid, one array per
    field and level), saved as a compressed npz read by sim_aifs.py.

    The fields are not hard-coded per checkpoint: the checkpoint metadata lists the input
    variables it needs (for example 't_500', '2t', 'stl1', 'cos_mwd', 'z') and this script
    takes them from the standard variables through the model card. This covers the 1.x
    checkpoints (13 levels, no waves) and AIFS ENS 2.0 (10 hPa, snow depth, wave fields,
    soil moisture) with the same code.
"""

# Built-in/Generics
import os
import re
import logging
from datetime import datetime, timezone
import json

# Third party
import numpy as np
import xarray as xr

# Local
from AIUQst_lib.functions import parse_arguments, read_config
from AIUQst_lib.cards import read_model_card, read_std_version
from AIUQst_lib.support_data import static_fields_flipped
from AIUQst_lib.waves import direction_to_cos_sin

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

NPTS_N320 = 542080                       # N320 point count
LEVEL_RE = re.compile(r"^(?P<base>.+)_(?P<lev>\d+)$")

# Model input names derived from a standard variable rather than mapped one-to-one by the card.
DERIVED_INPUTS = {
    "stl1": ("sot", "soil_level", 0),          # soil temperature, first layer
    "stl2": ("sot", "soil_level", 1),          # soil temperature, second layer
    "cos_mwd": ("mwd", "cos", None),           # mean wave direction as cos/sin components
    "sin_mwd": ("mwd", "sin", None),
}


def split_field_name(name: str):
    """'t_500' -> ('t', 500); '2t' -> ('2t', None)."""
    match = LEVEL_RE.match(name)
    if not match:
        return name, None
    return match.group("base"), int(match.group("lev"))


def plan_input_fields(required, model_card: dict, standard_variables: dict) -> dict:
    """Map the input fields of a checkpoint to the standard variables.

    ``required``: input names of the checkpoint (model names, levels as suffix).
    Returns a dict with
      ``fields``:  {input name: (standard short name, level or None, derivation or None)}
      ``levels``:  sorted pressure levels needed by any field
      ``missing``: input names that no card entry or derivation provides.
    Pressure-level and surface fields may share a model name ('z' is geopotential on levels
    and orography at the surface): the standard variable is chosen by its type.
    """
    by_name = {}                                 # (model name, on levels?) -> standard short name
    for code, model_name in model_card["variables"].items():
        spec = standard_variables.get(code)
        if spec is None:
            continue
        by_name[(model_name, spec.get("type") == "pl")] = spec["short_name"]

    fields, levels, missing = {}, set(), []
    for name in sorted(required):
        base, level = split_field_name(name)
        if level is None and name in DERIVED_INPUTS:
            source, how, index = DERIVED_INPUTS[name]
            fields[name] = (source, None, (how, index))
            continue
        standard_name = by_name.get((base, level is not None))
        if standard_name is None:
            missing.append(name)
            continue
        fields[name] = (standard_name, level, None)
        if level is not None:
            levels.add(level)
    return {"fields": fields, "levels": sorted(levels, reverse=True), "missing": missing}


def inputs_from_categories(categories: dict) -> list:
    """Input fields of a checkpoint from ``Checkpoint.variable_categories()``.

    Prognostic variables and the forcings read from the input (constant fields such as
    z, lsm, sdor, slor, wmb); not the forcings the runner computes (insolation, cos/sin of
    latitude, time of day ...) nor the diagnostic outputs.
    """
    inputs = []
    for name, cats in categories.items():
        cats = set(cats)
        if "computed" in cats or "diagnostic" in cats:
            continue
        if "prognostic" in cats or "forcing" in cats:
            inputs.append(name)
    return sorted(inputs)


def checkpoint_inputs(checkpoint) -> list:
    """Input fields of an anemoi checkpoint, across anemoi-inference versions."""
    if hasattr(checkpoint, "variable_categories"):             # 0.6 and 0.8
        return inputs_from_categories(checkpoint.variable_categories())
    return sorted(checkpoint.variables_from_input(include_forcings=True))   # older releases


def checkpoint_options(model_card: dict, checkpoint_path: str) -> dict:
    options = dict((model_card.get("checkpoints") or {}).get("default") or {})
    options.update((model_card.get("checkpoints") or {}).get(os.path.basename(checkpoint_path)) or {})
    return options


def regrid_to_n320(values: np.ndarray) -> np.ndarray:
    """(latitude, longitude) 0.25 deg regular grid -> N320 points, linear interpolation."""
    import earthkit.regrid as ekr                    # anemoi image only; tests import the planning
    return ekr.interpolate(values, {"grid": (0.25, 0.25)}, {"grid": "N320"},
                           method="linear").astype(np.float32, copy=False)


def main():

    # Read config
    args = parse_arguments()
    config = read_config(args.config)

    _INI_DATA_PATH      = config.get('INI_DATA_PATH', "")
    _START_TIME         = config.get("START_TIME", "")
    _HPCROOTDIR         = config.get("HPCROOTDIR", "")
    _MODEL_NAME         = config.get("MODEL_NAME", "")
    _ICS_TEMP_DIR       = config.get("ICS_TEMP_DIR", "")
    _STD_VERSION        = config.get("STD_VERSION", "")
    _MODEL_CHECKPOINT   = config.get("MODEL_CHECKPOINT", "")

    # Model card, standard and the input fields the checkpoint asks for
    model_card = read_model_card(_HPCROOTDIR, _MODEL_NAME)
    standard_variables = read_std_version(_HPCROOTDIR, _STD_VERSION)["variables"]["data"]
    options = checkpoint_options(model_card, _MODEL_CHECKPOINT)

    from anemoi.inference.checkpoint import Checkpoint   # anemoi image only
    required = checkpoint_inputs(Checkpoint(_MODEL_CHECKPOINT))
    logging.info(f"Checkpoint {os.path.basename(_MODEL_CHECKPOINT)} needs {len(required)} input fields")

    plan = plan_input_fields(required, model_card, standard_variables)
    if plan["missing"]:
        raise ValueError(f"The model card {_MODEL_NAME} has no standard variable for the checkpoint "
                         f"inputs {plan['missing']} (conf/cards/models/{_MODEL_NAME}_card.yml)")
    levels = plan["levels"]
    logging.info(f"Pressure levels: {levels} hPa")

    start_date = datetime.strptime(_START_TIME, "%Y-%m-%d")

    # Load ICs
    data_original = xr.open_zarr(_INI_DATA_PATH, chunks=None)

    # ERA5 ICs store the static fields upside down (coordinates overwritten when merging
    # the static file); EERIE and ECMWF-analysis ICs align them by value. Flip only when
    # the land-sea mask says the fields do not match their latitudes.
    if static_fields_flipped(data_original):
        logging.info("Static fields stored upside down: flipping orog, slor, sdor, lsm")
        for name in ("orog", "slor", "sdor", "lsm"):
            data_original[name] = (("latitude", "longitude"),
                                   np.flip(data_original[name].values, axis=0))

    needed = {source for source, _, _ in plan["fields"].values()} | {"lsm"}
    absent = sorted(needed - set(data_original.data_vars))
    if absent:
        raise ValueError(f"Initial conditions at {_INI_DATA_PATH} lack the standard variables {absent} "
                         f"needed by {os.path.basename(_MODEL_CHECKPOINT)} (optional variables of the "
                         "standard, such as the wave fields, may have no source for this IC)")

    # Interpolate on the pressure levels the checkpoint uses (the standard has them all)
    data_original = data_original.interp(level=levels)
    times = data_original["time"].values
    n_times = times.size

    # Regrid each needed (variable, level) to N320: fields[name] has shape (time, points)
    def to_n320(da: xr.DataArray) -> np.ndarray:
        if "time" not in da.dims:                       # constants
            da = da.expand_dims(time=times)
        da = da.transpose("time", "latitude", "longitude")
        out = np.empty((n_times, NPTS_N320), dtype=np.float32)
        for ti in range(n_times):
            out[ti, :] = regrid_to_n320(da.values[ti, :, :])
        return out

    cache = {}

    def surface(standard_name: str) -> np.ndarray:
        if standard_name not in cache:
            cache[standard_name] = to_n320(data_original[standard_name])
        return cache[standard_name]

    fields = {}
    for name, (standard_name, level, derivation) in plan["fields"].items():
        if level is not None:
            fields[name] = to_n320(data_original[standard_name].sel(level=level))
        elif derivation is None:
            fields[name] = surface(standard_name)
        elif derivation[0] == "soil_level":
            fields[name] = to_n320(data_original[standard_name].isel(SoilLevel=derivation[1]))
        elif derivation[0] in ("cos", "sin"):
            cos_component, sin_component = direction_to_cos_sin(surface(standard_name))
            fields[name] = (cos_component if derivation[0] == "cos" else sin_component).astype(np.float32)
        else:
            raise ValueError(f"Unknown derivation {derivation} for {name}")

    # Fields the checkpoint expects undefined over sea (AIFS ENS 2.0: sd, swvl1, swvl2)
    mask_over_sea = [v for v in options.get("mask_over_sea") or [] if v in fields]
    if mask_over_sea:
        sea = surface("lsm") < 0.5
        for name in mask_over_sea:
            fields[name] = np.where(sea, np.nan, fields[name]).astype(np.float32)
        logging.info(f"Masked over sea: {mask_over_sea}")

    for name, values in fields.items():
        if not np.isfinite(values).any():
            raise ValueError(f"Input field {name} is entirely undefined")

    t64 = data_original["time"].isel(time=-1).values
    DATE = datetime.utcfromtimestamp(t64.astype("datetime64[s]").astype(int))
    timestamp = int(DATE.replace(tzinfo=timezone.utc).timestamp())

    # use YYYYMMDD in filename to avoid spaces/colons
    ics_basename = f"ics_aifs_{start_date.strftime('%Y%m%d')}"
    ics_file = os.path.join(_ICS_TEMP_DIR, f"{ics_basename}.npz")
    ics_names_file = os.path.join(_ICS_TEMP_DIR, f"{ics_basename}.names.json")

    # Ensure dir exists
    os.makedirs(_ICS_TEMP_DIR, exist_ok=True)
    print(f"Writing ICs to {ics_file} ...")

    # Prepare dict for np.savez: prefix field arrays to avoid name collisions
    np_dict = {f"f__{k}": np.asarray(arr, dtype=np.float32) for k, arr in fields.items()}
    field_names = list(fields)

    # Add timestamp as scalar array
    np_dict["timestamp"] = np.array([timestamp], dtype=np.int64)

    # Save compressed .npz
    np.savez_compressed(ics_file, **np_dict)

    # Save the ordered list of field names in json for human inspection
    with open(ics_names_file, "w") as nf:
        json.dump({"fields": field_names, "timestamp": timestamp, "levels": levels,
                   "checkpoint": os.path.basename(_MODEL_CHECKPOINT),
                   "mask_over_sea": mask_over_sea}, nf)

    print(f"Wrote: {ics_file} ({len(field_names)} fields)")
    print(f"Wrote names meta: {ics_names_file}")


if __name__ == "__main__":
    main()
