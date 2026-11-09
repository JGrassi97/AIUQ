"""
Author: Jacopo Grassi
Institution: Politecnico di Torino
Email: jacopo.grassi@polito.it

Created: 2025-01-12
Last modified: 2025-02-13

Description:

"""

# Built-in/Generics
import os 
from typing import Dict, List, Tuple, Set
from concurrent.futures import ThreadPoolExecutor

# Third party
import xarray as xr
import numpy as np
import cfgrib
from ecmwfapi import ECMWFDataServer

# Local
from AIUQst_lib.functions import parse_arguments, read_config
from AIUQst_lib.pressure_levels import check_pressure_levels
from AIUQst_lib.cards import read_ic_card, read_std_version
from AIUQst_lib.variables import reassign_long_names_units, define_ics_mappers

# MARS has problems if 2m variables (167, 168) are requested together with other sfc variables.
# Here we define the codes to be requested separately.
_SFC_SEPARATE_2M_CODES: Set[int] = {167, 168}  # 2t, 2d

def build_base_request(start_date: str, grid: str, expver: str = "0001") -> dict:
    return {
        "activity": "cmip6",
        "class": "ed",
        "dataset": "research",
        "date": f"{start_date}",
        "experiment": "hist",
        "expver": str(expver).zfill(4),
        "generation": "1",
        "grid": grid,
        "model": "ifs",
        "realization": "1",
        "resolution": "high",
        "stream": "clte",
        "time": "00:00:00/06:00:00",
        "type": "fc",
    }

def _dedupe_preserve_order(vals: List[int]) -> List[int]:
    seen = set()
    out: List[int] = []
    for v in vals:
        v = int(v)
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out

def _params_to_str(params: List[int]) -> str:
    return "/".join(str(int(p)) for p in params)
    
def build_request(base: dict, var_type: str, params: List[int]) -> dict:

    req = dict(base)
    req["levtype"] = var_type
    req["param"] = _params_to_str(params)

    req.pop("levelist", None)

    if var_type == "pl":
        req["levelist"] = "1/5/10/20/30/50/70/100/150/200/250/300/400/500/600/700/850/925/1000"

    if var_type == "sol":
        req["levelist"] = "1/2"

    return req

def _split_sfc_separate_2m(vars_to_take_mars: Dict[str, List[int]]) -> Tuple[Dict[str, List[int]], List[int]]:
    out = {k: list(v) for k, v in vars_to_take_mars.items()}
    sfc = [int(x) for x in out.get("sfc", [])]

    params_2m = [p for p in sfc if p in _SFC_SEPARATE_2M_CODES]
    if params_2m:
        out["sfc"] = [p for p in sfc if p not in _SFC_SEPARATE_2M_CODES]

    return out, _dedupe_preserve_order(params_2m)

def retrieve_ecmwf(request: dict, target_path: str) -> None:
    request = dict(request)
    request["target"] = target_path
    server = ECMWFDataServer()
    server.retrieve(request)

def download_datasets(
    start_date: str,
    temp_dir: str,
    vars_to_take_mars: Dict[str, List[int]],
    grid: str,
    expver: str = "0001",
    area: str | None = None,
) -> Dict[str, str]:
    os.makedirs(temp_dir, exist_ok=True)

    base = build_base_request(start_date, grid, expver)
    if area:
        base["area"] = area
    outputs: Dict[str, str] = {}

    vars_main, params_2m = _split_sfc_separate_2m(vars_to_take_mars)

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = []

        for var_type, params in vars_main.items():
            if not params:
                continue

            out_path = os.path.join(temp_dir, f"output_{var_type}.grib")
            req = build_request(base, var_type, _dedupe_preserve_order(params))
            futures.append((var_type, out_path, ex.submit(retrieve_ecmwf, req, out_path)))

        if params_2m:
            out_path_2m = os.path.join(temp_dir, "output_sfc_2m.grib")
            req_2m = build_request(base, "sfc", params_2m)
            futures.append(("sfc_2m", out_path_2m, ex.submit(retrieve_ecmwf, req_2m, out_path_2m)))

        for key, out_path, fut in futures:
            fut.result()
            outputs[key] = out_path

    return outputs




def open_and_prepare_datasets(grib_paths: Dict[str, str]) -> xr.Dataset:
    """Read all GRIB hypercubes, including the distinct 2 m/10 m/soil types."""
    datasets = []
    for path in grib_paths.values():
        for ds in cfgrib.open_datasets(path, backend_kwargs={"indexpath": ""}):
            # Only retain spatial, pressure and soil coordinates before merging.
            ds = ds.drop_vars([
                c for c in ds.coords
                if c not in {"time", "latitude", "longitude", "isobaricInhPa", "soilLayer"}
            ])
            datasets.append(ds)
    return xr.merge(datasets, join="exact", compat="no_conflicts")


def mars_variables(ic_card: dict, standard_variables: dict) -> dict:
    """Map AIUQ variable IDs to archive IDs (soil moisture uses GRIB2 260199)."""
    result = {}
    for code, spec in ic_card["variables"].items():
        if code in standard_variables:
            params = result.setdefault(spec["levtype"], [])
            param = spec.get("mars_param", code)
            if param not in params:
                params.append(param)
    return result


def normalize_soil(dataset: xr.Dataset, ic_card: dict) -> xr.Dataset:
    """Expose the first two soil moisture layers under the standard's separate IDs."""
    for spec in ic_card["variables"].values():
        if "soil_level" in spec:
            dataset[spec["name"]] = dataset["vsw"].sel(soilLayer=spec["soil_level"], drop=True)
    return dataset


def align_support_grid(dataset: xr.Dataset, selected: xr.Dataset) -> xr.Dataset:
    """Align coordinates by value; never attach a new grid to unchanged data."""
    dataset = dataset.assign_coords(longitude=dataset.longitude % 360).sortby("longitude")
    # Periodic extension avoids NaNs around the Greenwich meridian.
    if dataset.longitude.size > 1 and float(dataset.longitude[-1] - dataset.longitude[0]) > 350:
        left = dataset.isel(longitude=[-1]).assign_coords(longitude=[float(dataset.longitude[-1]) - 360])
        right = dataset.isel(longitude=[0]).assign_coords(longitude=[float(dataset.longitude[0]) + 360])
        dataset = xr.concat([left, dataset, right], dim="longitude")
    return dataset.interp(latitude=selected.latitude, longitude=selected.longitude)


def add_fallback(selected: xr.Dataset, missing: dict | None, card: dict,
                 path: str, climatology: bool = False) -> tuple:
    if not missing:
        return selected, None
    names, rename, long_names, units, remaining = define_ics_mappers(card["variables"], missing)
    if not names:
        return selected, remaining
    with xr.open_dataset(path) as source:
        absent = set(names.values()) - set(source.data_vars)
        if absent:
            raise ValueError(f"Missing fallback variables {sorted(absent)} in {path}. "
                             "Extend the ERA5 climatology with u100/v100 for Aurora 1.5.")
        fallback = source[list(names.values())].rename(rename).load()
    if climatology:
        # Select actual UTC hours, not the first two entries (which may be 00/01).
        if "month" in fallback.dims:
            fallback = fallback.sel(month=selected.time.dt.month)
        if "dayofyear" in fallback.dims:
            fallback = fallback.sel(dayofyear=selected.time.dt.dayofyear)
        if "hour" not in fallback.dims:
            raise ValueError(f"ERA5 climatology {path} must have an 'hour' dimension (UTC).")
        fallback = fallback.sel(hour=selected.time.dt.hour)
        fallback = fallback.drop_vars([c for c in ["hour", "month", "dayofyear"] if c in fallback.coords])
    fallback = align_support_grid(fallback, selected)
    fallback = reassign_long_names_units(fallback, long_names, units)
    for var in fallback.data_vars:
        fallback[var].attrs["source"] = "ERA5 climatology" if climatology else "static support data"
        if climatology:
            fallback[var].attrs["fallback_reason"] = "Not archived in EERIE expver 0001"
        if var in {"u100", "v100"} and not bool(np.isfinite(fallback[var]).all()):
            raise ValueError(f"Non-finite {var} in ERA5 climatology after grid alignment: {path}")
    return xr.merge([selected, fallback], join="exact", compat="no_conflicts"), remaining


def prepare_initial_conditions(dataset: xr.Dataset, config: dict, ic_card: dict,
                               standard_dict: dict) -> xr.Dataset:
    standard_variables = standard_dict["variables"]["data"]
    names, rename, long_names, units, missing = define_ics_mappers(ic_card["variables"], standard_variables)
    dataset = normalize_soil(dataset, ic_card)
    times = np.datetime64(config["START_TIME"], "ns") + np.array([0, 6]) * np.timedelta64(1, "h")
    selected = dataset[list(names.values())].rename(rename).sel(time=times)
    selected = reassign_long_names_units(selected, long_names, units)
    selected = check_pressure_levels(selected, ic_card, standard_dict["pressure_levels"])
    selected = selected.assign_coords(longitude=selected.longitude % 360).sortby("longitude")
    for card_name, key, climatology in [("static", "STATIC_DATA", False),
                                        ("climatology", "CLIMATOLOGY_DATA", True)]:
        selected, missing = add_fallback(
            selected, missing, read_ic_card(config["HPCROOTDIR"], card_name),
            config[key], climatology,
        )
    if missing:
        raise ValueError(f"No EERIE/static/climatology source for variable IDs: {sorted(missing)}")
    if "soilLayer" in selected.dims:
        selected = selected.rename({"soilLayer": "SoilLevel"})
    # Land/ocean masks in SST, sea ice and soil variables are expected in EERIE.
    for name, da in selected.data_vars.items():
        if name in {"sst", "ci", "swvl1", "swvl2", "sot"}:
            continue
        if not bool(np.isfinite(da).all().compute()):
            raise ValueError(f"Non-finite values in EERIE initial conditions: {name}")
    selected.attrs.update(source="EERIE", eerie_expver=str(ic_card.get("expver", "0001")),
                          fallback_source="ERA5 climatology")
    return selected



def main() -> None:
    args = parse_arguments()
    config = read_config(args.config)
    output = config["INI_DATA_PATH"]
    ic_card = read_ic_card(config["HPCROOTDIR"], config["IC_NAME"])
    standard = read_std_version(config["HPCROOTDIR"], config["STD_VERSION"])
    if os.path.isdir(output) and os.listdir(output):
        with xr.open_zarr(output) as existing:
            expected = {v["short_name"] for v in standard["variables"]["data"].values()}
            if (existing.attrs.get("eerie_expver") != ic_card["expver"]
                    or not expected.issubset(existing.data_vars)):
                raise ValueError(f"Existing ICs at {output} do not match the EERIE card; "
                                 "use a new INI_DATA_PATH or explicitly remove the old ICs.")
        print(f"Output already exists at {output}, skipping download.")
        return
    grib_paths = download_datasets(
        start_date=config["START_TIME"],
        temp_dir=config.get("ICS_TEMP_DIR") or config["TEMP_DIR"],
        vars_to_take_mars=mars_variables(ic_card, standard["variables"]["data"]),
        grid=".25/.25", expver=ic_card["expver"],
    )
    dataset = open_and_prepare_datasets(grib_paths)
    selected = prepare_initial_conditions(dataset, config, ic_card, standard)
    # A failed write must not appear to be a complete set of initial conditions.
    temporary = output + ".partial"
    selected.chunk({"time": 1}).to_zarr(temporary, mode="w", zarr_format=2)
    os.replace(temporary, output)


if __name__ == "__main__":
    main()
