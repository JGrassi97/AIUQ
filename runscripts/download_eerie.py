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
import shutil
from typing import Dict, List, Tuple, Set
from concurrent.futures import ThreadPoolExecutor

# Third party
import xarray as xr
import zarr
from ecmwfapi import ECMWFDataServer

# Local
from AIUQst_lib.functions import parse_arguments, read_config
from AIUQst_lib.pressure_levels import check_pressure_levels
from AIUQst_lib.cards import read_model_card, read_ic_card, read_std_version
from AIUQst_lib.variables import reassign_long_names_units, define_ics_mappers

# MARS has problems if 2m variables (167, 168) are requested together with other sfc variables.
# Here we define the codes to be requested separately.
_SFC_SEPARATE_2M_CODES: Set[int] = {167, 168}  # 2t, 2d

def build_base_request(start_date: str, grid: str) -> dict:
    return {
        "activity": "cmip6",
        "class": "ed",
        "dataset": "research",
        "date": f"{start_date}",
        "experiment": "hist",
        "expver": "0002",
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
) -> Dict[str, str]:
    os.makedirs(temp_dir, exist_ok=True)

    base = build_base_request(start_date, grid)
    outputs: Dict[str, str] = {}

    vars_main, params_2m = _split_sfc_separate_2m(vars_to_take_mars)

    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = []

        for var_type, params in vars_main.items():
            if not params:
                continue

            out_path = os.path.join(temp_dir, f"output_{var_type}.grib")
            req = build_request(base, var_type, params)
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

    datasets = []

    sfc_ds = None
    pl_ds = None

    if "sfc" in grib_paths:
        sfc_ds = xr.open_dataset(grib_paths["sfc"], engine="cfgrib")
        datasets.append(sfc_ds)

    if "sfc_2m" in grib_paths:
        sfc_2m_ds = xr.open_dataset(grib_paths["sfc_2m"], engine="cfgrib")
        datasets.append(sfc_2m_ds)

        if sfc_ds is None:
            sfc_ds = sfc_2m_ds

    if "pl" in grib_paths:
        pl_ds = xr.open_dataset(grib_paths["pl"], engine="cfgrib")
        datasets.append(pl_ds)

    for k, path in grib_paths.items():
        if k in {"sfc", "sfc_2m", "pl"}:
            continue
        datasets.append(xr.open_dataset(path, engine="cfgrib"))

    dataset = xr.merge(datasets, compat='override')

    for var in ["tclw", "tciw", "r", "entireAtmosphere", "step", "surface", "valid_time"]:
        if var in dataset:
            dataset = dataset.drop_vars(var)

    return dataset



def main() -> None:

    # Read config
    args = parse_arguments()
    config = read_config(args.config)

    _INI_DATA_PATH  = config.get('INI_DATA_PATH', "")
    _START_TIME     = config.get("START_TIME", "")
    _END_TIME       = config.get("END_TIME", "")
    _STATIC_DATA    = config.get("STATIC_DATA", "")
    _CLIMATOLOGY_DATA = config.get("CLIMATOLOGY_DATA", "")
    _HPCROOTDIR     = config.get("HPCROOTDIR", "")
    _MODEL_NAME     = config.get("MODEL_NAME", "")
    _IC             = config.get("IC_NAME", "")
    _STD_VERSION    = config.get("STD_VERSION", "")
    _TEMP_DIR       = config.get("TEMP_DIR", "")

    # IC settings
    model_card = read_model_card(_HPCROOTDIR, _MODEL_NAME)
    ic_card = read_ic_card(_HPCROOTDIR, _IC)
    static_card = read_ic_card(_HPCROOTDIR, 'static')
    climatology_card = read_ic_card(_HPCROOTDIR, 'climatology')
    standard_dict = read_std_version(_HPCROOTDIR, _STD_VERSION)


    # Create the mappers between model requirement and IC variables
    ic_names, rename_dict, long_names_dict, units_dict, missing_vars = define_ics_mappers(
        ic_card['variables'], 
        standard_dict['variables']
        )
    
    vars_to_take_mars = {}

    for k in list(ic_names.keys()):
        lev = ic_card['variables'][k]['levtype']
        vars_to_take_mars.setdefault(lev, []).append(k)

    vars_to_take_mars

    # Here the specific code to retrieve EERIE from GCS
    grib_paths = download_datasets(
        start_date=_START_TIME,
        temp_dir=_TEMP_DIR,
        vars_to_take_mars=vars_to_take_mars,
        grid=".25/.25",
    )
    
    dataset = open_and_prepare_datasets(grib_paths)

    # Standard pipeline to make the data consistent with Auto-UQ requirements
    selected = (
        dataset[list(ic_names.values())]
        .rename(rename_dict)
        .sel(time=slice(_START_TIME, _END_TIME))
        .isel(time=slice(0, 2))
        .pipe(reassign_long_names_units, long_names_dict, units_dict)
        .pipe(check_pressure_levels, ic_card, standard_dict['pressure_levels'])
        .compute()
        )
    
    # Adjust longitudes to -0 - 360
    selected['longitude'] = selected['longitude'] % 360
    selected = selected.sortby('longitude')
    
    # FALLBACK FOR STATIC VARIABLES
    if missing_vars is not None:

        ic_names_static, rename_dict_static, long_names_dict_static, units_dict_static, missing_vars_static = define_ics_mappers(
            missing_vars,
            static_card['variables'],
            standard_dict['variables']
        )

        static_dataset = (
            (xr.open_dataset(_STATIC_DATA)[list(ic_names_static.values())])
            .rename(rename_dict_static)
            .pipe(reassign_long_names_units, long_names_dict_static, units_dict_static)
            .compute()
        )

        static_dataset['longitude'] = selected['longitude']
        static_dataset['latitude'] = selected['latitude']

        selected = (
            xr.merge([selected, static_dataset], join="exact")
        )

        # FALLBACK FOR CLIMATOLOGY VARIABLES
        if missing_vars_static is not None:

            ic_names_climatology, rename_dict_climatology, long_names_dict_climatology, units_dict_climatology, missing_vars_climatology = define_ics_mappers(
                missing_vars_static,
                climatology_card['variables'],
                standard_dict['variables']
            )

            climatology_dataset = (
                xr.open_dataset(_CLIMATOLOGY_DATA)[list(ic_names_climatology.values())]
                .rename(rename_dict_climatology)
                .isel(hour=slice(0, 2))
                .rename({'hour': 'time'})
                .pipe(reassign_long_names_units, long_names_dict_climatology, units_dict_climatology)
                .compute()
            )

            climatology_dataset['time'] = selected['time']

            selected = (
                xr.merge([selected, static_dataset], join="exact")
            ) 
            
    # Rename soilLayer in SoilLevel
    selected = selected.rename({'soilLayer': 'SoilLevel'}) 
        
    # Final part - Savimg in zarr
    final = selected.chunk({"time": 1})        # Chunking by time for efficient access

    shutil.rmtree(                          # Remove existing data if any - avoid conflicts
        _INI_DATA_PATH,
        ignore_errors=True)
    
    final.to_zarr(                          # Save to zarr format - using version 2
        f"{_INI_DATA_PATH}",                # Zarr version 3 has some issues with BytesBytesCodec
        mode="w",                           # See https://github.com/pydata/xarray/issues/10032 as reference    
        zarr_format=2)
    
if __name__ == "__main__":
    main()