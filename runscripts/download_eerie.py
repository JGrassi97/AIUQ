"""

"""

# Built-in/Generics
import os 
import shutil
import yaml
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor

# Third party
import xarray as xr
import zarr
from ecmwfapi import ECMWFDataServer

# Local
from AIUQst_lib.functions import parse_arguments, read_config
from AIUQst_lib.pressure_levels import check_pressure_levels
from AIUQst_lib.cards import read_model_card, read_ic_card, read_std_version
from AIUQst_lib.variables import reassign_long_names_units, define_mappers



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
    _EERIE_PATH     = config.get("EERIE_PATH", "")

    # IC settings
    model_card = read_model_card(_HPCROOTDIR, _MODEL_NAME)
    ic_card = read_ic_card(_HPCROOTDIR, _IC)
    static_card = read_ic_card(_HPCROOTDIR, 'static')
    climatology_card = read_ic_card(_HPCROOTDIR, 'climatology')
    standard_dict = read_std_version(_HPCROOTDIR, _STD_VERSION)


    # Create the mappers between model requirement and IC variables
    ic_names, rename_dict, long_names_dict, units_dict, missing_vars = define_mappers(
        model_card['variables'], 
        ic_card['variables'], 
        standard_dict['variables']
        )
    
    # Select vars to take from EERIE
    vars_to_take_eerie = list(ic_names.values())

    # Remove - from strt_time and end_time
    base_date = _START_TIME.replace("-", "")

    # Build the paths 
    eerie_paths = [os.path.join(_EERIE_PATH, var, "1",f"{var}_{base_date}.nc") for var in vars_to_take_eerie]
    print("EERIE PATHS: ", eerie_paths)
    print("VARS TO TAKE FROM EERIE: ", vars_to_take_eerie)

    selected = []
    for path in eerie_paths:
        dat = xr.open_dataset(path, decode_cf=False).isel(time=slice(0,2)).drop_vars('step')
        selected.append(dat)
    
    selected = xr.merge(selected, compat='override')
    selected = xr.decode_cf(selected)
    selected = (
        selected
        .pipe(reassign_long_names_units, long_names_dict, units_dict)
        .pipe(check_pressure_levels, ic_card, standard_dict['pressure_levels'])
        .compute()
        )
    # Adjust longitudes to -0 - 360
    #selected['longitude'] = selected['longitude'] % 360
    
    # FALLBACK FOR STATIC VARIABLES
    if missing_vars is not None:

        ic_names_static, rename_dict_static, long_names_dict_static, units_dict_static, missing_vars_static = define_mappers(
            missing_vars,
            static_card['variables'],
            standard_dict['variables']
        )

        if list(ic_names_static.values()) != []:
            static_dataset = (
                (xr.open_dataset(_STATIC_DATA)[list(ic_names_static.values())])
                .rename(rename_dict_static)
                .pipe(reassign_long_names_units, long_names_dict_static, units_dict_static)
                .compute()
            )

            latitudes = static_dataset['latitude'].values
            longitudes = static_dataset['longitude'].values
            
            # Interpolate selected to static grid
            selected = selected.interp(
                latitude=latitudes,
                longitude=longitudes,
                method="linear"
            )

            selected = (
                xr.merge([selected, static_dataset], join="exact")
            )

        # FALLBACK FOR CLIMATOLOGY VARIABLES
        if missing_vars_static is not None:

            ic_names_climatology, rename_dict_climatology, long_names_dict_climatology, units_dict_climatology, missing_vars_climatology = define_mappers(
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

            latitudes = climatology_dataset['latitude'].values
            longitudes = climatology_dataset['longitude'].values
            
            # Interpolate selected to static grid
            selected = selected.interp(
                latitude=latitudes,
                longitude=longitudes,
                method="linear"
            )

            selected = (
                xr.merge([selected, climatology_dataset], join="exact")
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