"""

"""

# Built-in/Generics
import os 
import shutil
import yaml

# Third party
import gcsfs
import xarray as xr
import zarr

# Local
from AIUQst_lib.functions import parse_arguments, read_config
from AIUQst_lib.pressure_levels import check_pressure_levels
from AIUQst_lib.cards import read_ic_card, read_std_version
from AIUQst_lib.variables import reassign_long_names_units, define_ics_mappers


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

    # IC settings
    ic_card = read_ic_card(_HPCROOTDIR, _IC)
    static_card = read_ic_card(_HPCROOTDIR, 'static')
    climatology_card = read_ic_card(_HPCROOTDIR, 'climatology')
    standard_dict = read_std_version(_HPCROOTDIR, _STD_VERSION)

    # Here the specific code to retrieve ERA5 from GCS
    gcs = gcsfs.GCSFileSystem(token="anon")
    ini_data_path_remote = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
    full_era5 = xr.open_zarr(gcs.get_mapper(ini_data_path_remote), chunks={"time":1})
    
    # Create the mappers between model requirement and IC variables
    ic_names, rename_dict, long_names_dict, units_dict, missing_vars = define_ics_mappers(
        ic_card['variables'], 
        standard_dict['variables']
        )

    # The following is a workaround for soil temperature levels - google saves them as different variable, one for each level
    # If the soil temperature variable is requested, we remove it from the normal processing and we add it later
    restore_sot = False
    if 260360 in ic_names.keys():
        del ic_names[260360]
        del rename_dict['soil_temperature']
        del long_names_dict['sot']
        del units_dict['sot']
        sol_1 = full_era5['soil_temperature_level_1'].rename('sot').sel(time=slice(_START_TIME, _END_TIME)).isel(time=slice(0, 2)).to_dataset().assign_coords(SoilLevel=1)
        sol_2 = full_era5['soil_temperature_level_2'].rename('sot').sel(time=slice(_START_TIME, _END_TIME)).isel(time=slice(0, 2)).to_dataset().assign_coords(SoilLevel=2)
        sol = xr.concat([sol_1, sol_2], dim='SoilLevel')
        restore_sot = True
    # End of workaround

    selected = (
        full_era5[list(ic_names.values())]
        .rename(rename_dict)
        .sel(time=slice(_START_TIME, _END_TIME))
        .isel(time=slice(0, 2))
        .pipe(reassign_long_names_units, long_names_dict, units_dict)
        .pipe(check_pressure_levels, ic_card, standard_dict['pressure_levels'])
        .compute()
        )
    
    if restore_sot:
        selected = xr.merge([selected, sol])
    
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