"""

"""

# Built-in/Generics
import os 
import shutil

# Third party
import gcsfs
import xarray as xr
import zarr

# Local
from AIUQst_lib.functions import parse_arguments, read_config
from AIUQst_lib.cards import read_ic_card
from AIUQst_lib.variables import reassign_long_names_units


def main() -> None:

    # Read config
    args = parse_arguments()
    config = read_config(args.config)

    _RUN_TYPE = config.get("RUN_TYPE", "hindcast").lower()
    _START_TIME = config.get("START_TIME", "")
    _END_TIME = config.get("END_TIME", "")
    _HPCROOTDIR = config.get("HPCROOTDIR", "")
    _AMIP_FORCING_PATH = config.get("AMIP_FORCING_PATH", "")

    # Only proceed if AMIP mode
    if _RUN_TYPE != "amip":
        print(f"RUN_TYPE={_RUN_TYPE}, skipping AMIP forcing download")
        return

    print(f"RUN_TYPE=amip detected. Downloading forcing data from ERA5...")

    # IC settings (ERA5 card)
    era5_card = read_ic_card(_HPCROOTDIR, "era5")

    # Access ERA5 from GCS
    gcs = gcsfs.GCSFileSystem(token="anon")
    ini_data_path_remote = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
    full_era5 = xr.open_zarr(gcs.get_mapper(ini_data_path_remote), chunks={"time": 1})

    # Extract only AMIP forcing variables: sea_ice_cover (31) and sea_surface_temperature (34)
    amip_var_codes = {31, 34}
    amip_vars = {code: var_info["name"] for code, var_info in era5_card["variables"].items() if code in amip_var_codes}
    
    print(f"AMIP forcing variables: {amip_vars}")

    # Build rename dictionary for extracted variables
    rename_dict = {}
    for var_name in amip_vars.values():
        rename_dict[var_name] = var_name.replace("_", "")  # Conservative rename (optional, can skip)

    # Select time period and extract forcing variables
    forcing = (
        full_era5[list(amip_vars.values())]
        .sel(time=slice(_START_TIME, _END_TIME))
        .compute()
    )

    # Adjust longitudes to [0, 360]
    forcing['longitude'] = forcing['longitude'] % 360
    forcing = forcing.sortby('longitude')

    # Chunk by time for efficiency
    forcing = forcing.chunk({"time": 1})

    print(f"Forcing dataset shape: {forcing.dims}")

    # Save to zarr
    if not _AMIP_FORCING_PATH:
        _AMIP_FORCING_PATH = os.path.join(_HPCROOTDIR, "forcing", "amip")
    
    os.makedirs(_AMIP_FORCING_PATH, exist_ok=True)

    shutil.rmtree(_AMIP_FORCING_PATH, ignore_errors=True)

    forcing.to_zarr(
        _AMIP_FORCING_PATH,
        mode="w",
        zarr_format=2
    )

    print(f"AMIP forcing saved to {_AMIP_FORCING_PATH}")


if __name__ == "__main__":
    main()
