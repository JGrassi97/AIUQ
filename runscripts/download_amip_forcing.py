"""

"""

# Built-in/Generics
import os
import shutil
from datetime import datetime, timedelta

# Third party
import gcsfs
import numpy as np
import xarray as xr

# Local
from AIUQst_lib.functions import parse_arguments, read_config
from AIUQst_lib.cards import read_ic_card


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

    # Save incrementally to zarr (one day at a time) to keep RAM usage bounded
    if not _AMIP_FORCING_PATH:
        _AMIP_FORCING_PATH = os.path.join(_HPCROOTDIR, "forcing", "amip")

    os.makedirs(_AMIP_FORCING_PATH, exist_ok=True)
    shutil.rmtree(_AMIP_FORCING_PATH, ignore_errors=True)

    start_dt = datetime.strptime(_START_TIME, "%Y-%m-%d")
    end_dt = datetime.strptime(_END_TIME, "%Y-%m-%d")
    current_dt = start_dt
    wrote_any_day = False

    while current_dt <= end_dt:
        next_dt = current_dt + timedelta(days=1)
        day_start = current_dt.strftime("%Y-%m-%d")
        next_day = next_dt.strftime("%Y-%m-%d")

        print(f"Processing day {day_start}")

        # Use [day_start, next_day) to avoid overlap between consecutive days
        day_ds = full_era5[list(amip_vars.values())].sel(time=slice(day_start, next_day))
        day_ds = day_ds.where(day_ds.time < np.datetime64(next_day), drop=True)

        if day_ds.sizes.get("time", 0) == 0:
            print(f"No data for {day_start}, skipping")
            current_dt = next_dt
            continue

        # Materialize only one day at a time
        day_ds = day_ds.compute()

        # Adjust longitudes to [0, 360]
        day_ds["longitude"] = day_ds["longitude"] % 360
        day_ds = day_ds.sortby("longitude")

        # Chunk by time for efficient downstream access
        day_ds = day_ds.chunk({"time": 1})

        if not wrote_any_day:
            day_ds.to_zarr(_AMIP_FORCING_PATH, mode="w", zarr_format=2)
            wrote_any_day = True
        else:
            day_ds.to_zarr(
                _AMIP_FORCING_PATH,
                mode="a",
                append_dim="time",
                zarr_format=2,
            )

        print(f"Written day {day_start} to {_AMIP_FORCING_PATH}")
        current_dt = next_dt

    if not wrote_any_day:
        raise RuntimeError(
            f"No AMIP forcing data found in requested range {_START_TIME} to {_END_TIME}"
        )

    print(f"AMIP forcing saved to {_AMIP_FORCING_PATH}")


if __name__ == "__main__":
    main()
