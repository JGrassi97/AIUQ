"""Add UTC-hour 100 m wind climatology from ARCO ERA5 to an existing support file.

Yearly sums are checkpointed so a long network calculation can resume. Existing
cloud fields are copied unchanged; a backup is retained before atomic replacement.
"""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil

import dask
import netCDF4
import numpy as np
import xarray as xr

ARCO_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
WIND_NAMES = {"100m_u_component_of_wind": "u100", "100m_v_component_of_wind": "v100"}


def yearly_sums(source, year, hours):
    selected = source.sel(time=slice(f"{year}-01-01", f"{year}-12-31T23:00:00"))
    selected = selected.isel(time=np.flatnonzero(np.isin(selected.time.dt.hour.values, hours)))
    expected_days = int((np.datetime64(f"{year + 1}-01-01") - np.datetime64(f"{year}-01-01"))
                        / np.timedelta64(1, "D"))
    counts = selected.time.groupby("time.hour").count()
    if counts.sizes["hour"] != len(hours) or not bool((counts == expected_days).all()):
        raise ValueError(f"Incomplete ARCO timestamps for {year}: {counts.values}")
    # Float64 accumulation and skipna=False prevent silently reducing the sample count.
    sums = selected.astype("float64").groupby("time.hour").sum("time", skipna=False)
    sums["sample_count"] = counts
    return sums


def update_climatology(path, first_year=1980, last_year=1990, workers=16, store=ARCO_STORE):
    path = Path(path).resolve()
    with xr.open_dataset(path) as original:
        hours = original.hour.values.copy()
        latitudes = original.latitude.values.copy()
        longitudes = original.longitude.values.copy()
    if not np.isin(hours, np.arange(24)).all():
        raise ValueError("The target hour coordinate must contain integer UTC hours.")
    checkpoint_dir = path.parent / f"wind100-{first_year}-{last_year}-checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    print(f"Opening {store}; UTC hours {hours.tolist()}; years {first_year}-{last_year}", flush=True)
    source = xr.open_zarr(store, chunks={}, storage_options={"token": "anon"})
    source = source[list(WIND_NAMES)].rename(WIND_NAMES)
    if not (np.array_equal(source.latitude, latitudes) and np.array_equal(source.longitude, longitudes)):
        raise ValueError("ARCO and target grids differ; refusing to re-label coordinates.")
    total = None
    with dask.config.set(scheduler="threads", num_workers=workers):
        for year in range(first_year, last_year + 1):
            checkpoint = checkpoint_dir / f"wind100-{year}.nc"
            if not checkpoint.exists():
                print(f"Computing {year} ...", flush=True)
                sums = yearly_sums(source, year, hours).compute()
                if not all(bool(np.isfinite(sums[v]).all()) for v in WIND_NAMES.values()):
                    raise ValueError(f"Non-finite ARCO wind values in {year}")
                sums.attrs.update(source_store=store, year=year)
                temporary = checkpoint.with_suffix(".nc.partial")
                sums.to_netcdf(temporary)
                os.replace(temporary, checkpoint)
            with xr.open_dataset(checkpoint) as saved:
                if (saved.attrs.get("source_store") != store or saved.attrs.get("year") != year
                        or not np.array_equal(saved.hour, hours)
                        or not np.array_equal(saved.latitude, latitudes)
                        or not np.array_equal(saved.longitude, longitudes)):
                    raise ValueError(f"Incompatible checkpoint {checkpoint}")
                sums = saved.load()
            total = sums if total is None else total + sums
            print(f"Finished {year}; accumulated samples per hour: {total.sample_count.values.tolist()}", flush=True)
    means = {v: (total[v] / total.sample_count).astype("float32") for v in WIND_NAMES.values()}
    temporary = path.with_name(path.name + ".wind100.partial")
    shutil.copy2(path, temporary)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    period = f"{first_year}-01-01/{last_year}-12-31"
    with netCDF4.Dataset(temporary, "a") as output:
        for name, data in means.items():
            if name in output.variables:
                raise ValueError(f"{name} already exists in {path}; refusing to overwrite it.")
            variable = output.createVariable(name, "f4", ("hour", "latitude", "longitude"),
                                             zlib=True, complevel=4)
            variable[:] = data.transpose("hour", "latitude", "longitude").values
            variable.setncatts({"units": "m s**-1", "long_name": f"100 metre {name[0].upper()} wind component",
                               "source": "ERA5 climatology (Google ARCO)", "source_store": store,
                               "climatology_period": period, "cell_methods": "time: mean within UTC hours over years",
                               "samples_per_hour": int(total.sample_count.min()),
                               "date_created": stamp})
        output.setncattr("wind100_climatology_period", period)
    with xr.open_dataset(temporary) as check:
        for name, expected in means.items():
            np.testing.assert_array_equal(check[name].values, expected.values)
    # Hard-link backup retains the previous file even after os.replace.
    backup = path.with_name(path.name + f".before-wind100-{stamp}.bak")
    os.link(path, backup)
    os.replace(temporary, path)
    print(f"Updated {path}; backup {backup}; samples per hour {total.sample_count.values.tolist()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--climatology", required=True)
    parser.add_argument("--first-year", type=int, default=1980)
    parser.add_argument("--last-year", type=int, default=1990)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if args.first_year > args.last_year:
        parser.error("first-year must be <= last-year")
    update_climatology(args.climatology, args.first_year, args.last_year, args.workers)
