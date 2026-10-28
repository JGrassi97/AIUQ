"""Restore the common output contract after CDO's AIFS grid remapping."""

import argparse
from datetime import datetime, timezone

import numpy as np
import xarray as xr

from AIUQst_lib.cards import read_std_version
from AIUQst_lib.functions import read_config
from AIUQst_lib.output import standardize_output, write_forecast


def finalize(remapped_path, reference_path, output_path, config):
    with xr.open_dataset(reference_path) as reference, xr.open_dataset(remapped_path) as remapped:
        ds = remapped.load()
        # CDO may rename the temporal axis and discard scalar auxiliary coords.
        if "valid_time" not in ds.dims and "time" in ds.dims:
            ds = ds.rename({"time": "valid_time"})
        if ds.sizes.get("valid_time") != reference.sizes["valid_time"]:
            raise ValueError("CDO changed the number of forecast timestamps")
        if not np.array_equal(ds.valid_time.values, reference.valid_time.values):
            raise ValueError("CDO changed forecast valid times")
        standard = read_std_version(config["HPCROOTDIR"], config["STD_VERSION"])["variables"]
        metadata = dict(reference.attrs)
        metadata["date_postprocessed"] = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        metadata["horizontal_regridding"] = "CDO remapdis"
        ds = standardize_output(
            ds, {name: name for name in reference.data_vars}, standard,
            reference.time.values, metadata,
        )
        write_forecast(ds, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    finalize(args.input, args.reference, args.output, read_config(args.config))
