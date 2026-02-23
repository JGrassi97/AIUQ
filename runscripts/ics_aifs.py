"""

"""

# Built-in/Generics
import os
import logging
from datetime import datetime, timezone
import json

# Third party
import numpy as np
import xarray as xr
import earthkit.regrid as ekr

# Local
from AIUQst_lib.functions import parse_arguments, read_config
from AIUQst_lib.pressure_levels import check_pressure_levels
from AIUQst_lib.cards import read_model_card, read_ic_card, read_std_version
from AIUQst_lib.variables import reassign_long_names_units, name_mapper_for_model

logging.basicConfig(level=logging.DEBUG)


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

    # Reading model card for variable/level info
    model_card = read_model_card(_HPCROOTDIR, _MODEL_NAME)
    standard_dict = read_std_version(_HPCROOTDIR, _STD_VERSION)

    name_mapper = name_mapper_for_model(model_card['variables'], standard_dict['variables'])

    # Start date for filename
    start_date = datetime.strptime(_START_TIME, "%Y-%m-%d")

    # Load ICs
    data_original = xr.open_zarr(_INI_DATA_PATH, chunks=None)

    # Some fields are flipped upside down - flip them back to normal orientation (latitudes should be ascending)
    orog = data_original.orog.values
    orog = np.flip(orog, axis=0)
    data_original['orog'] = (('latitude', 'longitude'), orog)

    slor = data_original.slor.values
    slor = np.flip(slor, axis=0)
    data_original['slor'] = (('latitude', 'longitude'), slor)

    sdor = data_original.sdor.values
    sdor = np.flip(sdor, axis=0)
    data_original['sdor'] = (('latitude', 'longitude'), sdor)

    lsm = data_original.lsm.values
    lsm = np.flip(lsm, axis=0)
    data_original['lsm'] = (('latitude', 'longitude'), lsm)

    # Fixing soil level naming
    sot_1 = data_original["sot"].isel(SoilLevel=0).astype(np.float32).rename("stl1")
    sot_2 = data_original["sot"].isel(SoilLevel=1).astype(np.float32).rename("stl2")
    data_original = data_original.drop_vars("sot")
    data_original = data_original.drop_vars("SoilLevel")
    data_original = xr.merge([data_original, sot_1, sot_2], compat="override")

    # Interpolating on required pressure levels
    required_pressure_levels = model_card['pressure_levels']['values']
    data_original = data_original.interp(level=required_pressure_levels)
                                             
    # Rename variables to idiot ECWMF names
    NAME_MAP = {
        "t2m": "2t",
        "d2m": "2d",
        "u10": "10u",
        "v10": "10v",
    }

    data_original = data_original.rename(NAME_MAP)

    # Regrid everything to N320 as required by AIFS checkpoint
    regridded_vars = {}
    NPTS = 542080  # N320 point count

    for v in data_original.data_vars:
        da = data_original[v]

        # Ensure time dim exists (constants)
        if "time" not in da.dims:
            da = da.expand_dims(time=data_original["time"].values)

        # Pressure-level variable
        if "level" in da.dims:
            # expect (time, level, latitude, longitude)
            da = da.transpose("time", "level", "latitude", "longitude")
            out = np.empty((da.sizes["time"], da.sizes["level"], NPTS), dtype=np.float32)

            for ti in range(da.sizes["time"]):
                for li in range(da.sizes["level"]):
                    out[ti, li, :] = ekr.interpolate(
                        da.values[ti, li, :, :],
                        {"grid": (0.25, 0.25)},
                        {"grid": "N320"},
                        method="linear",
                    ).astype(np.float32, copy=False)

            regridded_vars[v] = (("time", "level", "point"), out)

        # Surface/constant variable
        else:
            da = da.transpose("time", "latitude", "longitude")
            out = np.empty((da.sizes["time"], NPTS), dtype=np.float32)

            for ti in range(da.sizes["time"]):
                out[ti, :] = ekr.interpolate(
                    da.values[ti, :, :],
                    {"grid": (0.25, 0.25)},
                    {"grid": "N320"},
                    method="linear",
                ).astype(np.float32, copy=False)

            regridded_vars[v] = (("time", "point"), out)

    coords = {"time": data_original["time"], "point": np.arange(NPTS, dtype=np.int32)}
    if "level" in data_original.coords:
        coords["level"] = data_original["level"]

    data_n320 = xr.Dataset(data_vars=regridded_vars, coords=coords)

    # Build fields dict in exact format expected by Anemoi
    fields = {}

    for v in data_n320.data_vars:
        da = data_n320[v]

        # Pressure-level variable should be split into separate arrays per level
        if "level" in da.dims:

            for lev in required_pressure_levels:
                fields[f"{v}_{lev}"] = da.sel(level=lev).values.astype(np.float32, copy=False)

        # Surface variable can be used as they are
        else:

            # Orography should be named 'z' in the pipeline
            if v == "orog":
                da = da.rename("z")
                v = "z"
            fields[v] = da.values.astype(np.float32, copy=False)

    t64 = data_n320["time"].isel(time=1).values
    DATE = datetime.utcfromtimestamp(t64.astype("datetime64[s]").astype(int))

    # Fields to keep
    to_keep = set([name_mapper[v] for v in name_mapper.keys()])
    
    # add stl1 and stl2 to the list of fields to keep
    to_keep.add("stl1")
    to_keep.add("stl2")

    def base_name(varname: str) -> str:
        if "_" in varname:
            return varname.split("_")[0]
        return varname

    fields = {
        k: v
        for k, v in fields.items()
        if base_name(k) in to_keep
    }

    timestamp = int(DATE.replace(tzinfo=timezone.utc).timestamp())

        # use YYYYMMDD in filename to avoid spaces/colons
    ics_basename = f"ics_aifs_{start_date.strftime('%Y%m%d')}"
    ics_file = os.path.join(_ICS_TEMP_DIR, f"{ics_basename}.npz")
    ics_names_file = os.path.join(_ICS_TEMP_DIR, f"{ics_basename}.names.json")

    # Ensure dir exists
    os.makedirs(_ICS_TEMP_DIR, exist_ok=True)
    print(f"Writing ICs to {ics_file} ...")

    # Prepare dict for np.savez: prefix field arrays to avoid name collisions
    np_dict = {}
    field_names = []
    for k, arr in fields.items():
        # Ensure ndarray of known dtype (float32)
        a = np.asarray(arr, dtype=np.float32)
        np_dict[f"f__{k}"] = a
        field_names.append(k)

    # Add timestamp as scalar array
    np_dict["timestamp"] = np.array([timestamp], dtype=np.int64)

    # Save compressed .npz
    np.savez_compressed(ics_file, **np_dict)

    # Save the ordered list of field names in json for human inspection
    with open(ics_names_file, "w") as nf:
        json.dump({"fields": field_names, "timestamp": timestamp}, nf)

    print(f"Wrote: {ics_file}")
    print(f"Wrote names meta: {ics_names_file}")


if __name__ == "__main__":
    main()