import re
import numpy as np
import xarray as xr
import earthkit.regrid as ekr

GRID_IN = {"grid": "N320"}
GRID_OUT = {"grid": [0.25, 0.25]}
N_POINTS = 542080
# coordinate standard per una 0.25° globale (ECMWF-style: lon 0..360)
LATS = np.arange(90.0, -90.25, -0.25)   # 721
LONS = np.arange(0.0, 360.0, 0.25)      # 1440

LEVEL_RE = re.compile(r"^(?P<base>.+)_(?P<lev>\d+)$")

def interp_to_025(data):
    #return ekr.interpolate(data, GRID_IN, GRID_OUT, "linear")
    return data

def parse_level_name(var_name: str):
    """
    Return (base, level_int) if var_name matches like 't_500', else (var_name, None).
    """
    m = LEVEL_RE.match(var_name)
    if not m:
        return var_name, None
    return m.group("base"), int(m.group("lev"))


def build_dataset_for_state(state, output_vars, output_levels):
    """
    state: dict con state['date'] e state['fields'] (var_name -> data)
    output_vars: set/list di variabili da tenere (nomi originali: '2t', 't_500', ...)
    """
    date = state["date"]

    # surface_vars[name] = 2D DataArray
    surface_vars = {}

    # level_groups[base] = list of (level, 2D DataArray)
    level_groups = {}

    for var_name, data in state["fields"].items():

        base, lev = parse_level_name(var_name)

        if base not in output_vars:
            continue

        if lev is not None and lev not in output_levels:
            continue

        data_interp = interp_to_025(data)  # UNA sola interpolazione per campo

        da2d = xr.DataArray(
            data_interp,
            dims=["points"],
            coords={"points": np.arange(N_POINTS)},
            name=base if lev is not None else var_name,
            attrs={"long_name": var_name, "units": "unknown"},
        )

        if lev is None:
            # surface
            surface_vars[var_name] = da2d
        else:
            # pressure-level: raggruppa per base
            level_groups.setdefault(base, []).append((lev, da2d))

    data_vars = {}

    # aggiungi surface
    data_vars.update(surface_vars)

    # crea i 3D per i gruppi su livello
    for base, items in level_groups.items():
        # ordina per livello
        items.sort(key=lambda x: x[0])
        levels = [lev for lev, _ in items]
        arrays = [da for _, da in items]

        da3d = xr.concat(arrays, dim=xr.IndexVariable("level", levels))
        da3d.name = base
        da3d.attrs["long_name"] = base
        data_vars[base] = da3d

    ds = xr.Dataset(data_vars).expand_dims(time=[date])
    return ds
























# """

# """

# # Built-in/Generics
# import os
# import yaml
# import re
# import ast
# import json
# import pickle
# import logging
# from copy import deepcopy
# from datetime import datetime, timedelta, timezone

# # Third party
# import netCDF4 as nc
# import numpy as np
# import xarray as xr
# import earthkit.regrid as ekr

# # Local
# from AIUQst_lib.functions import parse_arguments, read_config, normalize_out_vars
# from AIUQst_lib.pressure_levels import check_pressure_levels
# from AIUQst_lib.cards import read_model_card, read_ic_card, read_std_version
# from AIUQst_lib.variables import name_mapper_for_model


# def post_process_aifs(var_name, data, date, _OUT_LEVS) -> None:

#     if _OUT_LEVS != 'original':
#         desired_levels = [
#             int(plev)
#             for plev in _OUT_LEVS.strip('[]').split(',')
#         ]

#     dt_object = date

#     dataarrays = []
#     if '_' not in var_name:

#         data_interp = ekr.interpolate(
#                     data,
#                     {"grid": "N320"},
#                     {"grid": [0.25, 0.25]},
#                     "linear"
#                 )

#         dataarray = xr.DataArray(
#             data_interp,
#             dims=['latitude', 'longitude'],
#             attrs={'long_name': var_name, 'units': 'unknown'}
#         ).rename(var_name)
#         dataarrays.append(dataarray)

#     if '_' not in var_name:

#         for level in desired_levels:

#             data_interp = ekr.interpolate(
#                 data,
#                 {"grid": "N320"},
#                 {"grid": [0.25, 0.25]},
#                 "linear"
#             )

#             dataarray = xr.DataArray(
#                 data_interp,
#                 dims=['latitude', 'longitude'],
#                 attrs={'long_name': f'{var_name} at {level} hPa', 'units': 'unknown'}
#             ).rename(f'{var_name}').expand_dims('level').assign_coords(level=[level])
#             dataarrays.append(dataarray)

#     dataset_complete = xr.merge(dataarrays, join='outer')
#     dataset_complete = dataset_complete.assign_coords(time=[dt_object])

#     return dataset_complete
    
