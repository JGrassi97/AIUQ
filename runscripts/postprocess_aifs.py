"""

"""

# Built-in/Generics
import os
import yaml
import re
import ast
import json
import pickle
import logging
from copy import deepcopy
from datetime import datetime, timedelta, timezone

# Third party
import netCDF4 as nc
import numpy as np
import xarray as xr
import earthkit.regrid as ekr

# Local
from AIUQst_lib.functions import parse_arguments, read_config, normalize_out_vars
from AIUQst_lib.pressure_levels import check_pressure_levels
from AIUQst_lib.cards import read_model_card, read_ic_card, read_std_version
from AIUQst_lib.variables import name_mapper_for_model


def post_process_aifs(var_name, data, date, _OUT_LEVS) -> None:

    if _OUT_LEVS != 'original':
        desired_levels = [
            int(plev)
            for plev in _OUT_LEVS.strip('[]').split(',')
        ]

    dt_object = date

    dataarrays = []
    if '_' not in var_name:

        data_interp = ekr.interpolate(
                    data,
                    {"grid": "N320"},
                    {"grid": [0.25, 0.25]},
                    "linear"
                )

        dataarray = xr.DataArray(
            data_interp,
            dims=['latitude', 'longitude'],
            attrs={'long_name': var_name, 'units': 'unknown'}
        ).rename(var_name)
        dataarrays.append(dataarray)

    if '_' not in var_name:

        for level in desired_levels:

            data_interp = ekr.interpolate(
                data,
                {"grid": "N320"},
                {"grid": [0.25, 0.25]},
                "linear"
            )

            dataarray = xr.DataArray(
                data_interp,
                dims=['latitude', 'longitude'],
                attrs={'long_name': f'{var_name} at {level} hPa', 'units': 'unknown'}
            ).rename(f'{var_name}').expand_dims('level').assign_coords(level=[level])
            dataarrays.append(dataarray)

    dataset_complete = xr.merge(dataarrays, join='outer')
    dataset_complete = dataset_complete.assign_coords(time=[dt_object])

    return dataset_complete
    
