"""
Author: Jacopo Grassi
Institution: Politecnico di Torino
Email: jacopo.grassi@polito.it

Created: 2025-01-12
Last modified: 2025-02-13

Description:
    Utlity functions for handling variable according to the AIUQ standard.
    Allows to handle:
        - naming convention
        - units
"""   

# Built-in/Generics
import logging
import os
from typing import List

# Third party
import numpy as np
import xarray as xr

# Local

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def define_ics_mappers(ic_variables: dict, standard_variables: dict) -> tuple:

    """
    Defines the dictionaries for retrieving the variable specs from the card when saving the ICs.
    Before AIUQ-st v010, the variables required to the ICs were the variables needed by the model.
    Starting from AIUQ-st v010, the vaiable required to the ICs are all the variables defined by the standard.
    
    Returns:
        - ic_names:         {ID: IC_short_name}                         - variables required by the IC card 
        - missing_vars:     {ID: standard_variable_specs}               - variables required by the standard but missing in the IC card 
        - rename_dict:      {IC_short_name: standard_short_name}        - renaming dictionary to rename the IC short names to the standard short names 
        - long_names_dict:  {standard_short_name: standard_long_name}   - dictionary to retrieve the long names of the variables from the standard short names
        - units_dict:       {standard_short_name: units}                - dictionary to retrieve the units of the variables from the standard short names
    """

    logging.info("Defining the mappers for the IC variables...")

    # Do the intersection between the variables required by the standard by IDs
    var_to_take = set(standard_variables.keys())
    var_available = set(ic_variables.keys())
    vars = var_available.intersection(var_to_take)

    # -- Create the ic_names dictionary --
    ic_names = {
            v: ic_variables[v]['name']
            for v in vars
        }

    # -- Create the missing variables dictionary --
    if vars != var_to_take:
        missing_vars = var_to_take - vars
        logging.warning(f"The following variables are missing in the IC data: {missing_vars}. Falling back to another method if possible.")

        missing_vars = {
            v: standard_variables[v]
            for v in missing_vars
        }

    else:
        missing_vars = None
        logging.info("All required variables are available in the IC data.")

    # Create the mapper between the IC short names and the standard short names
    mapping = {
        v: standard_variables[v]
        for v in vars
    }

    # -- Create the renaming dictionary --
    rename_dict = {
        ic_names[v]: mapping[v]['short_name']
        for v in vars
    }

    # -- Create the long names and units dictionaries --
    long_names_dict = {
        mapping[v]['short_name']: mapping[v]['long_name']
        for v in vars
    }

    # -- Create the units dictionary --
    units_dict = {
        mapping[v]['short_name']: mapping[v]['units']
        for v in vars
    }

    return ic_names, rename_dict, long_names_dict, units_dict, missing_vars



def optional_variables(standard_variables: dict) -> set:
    """IDs of the standard variables flagged ``optional: true`` (AIUQ-st v100: the wave fields)."""
    return {code for code, spec in standard_variables.items() if spec.get("optional")}


def drop_optional_missing(missing: dict | None, standard_variables: dict) -> dict | None:
    """Remove the optional variables from ``missing`` ({ID: standard specs}), with a warning.

    Optional variables are not an error for the initial conditions: a model whose card
    needs them fails later (SAVE_ICS/SIM) with the list of the fields it cannot find.
    Returns the remaining (required) missing variables, or None.
    """
    if not missing:
        return None
    optional = optional_variables(standard_variables)
    dropped = sorted(spec["short_name"] for code, spec in missing.items() if code in optional)
    if dropped:
        logging.warning(f"Optional standard variables without a source, left out of the ICs: {dropped}")
    remaining = {code: spec for code, spec in missing.items() if code not in optional}
    return remaining or None


def variables_of_type(standard_variables: dict, kind: str) -> set:
    """Short names of the standard variables of a given type (sfc, pl, soil, wave)."""
    return {spec["short_name"] for spec in standard_variables.values() if spec.get("type") == kind}


def reassign_long_names_units(ds: xr.Dataset, long_names_dict: dict, units_dict: dict) -> xr.Dataset:

    """Reassigns the long names and units of the variables in the dataset according to the standard."""

    for var in ds.data_vars:

        if var in long_names_dict:
            ds[var].attrs['long_name'] = long_names_dict[var]

        if var in units_dict:
            current_unit = ds[var].attrs.get('units', None)
            if current_unit != units_dict[var]:
                # TODO: implement unit handling. For now, just print a warning if the units are different.
                # A possible solution can be reusing the unit handling from xclim. 
                # See: https://xclim.readthedocs.io/en/stable/notebooks/units.html
                logging.critical(f"Variable {var} has different units in the standard dictionary. \
                                  This warning will be addressed in future versions of the code. \
                                  Right now, no conversion will be applied so there might be inconsistencies in the produced dataset. \
                                  Please note that this warning may be triggered also by a difference in the unit format (e.g., 'K' vs 'kelvin'). \
                                  Current units: {units_dict[var]} - Required units: {current_unit}.")

            ds[var].attrs['units'] = units_dict[var]

    return ds



def name_mapper_for_model(model_variables: dict, standard_variable: dict) -> dict:

    """ Defines the mapper between the variable names used by the model and the variable names defined by the standard."""

    model_keys = set(model_variables.keys())
    standard_name = (standard_variable['data'])

    mapper = {
        standard_name[k]['short_name']: model_variables[k] 
        for k in model_keys
    }
    
    return mapper

# Testing =======================================================================================================

def output_translator(model_variables: dict, standard_variable: dict) -> dict:

    """ Translate the output names set in the main file and adjust to the chosen model standard"""

    model_keys = set(model_variables.keys())
    standard_name = (standard_variable['data'])

    translator ={}

    for k in model_keys:
        if k in standard_name:
            model_name = model_variables[k]
            short = standard_name[k]['short_name']
            long = standard_name[k]['long_name']

            translator[short] = model_name 
            translator[long] = model_name
    
    
    return translator

# Outputs =======================================================================================================

def standard_specs(standard_dict: dict) -> dict:
    """{code: spec} of the standard: the IC variables (variables.yml) and the output-only ones (outputs.yml)."""
    specs = dict((standard_dict.get("variables") or {}).get("data") or {})
    specs.update((standard_dict.get("outputs") or {}).get("data") or {})
    return specs


def output_specs(model_card: dict, standard_dict: dict) -> dict:
    """What a model can write, keyed by AIUQ-st short name.

    The card's ``variables`` (model inputs, also predicted) and ``outputs`` (output-only fields) map AIUQ-st
    codes to model names; an ``outputs`` entry may be a mapping with
      name          model name
      scale         factor applied to the model value to get the AIUQ-st units (default 1)
      window        period an accumulation or mean refers to (written to the file attributes)
      deaccumulate  the model gives a running total since the start: write the increment over each output step
    Returns {short_name: {"model", "scale", "window", "deaccumulate", "accumulated", "long_name", "units", "type"}}.
    """
    std = standard_specs(standard_dict)
    out = {}

    def add(code, entry, from_outputs):
        spec = std.get(code)
        if spec is None:
            return
        if not isinstance(entry, dict):
            entry = {"name": entry}
        short = spec["short_name"]
        if not from_outputs and short in out:
            return
        out[short] = {
            "model": str(entry["name"]),
            "scale": float(entry.get("scale", 1.0)),
            "window": entry.get("window"),
            "deaccumulate": bool(entry.get("deaccumulate", False)),
            "accumulated": bool(spec.get("accumulated", False)),
            "long_name": spec.get("long_name", short),
            "units": spec.get("units", ""),
            "type": spec.get("type", ""),
        }

    for code, name in (model_card.get("variables") or {}).items():
        add(code, name, False)
    for code, entry in (model_card.get("outputs") or {}).items():
        add(code, entry, True)
    return out


def checkpoint_entry(model_card: dict, checkpoint: str) -> dict:
    """The card's ``checkpoints`` entry for a checkpoint file (by base name), merged over ``default``."""
    blocks = model_card.get("checkpoints") or {}
    entry = dict(blocks.get("default") or {})
    entry.update(blocks.get(os.path.basename(str(checkpoint))) or {})
    return entry


def resolve_outputs(requested, model_card: dict, standard_dict: dict, checkpoint: str = ""):
    """Match the requested OUT_VARS (AIUQ-st short or long names) to what the model writes.

    Returns (selected, unknown): selected is a list of (short_name, spec) in the requested order (outputs are
    written under the short name); unknown lists the names this model card or checkpoint cannot write (a warning,
    not an error: the run writes the others).
    """
    specs = output_specs(model_card, standard_dict)
    by_long = {s["long_name"]: k for k, s in specs.items()}
    allowed = checkpoint_entry(model_card, checkpoint).get("outputs") if checkpoint else None
    selected, unknown = [], []
    for name in requested:
        short = name if name in specs else by_long.get(name)
        if short is None or (allowed is not None and short not in allowed):
            unknown.append(name)
            continue
        if short not in [k for k, _ in selected]:
            selected.append((short, specs[short]))
    return selected, unknown


def output_attrs(spec: dict, daily: bool = False) -> dict:
    """netCDF attributes of an output variable (``daily``: OUT_FREQ daily, see resample_daily)."""
    attrs = {"long_name": spec["long_name"], "units": spec["units"]}
    window = spec.get("window")
    if daily and spec["accumulated"]:
        window = f"day: sum of the totals over the periods ending 00-24 UTC ({window or 'model step'})"
    elif daily and window:
        window = f"day: mean of the values for the {window}"
    if window:
        attrs["aiuq_period"] = window
        attrs["cell_methods"] = f"valid_time: {'sum' if spec['accumulated'] else 'mean'} ({window})"
    return attrs


def deaccumulate(values, previous=None):
    """Increments along the first axis of a running total; ``previous`` is the total before the first value
    (zero at the start of the run)."""
    values = np.asarray(values)
    first = values[:1] - (0.0 if previous is None else previous)
    return np.concatenate([first, np.diff(values, axis=0)], axis=0)


def resample_daily(dataset, variable: str, accumulated: bool):
    """Daily output: mean of the steps of each day, or for accumulations the sum of the periods ending in
    (day 00 UTC, next day 00 UTC], labelled with the day."""
    if accumulated:
        # Day of each period: a total ending at 00 UTC belongs to the day before. groupby makes only the days
        # present (resample(closed="right") adds empty days at the edges).
        day = (dataset["valid_time"] - np.timedelta64(1, "s")).dt.floor("D").rename("day")
        out = dataset.drop_vars([c for c in dataset.coords if c != "valid_time" and "valid_time" in dataset[c].dims])
        out = out.groupby(day).sum(min_count=1).rename({"day": "valid_time"})
        return out
    return dataset.resample(valid_time="1D").mean()
