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

# Third party
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