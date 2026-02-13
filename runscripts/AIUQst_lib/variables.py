import xarray as xr


def define_mappers(ic_variables, standard_variables):

    """
    Defines the dictionaries for retrieving the variable specs from the card.
    Before AIUQ-st v010, the variables required to the ICs were the variables needed by the model.
    Starting from AIUQ-st v010, the vaiable required to the ICs are all the variables defined by the standard.
    
    Returns:
        - ic_names:         {ID: IC_short_name}                         - variables required by the IC card 
        - missing_vars:     {ID: standard_variable_specs}               - variables required by the standard but missing in the IC card 
        - rename_dict:      {IC_short_name: standard_short_name}        - renaming dictionary to rename the IC short names to the standard short names 
        - long_names_dict:  {standard_short_name: standard_long_name}   - dictionary to retrieve the long names of the variables from the standard short names
        - units_dict:       {standard_short_name: units}                - dictionary to retrieve the units of the variables from the standard short names
    """

    # Do the intersection between the variables required by the standard by IDs
    var_to_take = set(standard_variables['data'].keys())
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
        print(f"Warning: The following variables are missing in the IC data: {missing_vars}")

        missing_vars = {
            v: standard_variables['data'][v]
            for v in missing_vars
        }

    else:
        missing_vars = None
        print("All required variables are available in the IC data.")

    # Create the mapper between the IC short names and the standard short names
    mapping = {
        v: standard_variables['data'][v]
        for v in vars
    }


    rename_dict = {
        ic_names[v]: mapping[v]['short_name']
        for v in vars
    }

    long_names_dict = {
        mapping[v]['short_name']: mapping[v]['long_name']
        for v in vars
    }

    units_dict = {
        mapping[v]['short_name']: mapping[v]['units']
        for v in vars
    }

    return ic_names, missing_vars, rename_dict, long_names_dict, units_dict



def reassign_long_names_units(ds: xr.Dataset, long_names_dict, units_dict) -> xr.Dataset:
    for var in ds.data_vars:
        if var in long_names_dict:
            ds[var].attrs['long_name'] = long_names_dict[var]
        if var in units_dict:
            if units_dict[var] != units_dict[var]:
                print(f"Warning: Variable {var} has different units in the standard dictionary.")
            ds[var].attrs['units'] = units_dict[var]
    return ds

def name_mapper_for_model(model_variables, standard_variable):
    model_keys = set(model_variables.keys())
    standard_name = (standard_variable['data'])
    mapper = {
        standard_name[k]['short_name']: model_variables[k] for k in model_keys
    }
    return mapper