import xarray as xr


def define_mappers(model_variables, ic_variables, standard_variables):

    var_to_take = set(model_variables.keys())
    var_available = set(ic_variables.keys())

    # Check if var_available is a subset of var_to_take
    vars = var_available.intersection(var_to_take)

    if vars != var_to_take:
        missing_vars = var_to_take - vars
        print(f"Warning: The following variables are missing in the IC data: {missing_vars}")

        missing_vars = {
            v: model_variables[v]
            for v in missing_vars
        }

    else:
        missing_vars = None
        print("All required variables are available in the IC data.")

    mapping = {
        v: standard_variables['data'][v]
        for v in vars
    }

    ic_names = {
        v: ic_variables[v]['name']
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

    return ic_names, rename_dict, long_names_dict, units_dict, missing_vars

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