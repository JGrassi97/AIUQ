import xarray as xr



def check_pressure_levels(in_dataset: xr.Dataset, ic_card: dict, pressure_levels_std: dict) -> xr.Dataset:
    """Check and reinterpolate pressure levels in the dataset according to the IC card and standard."""

    # Expected values from IC card
    name_expected = ic_card['pressure_levels']['name']
    units_expected = ic_card['pressure_levels']['units']
    levels_expected = ic_card['pressure_levels']['values']

    # Required values from standard
    levels_required = pressure_levels_std['values']
    units_required = pressure_levels_std['units']['valid']
    units_aliases = pressure_levels_std['units']['aliases']

    # 0 - Rename level dimension to standard name if necessary
    if name_expected != 'level':
        in_dataset = in_dataset.rename({name_expected: 'level'})
        name_expected = 'level'

    # CHECK 1 - Presence of expected name
    level_array = in_dataset.get(name_expected)
    if level_array is None:
        raise KeyError(f"Level name '{name_expected}' not found in data variables")
    
    # Values found from dataset inspection
    levels_found = level_array.values
    units_found = level_array.attrs['units']
    
    # CHECK 2 - Units are as expected - Include renaming or conversion if necessary
    if units_found != units_expected:
        print(Warning(f"WARNING - Units found '{units_found}' do not match expected '{units_expected}'"))

    if units_found == units_required:
        print("Units match the standard")

    elif units_found in units_aliases:
        print("Units match the standard via alias - renaming")
        in_dataset[name_expected].attrs['units'] = units_required
    else:
        print(Warning(f"WARNING - Units found '{units_found}' do not match expected '{units_expected}'"))
        print("Looking for conversion methods available")

    # CHECK 3 - Levels are as expected - Include reinterpolation if necessary
    if len(levels_expected) != len(levels_found):
        print("Levels match the expected values")

        if len(levels_required) > len(levels_found):
            print(Warning("WARNING - Levels found are less than required values"))
        elif len(levels_required) < len(levels_found):
            print(Warning("WARNING - Levels found are more than required values"))
        in_dataset = in_dataset.interp(level=levels_required)

    else:
        if all(levels_expected == levels_found):
            print("Levels match the expected values")
        else:
            print("Reinterpolating to required levels")
            in_dataset = in_dataset.interp(level=levels_required)

    return in_dataset
