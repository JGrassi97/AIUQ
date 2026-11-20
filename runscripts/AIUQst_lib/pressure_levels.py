"""
Author: Jacopo Grassi
Institution: Politecnico di Torino
Email: jacopo.grassi@polito.it

Created: 2025-01-12
Last modified: 2025-02-13

Description:
    
"""   

# Built-in/Generics
import logging
from typing import Iterable

# Third party
import numpy as np
import xarray as xr

# Local

G = 9.80665                 # m s-2, converts geopotential height (gh) to geopotential (z)
RD = 287.05                 # J kg-1 K-1, dry air gas constant (hypsometric extrapolation)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



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
        logging.warning(f"Level units found '{units_found}' do not match expected '{units_expected}'")

    if units_found == units_required:
        logging.info("Units match the required standard.")

    elif units_found in units_aliases:
        logging.info("Units match the standard via alias - renaming")
        in_dataset[name_expected].attrs['units'] = units_required
    else:
        logging.warning(f"Units found '{units_found}' do not match expected '{units_expected}'")
        logging.info("Looking for conversion methods available")

    # CHECK 3 - Levels are as expected - Include reinterpolation if necessary
    if len(levels_expected) != len(levels_found):
        logging.info("Levels match the expected values")

        if len(levels_required) > len(levels_found):
            logging.warning("WARNING - Levels found are less than required values")
        elif len(levels_required) < len(levels_found):
            logging.warning("WARNING - Levels found are more than required values")
        in_dataset = in_dataset.interp(level=levels_required)

    else:
        if all(levels_expected == levels_found):
            logging.info("Levels match the expected values")
        else:
            logging.info("Reinterpolating to required levels")
            in_dataset = in_dataset.interp(level=levels_required)

    return in_dataset


# Interpolation to the standard levels (shared by the ECMWF Open Data and EC-Earth3 ICs) ---------

# US Standard Atmosphere 1976 layers: (base pressure Pa, base temperature K, lapse rate dT/dh K/m)
_STD_ATMOSPHERE = [
    (22632.06, 216.65, 0.0),
    (5474.889, 216.65, 0.001),
    (868.0187, 228.65, 0.0028),
    (110.9063, 270.65, 0.0),
    (66.93887, 270.65, -0.0028),
    (3.956420, 214.65, -0.002),
]


def standard_atmosphere_temperature(p_hpa):
    """US Standard Atmosphere 1976 temperature (K) at pressure p (hPa), above 226 hPa."""
    p = np.atleast_1d(np.asarray(p_hpa, dtype="float64")) * 100.0
    if (p > _STD_ATMOSPHERE[0][0]).any():
        raise ValueError("Standard-atmosphere profile defined only above 226.32 hPa")
    temperature = np.empty_like(p)
    for i, value in enumerate(p):
        base_p, base_t, lapse = [layer for layer in _STD_ATMOSPHERE if value <= layer[0]][-1]
        temperature[i] = base_t * (value / base_p) ** (-RD * lapse / G) if lapse else base_t
    return temperature if np.ndim(p_hpa) else temperature[0]


def extrapolate_above(top: xr.Dataset, p_top: float, above: np.ndarray, dim: str) -> xr.Dataset:
    """Levels above the highest available one.

    t follows the standard-atmosphere shape anchored at the top level,
    t(p) = t(p_top) + T_std(p) - T_std(p_top); z is integrated hydrostatically with the
    layer-mean temperature; the other variables (u, v, q, w) are held constant.
    """
    levels = np.sort(above)[::-1]                       # from just above p_top upwards
    profile = standard_atmosphere_temperature(np.concatenate([[p_top], levels]))
    out = top.expand_dims({dim: levels}).copy()
    if "t" in top:
        t_levels = [top["t"] + (profile[k + 1] - profile[0]) for k in range(levels.size)]
        out["t"] = xr.concat(t_levels, dim=dim).assign_coords({dim: levels})
        if "z" in top:
            z_levels, z_below, t_below, p_below = [], top["z"], top["t"], p_top
            for k, p in enumerate(levels):
                z_below = z_below + RD * 0.5 * (t_below + t_levels[k]) * np.log(p_below / p)
                t_below, p_below = t_levels[k], p
                z_levels.append(z_below)
            out["z"] = xr.concat(z_levels, dim=dim).assign_coords({dim: levels})
    return out.sortby(dim)


def to_standard_levels(dataset: xr.Dataset, required: Iterable[float], dim: str = "level") -> xr.Dataset:
    """Interpolate linearly in log(p); extrapolate above the top level with
    ``extrapolate_above``. Source levels are returned unchanged."""
    pl_vars = [v for v in dataset.data_vars if dim in dataset[v].dims]
    if not pl_vars:
        return dataset
    source = dataset[pl_vars].sortby(dim)
    p_source = source[dim].values.astype("float64")
    target = np.asarray(sorted(float(p) for p in required))
    if (target > p_source.max()).any():
        raise ValueError(f"Cannot extrapolate below the lowest level {p_source.max()} hPa")
    inside = target[target >= p_source.min()]
    above = target[target < p_source.min()]

    logp = source.assign_coords({dim: np.log(p_source)})
    parts = []
    if above.size:
        top = source.sel({dim: p_source.min()}, drop=True)
        parts.append(extrapolate_above(top, float(p_source.min()), above, dim))
    parts.append(logp.interp({dim: np.log(inside)}).assign_coords({dim: inside}))
    parts = [part.transpose(*source[pl_vars[0]].dims) for part in parts]
    levels_ds = xr.concat(parts, dim=dim)
    for v in pl_vars:
        levels_ds[v] = levels_ds[v].transpose(*dataset[v].dims).astype(dataset[v].dtype)
        levels_ds[v].attrs = dict(dataset[v].attrs)
    if np.all(target == np.round(target)):
        levels_ds = levels_ds.assign_coords({dim: target.astype("int64")})
    levels_ds[dim].attrs = dict(dataset[dim].attrs)
    out = dataset.drop_vars(pl_vars + [dim]).merge(levels_ds)
    out.attrs["levels_extrapolated_hPa"] = [float(p) for p in above]
    if above.size:
        out.attrs["levels_extrapolation"] = ("t: anchored US Standard Atmosphere 1976 profile; "
                                             "z: hydrostatic; u, v, q, w: constant")
    return out
