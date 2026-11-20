"""
Ocean-wave helpers for the AIUQ-st wave variables (type ``wave``, AIUQ-st v100).

- Mean wave direction is a circular quantity: the AIFS v2 checkpoints take and return it as
  cos/sin components (``cos_mwd``/``sin_mwd``, the anemoi ``cos_sin_mean_wave_direction`` filter).
- The significant wave heights per period band (h1012 ... h2530) are not archived in ERA5:
  they are approximated from the wave partitions (wind waves and the three swell partitions),
  assigning the energy (H**2) of each partition to the band that contains its mean period.
"""

# Third party
import numpy as np
import xarray as xr

# Period bands (s) of the ECMWF parameters 140114-140119, lower bound inclusive. The upper bound
# is exclusive except for the last band, so that every period between 10 s and 30 s falls in
# exactly one band.
WAVE_PERIOD_BANDS = {
    "h1012": (10.0, 12.0),
    "h1214": (12.0, 14.0),
    "h1417": (14.0, 17.0),
    "h1721": (17.0, 21.0),
    "h2125": (21.0, 25.0),
    "h2530": (25.0, 30.0),
}


def direction_to_cos_sin(direction):
    """Degrees (true) -> (cos, sin) components. Works on numpy arrays and DataArrays."""
    radians = np.deg2rad(direction)
    return np.cos(radians), np.sin(radians)


def cos_sin_to_direction(cos_component, sin_component):
    """(cos, sin) components -> degrees in [0, 360). Works on numpy arrays and DataArrays."""
    return np.rad2deg(np.arctan2(sin_component, cos_component)) % 360.0


def wave_period_bands(partitions, bands=WAVE_PERIOD_BANDS) -> dict:
    """Significant wave height per period band from wave partitions.

    ``partitions`` is a list of ``(height, period)`` pairs (DataArrays or arrays with the same
    shape): wind waves and swell partitions. Each partition is treated as narrow-banded around
    its mean period: its energy H**2 goes to the band containing that period, and the band
    height is sqrt of the summed energy. Points where every partition is undefined (land) stay
    NaN; partitions with no energy (NaN height but defined neighbours) count as zero.

    Returns ``{band_name: height}`` with the type of the inputs.
    """
    if not partitions:
        raise ValueError("At least one wave partition is required")
    last = list(bands)[-1]
    result = {}
    any_defined = None
    for height, _ in partitions:
        defined = height.notnull() if isinstance(height, xr.DataArray) else np.isfinite(height)
        any_defined = defined if any_defined is None else (any_defined | defined)
    for name, (low, high) in bands.items():
        energy = None
        for height, period in partitions:
            if name == last:
                inside = (period >= low) & (period <= high)
            else:
                inside = (period >= low) & (period < high)
            finite = (height.notnull() if isinstance(height, xr.DataArray) else np.isfinite(height))
            contribution = xr.where(inside & finite, height ** 2, 0.0) if isinstance(height, xr.DataArray) \
                else np.where(inside & finite, height ** 2, 0.0)
            energy = contribution if energy is None else energy + contribution
        band = np.sqrt(energy)
        if isinstance(band, xr.DataArray):
            band = band.where(any_defined)
            band.name = name
            band.attrs = {
                "derivation": "sqrt(sum of H**2 of the wave partitions whose mean period lies in the band)",
                "period_band_s": [low, high],
            }
        else:
            band = np.where(any_defined, band, np.nan)
        result[name] = band
    return result
