import numpy as np
import earthkit.regrid as ekr
import os

# Set the earthkit regrid cache directory
# from earthkit.regrid.utils.config import CONFIG
# EARTHKIT_REGRID_CACHE = os.environ.get('EARTHKIT_REGRID_CACHE', '')
# CONFIG.set("cache-policy", "user")
# CONFIG.set("user-cache-directory", EARTHKIT_REGRID_CACHE)

nlat = 721
nlon = 1440
dummy_ll = np.zeros((nlat, nlon), dtype=np.float32)

in_grid = {"grid": (0.25, 0.25)}
out_grid = {"grid": "N320"}

npoints_n320 = 542080
dummy_n320 = np.zeros(npoints_n320, dtype=np.float32)


nlat = 181
nlon = 360
dummy_1deg = np.zeros((nlat, nlon), dtype=np.float32)

ekr.interpolate(
    dummy_ll,
    in_grid=in_grid,
    out_grid=out_grid,
    method="linear",
)

print('Saved regrid 0.25° to N320 and back test successfully.')



in_grid = {"grid": "N320"}
out_grid = {"grid": [0.25, 0.25]}

ekr.interpolate(
    dummy_n320,
    in_grid=in_grid,
    out_grid=out_grid,
    method="linear",
)

print('Saved regrid N320 to 0.25° and back test successfully.')



in_grid = {"grid": "N320"}
out_grid = {"grid": [1, 1]}

ekr.interpolate(
    dummy_n320,
    in_grid=in_grid,
    out_grid=out_grid,
    method="linear",
)

print('Saved regrid N320 to 1° and back test successfully.')



in_grid = {"grid": [1, 1]}
out_grid = {"grid": "N320"}

ekr.interpolate(
    dummy_1deg,
    in_grid=in_grid,
    out_grid=out_grid,
    method="linear",
)

print('Saved regrid 1° to N320 and back test successfully.')