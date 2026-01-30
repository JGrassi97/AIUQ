import numpy as np
import earthkit.regrid as ekr

print(ekr._CACHE_DIR)


dummy = np.zeros((30, 30), dtype=np.float32)

in_grid = {"grid": (0.25, 0.25)}
out_grid = {"grid": "N320"}

ekr.interpolate(
    dummy,
    in_grid=in_grid,
    out_grid=out_grid,
    method="linear",
)

print('Saved regrid N320 to 0.25° and back test successfully.')

in_grid = {"grid": "N320"}
out_grid = {"grid": [0.25, 0.25]}

ekr.interpolate(
    dummy,
    in_grid=in_grid,
    out_grid=out_grid,
    method="linear",
)

print('Saved regrid 0.25° to N320 and back test successfully.')