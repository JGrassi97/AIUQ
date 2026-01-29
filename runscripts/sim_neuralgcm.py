"""

"""

# Built-in/Generics
import os
import pickle
import logging
from datetime import datetime, timedelta

# Third party
import numpy as np
import xarray as xr
import neuralgcm
from dinosaur import horizontal_interpolation, spherical_harmonic, xarray_utils

# Local
from AIUQst_lib.functions import parse_arguments, read_config, normalize_out_vars
from AIUQst_lib.cards import read_model_card, read_std_version
from AIUQst_lib.variables import name_mapper_for_model

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def main() -> None:

    # Read config
    args = parse_arguments()
    config = read_config(args.config)

    _INI_DATA_PATH      = config.get('INI_DATA_PATH', "")
    _START_TIME         = config.get("START_TIME", "")
    _END_TIME           = config.get("END_TIME", "")
    _HPCROOTDIR         = config.get("HPCROOTDIR", "")
    _MODEL_NAME         = config.get("MODEL_NAME", "")
    _STD_VERSION        = config.get("STD_VERSION", "")
    _MODEL_CHECKPOINT   = config.get("MODEL_CHECKPOINT", "")
    _INNER_STEPS        = config.get("INNER_STEPS", 1)
    _RNG_KEY            = config.get("RNG_KEY", 1)
    _OUTPUT_PATH        = config.get("OUTPUT_PATH", "")
    _OUT_VARS           = config.get("OUT_VARS", [])
    _OUT_FREQ           = config.get("OUT_FREQ", "")
    _OUT_RES            = config.get("OUT_RES", "")
    _OUT_LEVS           = config.get("OUT_LEVS", "")

    # IC settings
    model_card = read_model_card(_HPCROOTDIR, _MODEL_NAME)
    standard_dict = read_std_version(_HPCROOTDIR, _STD_VERSION)

    # Format time settings
    start_date = datetime.strptime(_START_TIME, '%Y-%m-%d')
    input_end_date = datetime.strptime(_START_TIME, '%Y-%m-%d') + timedelta(days=1)
    end_date = datetime.strptime(_END_TIME, '%Y-%m-%d')

    days_to_run = (end_date - start_date).days + 1
    outer_steps = days_to_run * 24 // _INNER_STEPS
    delta_t = np.timedelta64(_INNER_STEPS, 'h')
    times = np.arange(outer_steps) * _INNER_STEPS  # in hours

    # Load model
    with open(_MODEL_CHECKPOINT, 'rb') as f:
        ckpt = pickle.load(f)
    model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)

    # Load data
    initial_conditions = xr.open_zarr(_INI_DATA_PATH, chunks=None).load()

    # Map names for model specifics
    mapper = name_mapper_for_model(model_card['variables'], standard_dict['variables'])
    initial_conditions = initial_conditions.rename(mapper)

    # Setup grid and regridder
    data_grid = spherical_harmonic.Grid(
        latitude_nodes=initial_conditions.sizes['latitude'],
        longitude_nodes=initial_conditions.sizes['longitude'],
        latitude_spacing=xarray_utils.infer_latitude_spacing(initial_conditions.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(initial_conditions.longitude),
    )
    regridder = horizontal_interpolation.ConservativeRegridder(
        data_grid, model.data_coords.horizontal, skipna=True
    )

    # Regrid and fill NaNs
    regridded = xarray_utils.regrid(initial_conditions, regridder)
    data = xarray_utils.fill_nan_with_nearest(regridded)

    # Prepare inputs and forcings
    inputs = model.inputs_from_xarray(data.isel(time=0))
    input_forcings = model.forcings_from_xarray(data.isel(time=1))
    initial_state = model.encode(inputs, input_forcings, int(_RNG_KEY))
    all_forcings = model.forcings_from_xarray(data.head(time=1))

    # Forecast
    final_state, predictions = model.unroll(
        initial_state,
        all_forcings,
        steps=outer_steps,
        timedelta=delta_t,
        start_with_input=True,
    )

    # Prepare time coordinates for output
    initial_time = initial_conditions.time.isel(time=0).values
    valid_time = initial_time + np.arange(outer_steps) * delta_t
    steps = np.arange(outer_steps) * delta_t

    # Convert predictions to xarray
    predictions_ds = model.data_to_xarray(predictions, times=valid_time)

    # Rename time coordinate and assign new coordinates
    predictions_ds = predictions_ds.rename({'time':'valid_time'})
    predictions_ds = predictions_ds.assign_coords(time=initial_time)
    predictions_ds = predictions_ds.assign_coords(step=("valid_time", steps))
    predictions_ds = predictions_ds.drop_vars('sim_time')

    # Format output variables and select
    output_vars = normalize_out_vars(_OUT_VARS)
    if 'all' not in output_vars:
        predictions_ds = predictions_ds[output_vars]

    # Format output frequency
    if _OUT_FREQ == "daily":
        predictions_ds = predictions_ds.resample(valid_time="1D").mean()

    # Format output resolution
    if _OUT_RES == "0.25":
        latitudes = np.arange(-90, 90.25, 0.25)
        longitudes = np.arange(0, 360, 0.25)
    elif _OUT_RES == "0.5":
        latitudes = np.arange(-90, 90.5, 0.5)
        longitudes = np.arange(0, 360, 0.5)
    elif _OUT_RES == "1":
        latitudes = np.arange(-90, 91, 1.0)
        longitudes = np.arange(0, 360, 1.0)
    elif _OUT_RES == "1.5":
        latitudes = np.arange(-90, 91.5, 1.5)
        longitudes = np.arange(0, 360, 1.5)
    elif _OUT_RES == "2":
        latitudes = np.arange(-90, 92, 2.0)
        longitudes = np.arange(0, 360, 2.0)
    else:
        latitudes = predictions_ds.latitude.values
        longitudes = predictions_ds.longitude.values
    
    if _OUT_RES in ["0.25", "0.5", "1", "1.5", "2"]:
        predictions_ds = predictions_ds.interp(latitude=latitudes, longitude=longitudes, method="linear")
    
    # Format output pressure levels
    if _OUT_LEVS != 'original':
        desired_levels = [
            int(plev)
            for plev in _OUT_LEVS.strip('[]').split(',')
        ]
        predictions_ds = predictions_ds.interp(level=desired_levels)
    
    
    # Ensure output path exists
    os.makedirs(_OUTPUT_PATH, exist_ok=True)

    # Save to NetCDF
    predictions_ds.to_netcdf(f"{_OUTPUT_PATH}/model_state-{_START_TIME}-{_END_TIME}-{_RNG_KEY}.nc")


if __name__ == "__main__":
    main()