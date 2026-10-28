"""Forecast contract and real CDO integration, without GPU/model dependencies."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runscripts'))
from AIUQst_lib.output import (
    output_metadata, resolve_output_variables, standardize_output, write_forecast,
)
from finalize_aifs import finalize
from postprocess_aifs import build_dataset_for_state

STANDARD = yaml.safe_load((ROOT / 'conf/AIUQ-st/v010/variables.yml').read_text())
INIT = np.datetime64('2020-01-01', 'ns')


def config(model='neuralgcm'):
    return dict(MODEL_NAME=model, MODEL_CHECKPOINT='/models/test.ckpt', STD_VERSION='v010',
                HPCROOTDIR=str(ROOT), RNG_KEY=2, IC_NAME='era5', OUT_FREQ='original', OUT_RES='original')


def card(model):
    return yaml.safe_load((ROOT / f'conf/cards/models/{model}_card.yml').read_text())


def forecast(native='temperature'):
    return xr.Dataset(
        {native: (('valid_time', 'level', 'latitude', 'longitude'),
                  np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4),
                  {'units': 'unknown', 'GRIB_edition': 2})},
        coords={'valid_time': INIT + np.array([6, 12]) * np.timedelta64(1, 'h'),
                'latitude': [90., 0., -90.], 'longitude': [0., 90., 180., 270.],
                'level': [500, 850], 'step': ('valid_time', [99., 100.]),
                'sim_time': ('valid_time', [1., 2.])},
        attrs={'history': 'upstream GRIB commands', 'junk': 'discard'})


@pytest.mark.parametrize('model,native', [('neuralgcm','temperature'), ('aifs','t'), ('ms-aurora','t')])
def test_model_contract_and_netcdf_roundtrip(tmp_path, model, native):
    names, canonical = resolve_output_variables(['Temperature'], card(model), STANDARD)
    assert names == [native] and canonical == ['t']
    original = forecast(native)
    ds = standardize_output(original, dict(zip(names, canonical)), STANDARD, INIT, output_metadata(config(model)))
    assert list(ds.data_vars) == ['t']
    assert ds.t.dims == ('valid_time', 'level', 'latitude', 'longitude')
    np.testing.assert_array_equal(ds.longitude, [-180, -90, 0, 90])
    np.testing.assert_array_equal(ds.latitude, [-90, 0, 90])
    np.testing.assert_array_equal(ds.level, [850, 500])
    np.testing.assert_array_equal(ds.t.sel(longitude=-180, latitude=-90, level=500),
                                  original[native].sel(longitude=180, latitude=-90, level=500))
    assert ds.time.ndim == 0 and ds.time.values == INIT
    np.testing.assert_array_equal(ds.step, ds.valid_time - ds.time)
    assert 'sim_time' not in ds and 'history' not in ds.attrs
    assert 'GRIB_edition' not in ds.t.attrs
    assert original[native].attrs['units'] == 'unknown'
    assert ds.t.attrs['units'] == 'K'
    assert ds.time.attrs['standard_name'] == 'forecast_reference_time'
    path = tmp_path / 'out.nc'
    write_forecast(ds, path)
    with xr.open_dataset(path, decode_timedelta=True) as reopened:
        xr.testing.assert_equal(reopened, ds)
        for key in ('model', 'model_checkpoint', 'initial_conditions', 'initialization_time',
                    'date_created', 'aiuq_version', 'aiuq_st_version', 'source_revision', 'member'):
            assert key in reopened.attrs
        assert reopened.attrs['model'] == model
        assert reopened.attrs['member'] == '2'
    assert not path.with_name('out.nc.partial').exists()


def test_missing_unknown_duplicate_variables_fail():
    with pytest.raises(ValueError, match='Unsupported'):
        resolve_output_variables(['bogus'], card('neuralgcm'), STANDARD)
    with pytest.raises(ValueError, match='Duplicate'):
        resolve_output_variables(['t', 'Temperature'], card('neuralgcm'), STANDARD)
    with pytest.raises(ValueError, match='missing'):
        standardize_output(forecast(), {'q': 'q'}, STANDARD, INIT, output_metadata(config()))


def test_aifs_geopotential_selection_is_not_orography():
    assert resolve_output_variables(['z'], card('aifs'), STANDARD) == (['z'], ['z'])
    assert resolve_output_variables(['z'], card('ms-aurora'), STANDARD) == (['z'], ['z'])
    with pytest.raises(ValueError, match='orography'):
        resolve_output_variables(['orog'], card('ms-aurora'), STANDARD)


def test_daily_recomputes_step_and_keeps_initialization():
    ds = forecast().resample(valid_time='1D').mean()
    meta = output_metadata(config())
    meta['output_frequency'] = 'daily'
    ds = standardize_output(ds, {'temperature': 't'}, STANDARD, INIT, meta)
    assert ds.time.values == INIT
    assert ds.step.values[0] == np.timedelta64(0, 'h')
    assert ds.t.attrs['cell_methods'] == 'valid_time: mean'


@pytest.mark.parametrize('coord,values', [('longitude', [0., 90., 180., 360.]),
                                          ('latitude', [90., np.nan, -90.])])
def test_invalid_grid_rejected(coord, values):
    with pytest.raises(ValueError, match='finite and unique'):
        standardize_output(forecast().assign_coords({coord: values}), {'temperature': 't'},
                           STANDARD, INIT, output_metadata(config()))


def test_aifs_actual_state_dates_and_original_levels():
    state = {'date': datetime(2020, 1, 1, 6, tzinfo=timezone.utc),
             'fields': {'t_500': np.ones(4), 't_850': np.full(4, 2), '2t': np.full(4, 3)}}
    ds = build_dataset_for_state(state, ['t', '2t'], None)
    assert ds.time.values[0] == np.datetime64('2020-01-01T06:00')
    assert ds.sizes['points'] == 4
    np.testing.assert_array_equal(ds.level, [500, 850])
    selected = build_dataset_for_state(state, ['t'], [850])
    assert selected.sizes['level'] == 1


@pytest.mark.skipif(shutil.which('cdo') is None, reason='CDO is not installed')
def test_real_cdo_aifs_pipeline(tmp_path):
    # Small synthetic unstructured global grid, same dimensions as Anemoi outputs.
    ds = xr.Dataset({'t': (('valid_time', 'level', 'points'), np.ones((2, 2, 12), dtype=np.float32))},
                    coords={'valid_time': INIT + np.array([6, 12]) * np.timedelta64(1, 'h'),
                            'level': [500, 850], 'points': np.arange(12)})
    reference = standardize_output(ds, {'t': 't'}, STANDARD, INIT,
                                   output_metadata(config('aifs')), unstructured=True)
    ref = tmp_path / 'reference.nc'
    grid = tmp_path / 'grid.txt'
    gridded = tmp_path / 'gridded.nc'
    remapped = tmp_path / 'remapped.nc'
    final = tmp_path / 'final.nc'
    write_forecast(reference, ref)
    grid.write_text('gridtype = unstructured\ngridsize = 12\nxname = lon\nyname = lat\n'
                    'xunits = degrees_east\nyunits = degrees_north\n'
                    'xvals = 0 90 180 270 0 90 180 270 0 90 180 270\n'
                    'yvals = -90 -90 -90 -90 0 0 0 0 90 90 90 90\n')
    for args in (['-setgrid,' + str(grid), str(ref), str(gridded)],
                 ['-f', 'nc4', 'remapdis,r12x7', str(gridded), str(remapped)]):
        subprocess.run(['cdo'] + args, check=True, capture_output=True, text=True)
    finalize(remapped, ref, final, config('aifs'))
    with xr.open_dataset(final, decode_timedelta=True) as result:
        assert result.t.dims == ('valid_time', 'level', 'latitude', 'longitude')
        assert result.longitude.min() == -180 and result.longitude.max() < 180
        assert result.time.values == INIT
        np.testing.assert_array_equal(result.valid_time, reference.valid_time)
        np.testing.assert_array_equal(result.step, result.valid_time - result.time)
        np.testing.assert_allclose(result.t, 1.)
        assert result.attrs['model'] == 'aifs'
        assert 'CDO' not in result.attrs and 'history' not in result.attrs
        assert result.attrs['date_created'] == reference.attrs['date_created']


def test_exported_revision_is_used_without_git_checkout(tmp_path):
    from AIUQst_lib.output import source_revision
    (tmp_path / 'conf').mkdir()
    (tmp_path / 'conf/source_revision.txt').write_text('abc123-dirty\n')
    assert source_revision(tmp_path) == 'abc123-dirty'
    assert source_revision(tmp_path / 'missing') == 'unknown'


def test_surface_output_and_coordinate_aliases(tmp_path):
    original = forecast('2t').isel(level=0, drop=True).rename({'latitude': 'lat', 'longitude': 'lon'})
    ds = standardize_output(original, {'2t': 't2m'}, STANDARD, INIT, output_metadata(config('aifs')))
    assert ds.t2m.dims == ('valid_time', 'latitude', 'longitude')
    assert 'level' not in ds
    assert ds.t2m.attrs['units'] == 'K'
    write_forecast(ds, tmp_path / 'surface.nc')


def test_aurora_uses_prediction_timestamps():
    # Load just the conversion function, keeping this test independent of torch
    # and Aurora/checkpoint installations.
    import ast
    import logging
    from types import SimpleNamespace
    source = ast.parse((ROOT / 'runscripts/sim_ms-aurora.py').read_text())
    fn = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == '_preds_to_dataset')
    namespace = dict(np=np, xr=xr, Batch=object, logging=logging)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<aurora converter>', 'exec'), namespace)

    class Tensor:
        def __init__(self, array):
            self.array = np.asarray(array)
        def cpu(self):
            return self
        def numpy(self):
            return self.array
        def __getitem__(self, item):
            return Tensor(self.array[item])

    times = [datetime(2020, 1, 1, h) for h in (1, 3, 6)]
    preds = [SimpleNamespace(
        metadata=SimpleNamespace(lat=Tensor([0, 1]), lon=Tensor([0, 180]),
                                 atmos_levels=(850,), time=(time,)),
        surf_vars={'2t': Tensor(np.ones((1, 1, 2, 2)))}, atmos_vars={}) for time in times]
    ds = namespace['_preds_to_dataset'](preds, ['2t'], INIT)
    np.testing.assert_array_equal(ds.valid_time, np.array(times, dtype='datetime64[ns]'))
    np.testing.assert_array_equal(ds.step, ds.valid_time - INIT)
