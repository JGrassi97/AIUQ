"""AIFS input planning from the checkpoint metadata, wave helpers and the AIUQ-st v100 contracts."""
from pathlib import Path
import sys

import numpy as np
import pytest
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runscripts"))
from AIUQst_lib.variables import drop_optional_missing, optional_variables  # noqa: E402
from AIUQst_lib.waves import (WAVE_PERIOD_BANDS, cos_sin_to_direction,  # noqa: E402
                              direction_to_cos_sin, wave_period_bands)
import ics_aifs  # noqa: E402

STD = ROOT / "conf/AIUQ-st/v100"


def standard_variables():
    return yaml.safe_load((STD / "variables.yml").read_text())["data"]


def model_card(name="aifs"):
    return yaml.safe_load((ROOT / f"conf/cards/models/{name}_card.yml").read_text())


# Input names of the checkpoints as anemoi reports them (Checkpoint.variables_from_input)
LEVELS_V1 = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]
V1_INPUTS = ["10u", "10v", "2d", "2t", "lsm", "msl", "sdor", "skt", "slor", "sp", "stl1", "stl2", "tcw", "z"] + [
    f"{v}_{p}" for v in ("q", "t", "u", "v", "w", "z") for p in LEVELS_V1]
V2_INPUTS = (["10u", "10v", "2d", "2t", "lsm", "msl", "sdor", "skt", "slor", "sp", "stl1", "stl2", "swvl1",
              "swvl2", "tcw", "sd", "z", "wmb", "h1012", "h1214", "h1417", "h1721", "h2125", "h2530",
              "cos_mwd", "sin_mwd", "cdww", "mwp", "swh"]
             + [f"{v}_{p}" for v in ("t", "u", "v", "w", "z") for p in LEVELS_V1 + [10]]
             + [f"q_{p}" for p in LEVELS_V1 if p >= 100])


# ---------------------------------------------------------------- standard and cards

def test_v100_extends_v010_with_optional_wave_fields():
    v010 = yaml.safe_load((ROOT / "conf/AIUQ-st/v010/variables.yml").read_text())["data"]
    v100 = standard_variables()
    assert set(v010) <= set(v100)
    for code, spec in v010.items():
        assert v100[code] == spec, f"{code} changed between v010 and v100"
    new = {code: spec for code, spec in v100.items() if code not in v010}
    assert {spec["short_name"] for spec in new.values()} == {
        "h1012", "h1214", "h1417", "h1721", "h2125", "h2530", "wmb", "swh", "mwd", "mwp", "cdww"}
    assert all(spec["type"] == "wave" and spec["optional"] for spec in new.values())
    assert optional_variables(v100) == set(new)
    levels = yaml.safe_load((STD / "pressure_levels.yml").read_text())
    assert levels["AIUQ-st"] == "v100" and 10 in levels["values"]


def test_runtime_selects_v100():
    assert 'STD_VERSION: "v100"' in (ROOT / "templates/config.yml").read_text()
    assert 'STD_VERSION="v100"' in (ROOT / "templates/postprocess_aifs.sh").read_text()


@pytest.mark.parametrize("name", [p.name.removesuffix("_card.yml") for p in (ROOT / "conf/cards/models").glob("*_card.yml")])
def test_model_card_codes_exist_in_the_standard(name):
    unknown = set(model_card(name)["variables"]) - set(standard_variables())
    assert not unknown, f"{name}: codes {unknown} are not in AIUQ-st v100"


@pytest.mark.parametrize("name", ["era5", "ecmwf-an", "eerie", "static", "climatology"])
def test_ic_card_codes_exist_in_the_standard(name):
    card = yaml.safe_load((ROOT / f"conf/cards/ics/{name}_card.yml").read_text())
    unknown = set(card["variables"]) - set(standard_variables())
    assert not unknown, f"{name}: codes {unknown} are not in AIUQ-st v100"


def test_drop_optional_missing_keeps_required_variables():
    variables = standard_variables()
    missing = {140229: variables[140229], 167: variables[167]}
    assert drop_optional_missing(missing, variables) == {167: variables[167]}
    assert drop_optional_missing({140229: variables[140229]}, variables) is None
    assert drop_optional_missing(None, variables) is None


# ---------------------------------------------------------------- input planning

def test_v1_inputs_map_to_the_standard_without_waves():
    plan = ics_aifs.plan_input_fields(V1_INPUTS, model_card(), standard_variables())
    assert not plan["missing"]
    assert plan["levels"] == LEVELS_V1
    assert plan["fields"]["z"] == ("orog", None, None)
    assert plan["fields"]["z_500"] == ("z", 500, None)
    assert plan["fields"]["2t"] == ("t2m", None, None)
    assert plan["fields"]["stl2"] == ("sot", None, ("soil_level", 1))
    assert not {s for s, _, _ in plan["fields"].values()} & {"swh", "mwd", "sd", "swvl1"}


def test_v2_inputs_add_levels_snow_soil_moisture_and_waves():
    plan = ics_aifs.plan_input_fields(V2_INPUTS, model_card(), standard_variables())
    assert not plan["missing"]
    assert plan["levels"] == LEVELS_V1 + [10]
    assert plan["fields"]["t_10"] == ("t", 10, None) and "q_10" not in plan["fields"]
    assert plan["fields"]["cos_mwd"] == ("mwd", None, ("cos", None))
    assert plan["fields"]["sin_mwd"] == ("mwd", None, ("sin", None))
    assert plan["fields"]["h1012"] == ("h1012", None, None)
    assert plan["fields"]["swvl1"] == ("swvl1", None, None) and plan["fields"]["sd"] == ("sd", None, None)


def test_inputs_from_categories_keep_prognostic_and_input_forcings():
    categories = {"cos_latitude": ["computed", "constant", "forcing"], "insolation": ["computed", "forcing"],
                  "lsm": ["constant", "forcing"], "wmb": ["constant", "forcing"], "z": ["constant", "forcing"],
                  "tp": ["accumulation", "diagnostic"], "100u": ["diagnostic"],
                  "2t": ["prognostic"], "w_10": ["prognostic"], "cos_mwd": ["prognostic"]}
    assert ics_aifs.inputs_from_categories(categories) == ["2t", "cos_mwd", "lsm", "w_10", "wmb", "z"]


def test_unmapped_inputs_are_reported():
    plan = ics_aifs.plan_input_fields(["2t", "tp", "x_500"], model_card(), standard_variables())
    assert plan["missing"] == ["tp", "x_500"]


def test_checkpoint_options_merge_defaults():
    card = model_card()
    v1 = ics_aifs.checkpoint_options(card, "/x/aifs-ens-crps-1.0.ckpt")
    assert v1["mask_over_sea"] == [] and v1["image"] == "image_anemoi.sif"
    v2 = ics_aifs.checkpoint_options(card, "/x/aifs-ens-crps-2.0.ckpt")
    assert v2["mask_over_sea"] == ["sd", "swvl1", "swvl2"] and v2["image"] == "image_anemoi-2.sif"


def test_split_field_name():
    assert ics_aifs.split_field_name("t_500") == ("t", 500)
    assert ics_aifs.split_field_name("cos_mwd") == ("cos_mwd", None)
    assert ics_aifs.split_field_name("10u") == ("10u", None)


def test_shell_picks_the_image_like_the_card():
    import subprocess
    script = (ROOT / "lib/functions.sh").read_text()
    for ckpt, expected in [("aifs-single-mse-1.1.ckpt", "image_anemoi.sif"),
                           ("aifs-ens-crps-1.0.ckpt", "image_anemoi.sif"),
                           ("aifs-ens-crps-2.0.ckpt", "image_anemoi-2.sif")]:
        out = subprocess.run(["bash", "-c", script + f'\naifs_sif_name "{ckpt}"'], capture_output=True, text=True, check=True)
        assert out.stdout.strip() == expected == ics_aifs.checkpoint_options(model_card(), ckpt)["image"]


# ---------------------------------------------------------------- waves

def test_direction_round_trip():
    direction = np.array([0., 45., 180., 270., 359.5])
    cos_c, sin_c = direction_to_cos_sin(direction)
    np.testing.assert_allclose(cos_sin_to_direction(cos_c, sin_c), direction, atol=1e-9)
    da = xr.DataArray(direction, dims="x")
    back = cos_sin_to_direction(*direction_to_cos_sin(da))
    assert isinstance(back, xr.DataArray)


def test_wave_period_bands_assign_partition_energy():
    # three points: land (all NaN), swell at 15 s only, wind waves at 11 s + swell at 30 s
    heights = [np.array([np.nan, np.nan, 1.0]),          # wind waves
               np.array([np.nan, 2.0, 3.0]),             # first swell
               np.array([np.nan, np.nan, np.nan])]       # second swell (no energy)
    periods = [np.array([np.nan, np.nan, 11.0]),
               np.array([np.nan, 15.0, 30.0]),
               np.array([np.nan, np.nan, np.nan])]
    bands = wave_period_bands(list(zip(heights, periods)))
    assert set(bands) == set(WAVE_PERIOD_BANDS)
    assert all(np.isnan(b[0]) for b in bands.values())
    np.testing.assert_allclose(bands["h1417"], [np.nan, 2.0, 0.0])
    np.testing.assert_allclose(bands["h1012"], [np.nan, 0.0, 1.0])
    np.testing.assert_allclose(bands["h2530"], [np.nan, 0.0, 3.0])   # 30 s belongs to the last band
    assert bands["h1214"][1] == 0.0


def test_wave_period_bands_with_dataarrays_keep_attrs_and_mask():
    grid = {"latitude": [0., 1.], "longitude": [0.]}
    h = xr.DataArray([[np.nan], [1.5]], dims=("latitude", "longitude"), coords=grid)
    p = xr.DataArray([[np.nan], [12.0]], dims=("latitude", "longitude"), coords=grid)
    bands = wave_period_bands([(h, p)])
    assert bands["h1214"].sel(latitude=1).item() == 1.5 and np.isnan(bands["h1214"].sel(latitude=0).item())
    assert bands["h1012"].sel(latitude=1).item() == 0.0
    assert bands["h1214"].attrs["period_band_s"] == [12.0, 14.0]
