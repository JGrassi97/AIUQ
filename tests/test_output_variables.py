"""Output variables: AIUQ-st outputs.yml, the per-checkpoint output lists of the model cards and the SIM helpers."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runscripts"))
from AIUQst_lib.cards import read_std_version  # noqa: E402
from AIUQst_lib.output import time_cell, variable_attributes  # noqa: E402
from AIUQst_lib.variables import (checkpoint_entry, deaccumulate, output_specs,  # noqa: E402
                                  resample_daily, resolve_outputs)

STD = read_std_version(str(ROOT), "v100")
MODELS = ("aifs", "ms-aurora", "neuralgcm", "weathernext")


def card(name):
    return yaml.safe_load((ROOT / f"conf/cards/models/{name}_card.yml").read_text())


# What the checkpoints produce, read from the checkpoints on FELIPE (2026-11-20):
# AIFS: Checkpoint.variable_categories(), prognostic + diagnostic fields (model names, levels stripped).
AIFS_PRODUCED = {
    "aifs-single-mse-1.1.ckpt": {"10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "stl1", "stl2", "swvl1",
                                 "swvl2", "q", "t", "u", "v", "w", "z", "100u", "100v", "cp", "hcc", "lcc", "mcc",
                                 "ro", "sf", "ssrd", "strd", "tcc", "tp"},
    "aifs-ens-crps-1.0.ckpt": {"10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "stl1", "stl2", "q", "t", "u",
                               "v", "w", "z", "100u", "100v", "cp", "hcc", "lcc", "mcc", "ro", "sf", "ssrd",
                               "strd", "tcc", "tp"},
    "aifs-ens-crps-2.0.ckpt": {"10u", "10v", "2d", "2t", "msl", "skt", "sp", "tcw", "sd", "stl1", "stl2", "swvl1",
                               "swvl2", "cdww", "mwd", "mwp", "swh", "h1012", "h1214", "h1417", "h1721", "h2125",
                               "h2530", "q", "t", "u", "v", "w", "z", "100u", "100v", "cp", "hcc", "lcc", "mcc",
                               "ro", "sf", "snowc", "ssrd", "strd", "tcc", "tp"},   # mwd from cos_mwd/sin_mwd
}
# Aurora: surf_vars (without the prescribed insolation) and atmos_vars of the package classes.
AURORA_V1 = {"2t", "10u", "10v", "msl", "z", "u", "v", "t", "q"}
AURORA_V15 = AURORA_V1 | {"2d", "tcwv", "tcc", "100u", "100v", "sp", "lcc", "mcc", "hcc", "skt", "stl1", "swvl1",
                          "ci", "scaled_sd", "i10fg", "blh", "uvb_1h", "ssrd_1h", "ttr_1h", "scaled_tp_1h",
                          "scaled_sf_1h"}
AURORA_PRODUCED = {"aurora-0.25-pretrained.ckpt": AURORA_V1, "aurora-0.25-finetuned.ckpt": AURORA_V1,
                   "aurora-0.25-v1.5.ckpt": AURORA_V15, "aurora-0.25-v1.5-ensemble.ckpt": AURORA_V15}
# NeuralGCM: data_to_xarray of a 2-step unroll.
NGCM_BASE = {"geopotential", "temperature", "u_component_of_wind", "v_component_of_wind", "specific_humidity",
             "specific_cloud_liquid_water_content", "specific_cloud_ice_water_content"}
NGCM_PRECIP = NGCM_BASE | {"precipitation_cumulative_mean", "evaporation"}
NGCM_PRODUCED = {"deterministic_0_7_deg.pkl": NGCM_BASE, "deterministic_1_4_deg.pkl": NGCM_BASE,
                 "deterministic_2_8_deg.pkl": NGCM_BASE, "stochastic_1_4_deg.pkl": NGCM_BASE,
                 "stochastic_precip_2_8_deg.pkl": NGCM_PRECIP, "stochastic_evap_2_8_deg.pkl": NGCM_PRECIP,
                 "stochasti_evap_2_8_deg.pkl": NGCM_PRECIP}
# WeatherNext: task.target_variables of the configs shipped with weathernext v0.3.0 (weathernext2/configs/*.json),
# without the cyclone fields (cyclone_*), which are not written.
WN_MINI = {"temperature", "geopotential", "u_component_of_wind", "v_component_of_wind", "vertical_velocity",
           "specific_humidity", "2m_temperature", "mean_sea_level_pressure", "10m_u_component_of_wind",
           "10m_v_component_of_wind", "sea_surface_temperature", "total_precipitation_6hr"}
WN_PRODUCED = {"WeatherNext2_2025": WN_MINI | {"100m_u_component_of_wind", "100m_v_component_of_wind"},
               "WeatherNextCyclones_Mini_2024": WN_MINI}
PRODUCED = {"aifs": AIFS_PRODUCED, "ms-aurora": AURORA_PRODUCED, "neuralgcm": NGCM_PRODUCED,
            "weathernext": WN_PRODUCED}


def test_outputs_standard_is_separate_from_the_ic_variables():
    variables = STD["variables"]["data"]
    outputs = STD["outputs"]["data"]
    assert outputs, "outputs.yml not read"
    assert not set(variables) & set(outputs)
    names = [s["short_name"] for s in list(variables.values()) + list(outputs.values())]
    assert len(names) == len(set(names)), "short names must be unique across variables.yml and outputs.yml"
    for spec in outputs.values():
        assert {"short_name", "long_name", "units", "type"} <= set(spec)
        assert not spec.get("optional"), "output-only variables are never asked to the ICs"


def test_older_standard_versions_have_no_outputs_file():
    assert read_std_version(str(ROOT), "v010")["outputs"] == {"data": {}}


@pytest.mark.parametrize("model", MODELS)
def test_every_checkpoint_output_resolves_to_a_produced_model_field(model):
    c = card(model)
    specs = output_specs(c, STD)
    for ckpt, produced in PRODUCED[model].items():
        listed = checkpoint_entry(c, ckpt).get("outputs")
        assert listed, f"{model}: no output list for {ckpt}"
        for name in listed:
            assert name in specs, f"{model}/{ckpt}: {name} has no card mapping"
            assert specs[name]["model"] in produced, f"{model}/{ckpt}: {name} -> {specs[name]['model']} not produced"


@pytest.mark.parametrize("model", MODELS)
def test_every_produced_field_is_offered(model):
    """Nothing the checkpoints produce is left out of the output lists (static forcings are not outputs)."""
    c = card(model)
    specs = output_specs(c, STD)
    for ckpt, produced in PRODUCED[model].items():
        offered = {specs[n]["model"] for n in checkpoint_entry(c, ckpt)["outputs"]}
        assert produced - offered == set(), f"{model}/{ckpt}"


def test_weathernext_checkpoints_name_config_and_grid():
    c = card("weathernext")
    for ckpt, config, resolution in [("WeatherNext2_2025", "WeatherNext2", 0.25),
                                     ("WeatherNextCyclones_Mini_2024", "WeatherNextCyclones_Mini", 1.0)]:
        entry = checkpoint_entry(c, f"/support/models/weathernext/{ckpt}")     # a folder, by base name
        assert entry["config"] == config and float(entry["resolution"]) == resolution


def test_checkpoint_levels():
    c = card("aifs")
    assert 10 not in checkpoint_entry(c, "aifs-ens-crps-1.0.ckpt")["levels"]
    assert 10 in checkpoint_entry(c, "aifs-ens-crps-2.0.ckpt")["levels"]
    assert checkpoint_entry(c, "aifs-ens-crps-2.0.ckpt")["image"] == "image_anemoi-2.sif"
    assert checkpoint_entry(c, "aifs-single-mse-1.1.ckpt")["mask_over_sea"] == []


def test_resolve_outputs_skips_what_the_checkpoint_cannot_write():
    c = card("neuralgcm")
    sel, unknown = resolve_outputs(["t", "tp", "t2m", "Temperature", "avg_ie"], c, STD,
                                   "/x/models/neuralgcm/deterministic_2_8_deg.pkl")
    assert [n for n, _ in sel] == ["t"] and unknown == ["tp", "t2m", "avg_ie"]
    sel, _ = resolve_outputs(["tp", "avg_ie"], c, STD, "stochastic_precip_2_8_deg.pkl")
    tp = dict(sel)["tp"]
    assert tp["model"] == "precipitation_cumulative_mean" and tp["deaccumulate"] and tp["accumulated"]


def test_aurora_hourly_totals_become_mean_rates():
    specs = output_specs(card("ms-aurora"), STD)
    assert specs["avg_tprate"]["model"] == "scaled_tp_1h"
    assert specs["avg_tprate"]["scale"] == pytest.approx(1000 / 3600)
    assert specs["avg_sdswrf"]["scale"] == pytest.approx(1 / 3600)
    assert specs["sd"]["model"] == "scaled_sd" and specs["stl1"]["model"] == "stl1"
    attrs = variable_attributes("avg_tprate", specs["avg_tprate"])
    assert attrs["units"] == "kg m-2 s-1" and attrs["standard_name"] == "precipitation_flux"
    # a mean over the hour before each output time; as daily output, the mean of those over the day
    assert time_cell(specs["avg_tprate"], step_hours=6)[:2] == ("valid_time: mean", (-1.0, 0.0))
    assert time_cell(specs["avg_tprate"], daily=True, step_hours=6)[:2] == ("valid_time: mean (interval: 6 hours)", (0.0, 24.0))
    tp = output_specs(card("aifs"), STD)["tp"]
    assert time_cell(tp, step_hours=6)[:2] == ("valid_time: sum", (-6.0, 0.0))
    assert time_cell(tp, daily=True, step_hours=6)[:2] == ("valid_time: sum", (0.0, 24.0))


def test_aifs_names():
    specs = output_specs(card("aifs"), STD)
    assert specs["u100"]["model"] == "100u" and specs["fscov"]["model"] == "snowc"
    assert specs["tp"]["accumulated"] and not specs["tcc"]["accumulated"]


def test_deaccumulate_running_total():
    total = np.array([[0.0, 1.0], [1.0, 3.0], [4.0, 3.0]])
    assert np.allclose(deaccumulate(total), [[0, 1], [1, 2], [3, 0]])
    assert np.allclose(deaccumulate(total[1:], previous=total[0]), [[1, 2], [3, 0]])


def test_daily_totals_use_periods_ending_in_the_day():
    t = pd.date_range("2020-01-01T06", periods=8, freq="6h")          # 06 .. next day 00 (x2)
    ds = xr.Dataset({"tp": ("valid_time", np.ones(8))}, coords={"valid_time": t})
    daily = resample_daily(ds, "tp", accumulated=True)
    assert list(daily.tp.values) == [4.0, 4.0]
    assert [str(t)[:10] for t in daily.valid_time.values] == ["2020-01-01", "2020-01-02"]
    # a run starting at 18 UTC (ECMWF analysis) and ending mid-day: partial days, no empty ones
    t = pd.date_range("2020-01-01T18", periods=6, freq="6h")          # 18, 00, 06, 12, 18, 00
    ds = xr.Dataset({"tp": ("valid_time", np.ones(6))}, coords={"valid_time": t, "step": ("valid_time", np.arange(6))})
    daily = resample_daily(ds, "tp", accumulated=True)
    assert list(daily.tp.values) == [2.0, 4.0] and "step" not in daily.coords
    mean = resample_daily(ds, "tp", accumulated=False)
    assert list(mean.tp.values) == [1.0, 1.0, 1.0]
