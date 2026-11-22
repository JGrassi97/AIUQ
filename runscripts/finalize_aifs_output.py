"""
Final AIUQ output files of an AIFS member, written after the regridding.

AIFS predicts on the reduced Gaussian grid N320. SIM writes each variable on that grid
(``out-...-<var>_temp.nc``), the POSTPROCESS job gives it the grid description and regrids it with cdo
(``out-...-<var>_grid.nc``), and this script turns the cdo file into the final file
(``out-...-<var>.nc``) in the AIUQ output convention (``AIUQst_lib/output.py``): cdo keeps the values but
not the initial time of the forecast, the lead time, the names and the attributes.

usage: finalize_aifs_output.py -c <job config> --file VAR GRID_FILE SIM_FILE FINAL_FILE [--file ...]

The two intermediate files are removed once the final file is complete.
"""

# Built-in/Generics
import argparse
import logging
import os

# Third party
import xarray as xr

# Local
from AIUQst_lib.cards import read_model_card, read_std_version
from AIUQst_lib.functions import read_config
from AIUQst_lib.output import RunInfo, write_output
from AIUQst_lib.variables import resolve_outputs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def finalize(var: str, grid_file: str, sim_file: str, final_file: str, spec: dict, run: RunInfo,
             daily: bool, step_hours: float) -> str:
    """One variable: the cdo file with the metadata of the SIM file, as an AIUQ output file."""
    with xr.open_dataset(sim_file) as native:
        reference_time = native["time"].values            # initial time of the forecast (scalar coordinate)
    with xr.open_dataset(grid_file) as regridded:
        data = regridded[var]
        # Without a regridding (OUT_RES original) cdo names the horizontal coordinates lat/lon along one
        # dimension; the writer takes them as they are.
        write_output(final_file, data, var, spec, run, daily=daily, step_hours=step_hours,
                     reference_time=reference_time)
    for path in (grid_file, sim_file):
        os.remove(path)
    return final_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("-c", "--config", required=True, help="job configuration (YAML)")
    parser.add_argument("--file", nargs=4, action="append", default=[],
                        metavar=("VAR", "GRID_FILE", "SIM_FILE", "FINAL_FILE"))
    args = parser.parse_args()
    config = read_config(args.config)

    model_card = read_model_card(config.get("HPCROOTDIR", ""), config.get("MODEL_NAME", ""))
    standard_dict = read_std_version(config.get("HPCROOTDIR", ""), config.get("STD_VERSION", ""))
    names = [var for var, *_ in args.file]
    selected, unknown = resolve_outputs(names, model_card, standard_dict, config.get("MODEL_CHECKPOINT", ""))
    if unknown:
        raise ValueError(f"Not outputs of this checkpoint in the model card: {unknown}")
    specs = dict(selected)
    run = RunInfo.from_config(config)
    daily = str(config.get("OUT_FREQ", "")).strip() == "daily"
    step_hours = float(config.get("INNER_STEPS", 6))

    for var, grid_file, sim_file, final_file in args.file:
        finalize(var, grid_file, sim_file, final_file, specs[var], run, daily, step_hours)
        logging.info(f"Saved {final_file}")


if __name__ == "__main__":
    main()
