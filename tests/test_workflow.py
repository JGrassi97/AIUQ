"""The FELIPE workflow keeps the HPC scratch small: initial conditions at most two start dates ahead, the outputs of
each start date moved to the output store by its STORE job, ARCHIVE last (conf/jobs/FELIPE/models/, templates/).

The job graph these files give was checked with Autosubmit 4.1.17 itself: ``PREPARE_ICS`` of a date depends on every
``SIM`` job (members x chunks) two dates earlier and on nothing for the first two dates, ``STORE`` of a date on every
job of that date, ``ARCHIVE`` on every ``STORE``. (A ``-N`` distance to a section that runs once per date is ignored
by Autosubmit 4.1, which is why the limit points at ``SIM``, and a placeholder is not accepted as the distance.)
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
MODELS = sorted(p.name for p in (ROOT / "conf/jobs/FELIPE/models").iterdir() if p.is_dir())
PLACEHOLDER = re.compile(r"%[A-Z0-9_.]+%")


def jobs(model):
    return yaml.safe_load((ROOT / f"conf/jobs/FELIPE/models/{model}/{model}.yml").read_text())["JOBS"]


def deps(value):
    return set(value.split()) if isinstance(value, str) else set(value or {})


@pytest.mark.parametrize("model", MODELS)
def test_initial_conditions_at_most_two_dates_ahead(model):
    assert "SIM-2" in deps(jobs(model)["PREPARE_ICS"].get("DEPENDENCIES"))


@pytest.mark.parametrize("model", MODELS)
def test_each_start_date_is_stored_when_its_jobs_are_done(model):
    j = jobs(model)
    store = j["STORE"]
    assert store["FILE"] == "templates/store.sh"
    assert str(store["PLATFORM"]).lower() == "local" and str(store["RUNNING"]).lower() == "date"
    needed = {"SIM"} | ({"POSTPROCESS", "CLEAN"} & set(j))              # every job of the date that writes or cleans
    assert deps(store["DEPENDENCIES"]) == needed
    assert j["ARCHIVE"]["DEPENDENCIES"] == {"STORE": {"DATES_TO": "all"}}


@pytest.mark.parametrize("name", ["sim_neuralgcm.sh", "sim_ms-aurora.sh", "sim_weathernext.sh", "postprocess_aifs.sh"])
def test_jobs_that_write_outputs_list_them_for_store(name):
    text = (ROOT / "templates" / name).read_text()
    assert "AIUQ_OUTPUT_RECORD=" in text
    assert "store_pending/%SDATE%/${JOBNAME}.txt" in text


@pytest.mark.parametrize("path", ["templates/store.sh", "templates/archive.sh", "lib/output_store.sh"])
def test_store_scripts(path):
    text = (ROOT / path).read_text()
    # Autosubmit replaces %NAME% in templates: any other percent sign would be read as a placeholder.
    assert "%" not in PLACEHOLDER.sub("", text)
    if path.startswith("lib/"):
        assert not PLACEHOLDER.search(text)                                # sourced as is, never rendered
    if shutil.which("bash"):
        subprocess.run(["bash", "-n", str(ROOT / path)], check=True)
