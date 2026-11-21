"""The support matrix (docs/support-matrix.yml) is complete and the README table is in sync with it."""
from pathlib import Path
import importlib.util

import yaml

ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "docs" / "support-matrix.yml"
STATUSES = {"available", "wip", "not_implemented"}


def _generator():
    spec = importlib.util.spec_from_file_location("generate_table", ROOT / "docs" / "generate_table.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_checkpoint_has_a_valid_status_for_every_ic():
    data = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    ids = [c["id"] for m in data["models"] for c in m["checkpoints"]]
    assert len(ids) == len(set(ids)), "duplicate checkpoint ids"
    assert set(data["status"]) == set(data["ics"])
    for ic in data["ics"]:
        assert set(data["status"][ic]) == set(ids), f"{ic}: statuses do not match the checkpoints"
        assert set(data["status"][ic].values()) <= STATUSES, f"{ic}: unknown status"


def test_model_labels_match_model_cards():
    data = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    cards = {p.name.removesuffix("_card.yml") for p in (ROOT / "conf" / "cards" / "models").glob("*_card.yml")}
    for model in data["models"]:
        implemented = any(
            data["status"][ic][c["id"]] != "not_implemented" for ic in data["ics"] for c in model["checkpoints"]
        )
        if implemented:
            assert model["label"] in cards, f"{model['label']} has no model card"


def test_readme_table_is_generated_from_the_matrix():
    gen = _generator()
    html = gen.build_html(gen.load_master(MATRIX))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    block = readme.split(gen.START, 1)[1].split(gen.END, 1)[0]
    assert block.strip() == html.strip(), "README table is stale: run python docs/generate_table.py docs/support-matrix.yml README.md"


def test_launcher_reads_every_status_of_the_matrix(monkeypatch):
    """The Launcher takes its support badges from the matrix: every IC source and status must be understood."""
    import sys
    import yaml

    monkeypatch.syspath_prepend(str(ROOT / "tools" / "dashboard"))
    monkeypatch.setenv("AIUQ_STATE_DIR", str(ROOT / ".pytest_cache" / "aiuq-dash-state"))
    for name in [n for n in sys.modules if n == "aiuq_dash" or n.startswith("aiuq_dash.")]:
        monkeypatch.delitem(sys.modules, name)
    from aiuq_dash import launcher

    matrix = yaml.safe_load(MATRIX.read_text(encoding="utf-8"))
    assert set(matrix["status"]) == set(launcher.MATRIX_ICS), "new IC source: add it to launcher.MATRIX_ICS"
    table = launcher.parse_support_matrix(matrix)
    files = [ck["label"] for m in matrix["models"] for ck in m["checkpoints"]]
    assert set(table) == set(files)
    assert all(set(row) == set(launcher.MATRIX_ICS.values()) for row in table.values())
    assert {s for row in table.values() for s in row.values()} <= {"ok", "wip", "no"}
    # spot checks against the file itself
    assert table["aifs-ens-crps-2.0.ckpt"]["eerie"] == "no"
    assert table["stochastic_1_4_deg.pkl"]["era5"] == "ok"
