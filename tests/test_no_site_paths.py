"""The repository must not be tied to a person or to one installation.

Everything personal (HPC user, scratch and store folders, the hosts of an installation) belongs to the
experiment's ``main.yml`` or to the settings files in the user's home, never to tracked files. These checks
fail when a personal path or a built-in host comes back.
"""
import importlib.util
import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

# Absolute paths that name a user: /home/<name>, /Users/<name>, /work/users/<name>, /data/users/<name>.
# Placeholders (<user>, %PLATFORM.USER_CODE%, $USER, {user}) do not match the name pattern.
PERSONAL_PATH = re.compile(r"/(?:home|Users|work/users|data/users)/([A-Za-z0-9_.-]+)")
GENERIC_NAMES = {"me", "u", "user", "username", "jdoe"}


def tracked_text_files():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True).stdout
    for name in out.decode().split("\0"):
        path = REPO / name
        if not name or not path.is_file() or path.suffix in {".png", ".jpg", ".ico", ".gz", ".pkl", ".nc", ".pdf"}:
            continue
        try:
            yield name, path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue


def test_no_personal_paths_in_tracked_files():
    if not (REPO / ".git").exists():
        pytest.skip("not a git checkout")
    found = []
    for name, text in tracked_text_files():
        for n, line in enumerate(text.splitlines(), 1):
            for m in PERSONAL_PATH.finditer(line):
                if m.group(1) not in GENERIC_NAMES:
                    found.append(f"{name}:{n}: {m.group(0)}")
    assert not found, "personal paths in tracked files (use a placeholder or a setting):\n" + "\n".join(found)


def test_platforms_take_the_user_from_the_experiment():
    platforms = yaml.safe_load((REPO / "conf" / "platforms.yml").read_text())["PLATFORMS"]
    literal = {name: p["USER"] for name, p in platforms.items() if "%" not in str(p.get("USER", "%"))}
    assert not literal, f"conf/platforms.yml: USER must be a placeholder such as %PLATFORM.USER_CODE%, got {literal}"


def test_default_output_store_is_not_personal():
    store = yaml.safe_load((REPO / "conf" / "general.yml").read_text())["OUTPUT_STORE"]["DIR"]
    assert "%PLATFORM.USER_CODE%" in store, store


def test_dashboard_has_no_built_in_site(monkeypatch):
    """With an empty environment the dashboard knows no host, platform or folder, and says what is missing."""
    for key in list(os.environ):
        if key.startswith("AIUQ_"):
            monkeypatch.delenv(key)
    spec = importlib.util.spec_from_file_location(
        "aiuq_dash_settings_probe", REPO / "tools" / "dashboard" / "aiuq_dash" / "settings.py")
    settings = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(settings)
    assert settings.REMOTE_HOST == "" and settings.LAUNCH_PLATFORM == "" and settings.SUPPORT_FOLDER == ""
    assert settings.REMOTE_BASE == "" and settings.REMOTE_DELETE_ROOTS == []
    assert set(settings.missing()) == {"AIUQ_REMOTE_HOST", "AIUQ_REMOTE_BASE", "AIUQ_LAUNCH_PLATFORM",
                                       "AIUQ_SUPPORT_FOLDER", "AIUQ_REMOTE_DELETE_ROOTS"}
    assert "AIUQ_REMOTE_HOST" in settings.not_set("AIUQ_REMOTE_HOST")


def test_skill_cli_needs_a_config(tmp_path):
    """The `aiuq` command names no machine: without settings it stops before any ssh call."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("AIUQ_")}
    env["AIUQ_CONFIG"] = str(tmp_path / "missing")
    cli = REPO / "skill" / "aiuq" / "scripts" / "aiuq"
    r = subprocess.run(["bash", str(cli), "list"], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode != 0 and "not configured" in r.stderr and "AIUQ_AS_HOST" in r.stderr
    (tmp_path / "config").write_text('AIUQ_AS_HOST="as.example.org"\nAIUQ_HPC_HOST=hpc.example.org\nAIUQ_HPC_BASE=/scratch/x/y/\n')
    env["AIUQ_CONFIG"] = str(tmp_path / "config")
    r = subprocess.run(["bash", str(cli), "site"], env=env, capture_output=True, text=True, timeout=30)
    assert r.returncode == 0 and "AIUQ_AS_HOST=as.example.org" in r.stdout and "/scratch/x/y/<expid>" in r.stdout
