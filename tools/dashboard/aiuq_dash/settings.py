"""Runtime settings, overridable with environment variables (see deploy/aiuq-dashboard.env)."""
import getpass
import os
from pathlib import Path

HOME = Path.home()
USER = getpass.getuser()


def _env(name, default):
    return os.environ.get(name, default)


# Where Autosubmit keeps experiments on this machine (``[local] path`` in ~/.autosubmitrc).
AS_ROOT = Path(_env("AIUQ_AS_ROOT", str(HOME / "autosubmit")))
# Python interpreter that can import ``autosubmit`` (needed to unpickle job lists).
AS_PYTHON = _env("AIUQ_AS_PYTHON", str(HOME / "miniconda3" / "bin" / "python"))

# --- Site settings: no defaults, they describe your HPC (deploy/aiuq-dashboard.env lists them all).
# HPC host where jobs run. Must match the HOST of the platform in conf/platforms.yml.
REMOTE_HOST = _env("AIUQ_REMOTE_HOST", "").strip()
REMOTE_USER = _env("AIUQ_REMOTE_USER", "").strip() or USER
REMOTE_PYTHON = _env("AIUQ_REMOTE_PYTHON", "python3")
# Where experiments live on the HPC, <SCRATCH_DIR>/<PROJECT>/<USER> of the platform: used for an experiment
# that has no HPCROOTDIR recorded yet.
REMOTE_BASE = _env("AIUQ_REMOTE_BASE", "").strip().rstrip("/")
# Launcher: platform of conf/platforms.yml that new experiments use, and the AIUQ support folder on the HPC
# (checkpoints, images, static files).
LAUNCH_PLATFORM = _env("AIUQ_LAUNCH_PLATFORM", "").strip()
SUPPORT_FOLDER = _env("AIUQ_SUPPORT_FOLDER", "").strip().rstrip("/")
# Launcher: HPC folders may only be deleted below these roots (<root>/<project>/<user>/<expid>), ":"-separated.
REMOTE_DELETE_ROOTS = [r for r in _env("AIUQ_REMOTE_DELETE_ROOTS", "").split(":") if r.strip()]
ENV_FILE = "~/.config/aiuq-dashboard.env"


def missing(*names):
    """Site settings among ``names`` that are not set, as their environment variable names."""
    values = {"AIUQ_REMOTE_HOST": REMOTE_HOST, "AIUQ_REMOTE_BASE": REMOTE_BASE,
              "AIUQ_LAUNCH_PLATFORM": LAUNCH_PLATFORM, "AIUQ_SUPPORT_FOLDER": SUPPORT_FOLDER,
              "AIUQ_REMOTE_DELETE_ROOTS": REMOTE_DELETE_ROOTS}
    return [n for n in (names or values) if not values[n]]


def not_set(*names):
    """One sentence naming the unset settings among ``names``, or None when they are all set."""
    miss = missing(*names)
    return f"{', '.join(miss)} not set in {ENV_FILE} (restart the dashboard after editing it)" if miss else None
SSH_CONTROL_DIR = Path(_env("AIUQ_SSH_CONTROL_DIR", str(HOME / ".ssh" / "aiuq-dash")))

# Refresh cadence (seconds).
REFRESH = int(_env("AIUQ_REFRESH", "45"))          # collector cycle for active experiments
SLOW_REFRESH = int(_env("AIUQ_SLOW_REFRESH", "600"))  # remote refresh for idle experiments
REMOTE_TIMEOUT = int(_env("AIUQ_REMOTE_TIMEOUT", "90"))

HOST = _env("AIUQ_BIND", "127.0.0.1")
PORT = int(_env("AIUQ_PORT", "8765"))
# Default: listen on a Unix socket in a 0700 directory, so only this user can reach the server
# (the Launcher can create and start experiments). Set AIUQ_SOCKET="" to use HOST:PORT instead.
SOCKET = _env("AIUQ_SOCKET", str(HOME / ".aiuq-dashboard" / "dash.sock"))

# Small persisted history of node GPU use and queue length (never inside ~/autosubmit).
STATE_DIR = Path(_env("AIUQ_STATE_DIR", str(HOME / ".local" / "share" / "aiuq-dashboard")))
HISTORY_DAYS = int(_env("AIUQ_HISTORY_DAYS", "7"))

HELPERS = Path(__file__).resolve().parent / "helpers"
STATIC = Path(__file__).resolve().parent / "static"
