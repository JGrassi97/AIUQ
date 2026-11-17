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

# HPC host where jobs run. Must match the HPCHOST in the experiment config.
REMOTE_HOST = _env("AIUQ_REMOTE_HOST", "felipe.polito.it")
REMOTE_USER = _env("AIUQ_REMOTE_USER", USER)
REMOTE_PYTHON = _env("AIUQ_REMOTE_PYTHON", "python3")
# Fallback when an experiment has no HPCROOTDIR recorded yet.
REMOTE_BASE = _env("AIUQ_REMOTE_BASE", f"/work/users/{REMOTE_USER}/autosubmit_scratch/test/{REMOTE_USER}")
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
