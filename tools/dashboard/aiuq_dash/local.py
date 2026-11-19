"""Read-only access to Autosubmit data on the local machine (mafalda)."""
import json
import os
import re
import sqlite3
import subprocess
import time
from pathlib import Path

import yaml

from . import settings

EXPID_RE = re.compile(r"^[a-z0-9]{4}$")
STAT_RE = re.compile(r"^(?P<job>.+)_STAT_(?P<n>\d+)$")


def list_experiments():
    root = settings.AS_ROOT
    out = []
    for p in sorted(root.iterdir()) if root.is_dir() else []:
        if p.is_dir() and EXPID_RE.match(p.name) and (p / "conf").is_dir():
            out.append(p.name)
    return out


def _ro_sqlite(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)


def descriptions():
    db = settings.AS_ROOT / "autosubmit.db"
    if not db.exists():
        return {}
    try:
        with _ro_sqlite(db) as con:
            return {r[0]: r[1] for r in con.execute("select name, description from experiment")}
    except sqlite3.Error:
        return {}


def _load_yaml(path):
    try:
        with open(path) as fh:
            return yaml.safe_load(fh) or {}
    except (OSError, yaml.YAMLError):
        return {}


def expand_datelist(spec):
    """Expand Autosubmit DATELIST notation, e.g. '198601[01 11 21] 19870101'."""
    dates = []
    for m in re.finditer(r"(\d+)(?:\[([^\]]*)\])?", str(spec or "")):
        prefix, inner = m.group(1), m.group(2)
        if inner is None:
            dates.append(prefix)
        else:
            dates += [prefix + s for s in inner.split()]
    return dates


def read_conf(expid):
    conf = settings.AS_ROOT / expid / "conf"
    main = _load_yaml(conf / "main.yml")
    minimal = _load_yaml(conf / "minimal.yml")
    model = main.get("MODEL", {}) or {}
    exp = main.get("EXPERIMENT", {}) or {}
    git = minimal.get("GIT", {}) or {}
    dates = expand_datelist(exp.get("DATELIST"))
    return {
        "model": model.get("NAME"),
        "checkpoint": model.get("CHECKPOINT_NAME"),
        "ics": model.get("ICS"),
        "branch": git.get("PROJECT_BRANCH"),
        "origin": git.get("PROJECT_ORIGIN"),
        "run_type": exp.get("RUN_TYPE"),
        "members": str(exp.get("MEMBERS", "")).split(),
        "chunk_size": exp.get("CHUNKSIZE"),
        "chunk_unit": exp.get("CHUNKSIZEUNIT"),
        "num_chunks": exp.get("NUMCHUNKS"),
        "dates": dates,
        "first_date": dates[0] if dates else None,
        "last_date": dates[-1] if dates else None,
        "out_vars": exp.get("OUT_VARS") or [],
        "out_levs": exp.get("OUT_LEVS"),
        "out_res": exp.get("OUT_RES"),
        "out_freq": exp.get("OUT_FREQ"),
        "calendar": exp.get("CALENDAR"),
        "forcing": exp.get("FORCING_VERSION"),
        "output_store": (main.get("OUTPUT_STORE", {}) or {}).get("DIR"),
        "platform": (main.get("PLATFORM", {}) or {}).get("NAME") or (minimal.get("DEFAULT", {}) or {}).get("HPCARCH"),
        "as_version": (minimal.get("CONFIG", {}) or {}).get("AUTOSUBMIT_VERSION"),
    }


def store_marker(expid):
    """<expid>/OUTPUT_STORE.json written by the ARCHIVE job: {"status", "dir", "outputs", ...}, or None."""
    try:
        with open(settings.AS_ROOT / expid / "OUTPUT_STORE.json") as fh:
            rec = json.load(fh)
        return rec if isinstance(rec, dict) else None
    except (OSError, ValueError):
        return None


def conf_mtime(expid):
    conf = settings.AS_ROOT / expid / "conf"
    return max((f.stat().st_mtime for f in conf.glob("*.yml")), default=0)


_CLK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def _boot_time():
    try:
        with open("/proc/stat") as fh:
            for line in fh:
                if line.startswith("btime"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None


def running_autosubmit():
    """{expid: {"pid", "since"}} for live ``autosubmit run <expid>`` processes.

    Scans /proc directly: ``pgrep -af "autosubmit run"`` also matches the shell
    that invokes it, so we check that argv really is ``.../autosubmit run <expid>``.
    """
    found = {}
    btime = _boot_time()
    proc = Path("/proc")
    if not proc.is_dir():
        return found
    for p in proc.iterdir():
        if not p.name.isdigit():
            continue
        try:
            argv = (p / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        argv = [a.decode("utf-8", "replace") for a in argv if a]
        for i, a in enumerate(argv[:3]):
            if os.path.basename(a) == "autosubmit" and len(argv) > i + 1:
                rest = [x for x in argv[i + 1:] if not x.startswith("-")]
                if len(rest) >= 2 and rest[0] == "run" and EXPID_RE.match(rest[1]):
                    since = None
                    try:
                        fields = (p / "stat").read_text().rsplit(")", 1)[1].split()
                        if btime:
                            since = btime + int(fields[19]) / _CLK
                    except (OSError, IndexError, ValueError):
                        pass
                    found[rest[1]] = {"pid": int(p.name), "since": since}
                break
    return found


def job_data_path(expid):
    return settings.AS_ROOT / "metadata" / "data" / f"job_data_{expid}.db"


def read_job_data(expid):
    """Run history from Autosubmit's job_data DB (one row per submission)."""
    path = job_data_path(expid)
    res = {"runs": [], "rows": [], "hpc": {}}
    if not path.exists():
        return res
    try:
        with _ro_sqlite(path) as con:
            runs = con.execute(
                "select run_id, created, modified, start, finish, completed, total, failed, "
                "queuing, running, submitted, metadata from experiment_run order by run_id").fetchall()
            for r in runs:
                res["runs"].append(dict(zip(
                    ["run_id", "created", "modified", "start", "finish", "completed", "total",
                     "failed", "queuing", "running", "submitted"], r[:11])))
            if runs:
                try:
                    meta = json.loads(runs[-1][11] or "{}")
                    res["hpc"] = {k: meta.get(k) for k in ("HPCROOTDIR", "HPCLOGDIR", "HPCHOST", "HPCUSER", "HPCARCH")}
                    cfg = meta.get("CONFIG") or {}
                    plats = meta.get("PLATFORMS") or {}
                    jobs_conf = {}
                    for sec, jc in (meta.get("JOBS") or {}).items():
                        pname = str(jc.get("PLATFORM") or meta.get("DEFAULT", {}).get("HPCARCH") or "")
                        pc = plats.get(pname.upper(), {}) if pname.lower() != "local" else {"TYPE": "local"}
                        jobs_conf[sec] = {
                            "platform": pname, "type": pc.get("TYPE"),
                            "platform_total_jobs": pc.get("TOTAL_JOBS"), "platform_max_waiting": pc.get("MAX_WAITING_JOBS"),
                            "wallclock": jc.get("WALLCLOCK") or pc.get("MAX_WALLCLOCK"),
                            "gpu": "gpu" in str(jc.get("CUSTOM_DIRECTIVES", "")).lower(),
                            "processors": jc.get("PROCESSORS"), "running": jc.get("RUNNING"),
                        }
                    res["limits"] = {"total_jobs": cfg.get("TOTALJOBS"), "max_waiting": cfg.get("MAXWAITINGJOBS"), "jobs": jobs_conf}
                except (TypeError, ValueError):
                    pass
            rows = con.execute(
                "select job_name, status, submit, start, finish, job_id, platform, run_id, last, "
                "section, member, chunk, date from job_data order by id").fetchall()
            keys = ["job", "status", "submit", "start", "finish", "slurm_id", "platform", "run_id",
                    "last", "section", "member", "chunk", "date"]
            res["rows"] = [dict(zip(keys, r)) for r in rows]
    except sqlite3.Error as exc:
        res["error"] = str(exc)
    return res


def local_stats(expid):
    """STAT files of jobs that run on mafalda itself (e.g. SYNCHRONIZE)."""
    logdir = settings.AS_ROOT / expid / "tmp" / f"LOG_{expid}"
    out = []
    if not logdir.is_dir():
        return out
    for f in logdir.iterdir():
        m = STAT_RE.match(f.name)
        if not m:
            continue
        try:
            raw = [l.strip() for l in f.read_text().splitlines() if l.strip()]
        except OSError:
            continue
        nums = [int(l) for l in raw if l.isdigit()]
        words = [l for l in raw if not l.isdigit()]
        out.append({"job": m.group("job"), "n": int(m.group("n")),
                    "start": nums[0] if nums else None, "end": nums[1] if len(nums) > 1 else None,
                    "result": words[-1] if words else None, "mtime": f.stat().st_mtime, "local": True})
    return out


def pkl_path(expid):
    return settings.AS_ROOT / expid / "pkl" / f"job_list_{expid}.pkl"


def read_pkls(expids):
    """Unpickle job lists with Autosubmit's interpreter. Returns {expid: {"jobs"| "error"}}."""
    paths = {str(pkl_path(e)): e for e in expids if pkl_path(e).exists()}
    if not paths:
        return {}
    cmd = [settings.AS_PYTHON, "-I", str(settings.HELPERS / "pkl_reader.py"), *paths]
    proc = subprocess.run(cmd, capture_output=True, timeout=120, cwd="/tmp")
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
        return {e: {"error": "pkl_reader failed: " + " | ".join(msg)} for e in paths.values()}
    data = json.loads(proc.stdout)
    return {paths[p]: v for p, v in data.items()}


def mtime(path):
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0


def now():
    return time.time()
