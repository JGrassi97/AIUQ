"""AIUQ Launcher: create AIUQ experiments and start/stop their runs on the Autosubmit host.

Everything else in the dashboard is read-only; this module is the only place that writes,
and only through Autosubmit's own commands plus one generated conf/main.yml per new experiment.
Every action is appended to STATE_DIR/launcher.log.
"""
import datetime as dt
import json
import logging
import os
import re
import shlex
import subprocess
import threading
import time
import uuid

import yaml

from . import local, remote, settings

log = logging.getLogger("aiuq_dash.launcher")

GIT_REPO = os.environ.get("AIUQ_GIT_REPO", "https://github.com/JGrassi97/AIUQ.git")
PLATFORM = os.environ.get("AIUQ_LAUNCH_PLATFORM", "FELIPE")
SUPPORT_FOLDER = os.environ.get("AIUQ_SUPPORT_FOLDER", f"/work/users/{settings.REMOTE_USER}/AIUQ")
REPO_DIR = settings.STATE_DIR / "repo.git"
AUDIT = settings.STATE_DIR / "launcher.log"
AS_BIN = os.path.join(os.path.dirname(settings.AS_PYTHON), "autosubmit")

MODEL_NAMES = {"aifs": "AIFS", "ms-aurora": "Aurora", "neuralgcm": "NeuralGCM", "ace": "ACE"}
# Support matrix from the README (ok / wip / no); checkpoints not listed are "unknown".
SUPPORT = {
    "aifs-single-mse-1.1.ckpt": {"era5": "ok", "eerie": "wip"},
    "aifs-ens-crps-1.0.ckpt": {"era5": "ok", "eerie": "ok"},
    "aurora-0.25-pretrained.ckpt": {"era5": "ok", "eerie": "wip"},
    "aurora-0.25-finetuned.ckpt": {"era5": "wip", "eerie": "wip"},
    "aurora-0.25-v1.5.ckpt": {"era5": "wip", "eerie": "wip"},
    "aurora-0.25-v1.5-ensemble.ckpt": {"era5": "ok", "eerie": "wip"},
}
NEURALGCM_OK = {"era5": "ok", "eerie": "ok"}
AMIP_MODELS = {"neuralgcm"}
FORCINGS = ["ESA-CCI-v3.0", "AIMIP"]
RESOLUTIONS = ["original", "0.25", "0.5", "1", "1.5", "2"]
FREQS = ["original", "daily"]
CHUNK_UNITS = ["day", "month"]
EXPID_RE = re.compile(r"^[a-z0-9]{4}$")
DATE_RE = re.compile(r"^\d{8}$")


def is_ensemble(ckpt):
    c = (ckpt or "").lower()
    return "ens" in c or "stochastic" in c or "stochasti_" in c


# ---------------------------------------------------------------------------- audit
def audit(action, **kw):
    try:
        settings.STATE_DIR.mkdir(parents=True, exist_ok=True)
        with open(AUDIT, "a") as fh:
            fh.write(json.dumps({"t": dt.datetime.now().isoformat(timespec="seconds"), "action": action, **kw}) + "\n")
    except OSError:
        log.warning("audit log not writable", exc_info=True)


# ---------------------------------------------------------------------------- git cache
_git_lock = threading.Lock()
_git_state = {"fetched": 0, "branches": None, "branches_t": 0}


def _git(*args, timeout=120):
    p = subprocess.run(["git", *args], capture_output=True, text=True, timeout=timeout)
    if p.returncode:
        raise RuntimeError(f"git {' '.join(args[:2])}: {p.stderr.strip()[-300:]}")
    return p.stdout


def branches():
    if _git_state["branches"] and time.time() - _git_state["branches_t"] < 300:
        return _git_state["branches"]
    out = _git("ls-remote", "--heads", GIT_REPO, timeout=30)
    names = sorted(line.split("refs/heads/", 1)[1] for line in out.splitlines() if "refs/heads/" in line)
    _git_state.update(branches=names, branches_t=time.time())
    return names


def _ensure_repo():
    """Bare, blob-less clone of the AIUQ repo in the dashboard's state dir, refreshed every 5 minutes."""
    with _git_lock:
        if not REPO_DIR.exists():
            settings.STATE_DIR.mkdir(parents=True, exist_ok=True)
            _git("clone", "--bare", "--filter=blob:none", GIT_REPO, str(REPO_DIR), timeout=300)
            _git_state["fetched"] = time.time()
        elif time.time() - _git_state["fetched"] > 300:
            _git("-C", str(REPO_DIR), "fetch", "--prune", "origin", "+refs/heads/*:refs/heads/*", timeout=120)
            _git_state["fetched"] = time.time()


def repo_ls(branch, path):
    _ensure_repo()
    out = _git("-C", str(REPO_DIR), "ls-tree", "--name-only", f"{branch}:{path}")
    return [x for x in out.splitlines() if x]


def repo_yaml(branch, path):
    _ensure_repo()
    return yaml.safe_load(_git("-C", str(REPO_DIR), "show", f"{branch}:{path}")) or {}


_opts_cache = {}


def options(branch):
    """Everything the form needs for one branch: models, checkpoints, ICS, variables, levels."""
    if branch not in branches():
        raise ValueError(f"unknown branch {branch!r}")
    hit = _opts_cache.get(branch)
    if hit and time.time() - hit[0] < 300:
        return hit[1]
    _ensure_repo()
    std_version = "v010"
    try:
        cfg_txt = _git("-C", str(REPO_DIR), "show", f"{branch}:templates/config.yml")
        m = re.search(r'STD_VERSION:\s*"?([\w.]+)', cfg_txt)
        std_version = m.group(1) if m else std_version
    except RuntimeError:
        pass
    std = repo_yaml(branch, f"conf/AIUQ-st/{std_version}/variables.yml").get("data", {})
    std_by_code = {int(k): v for k, v in std.items()}
    plat_models = repo_ls(branch, f"conf/jobs/{PLATFORM}/models")
    plat_ics = repo_ls(branch, f"conf/jobs/{PLATFORM}/ics")
    ckpts = checkpoints_on_hpc()
    models = []
    for card in repo_ls(branch, "conf/cards/models"):
        c = repo_yaml(branch, f"conf/cards/models/{card}")
        name = c.get("model")
        if not name:
            continue
        variables = []
        for code in (c.get("variables") or {}):
            s = std_by_code.get(int(code))
            if s and s.get("short_name") not in {x["name"] for x in variables}:
                variables.append({"name": s["short_name"], "long": s.get("long_name", ""), "units": s.get("units", ""), "type": s.get("type", "")})
        levels = (c.get("pressure_levels") or {}).get("values") or []
        on_hpc = ckpts.get(name)
        model_ck = sorted(set(on_hpc or []) | {k for k in SUPPORT if _ckpt_model(k) == name})
        models.append({
            "name": name, "label": MODEL_NAMES.get(name, name), "producer": c.get("producer"),
            "on_platform": name in plat_models, "variables": variables, "levels": levels,
            "amip": name in AMIP_MODELS,
            "checkpoints": [{"name": k, "present": on_hpc is not None and k in on_hpc, "ensemble": is_ensemble(k),
                             "support": SUPPORT.get(k) or (NEURALGCM_OK if name == "neuralgcm" else {})} for k in model_ck],
        })
    models.sort(key=lambda m: (not m["on_platform"], m["label"]))
    res = {"branch": branch, "platform": PLATFORM, "ics": plat_ics, "models": models, "std_version": std_version,
           "forcings": FORCINGS, "resolutions": RESOLUTIONS, "freqs": FREQS, "chunk_units": CHUNK_UNITS,
           "support_folder": SUPPORT_FOLDER, "user_code": settings.REMOTE_USER, "checkpoints_checked": bool(ckpts)}
    _opts_cache[branch] = (time.time(), res)
    return res


def _ckpt_model(ck):
    return "aifs" if ck.startswith("aifs") else "ms-aurora" if ck.startswith("aurora") else "neuralgcm"


_ck_cache = {"t": 0, "data": {}}


def checkpoints_on_hpc():
    """{model: [checkpoint files]} found under SUPPORT_FOLDER/models on the HPC (cached 10 min)."""
    if time.time() - _ck_cache["t"] < 600:
        return _ck_cache["data"]
    try:
        r = remote.run_helper("felipe_probe.py", {"user": settings.REMOTE_USER, "exps": {}, "ls": f"{SUPPORT_FOLDER}/models"}, timeout=60)
        _ck_cache.update(t=time.time(), data=r.get("ls") or {})
    except remote.RemoteError:
        log.warning("could not list checkpoints", exc_info=True)
    return _ck_cache["data"]


# ---------------------------------------------------------------------------- config
def compress_dates(dates):
    """['19860101','19860111','19860210'] -> '198601[01 11] 19860210' (Autosubmit DATELIST notation)."""
    groups = {}
    for d in dates:
        groups.setdefault(d[:6], []).append(d[6:])
    parts = []
    for ym in sorted(groups):
        days = groups[ym]
        parts.append(ym + days[0] if len(days) == 1 else f"{ym}[{' '.join(days)}]")
    return " ".join(parts)


def validate(cfg):
    """Normalise a form config; return (clean, errors, warnings)."""
    errors, warnings = [], []
    branch = str(cfg.get("branch") or "main")
    try:
        opts = options(branch)
    except (ValueError, RuntimeError) as exc:
        return None, [str(exc)], []
    model = next((m for m in opts["models"] if m["name"] == cfg.get("model")), None)
    if not model:
        errors.append("Choose a model.")
        return None, errors, warnings
    if not model["on_platform"]:
        errors.append(f"{model['label']} has no job configuration for {PLATFORM} on branch {branch}.")
    ck = next((c for c in model["checkpoints"] if c["name"] == cfg.get("checkpoint")), None)
    if not ck:
        errors.append("Choose a checkpoint.")
    elif opts["checkpoints_checked"] and not ck["present"]:
        errors.append(f"Checkpoint {ck['name']} is not in {SUPPORT_FOLDER}/models/{model['name']} on the HPC.")
    ics = cfg.get("ics")
    if ics not in opts["ics"]:
        errors.append(f"Choose initial conditions available for {PLATFORM}: {', '.join(opts['ics'])}.")
    elif ck:
        sup = ck["support"].get(ics)
        if sup == "wip":
            warnings.append(f"{ck['name']} with {ics.upper()} is marked as work in progress in the README.")
        elif sup is None:
            warnings.append(f"No support information for {ck['name']} with {ics.upper()}.")
    run_type = cfg.get("run_type") or "hindcast"
    if run_type not in ("hindcast", "amip"):
        errors.append("Run type must be hindcast or amip.")
    elif run_type == "amip" and not model["amip"]:
        errors.append(f"AMIP runs are only supported by NeuralGCM, not {model['label']}.")
    forcing = cfg.get("forcing") or FORCINGS[0]
    if forcing not in FORCINGS:
        errors.append("Unknown forcing version.")
    try:
        members = int(cfg.get("members") or 1)
    except (TypeError, ValueError):
        members = 0
    if not 1 <= members <= 100:
        errors.append("Members must be between 1 and 100.")
    elif ck and members > 1 and not ck["ensemble"]:
        warnings.append(f"{ck['name']} is deterministic: all {members} members will be identical.")
    dates = sorted({str(d) for d in (cfg.get("dates") or [])})
    bad = [d for d in dates if not DATE_RE.match(d)]
    for d in dates:
        try:
            dt.datetime.strptime(d, "%Y%m%d")
        except ValueError:
            bad.append(d)
    if bad:
        errors.append(f"Invalid start dates: {', '.join(sorted(set(bad))[:5])}.")
    if not dates:
        errors.append("Add at least one start date.")
    if len(dates) > 2000:
        errors.append("At most 2000 start dates.")
    if ics == "era5" and dates and dates[0] < "19400101":
        errors.append("ERA5 starts in 1940.")
    try:
        chunk_size, num_chunks = int(cfg.get("chunk_size") or 0), int(cfg.get("num_chunks") or 0)
    except (TypeError, ValueError):
        chunk_size = num_chunks = 0
    unit = cfg.get("chunk_unit") or "day"
    if chunk_size < 1 or num_chunks < 1:
        errors.append("Chunk size and number of chunks must be at least 1.")
    if unit not in CHUNK_UNITS:
        errors.append("Chunk unit must be day or month.")
    out_vars = [v for v in (cfg.get("out_vars") or []) if v]
    valid_vars = {v["name"] for v in model["variables"]}
    if not out_vars:
        errors.append("Choose at least one output variable.")
    unknown = [v for v in out_vars if v not in valid_vars]
    if unknown:
        errors.append(f"{model['label']} does not produce: {', '.join(unknown)}.")
    levs = cfg.get("out_levs")
    if levs != "original":
        levs = sorted({int(x) for x in (levs or [])}, reverse=True)
        if not levs:
            errors.append("Choose output pressure levels (or 'all levels').")
        bad_l = [l for l in levs if model["levels"] and l not in model["levels"]]
        if bad_l:
            errors.append(f"{model['label']} has no level {', '.join(map(str, bad_l))} hPa.")
    res = str(cfg.get("out_res") or "1")
    if res not in RESOLUTIONS:
        errors.append("Unknown output resolution.")
    freq = cfg.get("out_freq") or "daily"
    if freq not in FREQS:
        errors.append("Unknown output frequency.")
    desc = (cfg.get("description") or "").strip()
    if not desc:
        errors.append("Write a short description.")
    if len(desc) > 200 or any(ch in desc for ch in "\n\r"):
        errors.append("The description must be a single line under 200 characters.")
    sims = len(dates) * members * max(num_chunks, 0)
    if sims > 3000:
        warnings.append(f"This creates {sims} simulation jobs.")
    clean = {"branch": branch, "model": model["name"], "checkpoint": ck["name"] if ck else None, "ics": ics,
             "run_type": run_type, "forcing": forcing, "members": members, "dates": dates, "chunk_size": chunk_size,
             "chunk_unit": unit, "num_chunks": num_chunks, "out_vars": out_vars, "out_levs": levs, "out_res": res,
             "out_freq": freq, "description": desc, "support_folder": SUPPORT_FOLDER, "user_code": settings.REMOTE_USER,
             "sims": sims}
    return clean, errors, warnings


def main_yml(c):
    levs = "original" if c["out_levs"] == "original" else "[" + ", ".join(map(str, c["out_levs"])) + "]"
    vars_ = "\n".join(f"    - {v}" for v in c["out_vars"])
    stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""# Generated by AIUQ Launcher on {stamp}
MODEL:
  NAME: {c['model']}
  CHECKPOINT_NAME: {c['checkpoint']}
  ICS: {c['ics']}

EXPERIMENT:
  RUN_TYPE: "{c['run_type']}"
  MEMBERS: "{' '.join(str(i) for i in range(1, c['members'] + 1))}"
  CHUNKSIZEUNIT: {c['chunk_unit']}
  DATELIST: {compress_dates(c['dates'])}
  CHUNKSIZE: {c['chunk_size']}
  NUMCHUNKS: {c['num_chunks']}
  CALENDAR: standard
  OUT_VARS:
{vars_}
  OUT_FREQ: {c['out_freq']}
  OUT_RES: "{c['out_res']}"
  OUT_LEVS: {levs}
  FORCING_VERSION: {c['forcing']}

PATHS:
  SUPPORT_FOLDER: {c['support_folder']}
  SIF_FOLDER: "%PATHS.SUPPORT_FOLDER%/sif"

PLATFORM:
  NAME: {PLATFORM}
  USER_CODE: {c['user_code']}
"""


def expid_cmd(c):
    return [AS_BIN, "expid", "-min", "-H", PLATFORM, "-d", c["description"], "-repo", GIT_REPO, "-b", c["branch"],
            "-conf", "conf/bootstrap/"]


def preview(cfg):
    clean, errors, warnings = validate(cfg)
    if clean is None:
        return {"ok": False, "errors": errors, "warnings": warnings}
    # sanity: the generated DATELIST expands back to the same dates
    if local.expand_datelist(compress_dates(clean["dates"])) != clean["dates"]:
        errors.append("Internal error: the start-date list does not round-trip.")
    cmds = [" ".join(shlex.quote(x) for x in expid_cmd(clean)),
            "# write <expid>/conf/main.yml (shown above)",
            f"{AS_BIN} create <expid> -np"]
    return {"ok": not errors, "errors": errors, "warnings": warnings, "yaml": main_yml(clean), "commands": cmds,
            "summary": {k: clean[k] for k in ("sims", "members", "num_chunks", "chunk_size", "chunk_unit")} | {"n_dates": len(clean["dates"])}}


def template(expid):
    """Form values from an existing experiment, to start a new one from it."""
    if not EXPID_RE.match(expid):
        raise ValueError("bad expid")
    conf = local.read_conf(expid)
    lev = conf.get("out_levs")
    return {"branch": conf.get("branch") or "main", "model": conf.get("model"), "checkpoint": conf.get("checkpoint"),
            "ics": conf.get("ics"), "run_type": conf.get("run_type") or "hindcast", "forcing": conf.get("forcing") or FORCINGS[0],
            "members": len(conf.get("members") or []) or 1, "dates": conf.get("dates") or [],
            "chunk_size": conf.get("chunk_size") or 1, "chunk_unit": conf.get("chunk_unit") or "day",
            "num_chunks": conf.get("num_chunks") or 1, "out_vars": conf.get("out_vars") or [],
            "out_levs": lev if lev == "original" else (lev or []), "out_res": str(conf.get("out_res") or "1"),
            "out_freq": conf.get("out_freq") or "daily", "description": f"copy of {expid}"}


# ---------------------------------------------------------------------------- tasks
TASKS = {}
_task_lock = threading.Lock()


class Task:
    def __init__(self, kind, expid=None):
        self.id = uuid.uuid4().hex[:12]
        self.kind, self.expid = kind, expid
        self.status, self.error = "running", None
        self.lines, self.started, self.finished = [], time.time(), None
        self.steps = []

    def say(self, line):
        for l in str(line).rstrip().splitlines():
            self.lines.append(l)
        del self.lines[:-800]

    def step(self, name, state="running"):
        if self.steps and self.steps[-1]["state"] == "running":
            self.steps[-1]["state"] = "done"
        self.steps.append({"name": name, "state": state})

    def as_dict(self, tail=400):
        return {"id": self.id, "kind": self.kind, "expid": self.expid, "status": self.status, "error": self.error,
                "lines": self.lines[-tail:], "started": self.started, "finished": self.finished, "steps": self.steps}


def _env():
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(settings.AS_PYTHON) + ":" + env.get("PATH", "/usr/bin:/bin")
    env.setdefault("HOME", str(settings.HOME))
    return env


def _run(task, cmd, timeout):
    task.say("$ " + " ".join(shlex.quote(x) for x in cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True,
                         cwd=str(settings.AS_ROOT), env=_env(), bufsize=1)
    t0 = time.time()
    for line in p.stdout:
        task.say(line)
        if time.time() - t0 > timeout:
            p.kill()
            raise RuntimeError(f"timeout after {timeout}s")
    rc = p.wait()
    if rc:
        raise RuntimeError(f"command failed with exit code {rc}")


def _start(task, fn):
    with _task_lock:
        TASKS[task.id] = task
        for k in [k for k, t in TASKS.items() if t.finished and time.time() - t.finished > 86400]:
            del TASKS[k]

    def wrapper():
        try:
            fn(task)
            task.status = "done"
            if task.steps and task.steps[-1]["state"] == "running":
                task.steps[-1]["state"] = "done"
        except Exception as exc:  # noqa: BLE001
            task.status, task.error = "failed", str(exc)
            if task.steps:
                task.steps[-1]["state"] = "failed"
            task.say(f"ERROR: {exc}")
            log.warning("launcher task %s failed: %s", task.kind, exc)
        task.finished = time.time()
        audit(f"{task.kind}:{task.status}", expid=task.expid, error=task.error)

    threading.Thread(target=wrapper, daemon=True, name=f"launcher-{task.kind}").start()
    return task


def create(cfg):
    clean, errors, _ = validate(cfg)
    if errors:
        raise ValueError(" ".join(errors))
    task = Task("create")
    audit("create:start", config={k: v for k, v in clean.items() if k != "dates"}, n_dates=len(clean["dates"]))

    def work(t):
        before = set(local.list_experiments()) | {p.name for p in settings.AS_ROOT.iterdir() if p.is_dir()}
        t.step("Registering the experiment (autosubmit expid)")
        _run(t, expid_cmd(clean), timeout=300)
        new = sorted({p.name for p in settings.AS_ROOT.iterdir() if p.is_dir() and EXPID_RE.match(p.name)} - before)
        m = re.findall(r"\b([a-z0-9]{4})\b", "\n".join(l for l in t.lines if "created" in l.lower() or "experiment" in l.lower()))
        expid = new[0] if len(new) == 1 else next((x for x in m if x in new), None)
        if not expid:
            raise RuntimeError(f"could not find the new experiment id (new folders: {new})")
        t.expid = expid
        t.say(f"New experiment: {expid}")
        t.step("Writing conf/main.yml")
        path = settings.AS_ROOT / expid / "conf" / "main.yml"
        if path.exists():
            raise RuntimeError(f"{path} already exists, not overwriting")
        path.write_text(main_yml(clean))
        t.say(f"wrote {path}")
        t.step("Building the workflow (autosubmit create)")
        _run(t, [AS_BIN, "create", expid, "-np"], timeout=1800)
        t.step("Ready to run", "done")

    return _start(task, work)


def run(expid):
    if not EXPID_RE.match(expid) or not (settings.AS_ROOT / expid / "conf").is_dir():
        raise ValueError("unknown experiment")
    if expid in local.running_autosubmit():
        raise ValueError(f"{expid} is already running")
    if not (settings.AS_ROOT / expid / "pkl" / f"job_list_{expid}.pkl").exists():
        raise ValueError(f"{expid} has no workflow yet: run 'autosubmit create {expid}' first")
    task = Task("run", expid)
    audit("run:start", expid=expid)

    def work(t):
        t.step("Starting autosubmit run in its own systemd unit")
        out = settings.AS_ROOT / f"{expid}_run.out"
        unit = f"aiuq-run-{expid}-{int(time.time())}"
        inner = f"exec {shlex.quote(AS_BIN)} run {expid} >> {shlex.quote(str(out))} 2>&1 < /dev/null"
        _run(t, ["systemd-run", "--user", "--collect", f"--unit={unit}", f"--working-directory={settings.AS_ROOT}",
                 f"--setenv=PATH={_env()['PATH']}", f"--setenv=HOME={settings.HOME}", "/bin/bash", "-c", inner], timeout=30)
        t.say(f"output: {out}   ·   journal: journalctl --user -u {unit}")
        t.step("Waiting for the process to appear")
        for _ in range(20):
            time.sleep(1)
            if expid in local.running_autosubmit():
                t.say(f"autosubmit run {expid} is running (pid {local.running_autosubmit()[expid]['pid']})")
                return
        tail = out.read_text(errors="replace").splitlines()[-15:] if out.exists() else []
        for l in tail:
            t.say(l)
        raise RuntimeError("the run did not start (see the output above)")

    return _start(task, work)


def stop(expid, cancel=False):
    if not EXPID_RE.match(expid) or not (settings.AS_ROOT / expid).is_dir():
        raise ValueError("unknown experiment")
    if expid not in local.running_autosubmit():
        raise ValueError(f"{expid} is not running")
    task = Task("stop", expid)
    audit("stop:start", expid=expid, cancel=cancel)

    def work(t):
        t.step("Asking Autosubmit to stop" + (" and cancel queued/running jobs" if cancel else ""))
        _run(t, [AS_BIN, "stop", "-y", *(["-c"] if cancel else []), expid], timeout=600)
        t.step("Waiting for the run to end")
        for _ in range(60):
            if expid not in local.running_autosubmit():
                t.say(f"{expid} stopped")
                return
            time.sleep(2)
        t.say("Autosubmit was asked to stop; it finishes its current loop first and may take a few minutes.")

    return _start(task, work)


# ---------------------------------------------------------------------------- delete
# Remote folders may only be removed below these roots, at <root>/<project>/<user>/<expid>.
DELETE_ROOTS = [r for r in os.environ.get("AIUQ_REMOTE_DELETE_ROOTS",
                                          f"/work/users/{settings.REMOTE_USER}/autosubmit_scratch").split(":") if r]
LEFTOVERS = settings.STATE_DIR / "pending_remote_deletes.json"
_left_lock = threading.Lock()


def _remote_path(expid):
    """The experiment's HPCROOTDIR, read from its job history (gone after `autosubmit delete`)."""
    hpc = (local.read_job_data(expid).get("hpc") or {})
    host = hpc.get("HPCHOST")
    if host and host != settings.REMOTE_HOST:
        return None, f"its jobs ran on {host}; only {settings.REMOTE_HOST} folders are handled here"
    return (hpc.get("HPCROOTDIR") or f"{settings.REMOTE_BASE}/{expid}"), None


def _local_size(path):
    files = size = 0
    for dirpath, _dirs, fns in os.walk(path, followlinks=False):
        for fn in fns:
            try:
                size += os.lstat(os.path.join(dirpath, fn)).st_size
                files += 1
            except OSError:
                pass
    return files, size


def _slurm_jobs(expid):
    try:
        r = remote.run_helper("felipe_probe.py", {"user": settings.REMOTE_USER, "slurm": True, "exps": {}}, timeout=60)
    except remote.RemoteError as exc:
        raise RuntimeError(f"cannot read the SLURM queue: {exc}") from exc
    return [j for j in (r.get("slurm") or {}).get("jobs", []) if j.get("name", "").startswith(expid + "_")]


def _inspect_remote(expid, path):
    return remote.run_helper("felipe_delete.py", {"mode": "inspect", "expid": expid, "path": path, "roots": DELETE_ROOTS}, timeout=180)


def delete_plan(expid):
    """What deleting `expid` would remove, and what currently prevents it."""
    if not EXPID_RE.match(expid) or not (settings.AS_ROOT / expid).is_dir():
        raise ValueError("unknown experiment")
    lpath = settings.AS_ROOT / expid
    files, size = _local_size(lpath)
    rpath, rerr = _remote_path(expid)
    remote_info = {"ok": False, "error": rerr}
    if rpath:
        try:
            remote_info = _inspect_remote(expid, rpath)
        except remote.RemoteError as exc:
            remote_info = {"ok": False, "error": f"cannot inspect the HPC folder: {exc}"}
    jobs = _slurm_jobs(expid)
    blockers = []
    if expid in local.running_autosubmit():
        blockers.append("autosubmit run is active for this experiment: stop it first.")
    if jobs:
        blockers.append(f"{len(jobs)} job(s) of this experiment are still in the SLURM queue: stop it with “also cancel jobs” first.")
    if rpath and not remote_info.get("ok"):
        blockers.append(f"The HPC folder will not be touched: {remote_info.get('error')}. Fix this before deleting.")
    conf = local.read_conf(expid)
    return {"expid": expid, "description": local.descriptions().get(expid), "model": conf.get("model"), "ics": conf.get("ics"),
            "local": {"path": str(lpath), "files": files, "bytes": size,
                      "extra": [f"Autosubmit database entry of {expid}", f"job history (metadata/data/job_data_{expid}.*)",
                                f"structure database (metadata/structures/structure_{expid}.db)"]},
            "remote": {**remote_info, "path": rpath, "host": settings.REMOTE_HOST},
            "slurm_jobs": [{"id": j.get("id"), "name": j.get("name"), "state": j.get("state")} for j in jobs],
            "blockers": blockers}


def leftovers():
    try:
        return json.loads(LEFTOVERS.read_text())
    except (OSError, ValueError):
        return []


def _set_leftover(expid, path, add):
    with _left_lock:
        items = [x for x in leftovers() if x.get("expid") != expid]
        if add:
            items.append({"expid": expid, "path": path, "since": dt.datetime.now().isoformat(timespec="seconds")})
        settings.STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = LEFTOVERS.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=1))
        os.replace(tmp, LEFTOVERS)


def _delete_remote(t, expid, path):
    t.step(f"Deleting {path} on {settings.REMOTE_HOST}")
    _set_leftover(expid, path, True)  # recorded first, so a failure can be retried
    r = remote.run_helper("felipe_delete.py", {"mode": "delete", "expid": expid, "path": path, "roots": DELETE_ROOTS}, timeout=3600)
    if not r.get("ok"):
        for e in r.get("errors", []):
            t.say(e)
        raise RuntimeError(f"{r.get('error') or str(r.get('n_errors', '?')) + ' files could not be removed'}; "
                           "the folder is listed under “Leftover folders” so you can retry")
    _set_leftover(expid, path, False)
    t.say(f"deleted {path}")


def delete(expid, confirm):
    if confirm != expid:
        raise ValueError("type the experiment id to confirm")
    plan = delete_plan(expid)
    if plan["blockers"]:
        raise ValueError(" ".join(plan["blockers"]))
    task = Task("delete", expid)
    rem = plan["remote"]
    audit("delete:start", expid=expid, local_bytes=plan["local"]["bytes"], remote_path=rem.get("path"),
          remote_bytes=rem.get("bytes"), remote_files=rem.get("files"), nc_files=rem.get("nc_files"))

    def work(t):
        t.step("Removing the experiment from Autosubmit (autosubmit delete)")
        _run(t, [AS_BIN, "delete", "-f", expid], timeout=900)
        if (settings.AS_ROOT / expid).exists():
            raise RuntimeError("autosubmit delete finished but the experiment folder is still there; the HPC folder was not touched")
        if rem.get("path") and rem.get("ok") and rem.get("exists"):
            _delete_remote(t, expid, rem["path"])
        else:
            t.say(f"nothing to delete on {settings.REMOTE_HOST} ({rem.get('path') or 'no folder recorded'} does not exist)")
        t.step(f"{expid} deleted", "done")

    return _start(task, work)


def retry_leftover(expid):
    item = next((x for x in leftovers() if x.get("expid") == expid), None)
    if not item:
        raise ValueError("no leftover folder for this experiment")
    if (settings.AS_ROOT / expid).exists():
        raise ValueError(f"{expid} exists again in Autosubmit; not touching its HPC folder")
    task = Task("cleanup", expid)
    audit("cleanup:start", expid=expid, remote_path=item["path"])

    def work(t):
        info = _inspect_remote(expid, item["path"])
        if not info.get("ok"):
            raise RuntimeError(info.get("error"))
        if info.get("exists"):
            _delete_remote(t, expid, item["path"])
        else:
            _set_leftover(expid, item["path"], False)
            t.say(f"{item['path']} is already gone")
        t.step("Done", "done")

    return _start(task, work)
