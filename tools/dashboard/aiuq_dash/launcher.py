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
import shutil
import subprocess
import threading
import time
import uuid

import yaml

from . import local, remote, settings

log = logging.getLogger("aiuq_dash.launcher")

GIT_REPO = os.environ.get("AIUQ_GIT_REPO", "https://github.com/JGrassi97/AIUQ.git")
PLATFORM = settings.LAUNCH_PLATFORM
SUPPORT_FOLDER = settings.SUPPORT_FOLDER
# OUTPUT_STORE.DIR (a folder on this host) is written into main.yml only when the form sets it; otherwise the
# branch's default applies: conf/jobs/<platform>/ics/<ics>/<ics>.yml, else conf/general.yml.
STORE_RE = re.compile(r"^/[A-Za-z0-9_./-]+$")
REPO_DIR = settings.STATE_DIR / "repo.git"
AUDIT = settings.STATE_DIR / "launcher.log"
AS_BIN = os.path.join(os.path.dirname(settings.AS_PYTHON), "autosubmit")

MODEL_NAMES = {"aifs": "AIFS", "ms-aurora": "Aurora", "neuralgcm": "NeuralGCM", "ace": "ACE"}
# Support status per checkpoint and IC source (ok / wip / no), read from docs/support-matrix.yml of the chosen
# branch (the file behind the README table), so the Launcher never disagrees with the documentation.
MATRIX_ICS = {"ERA5": "era5", "EERIE (AMIP)": "eerie", "ECMWF analysis": "ecmwf-an", "EC-Earth3 (CMIP6)": "ec-earth3"}
MATRIX_STATUS = {"available": "ok", "wip": "wip", "not_implemented": "no"}


def parse_support_matrix(matrix):
    """{checkpoint file: {ics: 'ok' | 'wip' | 'no'}} from the content of docs/support-matrix.yml."""
    label = {ck["id"]: ck["label"] for m in matrix.get("models") or [] for ck in m.get("checkpoints") or []}
    out = {}
    for ics_label, per_ckpt in (matrix.get("status") or {}).items():
        ics = MATRIX_ICS.get(ics_label)
        if not ics:
            continue
        for ckpt_id, status in (per_ckpt or {}).items():
            if ckpt_id in label and status in MATRIX_STATUS:
                out.setdefault(label[ckpt_id], {})[ics] = MATRIX_STATUS[status]
    return out


def support_matrix(branch):
    """Support table of a branch; empty (every combination 'unknown') when the branch has no matrix."""
    try:
        return parse_support_matrix(repo_yaml(branch, "docs/support-matrix.yml"))
    except (RuntimeError, yaml.YAMLError, KeyError, TypeError, AttributeError):
        log.warning("no usable docs/support-matrix.yml on %s", branch, exc_info=True)
        return {}


AMIP_MODELS = {"neuralgcm"}
FORCINGS = ["ESA-CCI-v3.0", "AIMIP"]
RESOLUTIONS = ["original", "0.25", "0.5", "1", "1.5", "2"]
FREQS = ["original", "daily"]
CHUNK_UNITS = ["day", "month"]
EXPID_RE = re.compile(r"^[a-z0-9]{4}$")
DATE_RE = re.compile(r"^\d{8}$")

# ---------------------------------------------------------------------------- output compression
# EXPERIMENT.OUT_COMPRESSION of the experiment (runscripts/AIUQst_lib/compression.py reads it in the jobs; the
# rules here are the same, tests/test_launcher_compression.py keeps the two in step). The steps of 'integer' and
# 'mixed' come from 'pack' of the AIUQ-st standard of the chosen branch.
COMPRESSION_PRESETS = [
    ("none", "None", "As the model writes them: not compressed. For debugging."),
    ("lossless", "Lossless", "float32 (what the models compute in), compressed. Nothing is rounded: research, verification, hindcasts."),
    ("bits", "Rounded bits", "Lossless, with the values rounded to a number of significant bits (12: relative error below 0.025 %)."),
    ("integer", "Integers", "Whole °C, m/s, hPa, metres, millimetres and percent. Smallest files; light rain becomes zero."),
    ("mixed", "Mixed", "Integers, with precipitation at 0.1 mm. Made for operational forecasts."),
    ("custom", "Custom", "A step, a number of bits or lossless for each output variable."),
]
COMPRESSION_NAMES = [name for name, _, _ in COMPRESSION_PRESETS]
KEEPBITS_MIN, KEEPBITS_MAX, KEEPBITS_DEFAULT = 4, 23, 12
CHUNK_NAMES = ("point", "map")
PRECIPITATION = ("tp", "cp", "sf", "ro", "avg_tprate", "avg_tsrwe")
# Size estimate: bytes per stored value, from the forecasts of November 2026 (0.25 degrees, 6-hourly) compressed
# with each scheme. Rough: a smooth field compresses better than a noisy one.
BYTES_UNCOMPRESSED = {"aifs": 4.0, "ms-aurora": 8.0, "neuralgcm": 8.0}      # the type each model writes
BYTES_LOSSLESS, BYTES_INTEGER, BYTES_INTEGER_FINE = 2.7, 0.33, 0.5
GRID_POINTS = {"0.25": 721 * 1440, "0.5": 361 * 720, "1": 181 * 360, "1.5": 121 * 240, "2": 91 * 180}
NATIVE_POINTS = {"aifs": 542080, "ms-aurora": 721 * 1440, "deterministic_0_7": 256 * 512, "_1_4_": 128 * 256, "_2_8_": 64 * 128}
BIG_RUN_BYTES = 20e9                         # 'none' above this size gets a warning


def default_compression(ics):
    """What the form proposes: forecasts from the latest analysis are kept small, everything else exact."""
    return "mixed" if ics == "ecmwf-an" else "lossless"


def compression_supported(branch):
    """True when the code of the branch reads OUT_COMPRESSION (its conf/general.yml sets the default)."""
    try:
        return bool((repo_yaml(branch, "conf/general.yml").get("EXPERIMENT") or {}).get("OUT_COMPRESSION"))
    except RuntimeError:
        return False


def _bits_bytes(keepbits):
    return 0.25 + 0.0625 * keepbits          # 12 bits: about 1 byte per value


def _rule_of(rules, name):
    return rules.get(name, rules.get("default"))


def variable_storage(preset, opts, variable):
    """(mode, value) for one variable of the form: ('integer', step), ('bits', n), ('lossless', None), ('none', None)."""
    keepbits = opts.get("keepbits", KEEPBITS_DEFAULT)
    if preset in ("none", "lossless"):
        return preset, None
    if preset == "bits":
        return "bits", keepbits
    if preset == "custom":
        rule = _rule_of(opts.get("custom") or {}, variable["name"])
        if not rule:
            return "bits", keepbits
        if "step" in rule:
            return "integer", float(rule["step"])
        if "keepbits" in rule:
            return "bits", int(rule["keepbits"])
        return str(rule.get("mode") or "lossless"), None
    step = (variable.get("pack") or {}).get(preset)
    return ("integer", float(step)) if step else ("bits", keepbits)


def clean_compression(cfg, ics, out_vars, variables, supported, errors, warnings):
    """Checked (preset, options) of a form config; problems are appended to ``errors`` and ``warnings``."""
    preset = cfg.get("out_compression")
    given = cfg.get("out_compression_opts") or {}
    if not supported:
        if preset or given:
            warnings.append("This branch does not read OUT_COMPRESSION: the outputs are written as its code does.")
        return None, {}
    preset = str(preset or default_compression(ics)).strip().lower()
    if preset not in COMPRESSION_NAMES:
        errors.append(f"Output compression must be one of {', '.join(COMPRESSION_NAMES)}.")
        return None, {}
    if not isinstance(given, dict):
        errors.append("The compression options must be a mapping.")
        return preset, {}
    opts = {}
    unknown = set(given) - {"keepbits", "chunks", "complevel", "custom"}
    if unknown:
        errors.append(f"Unknown compression option(s): {', '.join(sorted(unknown))}.")

    def whole(value, low, high, what):
        try:
            number = int(value)
            if number != float(value) or not low <= number <= high:
                raise ValueError
            return number
        except (TypeError, ValueError):
            errors.append(f"{what} must be a whole number from {low} to {high}.")
            return None

    if given.get("keepbits") not in (None, ""):
        bits = whole(given["keepbits"], KEEPBITS_MIN, KEEPBITS_MAX, "The number of significant bits")
        if bits is not None and bits != KEEPBITS_DEFAULT:
            opts["keepbits"] = bits
    if given.get("complevel") not in (None, ""):
        level = whole(given["complevel"], 1, 9, "The zlib level")
        if level is not None and level != 1:
            opts["complevel"] = level
    chunks = given.get("chunks")
    if chunks not in (None, "", "point"):
        if isinstance(chunks, str):
            if chunks.strip().lower() not in CHUNK_NAMES:
                errors.append("Chunks must be point, map or three numbers: valid times, latitudes, longitudes.")
            else:
                opts["chunks"] = chunks.strip().lower()
        else:
            try:
                sizes = [int(x) for x in chunks]
                if len(sizes) != 3 or any(x < 1 for x in sizes) or any(int(x) != float(x) for x in chunks):
                    raise ValueError
                opts["chunks"] = sizes
            except (TypeError, ValueError):
                errors.append("Chunks must be point, map or three whole numbers of at least 1: valid times, latitudes, longitudes.")

    custom = given.get("custom")
    if preset == "custom":
        rules = {}
        if not isinstance(custom, dict) or not custom:
            errors.append("Custom compression needs a rule for at least one output variable, or a default.")
        else:
            for name, rule in custom.items():
                name = str(name).strip().lower()
                if name != "default" and name not in out_vars:
                    errors.append(f"Custom compression: '{name}' is not among the output variables.")
                    continue
                if not isinstance(rule, dict):
                    errors.append(f"Custom compression of '{name}': give a step, a number of bits or lossless.")
                    continue
                rule = {str(k).lower(): v for k, v in rule.items()}
                if "step" in rule:
                    try:
                        step, offset = float(rule["step"]), float(rule.get("offset") or 0.0)
                        if not step > 0 or step != step or abs(step) == float("inf") or offset != offset:
                            raise ValueError
                        rules[name] = {"step": step, **({"offset": offset} if offset else {})}
                    except (TypeError, ValueError):
                        errors.append(f"Custom compression of '{name}': the step must be a number greater than zero.")
                elif "keepbits" in rule:
                    bits = whole(rule["keepbits"], KEEPBITS_MIN, KEEPBITS_MAX, f"Custom compression of '{name}': the number of bits")
                    if bits is not None:
                        rules[name] = {"keepbits": bits}
                elif str(rule.get("mode", "")).lower() in ("lossless", "none"):
                    rules[name] = {"mode": str(rule["mode"]).lower()}
                else:
                    errors.append(f"Custom compression of '{name}': give a step, a number of bits or lossless.")
        if rules:
            opts["custom"] = rules
    elif custom:
        errors.append("Per-variable rules are only for the custom compression.")

    by_name = {v["name"]: v for v in variables}
    used = [by_name.get(v, {"name": v}) for v in out_vars]
    bits_used = [value for mode, value in (variable_storage(preset, opts, v) for v in used) if mode == "bits"]
    if bits_used and min(bits_used) < 8:
        warnings.append(f"{min(bits_used)} significant bits is a relative error of {100 * 2.0 ** -min(bits_used):.1f} %: visible in maps and in differences between fields.")
    rain = [v for v in out_vars if v in PRECIPITATION]
    if preset == "integer" and rain:
        warnings.append(f"Integer compression stores {', '.join(rain)} in whole millimetres: about a third of the grid points "
                        "with rain in 6 h become zero. 'mixed' keeps 0.1 mm.")
    return preset, opts


def estimate_size(model, checkpoint, preset, opts, out_vars, variables, n_levels, res, freq, members, n_dates, days):
    """Rough size in bytes of the outputs of an experiment, per compression preset: {preset: bytes}."""
    if res in GRID_POINTS:
        points = GRID_POINTS[res]
    else:
        points = NATIVE_POINTS.get(model) or next((n for key, n in NATIVE_POINTS.items() if key in str(checkpoint)), 128 * 256)
    by_name = {v["name"]: v for v in variables}
    steps = max(1.0, days * (1 if freq == "daily" else 4))
    sizes = {}
    for name in COMPRESSION_NAMES:
        chosen = opts if name == preset else {}
        if name == "custom" and name != preset:
            continue                                   # depends on rules that have not been given
        total = 0.0
        for var in out_vars:
            info = by_name.get(var, {"name": var})
            fields = n_levels if info.get("type") == "pl" else 1
            mode, value = variable_storage(name, chosen, info)
            if mode == "none":
                rate = BYTES_UNCOMPRESSED.get(model, 8.0)
            elif mode == "lossless":
                rate = BYTES_LOSSLESS
            elif mode == "bits":
                rate = _bits_bytes(value)
            else:
                coarse = (info.get("pack") or {}).get("integer")
                rate = BYTES_INTEGER_FINE if coarse and value < 0.5 * float(coarse) else BYTES_INTEGER
            total += fields * rate
        sizes[name] = total * points * steps * members * n_dates
    return sizes


def compression_yaml(preset, opts):
    """Lines of EXPERIMENT in main.yml. The per-variable rules are a list of texts: Autosubmit hands the jobs the
    single values of the configuration, never a nested mapping."""
    if not preset:
        return ""
    lines = [f"  OUT_COMPRESSION: {preset}"]
    body = []
    if "keepbits" in opts:
        body.append(f"    KEEPBITS: {opts['keepbits']}")
    if "chunks" in opts:
        body.append(f"    CHUNKS: {opts['chunks'] if isinstance(opts['chunks'], str) else '[' + ', '.join(map(str, opts['chunks'])) + ']'}")
    if "complevel" in opts:
        body.append(f"    COMPLEVEL: {opts['complevel']}")
    if opts.get("custom"):
        body.append("    CUSTOM:")
        for name, rule in opts["custom"].items():
            inner = ", ".join(f"{k}: {v!r}" if isinstance(v, float) else f"{k}: {v}" for k, v in rule.items())
            body.append(f'      - "{name}: {{{inner}}}"')
    if body:
        lines += ["  OUT_COMPRESSION_OPTS:"] + body
    return "\n".join(lines) + "\n"



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


def _store_dir(conf):
    """OUTPUT_STORE.DIR of a config file, with the placeholders the Launcher itself fills in main.yml resolved."""
    d = ((conf or {}).get("OUTPUT_STORE") or {}).get("DIR")
    if isinstance(d, str):
        d = d.replace("%PLATFORM.USER_CODE%", settings.REMOTE_USER).replace("%PLATFORM.NAME%", PLATFORM)
    return d


def output_store_defaults(branch, ics_list):
    """{ics: default OUTPUT_STORE.DIR} on this branch; empty when the branch has no output store (no ARCHIVE job)."""
    try:
        general = _store_dir(repo_yaml(branch, "conf/general.yml"))
    except RuntimeError:
        general = None
    if not general:
        return {}
    out = {}
    for ics in ics_list:
        try:
            out[ics] = _store_dir(repo_yaml(branch, f"conf/jobs/{PLATFORM}/ics/{ics}/{ics}.yml")) or general
        except RuntimeError:
            out[ics] = general
    return out


def options(branch):
    """Everything the form needs for one branch: models, checkpoints, ICS, variables, levels."""
    problem = settings.not_set("AIUQ_LAUNCH_PLATFORM", "AIUQ_SUPPORT_FOLDER", "AIUQ_REMOTE_HOST")
    if problem:
        raise ValueError(f"The Launcher is not configured: {problem}.")
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
    try:   # output-only variables (diagnostics, accumulations, mean rates), when the branch has them
        std_out = repo_yaml(branch, f"conf/AIUQ-st/{std_version}/outputs.yml").get("data", {}) or {}
    except RuntimeError:
        std_out = {}
    std_by_code = {int(k): v for k, v in {**std, **std_out}.items()}
    plat_models = repo_ls(branch, f"conf/jobs/{PLATFORM}/models")
    plat_ics = repo_ls(branch, f"conf/jobs/{PLATFORM}/ics")
    store_by_ics = output_store_defaults(branch, plat_ics)
    ckpts = checkpoints_on_hpc()
    support = support_matrix(branch)
    models = []
    for card in repo_ls(branch, "conf/cards/models"):
        c = repo_yaml(branch, f"conf/cards/models/{card}")
        name = c.get("model")
        if not name:
            continue
        known = card_variables(c, std_by_code)
        levels = (c.get("pressure_levels") or {}).get("values") or []
        on_hpc = ckpts.get(name)
        ck_blocks = {k: v for k, v in (c.get("checkpoints") or {}).items() if k != "default" and isinstance(v, dict)}
        model_ck = sorted(set(on_hpc or []) | {k for k in support if _ckpt_model(k) == name and k.endswith((".pkl", ".ckpt"))}
                          | {k for k, v in ck_blocks.items() if v.get("outputs")})
        checkpoints = []
        for k in model_ck:
            listed = (ck_blocks.get(k) or {}).get("outputs")
            checkpoints.append({
                "name": k, "present": on_hpc is not None and k in on_hpc, "ensemble": is_ensemble(k),
                "support": support.get(k) or {},
                # What this checkpoint writes (model card), or None: the model-wide list applies.
                "variables": [v for v in listed if v in known] if listed else None,
                "levels": (ck_blocks.get(k) or {}).get("levels") or None,
            })
        # Model-wide list: every output of the listed checkpoints (older branches: every card variable).
        listed_any = [n for ck in checkpoints for n in (ck["variables"] or [])]
        names = list(dict.fromkeys(listed_any)) if listed_any else list(known)
        models.append({
            "name": name, "label": MODEL_NAMES.get(name, name), "producer": c.get("producer"),
            "on_platform": name in plat_models, "variables": [known[n] for n in names], "levels": levels,
            "amip": name in AMIP_MODELS, "checkpoints": checkpoints,
        })
    models.sort(key=lambda m: (not m["on_platform"], m["label"]))
    res = {"branch": branch, "platform": PLATFORM, "ics": plat_ics, "models": models, "std_version": std_version,
           "forcings": FORCINGS, "resolutions": RESOLUTIONS, "freqs": FREQS, "chunk_units": CHUNK_UNITS,
           "support_folder": SUPPORT_FOLDER, "user_code": settings.REMOTE_USER, "checkpoints_checked": bool(ckpts),
           "store_by_ics": store_by_ics,
           "compression": {"supported": compression_supported(branch),
                           "presets": [{"name": n, "label": l, "description": d} for n, l, d in COMPRESSION_PRESETS],
                           "keepbits": {"min": KEEPBITS_MIN, "max": KEEPBITS_MAX, "default": KEEPBITS_DEFAULT},
                           "chunks": list(CHUNK_NAMES), "precipitation": list(PRECIPITATION),
                           "defaults": {ics: default_compression(ics) for ics in plat_ics}}}
    _opts_cache[branch] = (time.time(), res)
    return res


def card_variables(card, std_by_code):
    """{short name: variable info} of everything a model card maps (inputs, then output-only fields)."""
    out = {}
    for block in ("variables", "outputs"):
        for code, entry in (card.get(block) or {}).items():
            s = std_by_code.get(int(code))
            if not s or s.get("short_name") in out:
                continue
            window = entry.get("window") if isinstance(entry, dict) else None
            period = bool(s.get("accumulated") or window)
            out[s["short_name"]] = {
                "name": s["short_name"], "long": s.get("long_name", ""), "units": str(s.get("units", "")),
                "type": s.get("type", ""), "pack": s.get("pack"),
                "period": window or ("over the output step" if s.get("accumulated") else None),
                "group": "period" if period else s.get("type", ""),
            }
    return out


def _ckpt_model(ck):
    return "aifs" if ck.startswith("aifs") else "ms-aurora" if ck.startswith("aurora") else "neuralgcm"


_ck_cache = {"t": 0, "data": {}}


def checkpoints_on_hpc():
    """{model: [checkpoint files]} found under SUPPORT_FOLDER/models on the HPC (cached 10 min)."""
    if time.time() - _ck_cache["t"] < 600:
        return _ck_cache["data"]
    try:
        r = remote.run_helper("hpc_probe.py", {"user": settings.REMOTE_USER, "exps": {}, "ls": f"{SUPPORT_FOLDER}/models"}, timeout=60)
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
            warnings.append(f"{ck['name']} with {ics.upper()} is marked as work in progress in the support matrix.")
        elif sup == "no":
            warnings.append(f"{ck['name']} with {ics.upper()} is marked as not implemented in the support matrix.")
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
    valid_vars = set((ck or {}).get("variables") or [v["name"] for v in model["variables"]])
    who = f"{model['label']} ({ck['name']})" if ck and ck.get("variables") else model["label"]
    if not out_vars:
        errors.append("Choose at least one output variable.")
    unknown = [v for v in out_vars if v not in valid_vars]
    if unknown:
        errors.append(f"{who} does not produce: {', '.join(unknown)}.")
    model_levels = (ck or {}).get("levels") or model["levels"]
    levs = cfg.get("out_levs")
    if levs != "original":
        levs = sorted({int(x) for x in (levs or [])}, reverse=True)
        if not levs:
            errors.append("Choose output pressure levels (or 'all levels').")
        bad_l = [l for l in levs if model_levels and l not in model_levels]
        if bad_l:
            errors.append(f"{who} has no level {', '.join(map(str, bad_l))} hPa.")
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
    store = str(cfg.get("output_store") or "").strip().rstrip("/") or None
    if store is None:
        pass  # the branch default for these initial conditions (opts["store_by_ics"])
    elif not opts["store_by_ics"]:
        errors.append(f"Branch {branch} has no output store (no ARCHIVE job): leave the output store empty.")
    elif not STORE_RE.match(store) or ".." in store or store.count("/") < 2:
        errors.append("The output store must be an absolute folder on the Autosubmit host, e.g. /data/users/<user>/AIUQ-outputs.")
    elif not os.path.isdir(store):
        if os.path.isdir(os.path.dirname(store)):
            warnings.append(f"The output store {store} does not exist yet: the ARCHIVE job will create it.")
        else:
            errors.append(f"The parent folder of the output store ({os.path.dirname(store)}) does not exist on this host.")
    sims = len(dates) * members * max(num_chunks, 0)
    if sims > 3000:
        warnings.append(f"This creates {sims} simulation jobs.")
    compression, compression_opts = clean_compression(cfg, ics, out_vars, model["variables"], opts["compression"]["supported"],
                                                       errors, warnings)
    days = max(chunk_size, 0) * max(num_chunks, 0) * (30 if unit == "month" else 1)
    sizes = estimate_size(model["name"], ck["name"] if ck else "", compression, compression_opts, out_vars, model["variables"],
                          len(model_levels) if levs == "original" else len(levs or []), res, freq, max(members, 0), len(dates), days)
    if compression == "none" and sizes.get("none", 0) > BIG_RUN_BYTES:
        warnings.append(f"Without compression the outputs take about {sizes['none'] / 1e9:.0f} GB "
                        f"(lossless: about {sizes['lossless'] / 1e9:.0f} GB, with the same values).")
    clean = {"branch": branch, "model": model["name"], "checkpoint": ck["name"] if ck else None, "ics": ics,
             "run_type": run_type, "forcing": forcing, "members": members, "dates": dates, "chunk_size": chunk_size,
             "chunk_unit": unit, "num_chunks": num_chunks, "out_vars": out_vars, "out_levs": levs, "out_res": res,
             "out_freq": freq, "description": desc, "support_folder": SUPPORT_FOLDER, "user_code": settings.REMOTE_USER,
             "output_store": store, "sims": sims, "out_compression": compression,
             "out_compression_opts": compression_opts, "sizes": sizes}
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
{compression_yaml(c.get('out_compression'), c.get('out_compression_opts') or {})}
PATHS:
  SUPPORT_FOLDER: {c['support_folder']}
  SIF_FOLDER: "%PATHS.SUPPORT_FOLDER%/sif"

PLATFORM:
  NAME: {PLATFORM}
  USER_CODE: {c['user_code']}
""" + (f"""
# Finished outputs are moved here (on the Autosubmit host) by the ARCHIVE job.
OUTPUT_STORE:
  DIR: {c['output_store']}
""" if c.get("output_store") else "")


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
            "summary": {k: clean[k] for k in ("sims", "members", "num_chunks", "chunk_size", "chunk_unit")} | {"n_dates": len(clean["dates"])},
            "compression": {"preset": clean["out_compression"], "opts": clean["out_compression_opts"], "sizes": clean["sizes"]}}


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
            "out_freq": conf.get("out_freq") or "daily", "description": f"copy of {expid}",
            "output_store": conf.get("output_store"),
            # None: the experiment was written before compression could be chosen; the form proposes its default
            "out_compression": conf.get("out_compression"), "out_compression_opts": conf.get("out_compression_opts") or {}}


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
DELETE_ROOTS = settings.REMOTE_DELETE_ROOTS
LEFTOVERS = settings.STATE_DIR / "pending_remote_deletes.json"
_left_lock = threading.Lock()


def _remote_path(expid):
    """The experiment's HPCROOTDIR, read from its job history (gone after `autosubmit delete`)."""
    hpc = (local.read_job_data(expid).get("hpc") or {})
    host = hpc.get("HPCHOST")
    if not settings.REMOTE_HOST:
        raise RuntimeError(settings.not_set("AIUQ_REMOTE_HOST"))
    if host and host != settings.REMOTE_HOST:
        return None, f"its jobs ran on {host}; only {settings.REMOTE_HOST} folders are handled here"
    root = hpc.get("HPCROOTDIR") or (f"{settings.REMOTE_BASE}/{expid}" if settings.REMOTE_BASE else None)
    if not root:  # never guess: without the folder the experiment cannot be checked, so it is not deleted
        raise RuntimeError(f"the HPC folder of {expid} is unknown: {settings.not_set('AIUQ_REMOTE_BASE')}")
    return root, None


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
        r = remote.run_helper("hpc_probe.py", {"user": settings.REMOTE_USER, "slurm": True, "exps": {}}, timeout=60)
    except remote.RemoteError as exc:
        raise RuntimeError(f"cannot read the SLURM queue: {exc}") from exc
    return [j for j in (r.get("slurm") or {}).get("jobs", []) if j.get("name", "").startswith(expid + "_")]


def _store_path(expid):
    """Output-store folder of an archived experiment (recorded by the ARCHIVE job in <expid>/OUTPUT_STORE.json)."""
    rec = local.store_marker(expid) or {}
    return rec.get("dir") or None


def _inspect_store(expid, path):
    """Checks before removing an output-store folder (on this host): it must be <store>/<expid>_<suffix>, a real
    folder owned by us, holding LOG_<expid> or archive.json of this expid. Returns the size like the HPC inspect."""
    p = os.path.abspath(path)
    name = os.path.basename(p)
    if p != path.rstrip("/") or ".." in path or p.count("/") < 3:
        return {"ok": False, "error": f"unexpected path {path}"}
    if not re.match(rf"^{re.escape(expid)}_[0-9A-Za-z_]+$", name):
        return {"ok": False, "error": f"{name} is not named {expid}_<time>"}
    if not os.path.lexists(p):
        return {"ok": True, "exists": False}
    if os.path.islink(p) or not os.path.isdir(p):
        return {"ok": False, "error": f"{p} is a link or not a folder"}
    if os.stat(p).st_uid != os.getuid():
        return {"ok": False, "error": f"{p} is not owned by {settings.USER}"}
    try:
        with open(os.path.join(p, "archive.json")) as fh:
            arch_expid = json.load(fh).get("expid")
    except (OSError, ValueError):
        arch_expid = None
    if arch_expid not in (None, expid) or (arch_expid is None and not os.path.isdir(os.path.join(p, f"LOG_{expid}"))):
        return {"ok": False, "error": f"{p} does not hold experiment {expid}"}
    files = size = nc = nc_bytes = 0
    for dirpath, _dirs, fns in os.walk(p, followlinks=False):
        for fn in fns:
            try:
                sz = os.lstat(os.path.join(dirpath, fn)).st_size
            except OSError:
                continue
            files, size = files + 1, size + sz
            if fn.endswith(".nc"):
                nc, nc_bytes = nc + 1, nc_bytes + sz
    return {"ok": True, "exists": True, "files": files, "bytes": size, "nc_files": nc, "nc_bytes": nc_bytes,
            "top": sorted(os.listdir(p))}


def _delete_store(t, expid, path):
    t.step(f"Deleting {path} (output store)")
    _set_leftover(expid, path, True, kind="store")
    info = _inspect_store(expid, path)  # checked again right before removing
    if not info.get("ok"):
        raise RuntimeError(f"{info.get('error')}; listed under “Leftover folders”")
    if info.get("exists"):
        errors = []
        shutil.rmtree(path, onerror=lambda _f, pth, exc: errors.append(f"{pth}: {exc[1]}"))
        if os.path.lexists(path):
            for e in errors[:20]:
                t.say(e)
            raise RuntimeError(f"{len(errors)} files could not be removed; the folder is listed under “Leftover folders”")
    _set_leftover(expid, path, False, kind="store")
    t.say(f"deleted {path}")


def _inspect_remote(expid, path):
    if not DELETE_ROOTS:
        return {"ok": False, "error": settings.not_set("AIUQ_REMOTE_DELETE_ROOTS")}
    return remote.run_helper("hpc_delete.py", {"mode": "inspect", "expid": expid, "path": path, "roots": DELETE_ROOTS}, timeout=180)


def delete_plan(expid):
    """What deleting `expid` would remove, and what currently prevents it."""
    if not EXPID_RE.match(expid) or not (settings.AS_ROOT / expid).is_dir():
        raise ValueError("unknown experiment")
    lpath = settings.AS_ROOT / expid
    files, size = _local_size(lpath)
    blockers = []
    try:
        rpath, rerr = _remote_path(expid)
    except RuntimeError as exc:
        rpath, rerr = None, str(exc)
        blockers.append(f"Nothing will be deleted: {exc}.")
    remote_info = {"ok": False, "error": rerr}
    if rpath:
        try:
            remote_info = _inspect_remote(expid, rpath)
        except remote.RemoteError as exc:
            remote_info = {"ok": False, "error": f"cannot inspect the HPC folder: {exc}"}
    spath = _store_path(expid)
    store_info = _inspect_store(expid, spath) if spath else {"ok": True, "exists": False}
    jobs = _slurm_jobs(expid)
    if expid in local.running_autosubmit():
        blockers.append("autosubmit run is active for this experiment: stop it first.")
    if jobs:
        blockers.append(f"{len(jobs)} job(s) of this experiment are still in the SLURM queue: stop it with “also cancel jobs” first.")
    if rpath and not remote_info.get("ok"):
        blockers.append(f"The HPC folder will not be touched: {remote_info.get('error')}. Fix this before deleting.")
    if spath and not store_info.get("ok"):
        blockers.append(f"The output-store folder will not be touched: {store_info.get('error')}. Fix this before deleting.")
    conf = local.read_conf(expid)
    return {"expid": expid, "description": local.descriptions().get(expid), "model": conf.get("model"), "ics": conf.get("ics"),
            "local": {"path": str(lpath), "files": files, "bytes": size,
                      "extra": [f"Autosubmit database entry of {expid}", f"job history (metadata/data/job_data_{expid}.*)",
                                f"structure database (metadata/structures/structure_{expid}.db)"]},
            "remote": {**remote_info, "path": rpath, "host": settings.REMOTE_HOST},
            "store": {**store_info, "path": spath},
            "slurm_jobs": [{"id": j.get("id"), "name": j.get("name"), "state": j.get("state")} for j in jobs],
            "blockers": blockers}


def leftovers():
    try:
        return json.loads(LEFTOVERS.read_text())
    except (OSError, ValueError):
        return []


def _set_leftover(expid, path, add, kind="hpc"):
    with _left_lock:
        items = [x for x in leftovers() if not (x.get("expid") == expid and x.get("kind", "hpc") == kind)]
        if add:
            items.append({"expid": expid, "path": path, "kind": kind, "since": dt.datetime.now().isoformat(timespec="seconds")})
        settings.STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = LEFTOVERS.with_suffix(".tmp")
        tmp.write_text(json.dumps(items, indent=1))
        os.replace(tmp, LEFTOVERS)


def _delete_remote(t, expid, path):
    t.step(f"Deleting {path} on {settings.REMOTE_HOST}")
    _set_leftover(expid, path, True)  # recorded first, so a failure can be retried
    r = remote.run_helper("hpc_delete.py", {"mode": "delete", "expid": expid, "path": path, "roots": DELETE_ROOTS}, timeout=3600)
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
    rem, sto = plan["remote"], plan["store"]
    audit("delete:start", expid=expid, local_bytes=plan["local"]["bytes"], remote_path=rem.get("path"),
          remote_bytes=rem.get("bytes"), remote_files=rem.get("files"), nc_files=rem.get("nc_files"),
          store_path=sto.get("path"), store_bytes=sto.get("bytes"), store_nc_files=sto.get("nc_files"))

    def work(t):
        t.step("Removing the experiment from Autosubmit (autosubmit delete)")
        _run(t, [AS_BIN, "delete", "-f", expid], timeout=900)
        if (settings.AS_ROOT / expid).exists():
            raise RuntimeError("autosubmit delete finished but the experiment folder is still there; the HPC folder was not touched")
        if sto.get("path") and sto.get("exists"):  # the record that pointed to it is gone with the experiment
            _set_leftover(expid, sto["path"], True, kind="store")
        if rem.get("path") and rem.get("ok") and rem.get("exists"):
            _delete_remote(t, expid, rem["path"])
        else:
            t.say(f"nothing to delete on {settings.REMOTE_HOST} ({rem.get('path') or 'no folder recorded'} does not exist)")
        if sto.get("path") and sto.get("exists"):
            _delete_store(t, expid, sto["path"])
        t.step(f"{expid} deleted", "done")

    return _start(task, work)


def retry_leftover(expid):
    items = [x for x in leftovers() if x.get("expid") == expid]
    if not items:
        raise ValueError("no leftover folder for this experiment")
    if (settings.AS_ROOT / expid).exists() and any(x.get("kind", "hpc") == "hpc" for x in items):
        raise ValueError(f"{expid} exists again in Autosubmit; not touching its HPC folder")
    task = Task("cleanup", expid)
    audit("cleanup:start", expid=expid, paths=[x["path"] for x in items])

    def work(t):
        for item in items:
            if item.get("kind") == "store":  # unique <expid>_<time> folder: safe even if the expid is reused
                _delete_store(t, expid, item["path"])
            else:
                _cleanup_hpc(t, expid, item)
        t.step("Done", "done")

    return _start(task, work)


def _cleanup_hpc(t, expid, item):
    info = _inspect_remote(expid, item["path"])
    if not info.get("ok"):
        raise RuntimeError(info.get("error"))
    if info.get("exists"):
        _delete_remote(t, expid, item["path"])
    else:
        _set_leftover(expid, item["path"], False)
        t.say(f"{item['path']} is already gone")
