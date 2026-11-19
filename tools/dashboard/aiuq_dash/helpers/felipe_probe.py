"""Read-only probe executed on the HPC login node (felipe) once per refresh cycle.

The dashboard sends this file through ssh stdin (``python3 - <b64 json>``), so
nothing is installed on felipe. It uses only the standard library and is kept
compatible with older Python 3 versions.

Request (base64 JSON in argv[1]):
  {"user": "jgrassi", "slurm": true, "gpu": true,
   "exps": {"<expid>": {"root": "<HPCROOTDIR>", "stat_since": 0, "mars_since": 0,
                        "outputs": true, "failed": [job, ...], "running": [job, ...]}}}

"root" is the HPC working folder or, once the ARCHIVE job is done, the experiment's output-store folder
(same layout; felipe reads /data at the same path).

Response (JSON on stdout): see ``main``.
"""
import base64
import json
import os
import re
import subprocess
import sys
import time

STAT_RE = re.compile(r"^(?P<job>.+)_STAT_(?P<n>\d+)$")
TS_LOG_RE = re.compile(r"^(?P<job>.+)\.(?P<ts>\d{14})\.(?P<kind>out|err)$")
CMD_LOG_RE = re.compile(r"^(?P<job>.+)\.cmd\.(?P<kind>out|err)\.(?P<n>\d+)$")
TAIL_BYTES = 256 * 1024
MARS_BYTES = 4 * 1024 * 1024


def read_tail(path, nbytes=TAIL_BYTES):
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            if size > nbytes:
                fh.seek(size - nbytes)
            data = fh.read()
        text = data.decode("utf-8", "replace")
        if size > nbytes:  # drop the partial first line
            text = text.split("\n", 1)[-1]
        return text
    except (IOError, OSError):
        return ""


def useful_lines(text, script=None):
    """Drop the shell xtrace and carriage-return noise.

    xtrace lines start with '+'. A traced command with a multi-line argument is
    printed as a '+' line opening a single-quoted string, followed by unprefixed
    continuation lines until the quote closes; those are skipped too.
    """
    out, in_quote = [], False
    for line in text.splitlines():
        if "\r" in line:
            line = line.split("\r")[-1]
        if in_quote:
            if line.count("'") % 2 == 1:
                in_quote = False
            continue
        if line.lstrip().startswith("+"):
            if line.count("'") % 2 == 1:
                in_quote = True
            continue
        if not line.strip():
            continue
        if script and line.strip() in script:  # echoed script source (set -v)
            continue
        out.append(line.rstrip())
    return out


_LIB_CACHE = {}


def script_lines(logdir, job):
    """Stripped source lines of the job's .cmd and of <root>/lib/*.sh.

    Jobs run with ``set -v``, which echoes the script (and sourced libraries) to
    stderr; those lines are dropped from excerpts.
    """
    text = read_tail(os.path.join(logdir, job + ".cmd"), 1024 * 1024)
    lines = set(l.strip() for l in text.splitlines() if len(l.strip()) > 1)
    libdir = os.path.join(os.path.dirname(logdir), "lib")
    if libdir not in _LIB_CACHE:
        lib = set()
        if os.path.isdir(libdir):
            for fn in os.listdir(libdir):
                if fn.endswith(".sh"):
                    lib.update(l.strip() for l in read_tail(os.path.join(libdir, fn), 1024 * 1024).splitlines()
                               if len(l.strip()) > 1)
        _LIB_CACHE[libdir] = lib
    return lines | _LIB_CACHE[libdir]


def after_last_trace(text):
    """Only what was written after the last xtrace line (the running command's own output)."""
    lines = text.splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("+"):
            return "\n".join(lines[i + 1:])
    return text


def error_excerpt(text, max_lines=80, script=None):
    lines = useful_lines(text, script)
    idx = -1
    for i, line in enumerate(lines):
        if "Traceback (most recent call last)" in line:
            idx = i
    if idx >= 0:
        return lines[idx:idx + max_lines], True
    return lines[-40:], False


def parse_stat(path):
    try:
        with open(path) as fh:
            raw = [l.strip() for l in fh.read().splitlines() if l.strip()]
    except (IOError, OSError):
        return None
    nums = [int(l) for l in raw if l.isdigit()]
    words = [l for l in raw if not l.isdigit()]
    return {
        "start": nums[0] if len(nums) > 0 else None,
        "end": nums[1] if len(nums) > 1 else None,
        "result": words[-1] if words else None,
    }


TS_PREFIX = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
MARS_PATTERNS = [
    ("request_id", re.compile(r"Web API request id: (\S+)")),
    ("requesting", re.compile(r"Requesting (\d+) fields")),
    ("state", re.compile(r"Request is (\w+)")),
    ("cost", re.compile(r"Request cost: (\d+) fields, ([\d.]+) (\w)bytes")),
    ("transfer", re.compile(r"Transferr?ing (.*)")),
    ("done", re.compile(r"\bDone \((.*)\)")),
    # case-sensitive on purpose: MARS prints "No errors reported" on success
    ("error", re.compile(r"(\bERROR\b|APIException|Exception:|Traceback)")),
]
UNIT = {"K": 1.0 / 1024, "M": 1.0, "G": 1024.0, "T": 1024.0 ** 2}


def parse_mars(path):
    text = read_tail(path, MARS_BYTES)
    res = {"requests": 0, "fields": 0, "cost_fields": 0, "tape_mb": 0.0,
           "states": {}, "last_state": None, "transfers": 0, "done": 0,
           "errors": 0, "first_ts": None, "last_ts": None, "events": []}
    for line in text.splitlines():
        if line.lstrip().startswith("+"):
            continue
        m = TS_PREFIX.match(line)
        if m:
            res["first_ts"] = res["first_ts"] or m.group(1)
            res["last_ts"] = m.group(1)
        key_hit = None
        for key, rx in MARS_PATTERNS:
            mm = rx.search(line)
            if not mm:
                continue
            key_hit = key
            if key == "request_id":
                res["requests"] += 1
            elif key == "requesting":
                res["fields"] += int(mm.group(1))
            elif key == "state":
                st = mm.group(1).lower()
                res["states"][st] = res["states"].get(st, 0) + 1
                res["last_state"] = st
            elif key == "cost":
                res["cost_fields"] += int(mm.group(1))
                res["tape_mb"] += float(mm.group(2)) * UNIT.get(mm.group(3).upper(), 1.0)
            elif key == "transfer":
                res["transfers"] += 1
            elif key == "done":
                res["done"] += 1
            elif key == "error":
                res["errors"] += 1
            break
        if key_hit:
            res["events"].append(line.strip()[:240])
    res["events"] = res["events"][-40:]
    return res


def index_logs(logdir):
    """Single directory pass: STAT files and per-job stdout/stderr candidates."""
    stats, logs = [], {}
    if not os.path.isdir(logdir):
        return stats, logs
    for name in os.listdir(logdir):
        m = STAT_RE.match(name)
        if m:
            stats.append((m.group("job"), int(m.group("n")), name))
            continue
        m = CMD_LOG_RE.match(name) or TS_LOG_RE.match(name)
        if m:
            logs.setdefault(m.group("job"), {}).setdefault(m.group("kind"), []).append(name)
    return stats, logs


def newest(logdir, names):
    best, best_m = None, -1
    for n in names or []:
        try:
            mt = os.path.getmtime(os.path.join(logdir, n))
        except OSError:
            continue
        if mt > best_m:
            best, best_m = n, mt
    return best, best_m


def list_outputs(outdir):
    files = []
    if not os.path.isdir(outdir):
        return files
    for var in sorted(os.listdir(outdir)):
        vdir = os.path.join(outdir, var)
        if var in ("checkpoints", "tmp", "temp") or not os.path.isdir(vdir):
            continue
        for mem in sorted(os.listdir(vdir)):
            mdir = os.path.join(vdir, mem)
            if not os.path.isdir(mdir):
                continue
            for fn in os.listdir(mdir):
                if not fn.endswith(".nc"):
                    continue
                try:
                    st = os.stat(os.path.join(mdir, fn))
                except OSError:
                    continue
                files.append([var, mem, fn, st.st_size, st.st_mtime])
    return files


def probe_exp(expid, opt):
    root = opt["root"]
    logdir = os.path.join(root, "LOG_%s" % expid)
    res = {"exists": os.path.isdir(root), "logdir": logdir}
    outdir = os.path.join(root, "outputs")
    res["outputs_dir"] = outdir
    if not res["exists"]:
        return res
    stats, logs = index_logs(logdir)

    stat_since = opt.get("stat_since", 0) or 0
    out_stats, max_m = [], stat_since
    for job, n, name in stats:
        path = os.path.join(logdir, name)
        try:
            mt = os.path.getmtime(path)
        except OSError:
            continue
        if mt <= stat_since:
            continue
        rec = parse_stat(path)
        if rec is None:
            continue
        rec.update({"job": job, "n": n, "mtime": mt})
        out_stats.append(rec)
        max_m = max(max_m, mt)
    res["stat"] = out_stats
    res["stat_mtime"] = max_m

    mars_since = opt.get("mars_since", 0) or 0
    mars, mars_max = {}, mars_since
    for job, kinds in logs.items():
        if not job.endswith("_PREPARE_ICS"):
            continue
        name, mt = newest(logdir, kinds.get("out"))
        if name is None or mt <= mars_since:
            continue
        rec = parse_mars(os.path.join(logdir, name))
        rec.update({"file": name, "mtime": mt})
        mars[job] = rec
        mars_max = max(mars_max, mt)
    res["mars"] = mars
    res["mars_mtime"] = mars_max

    errs = {}
    for job in opt.get("failed", [])[:60]:
        name, mt = newest(logdir, logs.get(job, {}).get("err"))
        if name is None:
            errs[job] = {"file": None, "lines": [], "traceback": False}
            continue
        script = script_lines(logdir, job)
        lines, tb = error_excerpt(read_tail(os.path.join(logdir, name)), script=script)
        if not tb:  # Python tracebacks sometimes land in stdout
            oname, _ = newest(logdir, logs.get(job, {}).get("out"))
            if oname:
                olines, otb = error_excerpt(read_tail(os.path.join(logdir, oname)), script=script)
                if otb:
                    lines, tb, name = olines, otb, oname
        errs[job] = {"file": name, "mtime": mt, "lines": lines, "traceback": tb}
    res["err"] = errs

    tails = {}
    for job in opt.get("running", [])[:20]:
        rec = {}
        script = script_lines(logdir, job)
        for kind in ("out", "err"):
            name, mt = newest(logdir, logs.get(job, {}).get(kind))
            if name:
                text = read_tail(os.path.join(logdir, name), 64 * 1024)
                if kind == "err":
                    text = after_last_trace(text)
                rec[kind] = {"file": name, "mtime": mt, "lines": useful_lines(text, script)[-15:]}
        tails[job] = rec
    res["tail"] = tails

    if opt.get("outputs"):
        res["outputs"] = list_outputs(outdir)
    return res


def run(cmd, timeout=20):
    try:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate(timeout=timeout)
        return p.returncode, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        return -1, "", str(exc)


def slurm(user):
    fmt = "%i|%j|%T|%M|%l|%N|%R|%S|%V|%b|%P"
    rc, out, err = run(["squeue", "-h", "-u", user, "-o", fmt])
    keys = ["id", "name", "state", "time", "limit", "nodes", "reason", "start", "submit", "gres", "partition"]
    jobs = [dict(zip(keys, l.split("|"))) for l in out.splitlines() if l.strip()]
    rc2, out2, _ = run(["sinfo", "-h", "-N", "-o", "%N|%P|%T|%G|%C"])
    nodes = [dict(zip(["node", "partition", "state", "gres", "cpus"], l.split("|")))
             for l in out2.splitlines() if l.strip()]
    return {"ok": rc == 0, "error": err.strip() if rc else None, "jobs": jobs, "nodes": nodes}


def slurm_env(pid):
    try:
        with open("/proc/%s/environ" % pid, "rb") as fh:
            env = dict(kv.split(b"=", 1) for kv in fh.read().split(b"\0") if b"=" in kv)
        return env.get(b"SLURM_JOB_ID", b"").decode() or None, env.get(b"SLURM_JOB_NAME", b"").decode() or None
    except (IOError, OSError):
        return None, None


def gpu():
    q = "index,uuid,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw"
    rc, out, err = run(["nvidia-smi", "--query-gpu=" + q, "--format=csv,noheader,nounits"])
    if rc != 0:
        return {"ok": False, "error": (err or out).strip()[:300], "gpus": [], "procs": []}
    keys = q.split(",")
    gpus = []
    for l in out.splitlines():
        vals = [v.strip() for v in l.split(",")]
        if len(vals) == len(keys):
            gpus.append(dict(zip(keys, vals)))
    rc, out, _ = run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory,process_name",
                      "--format=csv,noheader,nounits"])
    procs = []
    for l in out.splitlines():
        vals = [v.strip() for v in l.split(",")]
        if len(vals) >= 4:
            jid, jname = slurm_env(vals[1])
            procs.append({"gpu_uuid": vals[0], "pid": vals[1], "mem": vals[2],
                          "process": vals[3].split("/")[-1], "slurm_id": jid, "slurm_name": jname})
    return {"ok": True, "gpus": gpus, "procs": procs}


def main():
    req = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
    t0 = time.time()
    out = {"time": t0, "exps": {}}
    for expid, opt in req.get("exps", {}).items():
        try:
            out["exps"][expid] = probe_exp(expid, opt)
        except Exception as exc:  # noqa: BLE001
            out["exps"][expid] = {"error": "%s: %s" % (type(exc).__name__, exc)}
    if req.get("slurm"):
        out["slurm"] = slurm(req.get("user") or os.environ.get("USER", ""))
    if req.get("gpu"):
        out["gpu"] = gpu()
    if req.get("ls"):  # {subdir: [files]} of one directory (used to list model checkpoints)
        base = req["ls"]
        listing = {}
        if os.path.isdir(base):
            for d in sorted(os.listdir(base)):
                if os.path.isdir(os.path.join(base, d)):
                    listing[d] = sorted(os.listdir(os.path.join(base, d)))
        out["ls"] = listing
    out["elapsed"] = time.time() - t0
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    main()
