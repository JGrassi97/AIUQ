"""Background collector: one refresh cycle every ``settings.REFRESH`` seconds.

Each cycle:
  1. local reads (no ssh): experiment list, conf/*.yml, ``autosubmit run`` processes,
     job_data DB, local STAT files, and pickles whose mtime changed;
  2. exactly one ssh call to the HPC host (``hpc_probe.py``) covering every
     experiment that is due, plus squeue and nvidia-smi. Fetches are incremental
     (only STAT / log files newer than the previous cycle);
  3. derived views (grid, Gantt, durations, ETA, ...) are rebuilt and swapped in
     atomically, so HTTP requests never wait on ssh.
"""
import collections
import json
import logging
import os
import re
import statistics
import threading
import time
import traceback

from . import local, remote, settings

log = logging.getLogger("aiuq_dash.collector")

BUSY = {"RUNNING", "SUBMITTED", "QUEUING", "READY", "PREPARED", "HELD", "DELAYED", "UNKNOWN"}
DONE = {"COMPLETED", "SKIPPED"}
STATUS_ORDER = ["COMPLETED", "RUNNING", "QUEUING", "SUBMITTED", "READY", "PREPARED", "DELAYED",
                "WAITING", "HELD", "SUSPENDED", "FAILED", "UNKNOWN", "SKIPPED"]


class ExpCache:
    def __init__(self, expid):
        self.expid = expid
        self.conf, self.conf_m = {}, -1
        self.jobs, self.pkl_m, self.pkl_error = [], -1, None
        self.jd, self.jd_m = {"runs": [], "rows": [], "hpc": {}}, -1
        self.stats, self.stat_since = {}, 0.0
        self.mars, self.mars_since = {}, 0.0
        self.err, self.err_sig = {}, {}
        self.tails = {}
        self.outputs, self.outputs_t = None, None
        self.outputs_tag = None                   # 'aiuq_compression' of the newest output file
        self.outputs_dir, self.archive = None, None
        self.remote_t, self.remote_error, self.remote_root = None, None, None
        self.local_stats = []


class Collector:
    def __init__(self):
        self.caches = {}
        self.views = {}
        self.overview = []
        self.meta = {"last_cycle": None, "cycle_seconds": None, "errors": [], "refresh": settings.REFRESH,
                     "remote_host": settings.REMOTE_HOST, "remote_ok": None, "remote_last_ok": None,
                     "remote_elapsed": None, "started": time.time()}
        self.slurm, self.gpu = None, None
        self.history = collections.deque()
        self._hist_dirty = 0
        self._load_history()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None

    # ------------------------------------------------------------------ loop
    def start(self):
        self._thread = threading.Thread(target=self._loop, name="collector", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._wake.set()

    def poke(self):
        self._wake.set()

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self.cycle()
            except Exception:  # noqa: BLE001 - never let the thread die
                log.exception("collector cycle failed")
                self.meta["errors"] = [traceback.format_exc(limit=3)]
            self.meta["cycle_seconds"] = round(time.time() - t0, 2)
            self._wake.wait(max(5.0, settings.REFRESH - (time.time() - t0)))
            self._wake.clear()

    # ----------------------------------------------------------------- cycle
    def cycle(self):
        now = time.time()
        errors = []
        expids = local.list_experiments()
        desc = local.descriptions()
        active = local.running_autosubmit()
        for e in list(self.caches):
            if e not in expids:
                del self.caches[e]
        for e in expids:
            self.caches.setdefault(e, ExpCache(e))

        # --- local reads
        stale_pkl = []
        for e in expids:
            c = self.caches[e]
            m = local.conf_mtime(e)
            if m != c.conf_m:
                c.conf, c.conf_m = local.read_conf(e), m
            m = local.mtime(local.job_data_path(e))
            if m != c.jd_m:
                c.jd, c.jd_m = local.read_job_data(e), m
            c.local_stats = local.local_stats(e)
            m = local.mtime(local.pkl_path(e))
            if m != c.pkl_m:
                stale_pkl.append(e)
        if stale_pkl:
            try:
                res = local.read_pkls(stale_pkl)
            except Exception as exc:  # noqa: BLE001
                res = {e: {"error": f"{type(exc).__name__}: {exc}"} for e in stale_pkl}
            for e in stale_pkl:
                c, r = self.caches[e], res.get(e, {"error": "pickle not found"})
                if "jobs" in r:
                    c.jobs, c.pkl_m, c.pkl_error = r["jobs"], local.mtime(local.pkl_path(e)), None
                else:  # keep last good state; retry next cycle
                    c.pkl_error = r.get("error")
                    errors.append(f"{e}: {c.pkl_error}")

        # --- one remote call
        req = {"user": settings.REMOTE_USER, "slurm": True, "gpu": True, "exps": {}}
        for e in expids:
            c = self.caches[e]
            hpc = c.jd.get("hpc") or {}
            host = hpc.get("HPCHOST")
            if not settings.REMOTE_HOST:
                c.remote_error = settings.not_set("AIUQ_REMOTE_HOST")
                continue
            if host and host != settings.REMOTE_HOST:
                c.remote_error = f"jobs run on {host}; only {settings.REMOTE_HOST} is probed"
                continue
            # Once the ARCHIVE job has finished, the experiment folder (logs, outputs) is in the output store.
            marker = local.store_marker(e)
            if marker and marker.get("status") == "done" and marker.get("dir"):
                root, c.archive = marker["dir"], marker
            else:
                root, c.archive = hpc.get("HPCROOTDIR") or (f"{settings.REMOTE_BASE}/{e}" if settings.REMOTE_BASE else None), None
            if not root:  # nothing has run yet and the site settings do not say where experiments live
                c.remote_error = settings.not_set("AIUQ_REMOTE_BASE")
                continue
            if root != c.remote_root:  # moved to the output store: read everything again from there now
                c.remote_root, c.remote_t, c.outputs_t, c.outputs_dir = root, None, None, None
            busy = e in active or any(j["status"] in BUSY for j in c.jobs)
            due = busy or c.remote_t is None or now - c.remote_t >= settings.SLOW_REFRESH
            if not due:
                continue
            failed = [j["name"] for j in c.jobs if j["status"] == "FAILED"]
            want_err = [j for j in failed if j not in c.err or c.err_sig.get(j) != self._sig(c, j)]
            req["exps"][e] = {
                "root": c.remote_root,
                "stat_since": max(0.0, c.stat_since - 5),
                "mars_since": max(0.0, c.mars_since - 5),
                "outputs": busy or c.outputs_t is None or now - c.outputs_t >= settings.SLOW_REFRESH,
                "failed": want_err,
                "running": [j["name"] for j in c.jobs if j["status"] == "RUNNING"],
            }
        try:
            t0 = time.time()
            res = remote.run_helper("hpc_probe.py", req)
            self.meta.update(remote_ok=True, remote_last_ok=time.time(), remote_elapsed=round(time.time() - t0, 2))
            self.slurm, self.gpu = res.get("slurm"), res.get("gpu")
            self._record_history()
            for e, r in res.get("exps", {}).items():
                self._merge_remote(self.caches[e], r, req["exps"][e], now)
        except remote.RemoteError as exc:
            self.meta["remote_ok"] = False
            errors.append(str(exc))
            for e in req["exps"]:
                self.caches[e].remote_error = str(exc)

        # --- derived views
        views, overview = {}, []
        for e in expids:
            try:
                v = build_view(self.caches[e], desc.get(e), active.get(e), self.slurm, self.gpu, now)
            except Exception as exc:  # noqa: BLE001
                log.exception("view %s", e)
                errors.append(f"{e}: view failed: {exc}")
                continue
            views[e] = v
            overview.append({k: v[k] for k in ("expid", "description", "conf", "active", "counts", "summary",
                                                "remote", "pkl_error")})
        borrow_eta(views)
        overview.sort(key=lambda x: (not x["active"], x["expid"]))
        self.views, self.overview = views, overview
        self.meta.update(last_cycle=time.time(), errors=errors)

    # ----------------------------------------------------- GPU / queue history
    def _hist_path(self):
        return settings.STATE_DIR / "history.json"

    def _load_history(self):
        try:
            data = json.loads(self._hist_path().read_text())
            cutoff = time.time() - settings.HISTORY_DAYS * 86400
            self.history.extend(h for h in data if h.get("t", 0) >= cutoff)
        except (OSError, ValueError):
            pass

    def _record_history(self):
        g, sl = self.gpu or {}, self.slurm or {}
        jobs = sl.get("jobs", [])
        entry = {
            "t": round(time.time()),
            "util": [float(x.get("utilization.gpu") or 0) for x in g.get("gpus", [])],
            "mem": [round(float(x.get("memory.used") or 0) / max(1.0, float(x.get("memory.total") or 1)) * 100, 1) for x in g.get("gpus", [])],
            "running": sum(1 for j in jobs if j.get("state") == "RUNNING"),
            "pending": sum(1 for j in jobs if j.get("state") == "PENDING"),
            "by_exp": dict(collections.Counter(j.get("name", "").split("_")[0] for j in jobs if j.get("state") == "RUNNING")),
        }
        self.history.append(entry)
        cutoff = time.time() - settings.HISTORY_DAYS * 86400
        while self.history and self.history[0]["t"] < cutoff:
            self.history.popleft()
        self._hist_dirty += 1
        if self._hist_dirty >= 4:  # write every few cycles, atomically
            self._hist_dirty = 0
            try:
                settings.STATE_DIR.mkdir(parents=True, exist_ok=True)
                tmp = self._hist_path().with_suffix(".tmp")
                tmp.write_text(json.dumps(list(self.history), separators=(",", ":")))
                os.replace(tmp, self._hist_path())
            except OSError:
                log.warning("could not save history", exc_info=True)

    @staticmethod
    def _sig(c, job):
        j = next((x for x in c.jobs if x["name"] == job), {})
        last = max((s["mtime"] for (n, _), s in c.stats.items() if n == job), default=0)
        return f'{j.get("fail_count")}:{last}'

    def _merge_remote(self, c, r, opt, now):
        if "error" in r:
            c.remote_error = r["error"]
            return
        c.outputs_dir = r.get("outputs_dir")
        if not r.get("exists"):
            c.remote_error = f"{c.remote_root} not found on {settings.REMOTE_HOST}"
            c.remote_t = now
            return
        c.remote_error = None
        for s in r.get("stat", []):
            c.stats[(s["job"], s["n"])] = s
        c.stat_since = max(c.stat_since, r.get("stat_mtime") or 0)
        c.mars.update(r.get("mars", {}))
        c.mars_since = max(c.mars_since, r.get("mars_mtime") or 0)
        for job, rec in r.get("err", {}).items():
            c.err[job] = rec
            c.err_sig[job] = self._sig(c, job)
        c.tails = r.get("tail", {})
        if "outputs" in r:
            c.outputs, c.outputs_t = r["outputs"], now
            c.outputs_tag = r.get("outputs_tag")
        c.remote_t = now


# ---------------------------------------------------------------------- views
def _q(values, q):
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def _parallelism(intervals):
    """Average number of concurrently running jobs over the busy time."""
    iv = sorted((a, b) for a, b in intervals if a and b and b > a)
    if not iv:
        return 1.0
    covered, cur_a, cur_b = 0.0, iv[0][0], iv[0][1]
    for a, b in iv[1:]:
        if a > cur_b:
            covered += cur_b - cur_a
            cur_a, cur_b = a, b
        else:
            cur_b = max(cur_b, b)
    covered += cur_b - cur_a
    total = sum(b - a for a, b in iv)
    return max(1.0, total / covered) if covered > 0 else 1.0


def _member_key(m):
    try:
        return (0, int(m))
    except (TypeError, ValueError):
        return (1, str(m))


LEGACY_BITS = re.compile(r"^float32, BitRound keepbits=(\d+),")


def compression_info(conf, tag):
    """How the outputs of an experiment are stored: {"preset", "source", "keepbits", "variables", "detail"}.

    The files say it themselves in their 'aiuq_compression' attribute (JSON; the forecasts compressed by the
    earlier scripts carry one of two texts), and that wins: it is what is on disk. Without files, the
    experiment configuration (EXPERIMENT.OUT_COMPRESSION), and failing that nothing is known.
    """
    conf = conf or {}
    if tag:
        info = None
        try:
            parsed = json.loads(tag)
            if isinstance(parsed, dict) and parsed.get("preset"):
                info = {"preset": parsed["preset"], "keepbits": parsed.get("keepbits"), "variables": parsed.get("variables") or {},
                        "chunks": parsed.get("chunks")}
        except ValueError:
            match = LEGACY_BITS.match(tag)
            if match:
                info = {"preset": "bits", "keepbits": int(match.group(1)), "variables": {}, "chunks": "point"}
            elif tag.startswith("AIUQ mixed integer packing v1"):
                info = {"preset": "mixed", "keepbits": 12, "variables": {}, "chunks": "point"}
        if info:
            return {**info, "source": "files", "detail": tag}
    if conf.get("out_compression"):
        opts = conf.get("out_compression_opts") or {}
        return {"preset": conf["out_compression"], "keepbits": opts.get("keepbits"), "variables": {}, "chunks": opts.get("chunks"),
                "source": "configuration", "detail": None}
    return None


def build_view(c, description, active, slurm, gpu, now):
    conf = c.conf or {}
    jobs = c.jobs
    by_name = {j["name"]: j for j in jobs}
    counts = {}
    for j in jobs:
        counts[j["status"]] = counts.get(j["status"], 0) + 1
    sections, section_counts = [], {}
    for j in jobs:
        if j["section"] not in sections:
            sections.append(j["section"])
        sc = section_counts.setdefault(j["section"], {})
        sc[j["status"]] = sc.get(j["status"], 0) + 1

    # ---- attempts (STAT files are the primary source; job_data DB as fallback)
    stats = list(c.stats.values()) + list(c.local_stats)
    attempts = []
    source = "STAT files"
    if stats:
        for s in stats:
            j = by_name.get(s["job"], {})
            result = s.get("result")
            end = s.get("end")
            if not result and not end:
                result = "RUNNING" if j.get("status") == "RUNNING" else "INTERRUPTED"
            attempts.append({"job": s["job"], "n": s["n"], "start": s.get("start"), "end": end,
                             "result": result or "UNKNOWN", "section": j.get("section") or _section_of(s["job"]),
                             "date": j.get("date"), "member": j.get("member"), "chunk": j.get("chunk")})
    else:
        source = "Autosubmit job_data DB"
        for r in c.jd.get("rows", []):
            if r.get("start"):
                attempts.append({"job": r["job"], "n": r.get("run_id"), "start": r["start"],
                                 "end": r.get("finish") or None, "result": r.get("status"),
                                 "section": r.get("section"), "date": r.get("date"),
                                 "member": r.get("member"), "chunk": r.get("chunk")})
    attempts.sort(key=lambda a: a["start"] or 0)
    last_attempt = {}
    for a in attempts:
        last_attempt[a["job"]] = a

    # ---- durations & ETA (estimates)
    durations, dur_stats, eta = {}, {}, {"sections": {}, "total": 0.0, "partial": False}
    for sec in sections:
        done = [a for a in attempts if a["section"] == sec and a["result"] == "COMPLETED"
                and a["start"] and a["end"] and a["end"] >= a["start"]]
        ds = [a["end"] - a["start"] for a in done]
        durations[sec] = ds
        if ds:
            dur_stats[sec] = {"n": len(ds), "median": statistics.median(ds), "p10": _q(ds, .1),
                              "p90": _q(ds, .9), "mean": statistics.fmean(ds), "max": max(ds)}
        sec_jobs = [j for j in jobs if j["section"] == sec]
        remaining = [j for j in sec_jobs if j["status"] not in DONE and j["status"] != "FAILED"]
        if not remaining:
            continue
        running_elapsed = []
        for j in remaining:
            a = last_attempt.get(j["name"])
            if j["status"] == "RUNNING" and a and a["start"] and not a["end"]:
                running_elapsed.append(now - a["start"])
        if not ds:
            eta["partial"] = True
            eta["sections"][sec] = {"remaining": len(remaining), "seconds": None, "running_elapsed": running_elapsed}
            continue
        med = statistics.median(ds)
        work = 0.0
        for j in remaining:
            a = last_attempt.get(j["name"])
            if j["status"] == "RUNNING" and a and a["start"] and not a["end"]:
                work += max(med - (now - a["start"]), 0.0)
            else:
                work += med
        par = _parallelism([(a["start"], a["end"]) for a in done])
        secs = work / par
        eta["sections"][sec] = {"remaining": len(remaining), "seconds": secs, "median": med, "parallel": par,
                                "overdue": any(x > med for x in running_elapsed)}
        eta["total"] += secs

    # ---- grid: date x (section, member, chunk)
    members_order = conf.get("members") or []
    col_defs, once = [], []
    for sec in sections:
        sj = [j for j in jobs if j["section"] == sec]
        if any(j.get("member") for j in sj):
            mems = sorted({j["member"] for j in sj if j.get("member")},
                          key=lambda m: (members_order.index(m) if m in members_order else 1e9, _member_key(m)))
            chunks = sorted({j.get("chunk") for j in sj if j.get("chunk") is not None})
            for m in mems:
                for ch in chunks or [None]:
                    label = f"{sec} m{m}" + (f" c{ch}" if len(chunks) > 1 else "")
                    col_defs.append({"key": (sec, m, ch), "section": sec, "member": m, "chunk": ch, "label": label})
        elif any(j.get("date") for j in sj):
            col_defs.append({"key": (sec, None, None), "section": sec, "member": None, "chunk": None, "label": sec})
        else:
            once += [{"name": j["name"], "status": j["status"], "section": sec} for j in sj]
    dates = sorted({j["date"] for j in jobs if j.get("date")})
    di = {d: i for i, d in enumerate(dates)}
    ci = {cd["key"]: i for i, cd in enumerate(col_defs)}
    cells = [[None] * len(col_defs) for _ in dates]
    for j in jobs:
        if not j.get("date"):
            continue
        key = (j["section"], j.get("member"), j.get("chunk") if j.get("member") else None)
        if key not in ci:
            key = (j["section"], None, None)
        if key in ci:
            a = last_attempt.get(j["name"])
            dur = (a["end"] - a["start"]) if a and a.get("end") and a.get("start") else None
            cells[di[j["date"]]][ci[key]] = [j["status"], j["name"], j.get("fail_count") or 0, dur]
    for cd in col_defs:
        cd.pop("key")

    # ---- ICS phase (MARS)
    ics = []
    for j in jobs:
        if j["section"] != "PREPARE_ICS":
            continue
        a = last_attempt.get(j["name"])
        ics.append({"job": j["name"], "date": j.get("date"), "status": j["status"],
                    "fail_count": j.get("fail_count"), "mars": c.mars.get(j["name"]),
                    "start": a["start"] if a else None, "end": a["end"] if a else None})
    ics.sort(key=lambda x: x["date"] or "")

    # ---- failed jobs
    failed = []
    for j in jobs:
        if j["status"] != "FAILED":
            continue
        a = last_attempt.get(j["name"])
        failed.append({**{k: j.get(k) for k in ("name", "section", "date", "member", "chunk", "fail_count",
                                                  "platform", "id")},
                       "start": a["start"] if a else None, "end": a["end"] if a else None,
                       "attempts": sum(1 for x in attempts if x["job"] == j["name"]),
                       "err": c.err.get(j["name"])})
    failed.sort(key=lambda f: -(f["end"] or 0))

    running = []
    for j in jobs:
        if j["status"] in ("RUNNING", "QUEUING", "SUBMITTED"):
            a = last_attempt.get(j["name"])
            running.append({"name": j["name"], "status": j["status"], "section": j["section"],
                            "platform": j.get("platform"), "id": j.get("id"),
                            "start": a["start"] if a and not a.get("end") else None,
                            "tail": c.tails.get(j["name"])})

    # ---- SLURM / GPU (shared host data, filtered to this experiment where possible)
    sl_jobs = [s for s in (slurm or {}).get("jobs", []) if s.get("name", "").startswith(c.expid + "_")]

    # ---- outputs
    out_files = c.outputs or []
    by_vm = {}
    for var, mem, fn, size, mt in out_files:
        d = by_vm.setdefault(var, {}).setdefault(mem, {"n": 0, "size": 0, "latest": 0})
        d["n"] += 1
        d["size"] += size
        d["latest"] = max(d["latest"], mt)

    # ---- summary
    total = len(jobs)
    completed = counts.get("COMPLETED", 0)
    runs = c.jd.get("runs") or []
    run = runs[-1] if runs else {}
    run_start = run.get("start") or (attempts[0]["start"] if attempts else None)
    last_activity = max([a["end"] or a["start"] or 0 for a in attempts] + [0]) or None
    if active:
        elapsed = now - run_start if run_start else None
    else:
        end_ref = run.get("finish") or last_activity
        elapsed = (end_ref - run_start) if (run_start and end_ref and end_ref >= run_start) else None
    summary = {
        "total": total, "completed": completed,
        "pct": (100.0 * completed / total) if total else 0.0,
        "running": counts.get("RUNNING", 0),
        "queued": sum(counts.get(s, 0) for s in ("SUBMITTED", "QUEUING", "READY", "PREPARED")),
        "failed": counts.get("FAILED", 0),
        "waiting": counts.get("WAITING", 0),
        "run_start": run_start, "run_finish": run.get("finish") or None, "runs": len(runs),
        "elapsed": elapsed, "last_activity": last_activity,
        "eta": eta if (total - completed) else None,
    }

    perf = build_perf(jobs, attempts, last_attempt, sections, c.jd.get("limits") or {}, now)

    return {
        "perf": perf,
        "expid": c.expid, "description": description, "conf": conf,
        "active": bool(active), "pid": (active or {}).get("pid"), "active_since": (active or {}).get("since"),
        "counts": counts, "summary": summary, "sections": sections, "status_order": STATUS_ORDER,
        "section_counts": section_counts,
        "grid": {"dates": dates, "columns": col_defs, "cells": cells}, "once_jobs": once,
        "attempts": attempts, "attempts_source": source,
        "durations": durations, "dur_stats": dur_stats,
        "ics": ics, "failed": failed, "running": running,
        "slurm": {"jobs": sl_jobs, "all_jobs": (slurm or {}).get("jobs", []), "ok": (slurm or {}).get("ok"),
                  "error": (slurm or {}).get("error"), "nodes": (slurm or {}).get("nodes", [])},
        "gpu": gpu,
        "outputs": {"files": out_files, "by_var_member": by_vm, "n": len(out_files),
                    "size": sum(f[3] for f in out_files), "fetched": c.outputs_t,
                    "dir": c.outputs_dir or f"{c.remote_root}/outputs", "archive": c.archive,
                    "compression": compression_info(c.conf, c.outputs_tag)},
        "remote": {"t": c.remote_t, "error": c.remote_error, "root": c.remote_root, "host": settings.REMOTE_HOST},
        "pkl_error": c.pkl_error, "pkl_mtime": c.pkl_m,
    }


MONTH = 30 * 86400


def build_perf(jobs, attempts, last_attempt, sections, limits, now):
    """Where the time goes, per job step: waiting for Autosubmit, queued in SLURM, running.

    ready/submit come from the job list (latest attempt), start/end from STAT files.
    """
    per = {sec: {"sched": [], "queue": [], "run": [], "attempts": 0, "failed": 0, "compute": 0.0} for sec in sections}
    for j in jobs:
        rec = per.get(j["section"])
        if rec is None:
            continue
        a = last_attempt.get(j["name"])
        ready, submit = j.get("ready"), j.get("submit")
        start = (a or {}).get("start") or j.get("start")
        end = (a or {}).get("end")
        if ready and submit and 0 <= submit - ready < MONTH:
            rec["sched"].append(submit - ready)
        if submit and start and -120 <= start - submit < MONTH:
            rec["queue"].append(max(0.0, start - submit))
        if start and end and 0 <= end - start < MONTH:
            rec["run"].append(end - start)
    for a in attempts:
        rec = per.get(a["section"])
        if rec is None or not a.get("start"):
            continue
        rec["attempts"] += 1
        rec["failed"] += a["result"] == "FAILED"
        rec["compute"] += max(0.0, (a.get("end") or now) - a["start"])
    # idle periods: nothing of this experiment was running
    iv = sorted((a["start"], a.get("end") or now) for a in attempts if a.get("start"))
    gaps, merged = [], []
    for s0, e0 in iv:
        if merged and s0 <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e0)
        else:
            if merged and s0 - merged[-1][1] > 900:
                gaps.append([merged[-1][1], s0])
            merged.append([s0, e0])
    busy = sum(e - s for s, e in merged)
    span = (merged[-1][1] - merged[0][0]) if merged else 0
    total_compute = sum(r["compute"] for r in per.values())
    return {
        "sections": per,
        "gaps": sorted(gaps, key=lambda g: g[0] - g[1])[:20],
        "gap_total": sum(e - s for s, e in gaps),
        "busy": busy, "span": span, "compute": total_compute,
        "parallelism": (total_compute / busy) if busy else None,
        "limits": limits,
    }


BORROW_KEYS = ("model", "checkpoint", "ics", "chunk_size", "chunk_unit")


def borrow_eta(views):
    """For sections with no completed job yet, estimate from comparable experiments.

    PREPARE_ICS matches on the ICS source; other sections need the same model,
    checkpoint, ICS and chunk length. The donors are recorded and shown in the UI.
    """
    for v in views.values():
        eta = v["summary"].get("eta")
        if not eta:
            continue
        changed = False
        for sec, e in eta["sections"].items():
            if e.get("seconds") is not None:
                continue
            pool, donors = [], []
            for o in views.values():
                if o is v or not o["durations"].get(sec):
                    continue
                keys = ("ics",) if sec == "PREPARE_ICS" else BORROW_KEYS
                if all(o["conf"].get(k) == v["conf"].get(k) for k in keys):
                    pool += o["durations"][sec]
                    donors.append(o["expid"])
            if not pool:
                continue
            med = statistics.median(pool)
            run_el = e.get("running_elapsed") or []
            work = (e["remaining"] - len(run_el)) * med + sum(max(med - x, 0.0) for x in run_el)
            e.update(seconds=work, median=med, parallel=1.0, borrowed=donors,
                     overdue=any(x > med for x in run_el))
            changed = True
        if changed:
            eta["total"] = sum(x["seconds"] or 0 for x in eta["sections"].values())
            eta["partial"] = any(x.get("seconds") is None for x in eta["sections"].values())
            eta["borrowed"] = True


def _section_of(job):
    """'a00a_19860101_1_1_SIM' -> 'SIM', 'a00a_19860101_PREPARE_ICS' -> 'PREPARE_ICS'."""
    parts = job.split("_")[1:]
    return "_".join(p for p in parts if not p.isdigit()) or job
