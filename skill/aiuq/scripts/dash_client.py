"""Client for the AIUQ dashboard API. Runs ON the Autosubmit host (mafalda), streamed over ssh stdin.

    ssh mafalda python3 - METHOD PATH MODE [BODY_B64] < dash_client.py

Talks HTTP over the dashboard's private Unix socket (~/.aiuq-dashboard/dash.sock) and prints a
compact, agent-friendly text summary (MODE) or the raw JSON (MODE=raw). Standard library only.
Exit code: 0 ok, 2 HTTP/API error, 3 dashboard unreachable.
"""
import base64
import http.client
import json
import os
import socket
import sys
import time

SOCK = os.environ.get("AIUQ_DASH_SOCK", os.path.expanduser("~/.aiuq-dashboard/dash.sock"))


class UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, path, timeout=180):
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect(self._path)
        self.sock = s


def call(method, path, body=None):
    conn = UnixHTTPConnection(SOCK)
    headers = {"Host": "localhost", "X-AIUQ-Launcher": "1"}
    if body is not None:
        headers["Content-Type"] = "application/json"
    try:
        conn.request(method, path, body=body, headers=headers)
        r = conn.getresponse()
        raw = r.read().decode("utf-8", "replace")
    except (OSError, socket.error) as exc:
        print("dashboard unreachable on %s: %s" % (SOCK, exc), file=sys.stderr)
        print("hint: on mafalda run `systemctl --user status aiuq-dashboard`; or use the raw autosubmit "
              "commands (references/autosubmit-cli.md)", file=sys.stderr)
        sys.exit(3)
    try:
        data = json.loads(raw)
    except ValueError:
        data = {"detail": raw[:2000]}
    return r.status, data


# ----------------------------------------------------------------------------- helpers
def ago(t):
    if not t:
        return "?"
    s = time.time() - t
    return dur(s) + " ago" if s >= 0 else "in " + dur(-s)


def dur(s):
    if s is None:
        return "?"
    s = int(s)
    if s < 90:
        return "%ds" % s
    if s < 5400:
        return "%dm" % round(s / 60)
    if s < 172800:
        return "%.1fh" % (s / 3600)
    return "%.1fd" % (s / 86400)


def ts(t):
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(t)) if t else "?"


def gb(n):
    return "%.1f GB" % ((n or 0) / 1e9)


def counts_str(c):
    order = ["COMPLETED", "RUNNING", "QUEUING", "SUBMITTED", "READY", "WAITING", "FAILED", "UNKNOWN", "SUSPENDED", "HELD"]
    keys = [k for k in order if k in c] + sorted(k for k in c if k not in order)
    return " ".join("%s=%s" % (k, c[k]) for k in keys) or "-"


def conf_line(c):
    return ("%s %s · ICS %s · %s · branch %s\n  start dates %d (%s..%s) · members %d · %s chunk(s) of %s %s · "
            "out vars %s levs %s res %s freq %s" % (
                c.get("model"), c.get("checkpoint"), c.get("ics"), c.get("run_type"), c.get("branch"),
                len(c.get("dates") or []), c.get("first_date"), c.get("last_date"), len(c.get("members") or []),
                c.get("num_chunks"), c.get("chunk_size"), c.get("chunk_unit"), ",".join(c.get("out_vars") or []),
                c.get("out_levs"), c.get("out_res"), c.get("out_freq")))


# ----------------------------------------------------------------------------- modes
def m_health(st, d):
    print("dashboard HTTP %s ok=%s" % (st, d.get("ok")))
    print("last collector cycle %s · HPC %s reachable=%s (last ok %s)" % (
        ago(d.get("last_cycle")), d.get("remote_host"), d.get("remote_ok"), ago(d.get("remote_last_ok"))))
    for e in d.get("errors") or []:
        print("  error:", e)


def m_list(st, d):
    exps = d.get("experiments") or []
    print("%-5s %-7s %-10s %-31s %-9s %-9s %6s  %s" % ("EXPID", "STATE", "MODEL", "CHECKPOINT", "ICS", "RUN", "DONE", "COUNTS / DESCRIPTION"))
    for e in sorted(exps, key=lambda x: x["expid"]):
        c, s = e.get("conf") or {}, e.get("summary") or {}
        state = "ACTIVE" if e.get("active") else "stopped"
        pct = "%5.1f%%" % s["pct"] if s.get("pct") is not None else "    ?"
        print("%-5s %-7s %-10s %-31s %-9s %-9s %6s  %s · %s" % (
            e["expid"], state, c.get("model"), c.get("checkpoint"), c.get("ics"), c.get("run_type"), pct,
            counts_str(e.get("counts") or {}), e.get("description") or ""))
    m = d.get("meta") or {}
    print("(data collected %s; HPC reachable=%s)" % (ago(m.get("last_cycle")), m.get("remote_ok")))


def m_status(st, d):
    v = d["exp"]
    c, s = v.get("conf") or {}, v.get("summary") or {}
    print("%s — %s" % (v["expid"], v.get("description") or ""))
    print("  " + conf_line(c))
    if v.get("active"):
        print("autosubmit run: ACTIVE (pid %s, since %s)" % (v.get("pid"), ts(v.get("active_since"))))
    else:
        print("autosubmit run: not running")
    print("progress: %s/%s jobs (%.1f%%) · running %s · queued %s · waiting %s · failed %s" % (
        s.get("completed"), s.get("total"), s.get("pct") or 0, s.get("running"), s.get("queued"),
        s.get("waiting"), s.get("failed")))
    print("run started %s · elapsed %s · last activity %s · finished %s" % (
        ts(s.get("run_start")), dur(s.get("elapsed")), ago(s.get("last_activity")), ts(s.get("run_finish"))))
    eta = s.get("eta") or {}
    if eta.get("sections"):
        parts = ["%s: %s left, median %s, x%.1f parallel%s" % (k, x.get("remaining"), dur(x.get("median")),
                 x.get("parallel") or 1, ", OVERDUE" if x.get("overdue") else "") for k, x in (eta.get("sections") or {}).items()]
        unknown = [k for k, x in eta["sections"].items() if x.get("median") is None]
        total = "~%s" % dur(eta.get("total")) if eta.get("total") else "unknown"
        if unknown:
            total += " + unknown for %s (no finished job yet; queue waits not included)" % ",".join(unknown)
        print("ETA (estimate) %s — %s" % (total, "; ".join(parts)))
    print("sections: " + " | ".join("%s %s" % (k, counts_str(x)) for k, x in (v.get("section_counts") or {}).items()))
    ds = v.get("dur_stats") or {}
    if ds:
        print("durations (median): " + ", ".join("%s %s (n=%s)" % (k, dur(x.get("median")), x.get("n")) for k, x in ds.items()))
    run = v.get("running") or []
    if run:
        print("running/submitted (%d):" % len(run))
        for j in run[:12]:
            last = ""
            out = ((j.get("tail") or {}).get("out") or {}).get("lines") or []
            if out:
                last = " · last out: " + out[-1][:140]
            print("  %s %s slurm=%s%s" % (j.get("name"), j.get("status"), j.get("id"), last))
        if len(run) > 12:
            print("  ... %d more" % (len(run) - 12))
    sj = [j for j in ((v.get("slurm") or {}).get("jobs") or []) if str(j.get("name", "")).startswith(v["expid"] + "_")]
    if sj:
        print("SLURM queue for %s (%d):" % (v["expid"], len(sj)))
        for j in sj[:12]:
            print("  %s %s %s %s part=%s elapsed=%s limit=%s est.start=%s" % (
                j.get("id"), j.get("name"), j.get("state"), j.get("reason") or "", j.get("partition"),
                j.get("time"), j.get("limit"), j.get("start")))
    for i in v.get("ics") or []:
        if i.get("status") not in ("COMPLETED",):
            print("ICS %s %s (fails %s)" % (i.get("job"), i.get("status"), i.get("fail_count")))
    f = v.get("failed") or []
    print("failed: %s" % (", ".join(x["name"] for x in f) if f else "none"))
    files = ((v.get("outputs") or {}).get("files")) or []
    if files:
        vars_ = sorted({x[0] for x in files})
        mems = sorted({x[1] for x in files}, key=lambda m: (len(m), m))
        print("outputs: %d files, %s, vars %s, members %s" % (len(files), gb(sum(x[3] or 0 for x in files)),
                                                               ",".join(vars_), len(mems)))
    else:
        print("outputs: none yet")
    r = v.get("remote") or {}
    print("HPC root: %s:%s%s" % (r.get("host"), r.get("root"), (" · remote error: %s" % r["error"]) if r.get("error") else ""))
    if v.get("pkl_error"):
        print("pkl error:", v["pkl_error"])
    print("(data collected %s)" % ago((d.get("meta") or {}).get("last_cycle")))


def m_failed(st, d):
    v = d["exp"]
    f = v.get("failed") or []
    if not f:
        print("%s: no FAILED jobs" % v["expid"])
        return
    for x in f:
        err = x.get("err") or {}
        print("=" * 100)
        print("%s  section=%s date=%s member=%s chunk=%s fail_count=%s attempts=%s ended %s slurm=%s" % (
            x["name"], x.get("section"), x.get("date"), x.get("member"), x.get("chunk"), x.get("fail_count"),
            x.get("attempts"), ts(x.get("end")), x.get("id")))
        print("log: %s (%s)" % (err.get("file"), "traceback" if err.get("traceback") else "last lines"))
        for line in err.get("lines") or ["(no stderr collected yet — the dashboard fetches it on its next cycle)"]:
            print("  " + line)


def m_outputs(st, d):
    v = d["exp"]
    files = ((v.get("outputs") or {}).get("files")) or []
    root = (v.get("remote") or {}).get("root")
    print("root: %s/outputs  (%d files, %s)" % (root, len(files), gb(sum(x[3] or 0 for x in files))))
    for var, mem, fn, size, mt in sorted(files, key=lambda x: (x[0], len(x[1]), x[1], x[2])):
        print("%s/%s/%s  %s  %s" % (var, mem, fn, gb(size), ts(mt)))


def m_options(st, d):
    print("branch %s · platform %s · std %s · support folder %s · HPC user %s" % (
        d.get("branch"), d.get("platform"), d.get("std_version"), d.get("support_folder"), d.get("user_code")))
    print("ICS available on platform: %s" % ", ".join(d.get("ics") or []))
    print("forcings %s · out_res %s · out_freq %s · chunk units %s" % (
        d.get("forcings"), d.get("resolutions"), d.get("freqs"), d.get("chunk_units")))
    if not d.get("checkpoints_checked"):
        print("WARNING: checkpoints on the HPC could not be listed")
    for m in d.get("models") or []:
        print("-" * 100)
        print("model %s (%s, %s) · on platform: %s · AMIP: %s" % (m["name"], m["label"], m.get("producer"),
                                                                  m["on_platform"], m["amip"]))
        for ck in m["checkpoints"]:
            sup = " ".join("%s=%s" % kv for kv in sorted((ck.get("support") or {}).items())) or "-"
            print("  ckpt %-32s %-11s %-13s README support: %s" % (ck["name"], "ON HPC" if ck["present"] else "MISSING",
                                                                 "ensemble" if ck["ensemble"] else "deterministic", sup))
        print("  variables: " + ", ".join("%s(%s)" % (x["name"], x.get("type")) for x in m["variables"]))
        print("  levels (hPa): %s" % m.get("levels"))


def m_preview(st, d):
    print("VALID" if d.get("ok") else "INVALID")
    for e in d.get("errors") or []:
        print("  error:", e)
    for w in d.get("warnings") or []:
        print("  warning:", w)
    s = d.get("summary")
    if s:
        print("summary: %s simulation jobs = %s start dates x %s members x %s chunks of %s %s" % (
            s["sims"], s["n_dates"], s["members"], s["num_chunks"], s["chunk_size"], s["chunk_unit"]))
    e = d.get("estimate")
    if e:
        print("estimate (from %s): GPU time %s, ICS prep %s, wall ~%s on %s GPU(s) — excludes queue waits" % (
            ",".join(e.get("based_on") or []), dur(e["gpu_s"]), dur(e["prep_s"]), dur(e["wall_s"]), e["gpus"]))
    if d.get("yaml"):
        print("----- conf/main.yml -----")
        print(d["yaml"].rstrip())
    for c in d.get("commands") or []:
        print("$ " + c)
    if not d.get("ok"):
        sys.exit(2)


def m_task(st, d):
    print("task %s kind=%s expid=%s status=%s%s" % (d.get("id"), d.get("kind"), d.get("expid"), d.get("status"),
                                                    (" error=%s" % d["error"]) if d.get("error") else ""))
    for s in d.get("steps") or []:
        print("  [%s] %s" % (s["state"], s["name"]))
    lines = d.get("lines") or []
    n = int(os.environ.get("AIUQ_TAIL", "25"))
    if lines:
        print("  --- last %d lines ---" % min(n, len(lines)))
        for line in lines[-n:]:
            print("  " + line)


def m_tasks(st, d):
    for t in d.get("tasks") or []:
        print("%s %-7s %-5s %-8s started %s%s" % (t["id"], t["kind"], t.get("expid") or "-", t["status"], ago(t["started"]),
                                                (" error: %s" % t["error"]) if t.get("error") else ""))


def m_deleteplan(st, d):
    l, r = d.get("local") or {}, d.get("remote") or {}
    print("delete plan for %s (%s, %s %s)" % (d.get("expid"), d.get("description"), d.get("model"), d.get("ics")))
    print("  local: %s (%s files, %s) + %s" % (l.get("path"), l.get("files"), gb(l.get("bytes")), "; ".join(l.get("extra") or [])))
    print("  HPC: %s:%s exists=%s files=%s nc=%s size=%s %s" % (r.get("host"), r.get("path"), r.get("exists"), r.get("files"),
                                                                r.get("nc_files"), gb(r.get("bytes")), r.get("error") or ""))
    for j in d.get("slurm_jobs") or []:
        print("  slurm job still queued: %s" % j)
    for b in d.get("blockers") or []:
        print("  BLOCKER:", b)
    if not d.get("blockers"):
        print("  no blockers")


def act(method, path, body):
    """POST a launcher action, then follow its task in this same process until it ends."""
    st, d = call(method, path, body)
    if st >= 400:
        print("REFUSED (HTTP %s): %s" % (st, d.get("detail")), file=sys.stderr)
        sys.exit(2)
    tid = d.get("id")
    timeout = float(os.environ.get("AIUQ_WAIT", "2400"))
    t0, seen = time.time(), 0
    while d.get("status") == "running" and time.time() - t0 < timeout:
        steps = d.get("steps") or []
        for s in steps[seen:]:
            print("[%s] %s" % (time.strftime("%H:%M:%S"), s["name"]), flush=True)
        seen = len(steps)
        time.sleep(3)
        st, d = call("GET", "/api/launcher/tasks/%s" % tid)
        if st >= 400:
            print("lost the task (HTTP %s): %s" % (st, d.get("detail")), file=sys.stderr)
            sys.exit(2)
    m_task(st, d)
    if d.get("status") == "running":
        print("still running after %ds; follow it with: aiuq task %s" % (timeout, tid))
    elif d.get("status") != "done":
        sys.exit(2)


MODES = {"health": m_health, "list": m_list, "status": m_status, "failed": m_failed, "outputs": m_outputs,
         "options": m_options, "preview": m_preview, "task": m_task, "tasks": m_tasks, "deleteplan": m_deleteplan}


def main():
    method, path, mode = sys.argv[1], sys.argv[2], sys.argv[3]
    body = base64.b64decode(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None
    if mode == "act":
        return act(method, path, body)
    st, d = call(method, path, body)
    if mode in ("raw", "json"):
        print(json.dumps(d, indent=1 if mode == "raw" else None, default=str))
    elif st >= 400 and not (mode == "health" and st == 503):
        print("HTTP %s: %s" % (st, d.get("detail") if isinstance(d, dict) else d), file=sys.stderr)
        sys.exit(2)
    else:
        MODES[mode](st, d)
    if st >= 400:
        sys.exit(2)


if __name__ == "__main__":
    main()
