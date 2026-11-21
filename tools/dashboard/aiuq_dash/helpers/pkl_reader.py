"""Read Autosubmit job_list pickles and print them as JSON.

Runs with the Autosubmit Python interpreter (``AIUQ_AS_PYTHON``), because
unpickling needs the ``autosubmit`` package. It only reads the pickle files and
never writes anything.

Usage: python pkl_reader.py <pkl_path> [<pkl_path> ...]
"""
import datetime
import json
import pickle
import sys
import time

STATUS = {
    0: "WAITING", 1: "READY", 2: "SUBMITTED", 3: "QUEUING", 4: "RUNNING",
    5: "COMPLETED", 6: "HELD", 7: "PREPARED", 8: "SKIPPED", 9: "DELAYED",
    -1: "FAILED", -2: "UNKNOWN", -3: "SUSPENDED",
}
try:  # Prefer Autosubmit's own mapping when available.
    from autosubmit.job.job_common import Status  # noqa: E402

    STATUS.update(Status.VALUE_TO_KEY)
except Exception:  # pragma: no cover - fallback map above is complete for AS 4.1
    pass


def _ts(value):
    """Autosubmit stores times as 'YYYYmmddHHMMSS' strings, ints or datetimes."""
    if not value:
        return None
    try:
        if isinstance(value, datetime.datetime):
            return value.timestamp()
        s = str(value)
        if len(s) == 14 and s.isdigit():
            return time.mktime(time.strptime(s, "%Y%m%d%H%M%S"))
    except Exception:
        return None
    return None


def _job_from_dict(d):
    date = d.get("date")
    return {
        "name": d.get("_name") or d.get("name"),
        "status": STATUS.get(d.get("_status", d.get("status")), str(d.get("_status"))),
        "section": d.get("_section") or d.get("section"),
        "date": date.strftime("%Y%m%d") if isinstance(date, datetime.datetime) else None,
        "member": d.get("_member"),
        "chunk": d.get("_chunk"),
        "split": d.get("_split"),
        "fail_count": d.get("_fail_count", 0) or 0,
        "retrials": d.get("_retrials"),
        "platform": d.get("platform_name"),
        "id": d.get("id") or None,
        "ready": _ts(d.get("ready_date")),
        "submit": _ts(d.get("submit_time_timestamp")),
        "start": _ts(d.get("start_time_timestamp")),
        "finish": _ts(d.get("finish_time_timestamp")),
        "wallclock": d.get("_wallclock"),
        "local_logs": list(d.get("_local_logs") or []),
        "remote_logs": list(d.get("_remote_logs") or []),
    }


def read(path):
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    if isinstance(obj, dict):
        items = obj.values()
    elif hasattr(obj, "nodes"):  # older networkx-based pickles
        items = [vars(data["job"]) for _, data in obj.nodes(data=True)]
    else:
        raise TypeError("unsupported pickle type %s" % type(obj).__name__)
    return [_job_from_dict(d if isinstance(d, dict) else vars(d)) for d in items]


def main():
    out = {}
    for path in sys.argv[1:]:
        for attempt in range(3):  # Autosubmit may be rewriting the file right now.
            try:
                out[path] = {"jobs": read(path)}
                break
            except Exception as exc:  # noqa: BLE001
                out[path] = {"error": "%s: %s" % (type(exc).__name__, exc)}
                time.sleep(0.5)
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
