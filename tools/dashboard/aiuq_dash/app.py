"""FastAPI app: serves the single-page dashboard and JSON snapshots from the collector."""
import collections
import logging
import re
import threading
import time
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import launcher, remote, settings
from .collector import Collector

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("aiuq_dash")

collector = Collector()


@asynccontextmanager
async def lifespan(_app):
    import os

    collector.start()
    log.info("collector started (refresh %ss, remote %s)", settings.REFRESH, settings.REMOTE_HOST or "not set")
    if settings.missing():
        log.warning("site settings: %s", settings.not_set())
    yield
    collector.stop()


app = FastAPI(title="AIUQ dashboard", lifespan=lifespan, docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=settings.STATIC), name="static")

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "[::1]"}


@app.middleware("http")
async def guard(request: Request, call_next):
    """Only answer to a browser on localhost (blocks DNS rebinding); write actions need a custom
    header that cross-site pages cannot send without a CORS preflight (blocks CSRF)."""
    host = (request.headers.get("host") or "").rsplit(":", 1)[0]
    if host and host not in _LOCAL_HOSTS:
        return JSONResponse({"detail": "forbidden host"}, status_code=403)
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin and origin.split("://", 1)[-1].rsplit(":", 1)[0] not in _LOCAL_HOSTS:
            return JSONResponse({"detail": "forbidden origin"}, status_code=403)
        if request.url.path.startswith("/api/launcher/") and request.headers.get("x-aiuq-launcher") != "1":
            return JSONResponse({"detail": "missing launcher header"}, status_code=403)
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/static/vendor/"):
        response.headers["Cache-Control"] = "public, max-age=604800"  # third-party libraries: 1 week
    elif path == "/" or path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"  # always revalidate (ETag -> 304), so updates show up at once
    return response


def _meta():
    return {**collector.meta, "server_time": time.time()}


@app.get("/")
def index():
    return FileResponse(settings.STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/health")
def health():
    m = collector.meta
    stale = m["last_cycle"] is None or time.time() - m["last_cycle"] > 4 * settings.REFRESH
    return JSONResponse({"ok": not stale, **_meta()}, status_code=503 if stale else 200)


@app.get("/api/experiments")
def experiments():
    return {"meta": _meta(), "experiments": collector.overview}


@app.get("/api/exp/{expid}")
def experiment(expid: str):
    v = collector.views.get(expid)
    if v is None:
        raise HTTPException(404, f"unknown experiment {expid!r} (or first cycle still running)")
    return {"meta": _meta(), "exp": v}


@app.get("/api/history")
def history(hours: float = 24.0):
    """Node GPU use and SLURM queue length sampled every collector cycle (persisted, node-wide)."""
    cutoff = time.time() - hours * 3600
    return {"meta": _meta(), "history": [h for h in list(collector.history) if h["t"] >= cutoff]}


@app.post("/api/refresh")
def refresh():
    """Wake the collector early. Rate-limited by the collector's own minimum sleep."""
    collector.poke()
    return {"ok": True}


# ---------------------------------------------------------------- previews
_preview_cache = collections.OrderedDict()
_preview_lock = threading.Lock()


@app.get("/api/preview")
def preview(exp: str, var: str, member: str, file: str,
            level: float | None = Query(None), time_index: int = Query(-1, alias="time")):
    v = collector.views.get(exp)
    if v is None:
        raise HTTPException(404, "unknown experiment")
    match = next((f for f in v["outputs"]["files"] if f[0] == var and f[1] == member and f[2] == file), None)
    if match is None:  # only files we listed ourselves may be opened
        raise HTTPException(404, "file not in the experiment's output listing")
    path = f'{v["outputs"]["dir"]}/{var}/{member}/{file}'
    key = (path, match[4], var, level, time_index)
    with _preview_lock:
        if key in _preview_cache:
            _preview_cache.move_to_end(key)
            return _preview_cache[key]
    try:
        data = remote.run_helper("hpc_preview.py",
                                 {"path": path, "var": var, "level": level, "time": time_index}, timeout=60)
    except remote.RemoteError as exc:
        raise HTTPException(502, str(exc)) from exc
    data["file"], data["member"] = file, member
    with _preview_lock:
        _preview_cache[key] = data
        while len(_preview_cache) > 24:
            _preview_cache.popitem(last=False)
    return data


def parse_output_name(fn):
    """'out-1986-01-01-1986-01-31-2-t.nc' -> ('1986-01-01', '1986-01-31', '2', 't')."""
    p = fn[4:-3].split("-") if fn.startswith("out-") and fn.endswith(".nc") else []
    if len(p) < 8:
        return None
    return "-".join(p[0:3]), "-".join(p[3:6]), "-".join(p[6:-1]), p[-1]


_series_cache = collections.OrderedDict()


@app.get("/api/series")
def point_series(exp: str, var: str, start: str, lat: float | None = None, lon: float | None = None,
                 level: float | None = Query(None), lon0: float | None = None, lon1: float | None = None,
                 lat0: float | None = None, lat1: float | None = None):
    """Time series for every member with an output for this start date: at one grid point (lat, lon)
    or averaged over a box (lon0, lon1, lat0, lat1; cos-latitude weighted)."""
    box = [lon0, lon1, lat0, lat1] if None not in (lon0, lon1, lat0, lat1) else None
    if box is None and (lat is None or lon is None):
        raise HTTPException(400, "give lat/lon or a box lon0/lon1/lat0/lat1")
    v = collector.views.get(exp)
    if v is None:
        raise HTTPException(404, "unknown experiment")
    paths, mt = {}, 0
    for fvar, member, fn, _size, fmt in v["outputs"]["files"]:
        meta = parse_output_name(fn)
        if fvar == var and meta and meta[0] == start and meta[3] == var:
            paths[member] = f'{v["outputs"]["dir"]}/{var}/{member}/{fn}'
            mt = max(mt, fmt)
    if not paths:
        raise HTTPException(404, "no output files for this variable and start date")
    where = tuple(round(x, 2) for x in box) if box else (round(lat, 2), round(lon, 2))
    key = (exp, var, start, where, level, mt, len(paths))
    with _preview_lock:
        if key in _series_cache:
            return _series_cache[key]
    try:
        data = remote.run_helper("hpc_preview.py", {"mode": "series", "paths": paths, "var": var, "level": level,
                                                        "lat": lat, "lon": lon, "box": box}, timeout=120)
    except remote.RemoteError as exc:
        raise HTTPException(502, str(exc)) from exc
    with _preview_lock:
        _series_cache[key] = data
        while len(_series_cache) > 32:
            _series_cache.popitem(last=False)
    return data


# ---------------------------------------------------------------- ensemble analyses
_analysis_cache = collections.OrderedDict()


def _member_paths(v, var, start):
    """{member: remote path} of every output of `var` for one start date, from our own listing."""
    paths, mt = {}, 0
    for fvar, member, fn, _size, fmt in v["outputs"]["files"]:
        meta = parse_output_name(fn)
        if fvar == var and meta and meta[0] == start and meta[3] == var:
            paths[member] = f'{v["outputs"]["dir"]}/{var}/{member}/{fn}'
            mt = max(mt, fmt)
    if not paths:
        raise HTTPException(404, "no output files for this variable and start date")
    return paths, mt


def _analysis(key, payload, timeout=120):
    with _preview_lock:
        if key in _analysis_cache:
            _analysis_cache.move_to_end(key)
            return _analysis_cache[key]
    try:
        data = remote.run_helper("hpc_preview.py", payload, timeout=timeout)
    except remote.RemoteError as exc:
        raise HTTPException(502, str(exc)) from exc
    with _preview_lock:
        _analysis_cache[key] = data
        while len(_analysis_cache) > 24:
            _analysis_cache.popitem(last=False)
    return data


def _view(exp):
    v = collector.views.get(exp)
    if v is None:
        raise HTTPException(404, "unknown experiment")
    return v


@app.get("/api/ens")
def ens_field(exp: str, var: str, start: str, level: float | None = None, time_index: int = Query(-1, alias="time")):
    """Ensemble mean and spread (std) of one time step and level."""
    paths, mt = _member_paths(_view(exp), var, start)
    return _analysis(("ens", exp, var, start, level, time_index, mt, len(paths)),
                     {"mode": "ens", "paths": paths, "var": var, "level": level, "time": time_index})


@app.get("/api/spread")
def spread_growth(exp: str, var: str, start: str, lon0: float, lon1: float, lat0: float, lat1: float, level: float | None = None):
    """Area-weighted ensemble spread per lead time for one start date."""
    paths, mt = _member_paths(_view(exp), var, start)
    box = [lon0, lon1, lat0, lat1]
    return _analysis(("spread", exp, var, start, level, tuple(box), mt, len(paths)),
                     {"mode": "spread", "paths": paths, "var": var, "level": level, "box": box})


@app.get("/api/hovmoller")
def hovmoller(exp: str, var: str, start: str, lat0: float, lat1: float, level: float | None = None, member: str = "mean"):
    """Longitude x time over a latitude band, for one member or the ensemble mean."""
    paths, mt = _member_paths(_view(exp), var, start)
    if member != "mean":
        if member not in paths:
            raise HTTPException(404, "no file for this member")
        paths = {member: paths[member]}
    return _analysis(("hov", exp, var, start, level, lat0, lat1, member, mt, len(paths)),
                     {"mode": "hov", "paths": paths, "var": var, "level": level, "lat0": lat0, "lat1": lat1})


@app.get("/api/hindcast")
def hindcast(exp: str, var: str, lon0: float, lon1: float, lat0: float, lat1: float, tidx: str, level: float | None = None):
    """Area mean at the given time indices for every start date and member (reads one file per pair)."""
    v = _view(exp)
    files, mt = [], 0
    for fvar, member, fn, _size, fmt in v["outputs"]["files"]:
        meta = parse_output_name(fn)
        if fvar == var and meta and meta[3] == var:
            files.append([meta[0], member, f'{v["outputs"]["dir"]}/{var}/{member}/{fn}'])
            mt = max(mt, fmt)
    if not files:
        raise HTTPException(404, "no output files for this variable")
    try:
        ts = sorted({int(x) for x in tidx.split(",") if x.strip() != ""})[:12]
    except ValueError as exc:
        raise HTTPException(400, "tidx must be a comma-separated list of integers") from exc
    box = [lon0, lon1, lat0, lat1]
    return _analysis(("hindcast", exp, var, level, tuple(box), tuple(ts), mt, len(files)),
                     {"mode": "hindcast", "files": files, "var": var, "level": level, "box": box, "tidx": ts}, timeout=600)


# ---------------------------------------------------------------- AIUQ Launcher (writes)
def _bad(exc):
    raise HTTPException(400, str(exc)) from exc


@app.get("/api/launcher/branches")
def launcher_branches():
    try:
        return {"branches": launcher.branches(), "repo": launcher.GIT_REPO}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"cannot list branches: {exc}") from exc


@app.get("/api/launcher/options")
def launcher_options(branch: str = "main"):
    try:
        return launcher.options(branch)
    except ValueError as exc:
        _bad(exc)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"cannot read the repository: {exc}") from exc


@app.get("/api/launcher/template/{expid}")
def launcher_template(expid: str):
    try:
        return launcher.template(expid)
    except ValueError as exc:
        _bad(exc)


@app.post("/api/launcher/preview")
def launcher_preview(cfg: dict = Body(...)):
    res = launcher.preview(cfg)
    if res.get("summary"):
        res["estimate"] = _estimate(cfg, res["summary"])
    return res


def _estimate(cfg, s):
    """Rough cost of a new experiment from finished ones with the same model (and checkpoint when possible)."""
    import statistics

    days = lambda size, unit: (size or 1) * (30 if unit == "month" else 1)
    rates = []
    for v in collector.views.values():
        c, d = v["conf"], v["dur_stats"].get("SIM")
        if c.get("model") == cfg.get("model") and d:
            rates.append((c.get("checkpoint") == cfg.get("checkpoint"), d["median"] / days(c.get("chunk_size"), c.get("chunk_unit")), v["expid"]))
    same = [r for r in rates if r[0]] or rates
    if not same:
        return None
    rate = statistics.median(r[1] for r in same)
    gpu_s = s["sims"] * days(s["chunk_size"], s["chunk_unit"]) * rate
    prep = [v["dur_stats"]["PREPARE_ICS"]["median"] for v in collector.views.values()
            if v["conf"].get("ics") == cfg.get("ics") and "PREPARE_ICS" in v["dur_stats"]]
    prep_s = s["n_dates"] * statistics.median(prep) if prep else 0
    gpus = 0
    for nd in (collector.slurm or {}).get("nodes", []):
        m = re.search(r"gpu(?::[^:(]+)?:(\d+)", nd.get("gres") or "")
        gpus = max(gpus, int(m.group(1)) if m else 0)
    gpus = gpus or 1
    return {"gpu_s": gpu_s, "prep_s": prep_s, "wall_s": prep_s + gpu_s / gpus, "gpus": gpus,
            "based_on": sorted({r[2] for r in same})}


@app.post("/api/launcher/create")
def launcher_create(cfg: dict = Body(...)):
    try:
        t = launcher.create(cfg)
    except ValueError as exc:
        _bad(exc)
    return t.as_dict()


@app.post("/api/launcher/run/{expid}")
def launcher_run(expid: str):
    try:
        t = launcher.run(expid)
    except ValueError as exc:
        _bad(exc)
    collector.poke()
    return t.as_dict()


@app.post("/api/launcher/stop/{expid}")
def launcher_stop(expid: str, cancel: bool = False):
    try:
        t = launcher.stop(expid, cancel)
    except ValueError as exc:
        _bad(exc)
    collector.poke()
    return t.as_dict()


@app.get("/api/launcher/delete-plan/{expid}")
def launcher_delete_plan(expid: str):
    try:
        return launcher.delete_plan(expid)
    except ValueError as exc:
        _bad(exc)
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc


@app.post("/api/launcher/delete/{expid}")
def launcher_delete(expid: str, body: dict = Body(...)):
    try:
        t = launcher.delete(expid, body.get("confirm"))
    except ValueError as exc:
        _bad(exc)
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc
    collector.poke()
    return t.as_dict()


@app.get("/api/launcher/leftovers")
def launcher_leftovers():
    return {"leftovers": launcher.leftovers()}


@app.post("/api/launcher/leftovers/{expid}/retry")
def launcher_leftover_retry(expid: str):
    try:
        return launcher.retry_leftover(expid).as_dict()
    except ValueError as exc:
        _bad(exc)


@app.get("/api/launcher/tasks/{task_id}")
def launcher_task(task_id: str):
    t = launcher.TASKS.get(task_id)
    if not t:
        raise HTTPException(404, "unknown task")
    if t.finished:
        collector.poke()
    return t.as_dict()


@app.get("/api/launcher/tasks")
def launcher_tasks():
    return {"tasks": [t.as_dict(tail=30) for t in sorted(launcher.TASKS.values(), key=lambda t: -t.started)][:20]}


def main():
    import os

    import uvicorn

    if settings.SOCKET:
        sock = settings.SOCKET
        os.makedirs(os.path.dirname(sock), mode=0o700, exist_ok=True)
        os.chmod(os.path.dirname(sock), 0o700)
        if os.path.exists(sock):
            os.unlink(sock)
        # Bind the socket ourselves so it is 0600 (uvicorn's own uds= option makes it 0666).
        import socket

        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        old = os.umask(0o077)
        try:
            srv.bind(sock)
        finally:
            os.umask(old)
        os.chmod(sock, 0o600)
        srv.listen(128)
        log.info("listening on unix socket %s", sock)
        uvicorn.run(app, fd=srv.fileno(), log_level="info", access_log=False)
    else:
        uvicorn.run(app, host=settings.HOST, port=settings.PORT, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
