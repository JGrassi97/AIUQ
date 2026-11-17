"""Read-only preview of one 2-D slice of an AIUQ NetCDF output, run on felipe.

Sent through ssh stdin (``python3 - <b64 json>``). Reads a single time step and a
single level, subsamples to at most ~360 x 181 points and prints JSON.

Request: {"path": ..., "var": "t" | null, "level": 850 | null, "time": -1}
"""
import base64
import json
import math
import sys

import netCDF4
import numpy as np

LAT_NAMES = ("lat", "latitude", "y")
LON_NAMES = ("lon", "longitude", "x")
LEV_NAMES = ("level", "plev", "pressure", "isobaricinhpa", "lev", "pressure_level")
MAX_LON = 360


def find_dim(dims, names):
    for d in dims:
        if d.lower() in names:
            return d
    return None


def time_labels(ds, d):
    tv = ds.variables.get(d)
    n = len(ds.dimensions[d])
    if tv is not None and getattr(tv, "units", None):
        try:
            dates = netCDF4.num2date(tv[:], tv.units, getattr(tv, "calendar", "standard"))
            return [str(x)[:16] for x in dates]
        except Exception:  # noqa: BLE001
            return [str(x) for x in tv[:]]
    return [str(i) for i in range(n)]


def box_mean(ds, v, dims, latd, lond, levd, timed, lat, lon, box, level):
    """cos(lat)-weighted mean over a lon/lat box (lon in -180..180, may cross the dateline), one value per time."""
    lon0, lon1, lat0, lat1 = [float(x) for x in box]
    lo, hi = min(lat0, lat1), max(lat0, lat1)
    rows = np.where((lat >= lo) & (lat <= hi))[0]
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    if lon0 <= lon1:
        cols = np.where((lon180 >= lon0) & (lon180 <= lon1))[0]
    else:  # crosses the dateline
        cols = np.where((lon180 >= lon0) | (lon180 <= lon1))[0]
    if rows.size == 0 or cols.size == 0:  # box smaller than a grid cell: nearest point
        rows = np.array([int(np.argmin(np.abs(lat - (lo + hi) / 2)))])
        cols = np.array([int(np.argmin(np.abs(((lon180 - (lon0 + lon1) / 2) + 180) % 360 - 180)))])
    r0, r1 = int(rows.min()), int(rows.max()) + 1  # contiguous latitude band, read once
    idx = []
    for d in dims:
        if d == timed:
            idx.append(slice(None))
        elif d == levd:
            lv = np.asarray(ds.variables[d][:], dtype=float)
            idx.append(int(np.argmin(np.abs(lv - float(level)))) if level is not None else 0)
        elif d == latd:
            idx.append(slice(r0, r1))
        elif d == lond:
            idx.append(slice(None))
        else:
            idx.append(0)
    kept = [d for d in dims if d in (timed, latd, lond)]
    data = np.ma.filled(np.ma.asarray(v[tuple(idx)], dtype=float), np.nan)
    data = np.moveaxis(data, [kept.index(timed), kept.index(latd), kept.index(lond)], [0, 1, 2]) if timed else data[None]
    data = data[:, rows - r0][:, :, cols]
    w = np.cos(np.deg2rad(lat[rows]))[None, :, None] * np.ones((1, 1, cols.size))
    w = np.broadcast_to(w, data.shape)
    ok = np.isfinite(data)
    num = np.where(ok, data * w, 0).sum(axis=(1, 2))
    den = np.where(ok, w, 0).sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        vals = num / den
    return vals, {"box": [lon0, lon1, lo, hi], "npoints": int(rows.size * cols.size)}


def series(req):
    """Time series at the grid point nearest to (lat, lon), one per member file."""
    out = {"members": {}, "times": [], "var": req.get("var")}
    for member, path in sorted(req["paths"].items()):
        try:
            ds = netCDF4.Dataset(path)
        except (IOError, OSError) as exc:
            out["members"][member] = {"error": str(exc)}
            continue
        var = req.get("var") if req.get("var") in ds.variables else None
        if var is None:
            ds.close()
            continue
        v = ds.variables[var]
        dims = v.dimensions
        latd, lond = find_dim(dims, LAT_NAMES), find_dim(dims, LON_NAMES)
        levd = find_dim(dims, LEV_NAMES)
        timed = next((d for d in dims if "time" in d.lower()), None)
        lat = np.asarray(ds.variables[latd][:], dtype=float)
        lon = np.asarray(ds.variables[lond][:], dtype=float)
        box = req.get("box")
        if box:
            vals, info = box_mean(ds, v, dims, latd, lond, levd, timed, lat, lon, box, req.get("level"))
            if not out["times"] and timed:
                out["times"] = time_labels(ds, timed)
            out["members"][member] = [None if not np.isfinite(x) else float("%.6g" % x) for x in vals]
            out.update(info)
            out["units"] = getattr(v, "units", "")
            ds.close()
            continue
        want_lon = float(req["lon"]) % 360.0 if lon.max() > 180 else ((float(req["lon"]) + 180) % 360) - 180
        iy = int(np.argmin(np.abs(lat - float(req["lat"]))))
        ix = int(np.argmin(np.abs(((lon - want_lon) + 180) % 360 - 180)))
        idx = []
        for d in dims:
            if d == timed:
                idx.append(slice(None))
            elif d == levd:
                lv = np.asarray(ds.variables[d][:], dtype=float)
                want = req.get("level")
                idx.append(int(np.argmin(np.abs(lv - float(want)))) if want is not None else 0)
            elif d == latd:
                idx.append(iy)
            elif d == lond:
                idx.append(ix)
            else:
                idx.append(0)
        vals = np.ma.filled(np.ma.asarray(v[tuple(idx)], dtype=float), np.nan).ravel()
        if not out["times"] and timed:
            out["times"] = time_labels(ds, timed)
        out["members"][member] = [None if not np.isfinite(x) else float("%.6g" % x) for x in vals]
        out["lat"], out["lon"] = float(lat[iy]), float(((lon[ix] + 180) % 360) - 180)
        out["units"] = getattr(v, "units", "")
        ds.close()
    return out


# ------------------------------------------------------------------ ensemble / analysis helpers
def open_var(path, var):
    ds = netCDF4.Dataset(path)
    v = ds.variables[var]
    dims = v.dimensions
    latd, lond = find_dim(dims, LAT_NAMES), find_dim(dims, LON_NAMES)
    levd = find_dim(dims, LEV_NAMES)
    timed = next((d for d in dims if "time" in d.lower()), None)
    lat = np.asarray(ds.variables[latd][:], dtype=float)
    lon = np.asarray(ds.variables[lond][:], dtype=float)
    return ds, v, dims, latd, lond, levd, timed, lat, lon


def read_slab(ds, v, dims, latd, lond, levd, timed, level, tsel=slice(None), latsel=slice(None), lonsel=slice(None)):
    """Return data as [time, lat, lon] float64 (time axis of length 1 when tsel is an int)."""
    idx, keep = [], []
    for d in dims:
        if d == timed:
            idx.append(tsel); keep.append(d)
        elif d == levd:
            lv = np.asarray(ds.variables[d][:], dtype=float)
            idx.append(int(np.argmin(np.abs(lv - float(level)))) if level is not None else 0)
        elif d == latd:
            idx.append(latsel); keep.append(d)
        elif d == lond:
            idx.append(lonsel); keep.append(d)
        else:
            idx.append(0)
    a = np.ma.filled(np.ma.asarray(v[tuple(idx)], dtype=float), np.nan)
    if timed is None or isinstance(tsel, int):
        keep = [k for k in keep if k != timed]
        a = np.moveaxis(a, [keep.index(latd), keep.index(lond)], [0, 1])[None]
    else:
        a = np.moveaxis(a, [keep.index(timed), keep.index(latd), keep.index(lond)], [0, 1, 2])
    return a


def to_lists(a, nd=5):
    return [[None if not np.isfinite(x) else float("%.*g" % (nd, x)) for x in row] for row in a]


def box_index(lat, lon, box):
    lon0, lon1, lat0, lat1 = [float(x) for x in box]
    lo, hi = min(lat0, lat1), max(lat0, lat1)
    rows = np.where((lat >= lo) & (lat <= hi))[0]
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    cols = np.where((lon180 >= lon0) & (lon180 <= lon1))[0] if lon0 <= lon1 else np.where((lon180 >= lon0) | (lon180 <= lon1))[0]
    return rows, cols


def ens(req):
    """Ensemble mean and standard deviation of one time step / level, subsampled like the map preview."""
    s1 = s2 = None
    n = 0
    lat = lon = None
    for member, path in sorted(req["paths"].items()):
        ds, v, dims, latd, lond, levd, timed, la, lo = open_var(path, req["var"])
        nt = len(ds.dimensions[timed]) if timed else 1
        t = int(req.get("time", -1))
        t = min(max(t + nt if t < 0 else t, 0), nt - 1)
        stride = max(1, int(math.ceil(len(lo) / float(MAX_LON))))
        a = read_slab(ds, v, dims, latd, lond, levd, timed, req.get("level"), t, slice(None, None, stride), slice(None, None, stride))[0]
        ds.close()
        s1 = a if s1 is None else s1 + a
        s2 = a * a if s2 is None else s2 + a * a
        n += 1
        lat, lon = la[::stride], lo[::stride]
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean * mean, 0)) if n > 1 else np.zeros_like(mean)
    lon = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon)
    lon, mean, std = lon[order], mean[:, order], std[:, order]
    if lat[0] > lat[-1]:
        lat, mean, std = lat[::-1], mean[::-1], std[::-1]
    f = lambda a: (float(np.nanmin(a)), float(np.nanmax(a)))
    return {"lat": [round(float(x), 3) for x in lat], "lon": [round(float(x), 3) for x in lon], "n": n, "time_index": t,
            "mean": to_lists(mean), "std": to_lists(std), "mean_range": f(mean), "std_range": f(std)}


def spread(req):
    """Area-weighted ensemble spread (sqrt of mean variance) and area mean of the ensemble mean, per time step."""
    s1 = s2 = None
    n = 0
    times = []
    for member, path in sorted(req["paths"].items()):
        ds, v, dims, latd, lond, levd, timed, lat, lon = open_var(path, req["var"])
        rows, cols = box_index(lat, lon, req["box"])
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        a = read_slab(ds, v, dims, latd, lond, levd, timed, req.get("level"), slice(None), slice(r0, r1))
        a = a[:, rows - r0][:, :, cols]
        if not times and timed:
            times = time_labels(ds, timed)
        ds.close()
        s1 = a if s1 is None else s1 + a
        s2 = a * a if s2 is None else s2 + a * a
        n += 1
        w = np.cos(np.deg2rad(lat[rows]))[None, :, None] * np.ones((1, 1, cols.size))
    mean = s1 / n
    var = np.maximum(s2 / n - mean * mean, 0) * (n / (n - 1.0) if n > 1 else 0)
    w = np.broadcast_to(w, mean.shape)
    ok = np.isfinite(var)
    wsum = np.where(ok, w, 0).sum(axis=(1, 2))
    spr = np.sqrt(np.where(ok, var * w, 0).sum(axis=(1, 2)) / wsum)
    amean = np.where(ok, mean * w, 0).sum(axis=(1, 2)) / wsum
    return {"times": times, "n": n, "spread": [float("%.5g" % x) for x in spr], "mean": [float("%.6g" % x) for x in amean],
            "npoints": int(rows.size * cols.size)}


def hov(req):
    """Hovmöller: cos-lat weighted mean over a latitude band -> [time, lon], ensemble mean of the given files."""
    acc, n, times, lon_out = None, 0, [], None
    for member, path in sorted(req["paths"].items()):
        ds, v, dims, latd, lond, levd, timed, lat, lon = open_var(path, req["var"])
        lo, hi = sorted([float(req["lat0"]), float(req["lat1"])])
        rows = np.where((lat >= lo) & (lat <= hi))[0]
        r0, r1 = int(rows.min()), int(rows.max()) + 1
        stride = max(1, int(math.ceil(len(lon) / float(MAX_LON))))
        a = read_slab(ds, v, dims, latd, lond, levd, timed, req.get("level"), slice(None), slice(r0, r1), slice(None, None, stride))
        a = a[:, rows - r0]
        w = np.cos(np.deg2rad(lat[rows]))[None, :, None]
        band = np.nansum(a * w, axis=1) / np.nansum(np.where(np.isfinite(a), w, 0), axis=1)
        if not times and timed:
            times = time_labels(ds, timed)
        lon_out = lon[::stride]
        ds.close()
        acc = band if acc is None else acc + band
        n += 1
    z = acc / n
    lon_out = ((lon_out + 180.0) % 360.0) - 180.0
    order = np.argsort(lon_out)
    return {"times": times, "lon": [round(float(x), 3) for x in lon_out[order]], "z": to_lists(z[:, order]), "n": n,
            "min": float(np.nanmin(z)), "max": float(np.nanmax(z))}


def hindcast(req):
    """Area mean at selected time indices for many files (start dates x members)."""
    out, w_cache = [], {}
    tidx = sorted(set(int(t) for t in req["tidx"]))
    for start, member, path in req["files"]:
        try:
            ds, v, dims, latd, lond, levd, timed, lat, lon = open_var(path, req["var"])
            nt = len(ds.dimensions[timed]) if timed else 1
            ts = [t for t in tidx if t < nt]
            rows, cols = box_index(lat, lon, req["box"])
            r0, r1 = int(rows.min()), int(rows.max()) + 1
            a = read_slab(ds, v, dims, latd, lond, levd, timed, req.get("level"), ts, slice(r0, r1))
            ds.close()
            a = a[:, rows - r0][:, :, cols]
            key = (r0, r1, cols.size)
            if key not in w_cache:
                w_cache[key] = np.cos(np.deg2rad(lat[rows]))[None, :, None] * np.ones((1, 1, cols.size))
            w = np.broadcast_to(w_cache[key], a.shape)
            ok = np.isfinite(a)
            vals = np.where(ok, a * w, 0).sum(axis=(1, 2)) / np.where(ok, w, 0).sum(axis=(1, 2))
            row = dict(zip(ts, vals))
            out.append([start, member, [None if t not in row or not np.isfinite(row[t]) else float("%.6g" % row[t]) for t in tidx]])
        except Exception as exc:  # noqa: BLE001 - one bad file must not kill the analysis
            out.append([start, member, None, str(exc)[:200]])
    return {"tidx": tidx, "rows": out}


MODES = {"series": lambda r: series(r), "ens": ens, "spread": spread, "hov": hov, "hindcast": hindcast}


def main():
    req = json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
    if req.get("mode") in MODES:
        sys.stdout.write(json.dumps(MODES[req["mode"]](req)))
        return
    ds = netCDF4.Dataset(req["path"])
    ds.set_auto_mask(True)
    var = req.get("var")
    if var not in ds.variables:
        cands = [k for k, v in ds.variables.items() if v.ndim >= 2 and k not in ds.dimensions]
        var = cands[0]
    v = ds.variables[var]
    dims = v.dimensions
    latd, lond = find_dim(dims, LAT_NAMES), find_dim(dims, LON_NAMES)
    levd = find_dim(dims, LEV_NAMES)
    timed = next((d for d in dims if "time" in d.lower()), None)

    lat = np.asarray(ds.variables[latd][:], dtype=float)
    lon = np.asarray(ds.variables[lond][:], dtype=float)
    stride = max(1, int(math.ceil(len(lon) / float(MAX_LON))))

    levels, lev_val = [], None
    times, t_idx = [], None
    idx = []
    for d in dims:
        if d == timed:
            n = len(ds.dimensions[d])
            t_idx = int(req.get("time", -1))
            if t_idx < 0:
                t_idx += n
            t_idx = min(max(t_idx, 0), n - 1)
            times = time_labels(ds, d)
            idx.append(t_idx)
        elif d == levd:
            lv = np.asarray(ds.variables[d][:], dtype=float) if d in ds.variables else np.arange(len(ds.dimensions[d]))
            levels = [float(x) for x in lv]
            want = req.get("level")
            li = int(np.argmin(np.abs(lv - float(want)))) if want is not None else 0
            lev_val = levels[li]
            idx.append(li)
        elif d in (latd, lond):
            idx.append(slice(None, None, stride))
        else:
            idx.append(0)
    data = np.ma.filled(np.ma.asarray(v[tuple(idx)], dtype=float), np.nan)
    if dims.index(latd) > dims.index(lond):
        data = data.T
    lat, lon = lat[::stride], lon[::stride]

    lon = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon)
    lon, data = lon[order], data[:, order]
    if lat[0] > lat[-1]:
        lat, data = lat[::-1], data[::-1, :]

    finite = data[np.isfinite(data)]
    z = [[None if not np.isfinite(x) else float("%.5g" % x) for x in row] for row in data]
    out = {
        "var": var, "units": getattr(v, "units", ""), "long_name": getattr(v, "long_name", var),
        "level": lev_val, "levels": levels, "time_index": t_idx, "times": times,
        "lat": [round(float(x), 3) for x in lat], "lon": [round(float(x), 3) for x in lon], "z": z,
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "stride": stride, "shape": list(v.shape),
        "vars": [k for k, vv in ds.variables.items() if vv.ndim >= 2 and k not in ds.dimensions],
    }
    ds.close()
    sys.stdout.write(json.dumps(out))


if __name__ == "__main__":
    main()
