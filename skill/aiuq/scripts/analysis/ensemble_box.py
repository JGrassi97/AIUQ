"""Ensemble statistics of one variable over a lat/lon box, per valid time. Runs ON the HPC.

    aiuq py <expid> scripts/analysis/ensemble_box.py VAR [--level HPA] [--box LON0 LON1 LAT0 LAT1]
                                                       [--start YYYY-MM-DD] [--daily] [--members]

Reads outputs/<VAR>/<member>/out-<start>-<end>-<member>-<VAR>.nc of every member (only the box and
level are read), takes the cos(latitude)-weighted box mean per member and prints CSV:
valid_time, n_members, mean, std (ensemble spread), min, max [, one column per member].
A single grid point: give LON0 == LON1 and LAT0 == LAT1 (nearest point is used).
Longitudes are 0..360 in AIUQ outputs; negative values are converted. Handles both the
latitude/longitude (NeuralGCM, Aurora) and lat/lon (AIFS after cdo) dimension names.
"""
import argparse
import glob
import os
import re
import sys
import warnings

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

p = argparse.ArgumentParser()
p.add_argument("var")
p.add_argument("--level", type=float)
p.add_argument("--box", type=float, nargs=4, metavar=("LON0", "LON1", "LAT0", "LAT1"), default=[0, 360, -90, 90])
p.add_argument("--start", help="output start date as in the file name (YYYY-MM-DD); default: all start dates")
p.add_argument("--daily", action="store_true", help="daily means before the statistics")
p.add_argument("--members", action="store_true", help="also print one column per member")
a = p.parse_args()

root = os.environ.get("AIUQ_OUTPUTS", "outputs")
pat = re.compile(r"^out-(\d{4}-\d{2}-\d{2})-(\d{4}-\d{2}-\d{2})-(.+)-%s\.nc$" % re.escape(a.var))
files = {}
for f in glob.glob(os.path.join(root, a.var, "*", "out-*-%s.nc" % a.var)):
    m = pat.match(os.path.basename(f))
    if m and (a.start is None or m.group(1) == a.start):
        files.setdefault(m.group(1), {})[m.group(3)] = f
if not files:
    sys.exit("no files for %s in %s (start=%s)" % (a.var, root, a.start))

lon0, lon1, lat0, lat1 = a.box
lon0, lon1 = lon0 % 360 if lon0 < 0 else lon0, lon1 % 360 if lon1 < 0 else lon1


def box_mean(path):
    ds = xr.open_dataset(path, decode_timedelta=True)
    da = ds[a.var]
    lat = "latitude" if "latitude" in da.dims else "lat"
    lon = "longitude" if "longitude" in da.dims else "lon"
    tdim = "valid_time" if "valid_time" in da.dims else "time"
    if a.level is not None and "level" in da.dims:
        da = da.sel(level=a.level)
    elif "level" in da.dims and da.sizes["level"] > 1:
        sys.exit("%s has levels %s: give --level" % (a.var, list(da["level"].values)))
    if lon0 == lon1 and lat0 == lat1:
        da = da.sel({lat: lat0, lon: lon0}, method="nearest")
    else:
        la = da[lat]
        da = da.sel({lat: slice(lat0, lat1) if la[0] < la[-1] else slice(lat1, lat0)})
        if lon0 <= lon1:
            da = da.sel({lon: slice(lon0, lon1)})
        else:  # box across the 0 meridian
            da = xr.concat([da.sel({lon: slice(lon0, 360)}), da.sel({lon: slice(0, lon1)})], dim=lon)
        w = np.cos(np.deg2rad(da[lat]))
        da = da.weighted(w).mean([lat, lon])
    da = da.load()
    if a.daily:
        da = da.resample({tdim: "1D"}).mean()
    return da.rename({tdim: "valid_time"})


for start in sorted(files):
    mem = sorted(files[start], key=lambda m: (len(m), m))
    series = xr.concat([box_mean(files[start][m]) for m in mem], dim="member").assign_coords(member=mem)
    print("# start %s · %s · level %s · box %s · %d members" % (start, a.var, a.level, a.box, len(mem)))
    head = "valid_time,n_members,mean,std,min,max" + ("," + ",".join("m" + m for m in mem) if a.members else "")
    print(head)
    for i, t in enumerate(series["valid_time"].values):
        x = series.isel(valid_time=i).values
        x = x[np.isfinite(x)]
        row = [str(t)[:16], str(len(x)), "%.6g" % x.mean(), "%.6g" % (x.std(ddof=1) if len(x) > 1 else 0),
               "%.6g" % x.min(), "%.6g" % x.max()]
        if a.members:
            row += ["%.6g" % v for v in series.isel(valid_time=i).values]
        print(",".join(row))
