# Outputs and analysis

## Layout on felipe

```
<root>/outputs/<var>/<member>/out-<first valid day>-<last valid day>-<member>-<var>.nc
<root>/outputs/checkpoints/          # NeuralGCM AMIP restarts, not model output
<root> = /work/users/<user>/autosubmit_scratch/test/<user>/<expid>
```

Once every simulation has finished, the `ARCHIVE` job (runs on mafalda) moves the whole experiment to the output
store (`OUTPUT_STORE.DIR` in `main.yml`, default from `conf/general.yml`: `/data/users/<user>/AIUQ-outputs`), in a
folder of its own because Autosubmit can reuse an expid:

```
<store>/<expid>_<YYYYmmddTHHMMSS>/outputs/...   # same layout as above, checkpoints included
<store>/<expid>_<YYYYmmddTHHMMSS>/LOG_<expid>/  # all job logs
<store>/<expid>_<YYYYmmddTHHMMSS>/archive.json  # source, time, files, bytes, git commit
~/autosubmit/<expid>/OUTPUT_STORE.json          # on mafalda: the folder (status "done")
```

Then `<root>` is removed from felipe. felipe mounts `/data` read-only at the same path, so `aiuq` commands and
analysis on felipe read the store directly. The ECMWF-analysis forecasts a00c…a010 are in
`/data/users/<user>/AIUQ-forecast`.

One file per variable × member × chunk (× start date: the start date shows up only as the first day
in the name). Example: `t/7/out-2026-11-16-2026-12-02-7-t.nc`.
AIFS writes intermediate `*_temp.nc` / `*_grid.nc` files that POSTPROCESS replaces with the final one;
ignore them (a leftover `_temp.nc` means POSTPROCESS did not run or failed).

## Conventions (check with `aiuq ncinfo` — never assume)

- One data variable named like the folder (AIUQ-st short names, ECMWF parameter database:
  `t z u v q w` on pressure levels; `t2m msl sp u10 v10 d2m tcw skt …` at the surface; output-only fields of
  `conf/AIUQ-st/v100/outputs.yml` such as `tp cp sf ro ssrd strd stl1 i10fg avg_tprate …`).
- Periods (attribute `aiuq_period` / `cell_methods`, written since the output-variables update, Nov 2026):
  AIFS `tp cp sf ro` (m) and `ssrd strd` (J m⁻²) are totals over the 6 h step ending at `valid_time`
  (daily output = daily total); NeuralGCM `tp` is the total over each output step (de-accumulated from the
  model's running total) and `avg_ie` the rate of its last 1 h step; Aurora 1.5 `avg_*` are mean rates
  (kg m⁻² s⁻¹) or fluxes (W m⁻²) over the **last hour** of each step only (the model writes 1-hour totals).
- **Dimension order and names differ between models — always select by name, never by position**
  (checked on real files, Nov 2026):

  | Model | Level field dims | Dtype | Notes |
  |---|---|---|---|
  | NeuralGCM | `(valid_time, level, longitude, latitude)` — longitude first | float64 | coords `time` (scalar), `step` |
  | Aurora | `(valid_time, level, latitude, longitude)`; surface `(valid_time, latitude, longitude)` | float64 | coord `time` (scalar) |
  | AIFS (cdo POSTPROCESS) | `(valid_time, level, lat, lon)` | float32 | `units: "unknown"`; `OUT_RES: original` keeps the N320 reduced Gaussian grid (no regular lat/lon) |

  Latitudes ascend −90…90 on the regular grids, longitudes run 0…360, `level` is in hPa.
- Units are the ECMWF/AIUQ-st ones (`conf/AIUQ-st/v100/variables.yml` and `outputs.yml`): K, m² s⁻² for `z`
  (geopotential, divide by 9.80665 for metres), m s⁻¹, kg kg⁻¹, Pa. Outputs written before the
  output-variables update (Nov 2026) carry **no meaningful `units` attribute**; newer ones carry
  `units` and `long_name` from the standard.
- Time: `OUT_FREQ: original` = 6-hourly; `daily` = daily means. With `ICS: ecmwf-an` the valid times
  follow the analysis cycle chosen when PREPARE_ICS ran, so they can start the day **before** the
  DATELIST date and differ between experiments with the same date (a00e: 2026-11-16 18 UTC; a00c:
  2026-11-17 06 UTC, both DATELIST 20261117). Read the times from the files; match members/experiments
  on `valid_time`, not on the index.
- AIFS outputs from experiments run before the 1.0.1 fix (commit "Fix AIFS output valid times") are
  labelled 12 h early; check the experiment's commit (`aiuq conf` shows the branch; ask before
  correcting anything).
- Members of a deterministic checkpoint are identical: spread 0 is not a bug.

## Analysing

Work next to the data. `aiuq py <expid> my_script.py args` streams a local script to felipe's
`python3` (xarray, numpy, netCDF4 available), cwd = experiment root, env `AIUQ_OUTPUTS` = outputs dir.
Print small results (CSV/JSON) and save them locally from the command output, or write a small
NetCDF on felipe under the experiment root's `analysis/` folder and `aiuq fetch` it.
For anything that reads many GB (all members × long runs × full globe), use
`aiuq py --srun <expid> …` (Slurm CPU job on partition `felipe`, 4 CPUs, 16 GB, 60 min; override with
`AIUQ_SRUN_PARTITION/CPUS/MEM/TIME`).

Ready-made: `scripts/analysis/ensemble_box.py VAR [--level HPA] [--box LON0 LON1 LAT0 LAT1] [--start YYYY-MM-DD] [--daily] [--members]`
→ CSV `valid_time, n_members, mean, std, min, max[, m1..mN]` of the cos-lat weighted box mean (or the
nearest grid point when LON0 = LON1 and LAT0 = LAT1). About 2 s for 20 members × 16 days at 0.25°.

Skeleton for your own scripts:
```python
import glob, os, xarray as xr
out = os.environ["AIUQ_OUTPUTS"]
files = sorted(glob.glob(f"{out}/t/*/out-*-t.nc"))
ens = xr.open_mfdataset(files, combine="nested", concat_dim="member", decode_timedelta=True)["t"]
lat = "latitude" if "latitude" in ens.dims else "lat"
ens850 = ens.sel(level=850)
mean, spread = ens850.mean("member"), ens850.std("member")
# select before computing; .load() only what you need
```
Comparisons across experiments: only gridpoint-wise when grids match (same `OUT_RES`); otherwise compare
area means or regrid explicitly and say so.

## Dashboard analysis API (read-only, computed on felipe, cached)

`aiuq api '<path>'` prints compact JSON. `start` is the first date in the output file name (`YYYY-MM-DD`),
`level` in hPa, boxes as `lon0,lon1,lat0,lat1` (0–360 longitudes).

| Path | Returns |
|---|---|
| `/api/series?exp=&var=&start=&lat=&lon=[&level=]` | time series at the nearest point for every member (`{"members": {m: [...]}, "times": [...]}`) |
| `/api/series?exp=&var=&start=&lon0=&lon1=&lat0=&lat1=[&level=]` | same, area mean over a box |
| `/api/ens?exp=&var=&start=[&level=][&time=-1]` | ensemble mean and spread map at one time index (thinned for display) |
| `/api/spread?exp=&var=&start=&lon0=&lon1=&lat0=&lat1=[&level=]` | area-weighted ensemble spread and mean per lead time |
| `/api/hovmoller?exp=&var=&start=&lat0=&lat1=[&level=][&member=mean]` | longitude × time over a latitude band |
| `/api/hindcast?exp=&var=&lon0=&lon1=&lat0=&lat1=&tidx=0,4,8[&level=]` | area mean at given time indices for every start date × member (slow on big hindcasts) |
| `/api/exp/<expid>` | the full collector view (config, jobs, attempts, durations, Slurm, outputs listing) |

Maps from `/api/ens` are subsampled; use your own script for full-resolution numbers.

## Bringing data to the laptop

`aiuq fetch <expid> <path relative to outputs/> [dest]` (globs allowed, default dest `./aiuq-data/<expid>/`)
uses rsync over the shared connection and refuses selections above 5 GB (`AIUQ_FETCH_MAX_GB`) unless the
user agreed (`--yes`). Prefer reduced products. The laptop's rsync is openrsync: no `--info=progress2`.

## Other tools of the user (if present)

- **AIUQ-store** (`~/Documents/AIUQ-store`, deployed on felipe at `/work/users/<user>/code/AIUQ-store`):
  catalogue of all outputs as one database, with a Jupyter kernel `aiuq-store`.
- **AIUQ Compare** (`AIUQ/compare/`, gitignored): multi-experiment comparison dashboard on the native
  grid, opened with `compare/deploy/open.sh`.
Mention them when the user wants cross-experiment work; don't modify or deploy them unless asked.
