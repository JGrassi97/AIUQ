# Outputs and analysis

## Layout on the HPC

```
<root>/outputs/<var>/<member>/out-<first valid day>-<last valid day>-<member>-<var>.nc
<root>/outputs/checkpoints/          # NeuralGCM AMIP restarts, not model output
<root> = <SCRATCH_DIR>/<PROJECT>/<USER>/<expid>      # of the platform; `aiuq site` prints the base folder
```

As soon as every simulation of a start date has finished, the `STORE` job of that date (Autosubmit host) moves its outputs to
the output store; the `ARCHIVE` job moves what is left (checkpoints, logs, code) at the end. While the experiment runs,
`OUTPUT_STORE.json` says `partial` and `aiuq outputs` lists only the dates still on the HPC; the dates already stored are
in `<store>/<expid>_<time>/outputs/` (readable from the HPC at the same path). Once `ARCHIVE` is done the whole experiment
is in the output store (`OUTPUT_STORE.DIR` in `main.yml`, default from `conf/general.yml`: `/data/users/<PLATFORM.USER_CODE>/AIUQ-outputs`), in a
folder of its own because Autosubmit can reuse an expid:

```
<store>/<expid>_<YYYYmmddTHHMMSS>/outputs/...   # same layout as above, checkpoints included
<store>/<expid>_<YYYYmmddTHHMMSS>/LOG_<expid>/  # all job logs
<store>/<expid>_<YYYYmmddTHHMMSS>/archive.json  # source, time, files, bytes, git commit
~/autosubmit/<expid>/OUTPUT_STORE.json          # on the Autosubmit host: the folder (status "done")
```

Then `<root>` is removed from the HPC. `aiuq` commands and analysis read the store from the HPC, so the HPC must
see the store at the same path (e.g. a read-only mount). The site notes list the stores in use.

One file per variable × member × chunk (× start date: the start date shows up only as the first day
in the name). Example: `t/7/out-2026-11-16-2026-12-02-7-t.nc`.
AIFS writes intermediate `*_temp.nc` / `*_grid.nc` files that POSTPROCESS replaces with the final one;
ignore them (a leftover `_temp.nc` means POSTPROCESS did not run or failed).

## Conventions (check with `aiuq ncinfo` — never assume)

- One data variable named like the folder (AIUQ-st short names, ECMWF parameter database:
  `t z u v q w` on pressure levels; `t2m msl sp u10 v10 d2m tcw skt …` at the surface; output-only fields of
  `conf/AIUQ-st/v100/outputs.yml` such as `tp cp sf ro ssrd strd stl1 i10fg avg_tprate …`).
- **Format** (global attribute `aiuq_output_format`; full description in `docs/source/outputs.rst`):
  files with `aiuq_output_format = "1.0"` are CF-1.8 and identical in layout for the three models:
  `(valid_time, [level,] latitude, longitude)`, float32, latitudes −90…90, longitudes 0…360, `level` in hPa
  from 1000 upwards; `time` (scalar) is the initial time of the forecast, `step` the lead time in hours,
  `number` the member; `standard_name`, `long_name`, `units` on every variable; global attributes say model,
  checkpoint, initial conditions, experiment and commit.
- **Storage** (global attribute `aiuq_compression`, JSON; `aiuq conf <expid>` shows `OUT_COMPRESSION`): depending on
  the experiment the data variable is float32 (`lossless`, or `bits` = rounded to N significant bits), float64
  (`none`) or **int16/int32 with `scale_factor` and `add_offset`** (`integer`, `mixed`, `custom`: the error is
  half a step, e.g. 0.5 °C, 0.5 hPa, 0.05 mm). xarray and netCDF4 decode them by themselves to floats in the
  units of the variable: never read the raw integers, and do not switch decoding off (`mask_and_scale=False`,
  `set_auto_maskandscale(False)`). Keep `decode_timedelta=True` as before. With a lossy scheme, differences
  between fields or members smaller than the step are not meaningful: check the tag before computing spreads of
  slowly varying fields. The forecasts archived in November 2026 carry an older text in the same attribute
  (`AIUQ mixed integer packing v1 (...)` = `mixed`, `float32, BitRound keepbits=12, ...` = `bits`).
- **Files without `aiuq_output_format`** (written before the format, AIUQ ≤ 1.0 and the stores of Nov 2026)
  differ between models — **select by name, never by position**:

  | Model | Level field dims | Notes |
  |---|---|---|
  | NeuralGCM | `(valid_time, level, longitude, latitude)` — longitude first | coords `time` (scalar), `step`; no coordinate attributes |
  | Aurora | `(valid_time, level, latitude, longitude)` | coord `time` (scalar); no coordinate attributes |
  | AIFS (cdo) | `(valid_time, level, lat, lon)`, levels 250 → 850 | no initial time in the file; oldest ones `units: "unknown"` |

- Periods (`cell_methods` + `valid_time_bnds`; old files: attribute `aiuq_period`, or nothing):
  AIFS `tp cp sf ro` (m) and `ssrd strd` (J m⁻²) are totals over the 6 h step ending at `valid_time`
  (daily output = daily total); NeuralGCM `tp` is the total over each output step (de-accumulated from the
  model's running total) and `avg_ie` the rate of its last 1 h step; Aurora 1.5 `avg_*` are mean rates
  (kg m⁻² s⁻¹) or fluxes (W m⁻²) over the **last hour** of each step only (the model writes 1-hour totals).
- Units are the ECMWF/AIUQ-st ones in CF notation (`conf/AIUQ-st/v100/variables.yml` and `outputs.yml`): K,
  `m2 s-2` for `z` (geopotential, divide by 9.80665 for metres), `m s-1`, `kg kg-1`, Pa, `1` for fractions.
  Old files write them the ECMWF way (`m2 s**-2`, `(0 - 1)`) or not at all.
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

Work next to the data. `aiuq py <expid> my_script.py args` streams a local script to the HPC's
`python3` (xarray, numpy, netCDF4 available), cwd = experiment root, env `AIUQ_OUTPUTS` = outputs dir.
Print small results (CSV/JSON) and save them locally from the command output, or write a small
NetCDF on the HPC under the experiment root's `analysis/` folder and `aiuq fetch` it.
For anything that reads many GB (all members × long runs × full globe), use
`aiuq py --srun <expid> …` (Slurm CPU job, 4 CPUs, 16 GB, 60 min, on the partition `AIUQ_SRUN_PARTITION` of the config; override with
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

## Dashboard analysis API (read-only, computed on the HPC, cached)

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

- **AIUQ-store** (its own repository, deployed on the HPC):
  catalogue of all outputs as one database, with a Jupyter kernel `aiuq-store`.
- **AIUQ Compare** (its own repository): multi-experiment comparison dashboard on the native grid.
Mention them when the user wants cross-experiment work; don't modify or deploy them unless asked.
