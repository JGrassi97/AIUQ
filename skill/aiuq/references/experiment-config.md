# Experiment configuration

## Launcher JSON (input of `aiuq preview` / `aiuq create`)

The dashboard Launcher validates this JSON, turns it into `<expid>/conf/main.yml` and runs
`autosubmit expid -min -H FELIPE -d <description> -repo https://github.com/JGrassi97/AIUQ.git -b <branch> -conf conf/bootstrap/`
followed by `autosubmit create <expid> -np`. Always run `aiuq preview` first; fix every `error:` line.

```json
{
  "branch": "main",
  "description": "NeuralGCM 1.4 stochastic from latest ECMWF analysis, 15 d, 10 members",
  "model": "neuralgcm",
  "checkpoint": "stochastic_1_4_deg.pkl",
  "ics": "ecmwf-an",
  "run_type": "hindcast",
  "forcing": "ESA-CCI-v3.0",
  "members": 10,
  "dates": ["20261117"],
  "chunk_size": 15,
  "chunk_unit": "day",
  "num_chunks": 1,
  "out_vars": ["t", "z"],
  "out_levs": [1000, 850, 500],
  "out_res": "1",
  "out_freq": "daily"
}
```

| Field | Values / rule |
|---|---|
| `branch` | A branch that exists on GitHub (`git ls-remote`); the experiment clones it. Default `main`. |
| `description` | Required, one line, < 200 chars. Shows up in `aiuq list`. |
| `model` | `neuralgcm`, `aifs`, `ms-aurora` (ACE is not implemented). |
| `checkpoint` | Must exist in `<support>/models/<model>/` on felipe (`aiuq options` shows ON HPC / MISSING). |
| `ics` | `era5`, `eerie`, `ecmwf-an`, `ec-earth3` (see below). |
| `run_type` | `hindcast` (all models) or `amip` (NeuralGCM only). |
| `forcing` | AMIP only: `ESA-CCI-v3.0` or `AIMIP`. Keep the default for hindcasts (ignored). |
| `members` | 1–100 → `MEMBERS: "1 2 … N"`. Member number = RNG seed for stochastic checkpoints. |
| `dates` | `YYYYMMDD` list, ≤ 2000, compressed to Autosubmit notation (`202001[01 15] 20200201`). ERA5 ≥ 1940. |
| `chunk_size`, `chunk_unit`, `num_chunks` | Forecast length = size × chunks (`day` or `month`). For hindcasts use `num_chunks: 1` and put the whole length in `chunk_size` (that is what every tested experiment does; multi-chunk restarts are meant for NeuralGCM AMIP). |
| `out_vars` | AIUQ-st short names the model produces (`aiuq options`): e.g. `t z u v q` on levels, `t2m msl sp u10 v10 …` at the surface for AIFS/Aurora. NeuralGCM has no surface atmospheric fields (only `sst ci` + levels). |
| `out_levs` | hPa levels from the model's level list, or `"original"` for all of them. |
| `out_res` | `"original"` (model grid; N320 reduced Gaussian for AIFS), `"0.25" "0.5" "1" "1.5" "2"` regular lat-lon. |
| `out_freq` | `original` (model step, 6-hourly) or `daily` (daily means). |

Number of SIM jobs = dates × members × num_chunks (each needs one GPU). Above ~3000 the Launcher warns.

### Recipes
- **Ensemble forecast from the latest analysis**: `ics: ecmwf-an`, `dates: [today]`, stochastic/ensemble
  checkpoint, 10–20 members, 10–15 days.
- **Hindcast set**: `ics: era5`, many `dates` (e.g. 1st and 15th of each month), few members.
- **Copy an experiment with changes**: `aiuq template <expid> > cfg.json`, edit, preview.
- **Deterministic checkpoints** (`deterministic_*`, `aifs-single-*`, Aurora except `v1.5-ensemble`)
  give identical members: use `members: 1` (IC perturbations are not implemented yet).

## Models and checkpoints (support folder on felipe: `/work/users/<user>/AIUQ/models/<model>/`)

| Model | Checkpoints | Kind | Native grid / levels | Modes |
|---|---|---|---|---|
| `neuralgcm` | `stochastic_1_4_deg.pkl`, `stochastic_precip_2_8_deg.pkl`, `stochastic_evap_2_8_deg.pkl` | ensemble | resolution in the name (0.7°/1.4°/2.8°), 37 levels (1–1000 hPa) | hindcast, amip |
| | `deterministic_0_7_deg.pkl`, `deterministic_1_4_deg.pkl`, `deterministic_2_8_deg.pkl` | deterministic | | |
| `aifs` | `aifs-ens-crps-1.0.ckpt` | ensemble | N320, 13 levels (50–1000 hPa) | hindcast |
| | `aifs-single-mse-1.1.ckpt` | deterministic | | |
| | `aifs-ens-crps-2.0.ckpt` | ensemble | N320, 14 levels (10–1000 hPa), wave fields (`swh mwd mwp cdww h1012…h2530`) in output; needs `image_anemoi-2.sif`; branch `models/aifs-ens-2`, not yet verified; no EERIE | hindcast |
| `ms-aurora` | `aurora-0.25-v1.5-ensemble.ckpt` | ensemble | 0.25°, 13 levels (50–1000 hPa) | hindcast |
| | `aurora-0.25-pretrained.ckpt`, `aurora-0.25-finetuned.ckpt`, `aurora-0.25-v1.5.ckpt` | deterministic | | |

Support status per IC lives in `docs/support-matrix.yml` (rendered in the README). In 1.0: everything is
✅ with ERA5 and `ecmwf-an`; with EERIE, NeuralGCM and AIFS-ens are ✅, AIFS-single and Aurora are ⚠️ (work
in progress). The Launcher's internal badge table is older (it calls some ✅ combinations "wip" and
has no `ecmwf-an` column, so it warns "No support information" for `ecmwf-an` — expected, ignore).
On felipe there is also a stray `stochasti_evap_2_8_deg.pkl` (typo copy); use `stochastic_evap_2_8_deg.pkl`.

## Initial conditions

| `ics` | Source | Notes |
|---|---|---|
| `era5` | ARCO ERA5 on Google Cloud | No credentials. 1940 onwards. |
| `eerie` | MARS via ECMWF Web API | Needs `~/.ecmwfapirc` on felipe (public EERIE data works; the user's key has **no** access to operational MARS `class=od`). PREPARE_ICS can sit in the MARS queue for a long time. |
| `ecmwf-an` | ECMWF Open Data, IFS HRES step 0 | No credentials. `dates: [today]` → newest analysis cycle T with T-6h and T published (searched back 24 h). Past dates (0.25° archive from Feb 2024, Google mirror) use that day's 18 UTC. Output valid times follow the cycle picked when PREPARE_ICS ran and can start the **previous day** (e.g. DATELIST 20261117 → files `out-2026-11-16-…`). `slt lsm sdor slor orog` from the static file, `clwc ciwc` from climatology; 1–7 hPa extrapolated (NeuralGCM only uses them). |
| `ec-earth3` | EC-Earth3 CMIP6 via ESGF OPeNDAP | Branch `ics/ec-earth3`, work in progress. No credentials. A climate-model state at 00/06 UTC of the date, not observed weather. ≤ 2014 `historical`, later the card's scenario (`ssp585`); override with `MODEL.ICS_EXPERIMENT`/`ICS_MEMBER` in main.yml (the Launcher does not set them). No wave fields: not for AIFS ENS v2. |

## The generated `conf/main.yml`

```yaml
MODEL:     {NAME: neuralgcm, CHECKPOINT_NAME: stochastic_1_4_deg.pkl, ICS: ecmwf-an}
EXPERIMENT:
  RUN_TYPE: "hindcast"
  MEMBERS: "1 2 3 4 5 6 7 8 9 10"
  CHUNKSIZEUNIT: day
  DATELIST: 20261117
  CHUNKSIZE: 15
  NUMCHUNKS: 1
  CALENDAR: standard
  OUT_VARS: [t, z]
  OUT_FREQ: daily
  OUT_RES: "1"
  OUT_LEVS: [1000, 850, 500]
  FORCING_VERSION: ESA-CCI-v3.0
PATHS:     {SUPPORT_FOLDER: /work/users/<user>/AIUQ, SIF_FOLDER: "%PATHS.SUPPORT_FOLDER%/sif"}
PLATFORM:  {NAME: FELIPE, USER_CODE: <user>}
```

Autosubmit merges it with `conf/platforms.yml`, `conf/general.yml`,
`conf/jobs/FELIPE/models/<model>/<model>.yml` and `conf/jobs/FELIPE/ics/<ics>/<ics>.yml` (see
`conf/bootstrap/include.yml`). Job resources come from those job files, not from `main.yml`.
MareNostrum 5 job files exist in the repo, but this setup (Launcher, paths) targets FELIPE only.

## Resources and timings on felipe (measured Nov 2026, L40S)

| Job | Resources | Typical time |
|---|---|---|
| PREPARE_ICS | `FELIPE` platform (direct process, no Slurm) | ~30 s (`ecmwf-an`); ERA5 varies with download speed; EERIE can wait hours in the MARS queue |
| NeuralGCM SIM | 1 GPU, 20 threads, 48 GB | 1.4°, 20 d: ~2.5 min; 0.7°, 30 d: ~15 min |
| AIFS SAVE_ICS | CPU, 16 GB | ~1 min |
| AIFS SIM | 1 GPU, 24 GB | 31 d, 14 vars × 13 levels: ~2.5 min |
| AIFS POSTPROCESS | CPU (cdo), 8 GB | seconds per member |
| Aurora SIM | 1 GPU, 48 GB | 1.5-ens, 30 d, 19 vars × 13 levels at 0.25°: ~6 min |

SIM wall-clock limit is 1 h; Slurm partitions: `gpu` (2 days), `fat_felipe` (12 h), `felipe` (CPU work).
With 2 GPUs, wall time ≈ SIM jobs × SIM time / 2 **plus queue waits** (other users; can be hours to a day).
The `preview` estimate comes from finished experiments with the same model and excludes queue waits.
