# Changelog

All notable changes to AIUQ are listed here. Versions follow [semantic versioning](https://semver.org/).

## [1.1.0] - unreleased

### Added
- **Outputs stored while the experiment runs** (branch `workflow/rolling-archive`, FELIPE): a `STORE` job per start
  date (Autosubmit host, `templates/store.sh`) moves the outputs of that date to the output store as soon as its
  simulations are done and removes its initial conditions; `ARCHIVE` moves only what is left at the end. The jobs
  list the files they write (`write_output` → `<HPCROOTDIR>/store_pending/<SDATE>/<job>.txt`), so `STORE` moves
  exactly those. `PREPARE_ICS` of a start date now waits for the `SIM` jobs two dates earlier: a long experiment
  (e.g. a 30-year climatology, 731 start dates) keeps the initial conditions and outputs of a few dates on the HPC
  instead of ~1 TB, and Slurm holds about two dates of `SIM` jobs instead of thousands. `OUTPUT_STORE.json` says
  `partial` while dates are being stored; shared code of the two jobs in `lib/output_store.sh`.
- **WeatherNext 2** (`MODEL.NAME: weathernext`, branch `models/weathernext-2`): Google DeepMind's ensemble model
  (FGN, [github.com/google-deepmind/weathernext](https://github.com/google-deepmind/weathernext), code Apache 2.0,
  weights CC BY 4.0, released August 2026). Two checkpoints, each a folder with one weights file per
  independently trained model ("seed"): `WeatherNext2_2025` (operational, 0.25°, 4 seeds, fine-tuned on HRES
  analyses) and `WeatherNextCyclones_Mini_2024` (1°, 1 seed). Member *m* runs seed ((*m* − 1) mod *n*) + 1 with its
  own noise, so an ensemble takes the same number of members from every seed, as in the paper. Inputs at T−6h
  and T from the AIUQ-st initial conditions (`z t u v w q` on 13 levels, `t2m msl u10 v10 u100 v100 sst`, surface
  geopotential and land-sea mask); outputs the same fields plus `tp` (6 h totals). New image
  `remote_set/image_weathernext.def` (weathernext v0.3.0, JAX with CUDA 12). Verified so far on FELIPE on CPU: with
  the same inputs the AIUQ code gives the same forecast as the upstream demo, to the bit; the 1° model from the
  ECMWF analysis gives physical fields. GPU runs and the Autosubmit chain are still to be tested.
- **Every output variable of every checkpoint** (branch `cards/output-variables`): the model cards list per checkpoint
  what `SIM` can write (`checkpoints` block), checked against the checkpoints themselves (AIFS metadata, Aurora model
  classes, NeuralGCM `data_to_xarray`) and the upstream documentation. New outputs: AIFS diagnostics `tp cp sf ro
  ssrd strd tcc lcc mcc hcc u100 v100`, soil temperature `stl1 stl2`, ENS 2.0 `fscov`; Aurora 1.5 `i10fg blh` and its
  hourly totals as mean rates/fluxes (`avg_tprate avg_tsrwe avg_sdswrf avg_tnlwrf avg_sduvrf`); NeuralGCM
  precipitation checkpoints `tp` (de-accumulated from the running total) and `avg_ie`. The output-only fields are
  defined in the new `conf/AIUQ-st/v100/outputs.yml` (the IC sources are not asked for them). Outputs now carry
  `units`, `long_name` and, for totals and means, the period (`aiuq_period`, `cell_methods`); daily output sums the
  accumulations instead of averaging them.
- The Launcher offers the variables and levels of the chosen checkpoint and rejects the others; the Monitor names and
  scales the new variables; Weather shows model gusts and, marked ≈, Aurora 1.5 precipitation estimated from its
  hourly rate; the `aiuq options` command prints the outputs per checkpoint.
- **EC-Earth3 CMIP6 initial conditions** (`ICS: ec-earth3`): the state of
  the EC-Earth3 `historical` and ScenarioMIP simulations (member `r4i1p1f1`) read from ESGF over OPeNDAP, only the
  time steps needed. Upper-air profiles combine the 6-hourly fields (7 levels) with the daily means (19 levels);
  the variables EC-Earth3 does not publish are derived or taken from the support files. Optional
  `MODEL.ICS_EXPERIMENT` / `MODEL.ICS_MEMBER`. The vertical-interpolation helpers of the ECMWF analysis moved to
  `AIUQst_lib/pressure_levels.py`. Verified on FELIPE with every checkpoint except AIFS ENS 2.0 (no wave fields);
  the NeuralGCM deterministic checkpoints do not pass (see Known issues).
- **AIFS ENS v2** (`MODEL.CHECKPOINT_NAME: aifs-ens-crps-2.0.ckpt`, ECMWF's operational ensemble model since
  May 2026): 14 pressure levels (10 hPa added), snow depth and soil moisture in input, the ocean-wave component
  (11 wave fields in input and output) and vertical velocity as a diagnostic. It needs a new container image
  (`remote_set/image_anemoi-2.def`: anemoi-inference 0.8.3, torch 2.7, FlashAttention 2.8), chosen by the AIFS
  templates from the checkpoint name; the 1.x checkpoints keep `image_anemoi.sif`. Verified on FELIPE from ERA5 and
  from the ECMWF analysis; not available with EERIE and EC-Earth3, which have no wave fields.
- **AIUQ-st v100**: the standard adds the wave fields (`swh`, `mwd`, `mwp`, `cdww`, `wmb`, `h1012` … `h2530`, type
  `wave`, sea points only) as *optional* variables; everything else is as in v010, which stays in
  `conf/AIUQ-st/v010` as reference. ERA5 takes them from ARCO ERA5 (the period bands `h1012` … `h2530`, not
  archived, are derived from the wave partitions), the ECMWF analysis from the `wave` stream of Open Data.
  Initial conditions written with v010 lack the wave fields and cannot initialise AIFS ENS v2, but remain valid
  for every other checkpoint.
- `SAVE_ICS` for AIFS builds the checkpoint input from the checkpoint metadata (`Checkpoint.variables_from_input`)
  instead of a hard-coded field list, so one model card covers the 1.x and 2.x checkpoints
  (`checkpoints` block of `conf/cards/models/aifs_card.yml`: fields masked over sea, image).
- **Output store**: on FELIPE a final `ARCHIVE` job (Autosubmit host, `templates/archive.sh`) moves the experiment
  folder from the HPC scratch to `<OUTPUT_STORE.DIR>/<expid>_<time>/` (unique even when Autosubmit reuses an expid;
  default `/data/users/<PLATFORM.USER_CODE>/AIUQ-outputs`, configurable per experiment in `main.yml` or in the Launcher): outputs,
  checkpoints, all job logs, the code that ran, the Autosubmit configuration and an `archive.json` record (git commit
  included). The HPC folder is removed only after verification, and `~/autosubmit/<expid>/OUTPUT_STORE.json` points
  to the store. The Monitor, the Launcher (new "Output store" field), Compare, Weather and the `aiuq` skill follow it.

### Changed
- **One CF-compliant output format for every model** (`aiuq_output_format` 1.0, described in the new documentation
  page "Output format"). NeuralGCM, Aurora and AIFS used to write their files each in its own way; they now go
  through one writer (`runscripts/AIUQst_lib/output.py`):
  - NetCDF-4 files following CF-1.8, dimensions `(valid_time, [level,] latitude, longitude)` for all models
    (NeuralGCM stored longitude before latitude, AIFS named them `lat`/`lon`), latitudes south to north, levels
    from 1000 hPa upwards, CF attributes on every coordinate;
  - the initial time of the forecast (`time`), the lead time (`step`) and the member (`number`) in every file:
    AIFS files had lost them in the cdo regridding;
  - `standard_name`, `long_name`, CF units and the ECMWF parameter number on every variable; totals and means
    carry `cell_methods` and the period they refer to (`valid_time_bnds`); fields at 2, 10 and 100 m a `height`;
  - global attributes that say what wrote the file: model, checkpoint, initial conditions, experiment, member,
    commit, and the institution when `OUTPUT_FORMAT.INSTITUTION` is set;
  - compressed and chunked as the experiment asks (next entry). Files are written under a temporary name, read
    back, compared with what was computed and renamed when complete.
  The AIFS `POSTPROCESS` job writes the final file in the AIFS image after cdo
  (`runscripts/finalize_aifs_output.py`). Verified with the IOOS compliance checker (no CF-1.8 error) and by
  rewriting real outputs in the four model images; the Monitor, Compare, Weather and AIUQ-store read the new
  files unchanged. Outputs written before this change keep their format.
- **Output compression, chosen per experiment**: `EXPERIMENT.OUT_COMPRESSION` = `none` | `lossless` (default) |
  `bits` | `integer` | `mixed` | `custom`, with `OUT_COMPRESSION_OPTS` (`KEEPBITS` 4-23, `CHUNKS` point / map /
  three numbers, `COMPLEVEL`, `CUSTOM`). Until now SIM wrote uncompressed files, `float64` for NeuralGCM and
  Aurora: an operational run (3 models, 10 members, 10 days at 0.25 degrees) took about 100 GB.
  - `lossless` is `float32` with zlib and shuffle; `bits` rounds to significant bits; `integer` and `mixed` store
    the variables that have a step as CF integers (`scale_factor` / `add_offset`): 1 degree C, 1 m/s, 1 hPa, 1 m,
    1 %, and precipitation at 1 mm (`integer`) or 0.1 mm (`mixed`). The steps are in the AIUQ-st standard (`pack`).
  - One library for the three models and for AIFS after cdo (`runscripts/AIUQst_lib/compression.py`); every file
    is verified before it gets its name and says how it is stored in `aiuq_compression` (JSON).
  - `runscripts/compress_outputs.py` compresses existing outputs in place: parallel, verified before the atomic
    replacement, repeatable. It recognises the files compressed by the earlier scripts and leaves alone a file
    stored with less precision than asked.
  - Launcher: the presets with a size estimate, advanced options and a rule table for `custom`; `mixed` proposed
    for `ics: ecmwf-an`, `lossless` otherwise; warnings and errors in the preview. New JSON fields
    `out_compression`, `out_compression_opts`. Monitor: the scheme and the size of every experiment.
  - The rules of `custom` are a list of texts in `main.yml`: Autosubmit hands the jobs single values only, so a
    nested mapping would be lost; a job that does not find its rules stops instead of using others.
- **AIUQ-st v100 describes every variable for CF**: `standard_name` (49 of 62 variables, checked against version
  95 of the CF table), `height`, and the units in CF notation (`m2 s-2` instead of `m2 s**-2`, `1` instead of
  `(0 - 1)` and `Numeric`, `m` instead of `m of water equivalent`). `z` is named "Geopotential", which is what
  its units say (it was "Geopotential height").
- **No personal or site values in the repository.** The FELIPE platforms take the HPC user and the scratch folder
  from `PLATFORM.USER_CODE` of the experiment's `main.yml` (as MareNostrum 5 already did), and the default output
  store is `/data/users/<PLATFORM.USER_CODE>/AIUQ-outputs`. Experiments whose `main.yml` sets `USER_CODE` (every
  experiment created by the Launcher) resolve to the same folders as before.
- **Dashboard**: the settings that describe an installation have no default any more and must be set in
  `~/.config/aiuq-dashboard.env`: `AIUQ_REMOTE_HOST`, `AIUQ_REMOTE_BASE`, `AIUQ_LAUNCH_PLATFORM`,
  `AIUQ_SUPPORT_FOLDER`, `AIUQ_REMOTE_DELETE_ROOTS` (`AIUQ_REMOTE_USER` defaults to the local user). `install.sh`
  lists the ones still empty, and the dashboard names a missing setting instead of guessing; a Launcher delete is
  refused while the HPC folder cannot be determined. **When updating an existing installation, add these settings
  before restarting the service.** The HPC helpers are now `helpers/hpc_probe.py`, `hpc_preview.py`, `hpc_delete.py`.
- **Launcher**: the support status of a checkpoint with an IC source comes from `docs/support-matrix.yml` of the
  chosen branch instead of a table in the code, so it follows the documentation (and knows `ecmwf-an`).
- **Agent skill**: `aiuq` reads the two hosts and the experiments folder from `~/.aiuq/config`
  (`skill/aiuq/config.example`) and has no built-in host; `aiuq site` prints the settings and the user's site notes
  (`~/.aiuq/site.md`, template `skill/aiuq/site.example.md`), which hold what is specific to an installation. New
  optional settings `AIUQ_AS_BIN`, `AIUQ_DASH_SOCK`, `AIUQ_GPU_PARTITIONS`, `AIUQ_SRUN_PARTITION`.
- Documentation: "Running AIUQ at a new site" (`docs/source/configuration.rst`) says where every site setting
  lives and how to add a machine. `tests/test_no_site_paths.py` fails when a personal path or a built-in host
  comes back.

### Fixed
- **Out of memory in the SIM post-processing with fine output grids**: NeuralGCM and Aurora regridded every model
  level to `OUT_RES` and only then selected `OUT_LEVS`. A 16-day NeuralGCM run with `OUT_RES: 0.25` and two levels
  needed about 60 GB and was killed (limit 48 GB). Levels are now selected first and the regridding works in blocks
  of time steps: the same run peaks at about 3.5 GB, and results are unchanged.
- **AIFS output times were 12 h early**: `valid_time` was rebuilt from the first IC state (T−6h) instead of the
  initial state used by AIFS (T), so every AIFS step was labelled 12 h before its true valid time (and daily means
  mixed the wrong hours). Times now come from the states returned by the model. Existing AIFS 6-hourly outputs can
  be corrected by adding 12 h to `valid_time`; daily means must be recomputed.
- **AIFS ignored `OUT_RES`**: the POSTPROCESS job always regridded to 1°. It now produces the requested grid
  (0.25° / 0.5° / 1° / 1.5° / 2°, the same regular grids as the other models) and keeps the native N320 grid for
  `original`. AIFS experiments run with another `OUT_RES` need SIM and POSTPROCESS again (the native SIM output is
  removed after POSTPROCESS).
- AIFS POSTPROCESS now fails when cdo fails (it used to finish as COMPLETED without the output), removes the SIM
  output only after the final file is written, and skips variables the SIM did not produce.

### Verified
- **Support matrix, 2026-11-21** (FELIPE, commit `d1880c0`): every combination that was "work in progress" was run
  as a 2-day forecast through the Launcher (19 experiments, 2 members for ensemble checkpoints) and its outputs
  checked: no NaN, temperature and geopotential height at 850 and 500 hPa within physical ranges, stable global
  means, different members. 16 are now available: AIFS ENS 2.0 from ERA5 and from the ECMWF analysis; AIFS single
  1.1 and the four Aurora checkpoints from EERIE; from EC-Earth3 the three NeuralGCM stochastic checkpoints, AIFS
  single 1.1, AIFS ENS 1.0 and the four Aurora checkpoints. The same runs exercised the site-settings change
  (user, scratch folder and output store resolved from `PLATFORM.USER_CODE`) and the `ARCHIVE` job.

### Known issues
- **NeuralGCM deterministic checkpoints from EC-Earth3** (`deterministic_0_7/1_4/2_8_deg.pkl`, still work in
  progress): the jobs complete, but the outputs have unphysical values at isolated points over the tropical oceans
  from the first output time (temperature at 850 hPa up to 3199 K with `deterministic_1_4`, 380 K with
  `deterministic_2_8`; a few points of geopotential height with `deterministic_0_7`). The same checkpoints are
  clean from ERA5 and the stochastic checkpoints are clean from EC-Earth3, which points to the EC-Earth3 initial
  state for these checkpoints.
- ECMWF accepts at most 5 queued MARS requests per user: several `ICS: eerie` experiments started together fail in
  `PREPARE_ICS` with `USER_QUEUED_LIMIT_EXCEEDED`. Start them one at a time.
- Aurora 1.5 writes precipitation, snowfall and radiation only for the last hour of each 6 h step; their units are
  not documented upstream (inferred: m and J m⁻²). None of the new outputs has been produced by a real run yet.
- AIFS output files (written by cdo) name the horizontal dimensions `lat`/`lon`; the other models use
  `latitude`/`longitude`.

## [1.0.0] - 2026-11-17

First stable release. The ERA5 and ECMWF-analysis combinations of the support matrix were run on FELIPE for this
release; the EERIE column keeps its 0.x status and will be re-tested. See the support matrix in the
[README](README.md) and the [documentation](https://aiuq.readthedocs.io).

### Added
- **Microsoft Aurora** (`MODEL.NAME: ms-aurora`): 0.25° pretrained, fine-tuned, 1.5 and 1.5 Ensemble checkpoints.
  The 1.5 Ensemble is stochastic: each member gets a reproducible noise seed from its member number.
- **Latest ECMWF analysis** as initial conditions (`ICS: ecmwf-an`): IFS HRES step 0 from ECMWF Open Data, with
  analysis-cycle selection, selective GRIB downloads and fallback fields. No ECMWF account needed.
- **FELIPE** (Politecnico di Torino) platform: Slurm job configuration with memory and wall-clock limits measured
  for each job.
- **Dashboard** (`tools/dashboard`): an Autosubmit Monitor (read-only) and an AIUQ Launcher to create, start, stop and
  delete experiments, served on a private Unix socket.
- Sphinx documentation on Read the Docs, with the support matrix generated from `docs/support-matrix.yml`;
  `docs/generate_table.py` regenerates the README table from the same file.
- `runscripts/update_wind_climatology.py`, a resumable tool that adds the 100 m wind climatology from ARCO ERA5
  to the climatology file.
- Test suite (`tests/`): ECMWF analysis and EERIE retrieval, fallbacks, wind climatology, output streaming, output
  regridding and support-matrix consistency.

### Changed
- Model outputs are written to disk step by step and post-processed one variable at a time, so memory no longer
  grows with forecast length or number of output variables. NeuralGCM hindcasts unroll in blocks of `CHUNK_STEPS`
  steps instead of the whole run at once.
- AIFS output files and folders use the AIUQ-st names, as for the other models (for example `t2m/` instead of
  `2t/`). Scripts that read AIFS surface outputs from 0.x experiments need the new names.
- EERIE initial conditions use expver 0001 and soil moisture, with validated ERA5 fallbacks for fields that are not
  archived.
- AMIP forcings are selected with `EXPERIMENT.FORCING_VERSION` (`ESA-CCI-v3.0` or `AIMIP`) and loaded from
  `<SUPPORT_FOLDER>/amip-forcings/<FORCING_VERSION>/`.
- The support folder is mounted read-only in the containers.
- The documentation and the README use the GitHub repository (`https://github.com/JGrassi97/AIUQ.git`).

### Fixed
- **NeuralGCM runs longer than one chunk** (`CHUNK_STEPS`, AMIP runs in 0.x): every chunk after the first skipped
  one model state and its outputs were labelled 6 h early. Each chunk now continues exactly from the previous one
  (verified bit-for-bit against a single-chunk run). AMIP outputs produced with 0.x should be regenerated.
- **Missing values at the grid edges with `OUT_RES`**: regridding NeuralGCM and Aurora outputs left all-NaN rows at
  the poles (latitudes beyond the outermost model row: ±88–90° at 2.8°, ±90° at 0.7°, −90° for Aurora) and, for
  NeuralGCM, all-NaN columns just below 360° (no periodic wrap). Longitude is now interpolated periodically and
  polar rows take the outermost model row; all other values are unchanged.
- A requested output variable that the model does not produce (for example `t2m` with NeuralGCM) is skipped with a
  warning instead of failing the SIM job after the forecast; the job fails only if none is available.
- AIFS with `OUT_LEVS: original` no longer fails.
- Checkpoint download links: AIFS links pointed to the Hugging Face web page instead of the file, and NeuralGCM
  links required a Google login.
- `EARTHKIT_REGRID_CACHE` was set to a literal string in the AIFS templates.
- FELIPE GPU jobs are submitted to the `gpu` partition, which FELIPE's Slurm now requires.

### Known issues
- Aurora output files are written by xarray 2025.12, which stores the `step` coordinate with a `dtype` attribute.
  Older xarray versions fail to open them; use xarray ≥ 2025.12 or `xr.open_dataset(..., decode_timedelta=False)`.
- AIFS forecasts on GPU are not bit-reproducible: two identical runs differ slightly from the first step.
- NeuralGCM specific humidity (`q`) can be slightly negative (down to about −1 g/kg at a few percent of the points),
  from the model's spectral representation; the initial conditions are positive. AIUQ does not clip it.
- NeuralGCM output files are laid out (time, level, longitude, latitude), the other models (time, level, latitude,
  longitude). Select by dimension name rather than by position.

### Removed
- The GitLab mirror workflow.
- ACE is listed as planned for a later release (it was marked as in development for ERA5 but has no integration yet).

## [0.2.0] - 2026-03-24

AMIP simulations with AIMIP and ESA-CCI-v3.0 forcings.

[1.1.0]: https://github.com/JGrassi97/AIUQ/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/JGrassi97/AIUQ/compare/v0.2.0...v1.0.0
[0.2.0]: https://github.com/JGrassi97/AIUQ/releases/tag/v0.2.0
