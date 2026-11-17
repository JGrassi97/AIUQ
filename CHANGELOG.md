# Changelog

All notable changes to AIUQ are listed here. Versions follow [semantic versioning](https://semver.org/).

## [1.0.1] - unreleased

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

### Known issues
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

[1.0.1]: https://github.com/JGrassi97/AIUQ/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/JGrassi97/AIUQ/compare/v0.2.0...v1.0.0
[0.2.0]: https://github.com/JGrassi97/AIUQ/releases/tag/v0.2.0
