# AIUQ dashboard

A web dashboard for the Autosubmit experiments of the AIUQ framework, made of two apps (switch in the header):

- **Autosubmit Monitor** — read-only: pickles, sqlite (opened `mode=ro`), YAML, log files, `squeue`, `nvidia-smi`.
- **AIUQ Launcher** — creates experiments and starts/stops their runs (see below). It is the only part that writes.

The start page (`/`) introduces both apps with live figures, the experiments running now and the latest ones.

In the Monitor, the experiments panel on the left groups experiments by state (running, needs attention, not
finished, completed; newest first) and has a filter (id, model, initial conditions, start date, description). It can be
hidden with the button at the left of the header; the choice is remembered per browser. Moving the pointer over an
experiment loads it in advance and visited experiments are kept in memory, so opening one is immediate.

Keyboard: `/` filter experiments · `J` / `K` next / previous experiment · `1`–`4` tabs · `R` refresh now ·
`[` hide the panel · `Esc` close the job details · `?` guide (also the "Guide" button in the header).

It runs permanently on the Autosubmit host as a `systemd --user` service and listens on a **private
Unix socket** (`~/.aiuq-dashboard/dash.sock`, folder 0700, socket 0600), so only your user can reach it.

## Open it

From your laptop:

```bash
ssh -N -L 8765:/home/<user>/.aiuq-dashboard/dash.sock <autosubmit-host>   # absolute socket path on the host
```

Then browse to <http://localhost:8765>. The selected experiment is part of the URL, so links can be shared
(for example <http://localhost:8765/?exp=a00b>).

## What it shows

| Panel | Source | Measured or estimated |
|---|---|---|
| Experiment list, active/stopped badge | `~/autosubmit/<expid>/conf/{main,minimal}.yml`, `autosubmit.db`, `/proc` (a live `.../autosubmit run <expid>` process) | measured |
| Job counts, status grid (date × section/member/chunk) | `pkl/job_list_<expid>.pkl`, read with Autosubmit's own Python | measured |
| Timeline (Gantt), durations | `<job>_STAT_<n>` files on the HPC host (start, end, result); falls back to `metadata/data/job_data_<expid>.db` | measured |
| Time to finish | jobs left × median duration ÷ observed parallelism, per section. When a section has no history, the median is borrowed from comparable experiments (labelled "from …") | **estimate** |
| ICS / MARS | key lines from the PREPARE_ICS stdout (`Requesting N fields`, `Request is …`, `Request cost …`, errors) | measured (parsed logs) |
| SLURM / GPU | `squeue -u $USER`, `sinfo`, `nvidia-smi` on the HPC host. `sacct` is not used (it is disabled on some sites) | snapshot |
| Running job logs | tail of the job stdout/stderr, with shell trace (`set -xv`) removed | measured |
| Outputs and map | `outputs/<var>/<member>/*.nc` listing (once `~/autosubmit/<expid>/OUTPUT_STORE.json` says the ARCHIVE job is done, logs and outputs are read from the store folder `<store>/<expid>_<time>/`); the map reads one time step and one level on the HPC host and subsamples to ≤ 360 longitudes | measured |
| Failed jobs | stderr of each FAILED job, from the last `Traceback` onwards (or its last lines) | measured |
| Experiment setup card | ICS → model → ensemble → outputs, workflow chain with per-section progress, start-date strip | measured / config |
| Job details (click a grid cell, a start date, a Gantt bar or a job name; opens as a panel on the right, `Esc` closes it) | the job's attempts, duration vs. section median, stderr / log tail / MARS lines | measured |
| Map: shaded field + contours | any two output variables (same member, start date and time step); colormap, reverse, min/max, contour count or interval, unit conversions (K→°C, geopotential→m/dam); style is remembered in the browser | measured |
| Point time series (click the map) | nearest grid point, every member of that start date, ensemble mean, member shown on the map highlighted; click a time to move the map there (`/api/series`) | measured |
| Area average (series card → "Area average") | preset regions, a box drawn on the map, or the visible map area; cos(latitude)-weighted mean per member, computed on the HPC host | measured |
| Place search (above the map) | Open-Meteo geocoding API, called **from your browser**: only the typed place name is sent; coordinates like `45.07, 7.69` work offline | external lookup |
| HPC performance | per step: waiting for Autosubmit (ready → submitted, from the pkl), queued (submitted → started), running (STAT); jobs running at once; finished jobs over time; pauses; failed attempts; platform limits from the experiment config; automatic "what is slowing things down" findings | measured; findings are heuristics |
| GPU use history | node `nvidia-smi` and `squeue` sampled every refresh, kept 7 days in `~/.local/share/aiuq-dashboard/history.json` (node-wide, shared with other users) | measured |

Each experiment page has three tabs in the bar under the header (kept in the URL as `&tab=live|review|data`; running experiments open on Live, stopped ones on Run review):

- **Live** — right now (running jobs vs. typical duration, SLURM-pending jobs and why, latest finished), progress grid, initial conditions / MARS, simulations & GPUs, problems.
- **Run review** — setup, timeline, durations & time left, HPC performance.
- **Data** — maps & time series, plus:
  - map field: one member, ensemble mean, ensemble spread (std) or member − mean (`/api/ens`);
  - time series as member lines, min–max range or daily box plots;
  - spread growth: area-weighted ensemble spread per forecast day for a region, with kept curves for comparison (`/api/spread`);
  - Hovmöller: longitude × time over a latitude band, ensemble mean or one member (`/api/hovmoller`);
  - across start dates: area mean at chosen forecast days for every start date and member, and the average drift with lead time (`/api/hindcast`, on demand: it reads one file per start date × member, ~12 s for 1080 files).

Every section can be collapsed (click its title, or "Collapse all" at the right of the tab bar); the choice is
remembered per browser. On a refresh only the parts that changed are redrawn, so logs keep their scroll position.

Why the pickle and not `autosubmit monitor -o txt`: reading the pickle takes about 0.1 s for all experiments,
includes WAITING/READY jobs, and writes nothing. `monitor` takes seconds and writes files into
`<expid>/status/`. The Autosubmit API configured in `~/.autosubmitrc` did not respond on our host, so it is not used.

## How data is collected

A single background thread runs one cycle every `AIUQ_REFRESH` seconds (45 by default):

1. Local reads on the Autosubmit host. Pickles are re-read only when their mtime changes.
2. **One** ssh call to the HPC host. It goes through a persistent ControlMaster (`~/.ssh/aiuq-dash/`) and
   streams `aiuq_dash/helpers/hpc_probe.py` over stdin, so nothing is installed on the HPC. Fetches are
   incremental: only STAT and log files newer than the previous cycle are read. Idle experiments are refreshed
   every `AIUQ_SLOW_REFRESH` seconds (600 by default).
3. Views are rebuilt and swapped in. HTTP requests never wait on ssh, except map previews, which are cached.

The page polls the server every 30, 45 or 60 s (selectable) and always shows when the data was collected.

## AIUQ Launcher

Four views in the tab bar (kept in the URL as `&view=runs|delete|activity`): **New experiment**, **Launch & stop**,
**Delete** and **Activity** (the actions of this dashboard session). The experiment tables have the same filter as the
Monitor, and "Manage in Launcher" on an experiment page opens Launch & stop on that experiment.

New experiment is a five-step form (start from blank or from an existing experiment · model & checkpoint · initial conditions & run type ·
members & start dates · outputs). Choices come from the selected Git branch (a blob-less clone kept in
`~/.local/share/aiuq-dashboard/repo.git`): models and ICS with a job configuration for the platform `AIUQ_LAUNCH_PLATFORM`, the model cards (variables,
levels) and the AIUQ-st standard. Checkpoints are checked on the HPC (`<support folder>/models/<model>/`) and the README
support matrix is shown as badges. The side panel shows the number of simulations, a recap of the five steps (click
one to jump to it), a time estimate from finished experiments with the same model, errors/warnings, the exact
`conf/main.yml` and the commands.

| Action | What runs on the Autosubmit host |
|---|---|
| Create | `autosubmit expid -min -H <AIUQ_LAUNCH_PLATFORM> -d … -repo … -b <branch> -conf conf/bootstrap/`, write `<expid>/conf/main.yml` (never overwrites; includes `OUTPUT_STORE.DIR` only if set in the form, otherwise the branch default for the chosen ICS applies), `autosubmit create <expid> -np`. Nothing is submitted. |
| Launch / Resume | `systemd-run --user --unit aiuq-run-<expid>-<ts> … autosubmit run <expid>` (own unit, so restarting the dashboard does not kill it); output in `~/autosubmit/<expid>_run.out` |
| Stop | `autosubmit stop -y [-c] <expid>` (`-c` also cancels queued/running jobs) |
| Delete | refused while `autosubmit run` is active or jobs are in the SLURM queue; shows a plan (local folder, DB entries, HPC folder with size and NetCDF count), needs the expid typed in; then `autosubmit delete -f <expid>` and removal of the HPC working folder (`HPCROOTDIR`, read before deleting) by `helpers/hpc_delete.py`, which only accepts `<scratch root>/<project>/<user>/<expid>` below `AIUQ_REMOTE_DELETE_ROOTS`, no symlinks, owned by you. A failed HPC step is kept in `pending_remote_deletes.json` and can be retried from the page. |

Every action needs a confirmation in the page and is appended to `~/.local/share/aiuq-dashboard/launcher.log`.
Write requests must carry the `X-AIUQ-Launcher: 1` header and come from `localhost` (blocks CSRF and DNS rebinding).

## Manage the service (on the Autosubmit host)

```bash
systemctl --user status aiuq-dashboard        # state
systemctl --user restart aiuq-dashboard       # restart (e.g. after updating the code)
systemctl --user stop aiuq-dashboard          # stop
systemctl --user disable --now aiuq-dashboard # stop and do not start at boot
journalctl --user -u aiuq-dashboard -f        # follow the logs
curl -s --unix-socket ~/.aiuq-dashboard/dash.sock http://localhost/api/health   # 200 when fresh, 503 when stale
```

The service restarts automatically on failure (`Restart=always`). Linger is enabled, so it keeps running
after logout and starts at boot.

## Install / update

```bash
# on the Autosubmit host, from a copy of tools/dashboard (e.g. ~/aiuq-dashboard)
bash deploy/install.sh
```

The script creates or updates the conda env `aiuq-dash` (python, fastapi, uvicorn, pyyaml; `base` is not
touched), downloads Plotly / topojson / coastlines into `aiuq_dash/static/vendor/`, installs the unit in
`~/.config/systemd/user/`, enables linger and restarts the service.

### Settings

Settings go in `~/.config/aiuq-dashboard.env`, created from [`deploy/aiuq-dashboard.env`](deploy/aiuq-dashboard.env)
at the first install. The ones that describe your site have no default: `install.sh` lists those still empty, and
until they are set the dashboard says which one is missing instead of guessing (the Monitor still shows what it reads
on the Autosubmit host). Restart the service after editing the file.

| Setting | Meaning | Example |
|---|---|---|
| `AIUQ_REMOTE_HOST` | HPC login node where jobs run (the `HOST` of the platform in `conf/platforms.yml`), reached by key-based ssh | `hpc.example.org` |
| `AIUQ_REMOTE_USER` | your user on the HPC (default: your user on the Autosubmit host) | `jdoe` |
| `AIUQ_REMOTE_BASE` | where experiments live on the HPC: `<SCRATCH_DIR>/<PROJECT>/<USER>` of the platform | `/scratch/jdoe/autosubmit_scratch/myproject/jdoe` |
| `AIUQ_LAUNCH_PLATFORM` | Launcher: platform passed to `autosubmit expid -H`, with job files in `conf/jobs/<name>/` | `FELIPE` |
| `AIUQ_SUPPORT_FOLDER` | Launcher: AIUQ support folder on the HPC, written to `PATHS.SUPPORT_FOLDER` | `/scratch/jdoe/AIUQ` |
| `AIUQ_REMOTE_DELETE_ROOTS` | Launcher: HPC folders may only be deleted below these roots (`:`-separated) | `/scratch/jdoe/autosubmit_scratch` |

The Launcher writes `PLATFORM.USER_CODE: <AIUQ_REMOTE_USER>` into every `main.yml`; `conf/platforms.yml` and the default
output store take the user from there.

## Look and feel

The interface shares its visual language with AIUQ Compare (header, sticky tab bar, cards, key-number tiles, segmented
controls, skeleton loaders, dialogs). The brand colours, fonts, radii and logo follow the documentation theme
(`docs/source/_static/aiuq.css`): navy header, teal accent, ink text. The design tokens at the top of `static/style.css` mirror those values (light and a
navy dark variant); change them together with the docs. Job-status colours (`--st-*`) and the data palettes of
charts and maps are semantic and kept independent of the brand.

## Layout

```
aiuq_dash/
  app.py             FastAPI app and JSON API (/api/experiments, /api/exp/<id>, /api/preview, /api/health)
  collector.py       refresh loop, incremental merge, derived views (grid, Gantt, durations, ETA)
  local.py           reads on the Autosubmit host
  remote.py          ssh ControlMaster wrapper
  helpers/
    pkl_reader.py    runs with Autosubmit's Python; prints the job list as JSON
    hpc_probe.py     runs on the HPC host (stdlib only): STAT, logs, MARS, outputs, squeue, nvidia-smi
    hpc_preview.py   runs on the HPC host (netCDF4 + numpy): one 2-D slice for the map
    hpc_delete.py    runs on the HPC host (stdlib only): checks and removes one experiment folder
  static/            index.html, app.js, style.css (vanilla JS + Plotly)
deploy/              install.sh, systemd unit, env template
```

## Limits

- Only experiments whose jobs run on `AIUQ_REMOTE_HOST` are probed remotely; others show local data only.
- GPU figures cover the whole node. Processes of other users cannot be attributed to a job.
- The ETA ignores failed jobs and anything blocked behind them. A MARS tape queue can make PREPARE_ICS far
  slower than history suggests.
