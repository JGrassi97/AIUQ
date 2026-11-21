# Troubleshooting

Start with `aiuq failed <expid>`, then `aiuq log <expid> <job> err 300` / `out 300`, `aiuq runlog <expid> 200`
and `aiuq queue`. Read the code that failed (`runscripts/`, `templates/`, `lib/functions.sh`,
`conf/jobs/<PLATFORM>/`) in the local repo, but remember the experiment runs the code of its branch
on GitHub at the time of `create`/`refresh`.

## Jobs and Slurm

| Symptom | Cause | What to do |
|---|---|---|
| SIM jobs `PENDING (Priority)` / `(Resources)` for hours | Few GPUs, shared with other users (`aiuq queue` shows who is queued) | Normal. Report the expected start (`status` shows Slurm's estimate). Don't cancel/resubmit to "unstick" them. |
| Jobs `UNKNOWN` in `aiuq status` while they are in `squeue` | Where `sacct` is disabled, Autosubmit cannot always read the state | Harmless while the job is queued/running. If it stays UNKNOWN after leaving the queue: `autosubmit recovery <expid>` (preview), then `-s` with consent. |
| `[ERROR] Command sacct ... accounting storage is disabled` in the run log | same | Harmless noise (`aiuq runlog` filters it). |
| `oom-kill`, `Killed`, exit 137, `OUT_OF_MEMORY` | The HPC enforces job RAM strictly (e.g. cgroup v2, no swap, a small default per CPU when no MEMORY is set) | Check `MEMORY` in `conf/jobs/<PLATFORM>/models/<model>/<model>.yml` and the experiment's commit: 1.0 streams outputs to disk; 1.0.1 fixed fine `OUT_RES` (0.25°) peaks for NeuralGCM/Aurora. Fix = newer code (refresh) or more memory (code change, needs push). |
| `DUE TO TIME LIMIT` | SIM wall clock 1 h | Shorter forecasts per job or higher `WALLCLOCK` in the job file (code change). |
| `autosubmit run: not running` but jobs unfinished and no FAILED | the run process died or was stopped | `aiuq runlog <expid> 200` for the reason; `autosubmit recovery <expid>` if statuses look wrong; then `aiuq run <expid> --yes` (consent). |
| `Experiment is locked` / run refuses to start | another `autosubmit run <expid>` is alive | `aiuq sh-as 'pgrep -af "autosubmit run <expid>"'`. Never remove lock files while a process exists. |
| Same job fails again after rerun | not transient | Don't loop reruns; diagnose, report to the user with the traceback. |

## Initial conditions (PREPARE_ICS)

| Symptom | Cause | What to do |
|---|---|---|
| `ecmwf-an`: HTTP 503 `SlowDown` | AWS Open Data mirror throttling | The card tries `data.ecmwf.int` then the Google mirror; if it still fails, rerun later. |
| `ecmwf-an`: no cycle found / 404 for a past date | data.ecmwf.int keeps ~4 days; the Google archive has 0.25° files only from Feb 2024 | Use a date ≥ 2024-02, or ERA5 for older dates. |
| `ecmwf-an` started from an older analysis than expected | the newest cycle needs both T-6h and T published | Expected behaviour; the chosen cycle is in the `analysis_time` attribute of the ICs. |
| EERIE PREPARE_ICS waits a long time | MARS queue (dashboard shows MARS request states) | Wait; tape retrievals can take hours. |
| `has no access to services/mars` | the user's ECMWF key has no operational MARS rights | Don't propose MARS `class=od` retrievals; use `ecmwf-an` (Open Data) or ERA5. |
| ERA5 date before 1940 | not available | Pick a later date. |

## Models and outputs

| Symptom | Cause | What to do |
|---|---|---|
| `WARNING: ... skipped` for a requested variable | the model does not produce it (e.g. `t2m` with NeuralGCM) | Expected; `aiuq options` lists what each model produces. |
| AIFS POSTPROCESS fails with cdo errors / `cdo: command not found` | cdo comes from the system on the HPC and can disappear with an OS update | `aiuq sh-hpc 'which cdo; cdo -V'`; tell the user if it is missing again. |
| AIFS output always at 1° / times 12 h early | experiment created before the 1.0.1 fixes | Needs SIM + POSTPROCESS again with the fixed code (consent). |
| All members identical | deterministic checkpoint | Expected; IC perturbations are not implemented. |
| AIFS members differ slightly between reruns of the same setup | AIFS on GPU is not bit-reproducible | Expected. |
| `_temp.nc` files left in `outputs/` (AIFS) | POSTPROCESS has not run or failed | Check the POSTPROCESS job. |

## Access

| Symptom | Cause | What to do |
|---|---|---|
| `kex_exchange_identification: Connection reset` from a host | Its sshd blocks bursts of new connections from the laptop | Stop all ssh attempts and wait (the site notes say how long; the user may have to clear the block). Always use `aiuq` (one multiplexed connection). |
| `Host key verification failed` for the HPC from the Autosubmit host | The laptop's ssh alias is unknown there | In commands run on the Autosubmit host use the HPC name of `conf/platforms.yml` (`HOST`). |
| `aiuq` says dashboard unreachable / `health` 503 stale | service down or collector stuck | `aiuq sh-as 'systemctl --user status aiuq-dashboard; journalctl --user -u aiuq-dashboard -n 50'`. Restart (`systemctl --user restart aiuq-dashboard`) after telling the user; meanwhile use the raw commands in autosubmit-cli.md. |
| `ControlPath too long` | macOS socket path limit (104 bytes) | Keep `AIUQ_SSH_CM_DIR` short (default `~/.ssh/cm`). |
