---
name: aiuq
description: Operate the AIUQ framework (AI weather/climate models NeuralGCM, AIFS and Microsoft Aurora run through Autosubmit) from the laptop over ssh — check what is running, create and launch experiments, follow their progress, debug failed jobs, stop runs, and inspect, analyse or download model outputs. Use whenever the user mentions AIUQ, an AIUQ experiment id (4 characters such as a00e), launching/running a forecast, hindcast, AMIP or ensemble simulation with NeuralGCM/AIFS/Aurora, Autosubmit experiments, the Autosubmit host or the HPC where they run, the AIUQ dashboard/Launcher, or the results/outputs of these simulations.
---

# AIUQ operator

AIUQ runs AI weather models (NeuralGCM, AIFS, MS Aurora) from initial conditions (ERA5, EERIE, latest
ECMWF analysis `ecmwf-an`) as Autosubmit workflows. You work from the user's laptop and reach two
machines over key-based ssh:

| Role | What lives there |
|---|---|
| **Autosubmit host** | `autosubmit` 4.1, experiments in `~/autosubmit/<expid>/` (`conf/main.yml`, `pkl/`, `tmp/LOG_<expid>/`), run output `~/autosubmit/<expid>_run.out`, **AIUQ dashboard** (Monitor + Launcher) on the private socket `~/.aiuq-dashboard/dash.sock`, **output stores** with finished experiments in `<store>/<expid>_<time>/` (`OUTPUT_STORE.DIR`, chosen per experiment), where the platform has an `ARCHIVE` job |
| **HPC** (Slurm, GPUs) | experiment root `<SCRATCH_DIR>/<PROJECT>/<USER>/<expid>/` (`LOG_<expid>/`, `outputs/`, `ics/`), support folder (checkpoints, `.sif` images, static, climatology) |

**Start with `aiuq site`.** Which machines these are, their paths, limits and habits belong to the
installation, not to this skill: `aiuq site` prints the settings in use (`~/.aiuq/config`) and the
user's site notes (`~/.aiuq/site.md`). Read them once per conversation before acting, and prefer
what they say over the generic text here. If `aiuq` answers "not configured", help the user fill in the
config from `config.example` (and the notes from `site.example.md`) instead of guessing hosts or paths.

The workflow per experiment: `SYNCHRONIZE` (Autosubmit host → HPC rsync of `lib/ runscripts/ conf/`) →
`PREPARE_ICS` (download initial conditions) → [AIFS: `WARMUP_REGRID*`, `SAVE_ICS`] → `SIM` (GPU, one per
start date × member × chunk) → [AIFS: `POSTPROCESS` (cdo regrid), `CLEAN`] → `ARCHIVE` (Autosubmit host, on
platforms that define it: moves the experiment folder to `<store>/<expid>_<time>/`, records it in
`~/autosubmit/<expid>/OUTPUT_STORE.json` and removes `<root>` from the HPC; older experiments lack it). `aiuq logs/log/outputs/ncinfo/py/fetch` follow the record. Job names are
`<expid>_<SDATE>_<member>_<chunk>_<SECTION>`, `<expid>_<SDATE>_<SECTION>` or `<expid>_<SECTION>`.

## The tool: `scripts/aiuq`

Always go through `scripts/aiuq` (paths in this file are relative to this skill's folder; call the
script and pass `scripts/analysis/*.py` by absolute path). It multiplexes all
ssh traffic over one ControlMaster connection per host — **some hosts block the laptop for a long time
after a burst of new ssh connections**, so never loop raw `ssh <host> ...` calls; batch work into one
command or use `aiuq sh-as '<cmd1>; <cmd2>'`.

```
aiuq site | health | list | status <expid> | failed <expid> | conf <expid> | runlog <expid> [N]
aiuq logs <expid> | log <expid> <job> [out|err|both] [N] | queue
aiuq outputs <expid> | ncinfo <expid> <rel.nc> | py [--srun] <expid> <script.py> [args] | fetch <expid> <rel|glob> [dest]
aiuq options [branch] | template <expid> | preview <cfg.json> | tasks | task <id> | delete-plan <expid> | api '<GET /api/...>'
aiuq create <cfg.json> --yes | run <expid> --yes | stop <expid> [--cancel] --yes | as <autosubmit args> --yes
aiuq as-ro describe <expid> | sh-as '<cmd>' | sh-hpc '<cmd>'
```

`status`, `list`, `failed` read the dashboard's collector (refreshed every ~45 s; it prints how old the
data is). Everything works without the browser. If the dashboard is down (`aiuq health` fails), use
the raw Autosubmit commands in [references/autosubmit-cli.md](references/autosubmit-cli.md).

## Rules

1. **Read freely, write only with consent.** Anything that creates, submits, stops, deletes, changes a
   job status, edits a config or re-syncs code needs the user's explicit OK for that specific action
   *in this conversation*. Show what will happen first (config, number of SIM jobs, estimate). Only then
   pass `--yes`. Approval for one experiment does not carry over to the next.
2. **Never delete** experiments or HPC folders unless the user names the experiment and asks for it.
   Show `aiuq delete-plan <expid>` and point them to the dashboard Launcher's Delete button (it checks
   paths, blocks while running and logs the action). Never `rm -rf` anything under the experiment
   roots by hand.
3. **One `autosubmit run` per experiment.** `aiuq run` refuses if it is already active. Never start
   `autosubmit run` in the foreground of an ssh session (it dies with the session) — `aiuq run` starts
   it in its own `systemd --user` unit.
4. **Experiments use the code on GitHub**, not the laptop checkout: `autosubmit expid` clones
   the Launcher's repository (`aiuq api /api/launcher/branches` names it; upstream is
   `https://github.com/JGrassi97/AIUQ.git`) at the chosen branch. Local changes reach an experiment
   only after the user commits and pushes them (ask; never push yourself unless told to).
5. **The GPUs are shared and scarce** (the site notes say how many, and how long the queue can be).
   Size experiments sensibly, and warn when a request implies hundreds of GPU jobs. Heavy analysis goes
   to a Slurm CPU job (`aiuq py --srun`), not the login node. Do not install persistent services on
   the HPC.
6. **Move results, not raw data.** Outputs are ~1 GB per variable × member at 0.25°. Reduce on the HPC
   with `aiuq py` and bring back small CSV/NetCDF; `fetch` refuses more than 5 GB without `--yes`.
7. Never print or copy credentials (`~/.ecmwfapirc`, tokens).

## Workflows

### What is going on?
`aiuq list` → `aiuq status <expid>` (progress, ETA, running/queued jobs with Slurm reason, failures,
outputs). `aiuq queue` for the whole GPU picture. Report in plain words: done/total, what is running,
what is waiting and why (e.g. `(Priority)`/`(Resources)` = waiting for a GPU), ETA, failures.

### Launch a new experiment
1. Gather intent: model + checkpoint, initial conditions, start date(s), forecast length, members,
   output variables/levels/resolution/frequency. Ask only for what you cannot infer; propose defaults.
2. `aiuq options` — models, checkpoints present on the HPC, ICS, variables and levels each model
   produces. Check the combination in the support matrix (`docs/support-matrix.yml` in the repo or the
   README table; the Launcher's own badge table is older — trust the README).
   Starting from an existing experiment: `aiuq template <expid> > cfg.json` and edit.
3. Write the launcher JSON (format and field rules: [references/experiment-config.md](references/experiment-config.md))
   in the scratchpad and run `aiuq preview cfg.json` until it prints `VALID`.
4. Show the user: the generated `conf/main.yml`, number of SIM jobs, the estimate (GPU time; queue waits
   excluded), warnings (e.g. deterministic checkpoint with several members ⇒ identical members). Ask
   for confirmation.
5. `aiuq create cfg.json --yes` (expid + main.yml + `autosubmit create`; nothing submitted yet). Note the
   new expid. Then, if the user wants it started, `aiuq run <expid> --yes`.
6. Confirm it started: `aiuq status <expid>` after a minute (SYNCHRONIZE/PREPARE_ICS should progress).

### Follow a run
Check `aiuq status` at a pace that matches the jobs (SIM takes 2–15 min; GPU queue waits can be hours).
Don't poll every few seconds. If the user wants to be told when it finishes, offer `/loop` with a
sensible interval. A run ends when `autosubmit run: not running` and all jobs are COMPLETED (or some
FAILED). `UNKNOWN` for submitted jobs in the pickle is normal while Autosubmit updates.

### Something failed
`aiuq failed <expid>` (traceback excerpt per FAILED job) → `aiuq log <expid> <job> err 200` for more →
diagnose with [references/troubleshooting.md](references/troubleshooting.md) and the code in the repo
(`runscripts/`, `templates/`, `lib/functions.sh`, `conf/jobs/<PLATFORM>/`). Then propose the fix:
- transient (queue, network, node): set the failed jobs back to WAITING and run again;
- config: new experiment from `template` (preferred) or edit `main.yml` + `autosubmit create`;
- code: user pushes the fix, then `autosubmit refresh` + re-sync + rerun.
Exact commands: [references/autosubmit-cli.md](references/autosubmit-cli.md). All of these need consent.

### Look at the results
- `aiuq outputs <expid>` lists `outputs/<var>/<member>/out-<start>-<end>-<member>-<var>.nc` (first line: the folder —
  the output store once `ARCHIVE` has run, else `<root>/outputs`; `ncinfo`, `py` and `fetch` use the same folder).
- `aiuq ncinfo <expid> <var>/<member>/<file>` for dimensions, coordinates, times.
- Analyse on the HPC: `aiuq py <expid> scripts/analysis/ensemble_box.py t --level 850 --box 6 10 44 47 --daily`
  (CSV of ensemble mean/spread/min/max per time; a point with `LON0=LON1, LAT0=LAT1`), or write your
  own script (it runs with the HPC's `python3`, which needs xarray/numpy/netCDF4; `AIUQ_OUTPUTS` points to the
  outputs folder).
- Quick numbers via the dashboard API: `aiuq api '/api/series?exp=<id>&var=t&start=<YYYY-MM-DD>&lat=45.07&lon=7.69&level=850'`.
- Plots: make them locally from the small CSV/NetCDF you computed, or point the user to the dashboard
  Data tab (`ssh -N -L 8765:/home/<user>/.aiuq-dashboard/dash.sock <autosubmit-host>`, then http://localhost:8765/?exp=<id>&tab=data).
Conventions and per-model quirks (dimension names, units, valid times) are in
[references/outputs-and-analysis.md](references/outputs-and-analysis.md) — read it before analysing.

### Stop
`aiuq stop <expid> --yes` stops Autosubmit (jobs already in Slurm keep running);
`aiuq stop <expid> --cancel --yes` also cancels queued/running jobs. Resume later with `aiuq run`.

## References
- [references/experiment-config.md](references/experiment-config.md) — launcher JSON, `main.yml`, models/ICS/checkpoints, resources and timings.
- [references/autosubmit-cli.md](references/autosubmit-cli.md) — raw Autosubmit/Slurm commands: create/run without the dashboard, rerun failed jobs, change config or code of an experiment.
- [references/outputs-and-analysis.md](references/outputs-and-analysis.md) — output layout, conventions, analysis recipes, dashboard analysis API.
- [references/troubleshooting.md](references/troubleshooting.md) — known failures and fixes.
