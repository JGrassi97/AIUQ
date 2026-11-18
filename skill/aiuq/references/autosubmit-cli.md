# Raw Autosubmit and Slurm commands

Use these when the dashboard is down, or for operations the Launcher does not offer (rerun failed jobs,
change an existing experiment). Run them through the helper so ssh stays multiplexed:

```bash
aiuq as <autosubmit args> --yes        # cd ~/autosubmit, PATH with ~/miniconda3/bin, stdin closed
aiuq sh-as '<shell>'                   # anything else on mafalda (one connection for several commands)
aiuq sh-hpc '<shell>'                  # on felipe
```

Every command below that changes state needs the user's explicit OK first. Verified flags: Autosubmit
4.1.17 (`aiuq as-ro <command> -h` prints the help).

## Create and start without the dashboard

```bash
# 1. register (prints the new expid, e.g. a00k)
aiuq as expid -min -H FELIPE -d "short description" \
   -repo https://github.com/JGrassi97/AIUQ.git -b main -conf conf/bootstrap/ --yes
# 2. write ~/autosubmit/<expid>/conf/main.yml (format: experiment-config.md). Never overwrite an existing one.
#    e.g. take the YAML printed by `aiuq preview cfg.json`, save it locally, then (set -C refuses to overwrite):
aiuq sh-as 'set -C; cat > ~/autosubmit/<expid>/conf/main.yml' < main.yml
# 3. build the workflow (no jobs submitted)
aiuq as create <expid> -np --yes
# 4. start detached, exactly as the Launcher does (survives the ssh session; output in <expid>_run.out)
aiuq sh-as 'cd ~/autosubmit && systemd-run --user --collect --unit=aiuq-run-<expid>-$(date +%s) \
   --working-directory=$HOME/autosubmit --setenv=PATH=$HOME/miniconda3/bin:/usr/bin:/bin --setenv=HOME=$HOME \
   /bin/bash -c "exec autosubmit run <expid> >> $HOME/autosubmit/<expid>_run.out 2>&1 < /dev/null"'
```

Is it running? `aiuq sh-as 'pgrep -af "autosubmit run <expid>"'` (one process, not the shell) or
`aiuq status <expid>`. Units: `systemctl --user list-units 'aiuq-run-*'`.

## Stop

```bash
aiuq as stop -y <expid> --yes          # Autosubmit stops after its current loop; Slurm jobs keep going
aiuq as stop -y -c <expid> --yes       # also cancels queued/running jobs (they end as FAILED)
```
Single Slurm job: `aiuq sh-hpc 'scancel <jobid>'` (Autosubmit will then see it as FAILED).

## Rerun failed jobs

Autosubmit does not retry by default. After understanding the failure (and fixing its cause):

```bash
# the experiment must not be running: check `aiuq status`, stop it if needed
aiuq as setstatus <expid> -fs FAILED -t WAITING -s -np --yes                 # every FAILED job
aiuq as setstatus <expid> -fl "<job1> <job2>" -t WAITING -s -np --yes        # specific jobs
aiuq as setstatus <expid> -ft SIM -fs FAILED -t WAITING -s -np --yes         # FAILED jobs of one section
aiuq run <expid> --yes                                                       # resume
```
`-s` saves the change (without it nothing happens). `-t` accepts READY, COMPLETED, WAITING, SUSPENDED,
UNKNOWN, HELD. To redo a COMPLETED job (e.g. after a postprocessing fix) set it to WAITING the same way;
jobs downstream that already completed are not rerun unless you set them too.

`autosubmit recovery <expid> -s` re-reads COMPLETED/STAT files from the platform and fixes the pickle
when Autosubmit lost track of jobs (e.g. after a crash of `autosubmit run`). Without `-s` it only shows
what it would change.

## Change an existing experiment

- **Configuration** (`main.yml`): only while it is not running. Edit the file on mafalda, then
  `aiuq as create <expid> -np --yes`; without `-f` the existing job statuses are kept and new jobs
  (e.g. more members or dates) are added. `-f` rebuilds the job list from scratch (all jobs back to
  WAITING) — only with explicit consent. Prefer a new experiment (`aiuq template` → edit → `create`)
  when the change alters what existing outputs mean (model, checkpoint, ICS, output grid).
- **Code** (runscripts, templates, conf/jobs): the experiment uses its own clone in
  `~/autosubmit/<expid>/proj/git_project` and the copy synced to felipe (`<root>/lib`, `runscripts`,
  `conf`). After the fix is pushed to the experiment's branch:
  1. `aiuq as refresh <expid> --yes` — re-clones the project on mafalda (templates are read from here at submission);
  2. re-sync to felipe, as SYNCHRONIZE does:
     `aiuq sh-as 'cd ~/autosubmit/<expid>/proj/git_project && for d in lib runscripts conf; do rsync -avpL $d felipe.polito.it:<HPC root>/; done'`
     (from mafalda use the full name `felipe.polito.it`, the short alias fails host-key checks there);
  3. set the affected jobs to WAITING and run again.

## Where to look on the hosts

| What | Where |
|---|---|
| Experiment list + descriptions | mafalda `~/autosubmit/autosubmit.db` (sqlite, table `experiment`); `aiuq list` |
| Config | mafalda `~/autosubmit/<expid>/conf/{main,minimal}.yml`; resolved per job on felipe `<root>/LOG_<expid>/config_<SDATE>_<member>_<chunk>_SIM` |
| Job list / statuses | mafalda `~/autosubmit/<expid>/pkl/job_list_<expid>.pkl` (read it via `aiuq status`) |
| Run history | mafalda `~/autosubmit/metadata/data/job_data_<expid>.db` |
| `autosubmit run` output | mafalda `~/autosubmit/<expid>_run.out` (`aiuq runlog`) |
| Logs of local jobs (SYNCHRONIZE) | mafalda `~/autosubmit/<expid>/tmp/LOG_<expid>/` |
| Logs of HPC jobs | felipe `<root>/LOG_<expid>/<job>.cmd.{out,err}.<attempt>` (Slurm) or `<job>.<timestamp>.{out,err}` (direct); `<job>_STAT_<n>` (start, end epoch, result); `<job>_COMPLETED` |
| Job scripts as submitted | felipe `<root>/LOG_<expid>/<job>.cmd` |
| Outputs | felipe `<root>/outputs/<var>/<member>/` |
| Initial conditions | felipe `<root>/ics/<ics>/<date>/` (zarr), temporary `ics_temp/` |
| Dashboard action log | mafalda `~/.local/share/aiuq-dashboard/launcher.log` |

`<root>` = `/work/users/<user>/autosubmit_scratch/test/<user>/<expid>` (`HPCROOTDIR`).

## Slurm on felipe

```bash
aiuq queue                                            # my jobs, GPU partitions (all users), GPUs
aiuq sh-hpc 'squeue -j <jobid> -o "%i %j %T %r %S"'   # reason and expected start
aiuq sh-hpc 'scontrol show job <jobid>'
```
`sacct` is disabled on felipe (the "Slurm accounting storage is disabled" errors in `<expid>_run.out`
are harmless). Memory actually used by a running job: `/sys/fs/cgroup/system.slice/felipe_slurmstepd.scope/job_<id>/memory.stat`
(`anon` line; `memory.peak` includes page cache).
