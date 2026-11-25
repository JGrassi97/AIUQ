#!/bin/bash
#
# Output store: what the STORE jobs (one per start date, templates/store.sh) and the ARCHIVE job (once,
# templates/archive.sh) share. Both run on the Autosubmit host and source this file from the experiment's
# copy of the project (<ROOTDIR>/proj/<PROJECT_DESTINATION>/lib/output_store.sh).
#
# The store is a folder on the Autosubmit host (OUTPUT_STORE.DIR). Each experiment gets its own folder
# <OUTPUT_STORE.DIR>/<expid>_<YYYYmmddTHHMMSS>/, chosen by the first STORE or ARCHIVE job that copies something
# and recorded in <ROOTDIR>/OUTPUT_STORE.json:
#
#   {"expid", "status", "dir", "source", "updated_at"[, "outputs"]}
#
#   status  partial   STORE jobs have moved the outputs of some start dates; the rest (and the logs) are still
#                     on the HPC, which readers keep using: there is no "outputs" key yet.
#           copying   the ARCHIVE job is copying what is left.
#           done      everything is in the store ("outputs" = <dir>/outputs) and the HPC folder is gone.
#
# The caller sets EXPID ROOTDIR HPCROOTDIR HPCUSER HPCHOST ENABLED STORE_DIR, then calls store_init.
# (No literal percent signs in this file's callers: Autosubmit reads them as placeholders. This file itself is
# not a template, but keep it free of them too.)

is_true() { case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in true|yes|1) return 0 ;; *) return 1 ;; esac; }
die() { echo "${STORE_JOB:-OUTPUT STORE} ERROR: $*" >&2; exit 1; }

# Files of outputs/ that are never moved: work files of running or failed jobs.
OUT_FILTERS=(--exclude=/tmp/ --exclude=/temp/ --exclude='*_temp.nc' --exclude='*_grid.nc' --exclude='*.writing')

store_check_paths() {  # sanity checks (a wrong value here could copy into or delete the wrong place)
    if is_true "${ENABLED:-true}"; then
        while [[ "$STORE_DIR" == */ ]]; do STORE_DIR="${STORE_DIR:0:${#STORE_DIR}-1}"; done
        [[ "$STORE_DIR" == /* ]] || die "OUTPUT_STORE.DIR must be an absolute path, got '${STORE_DIR}'"
        [[ "$STORE_DIR" != *..* && "$STORE_DIR" =~ ^/[A-Za-z0-9_./-]+$ ]] || die "unsafe OUTPUT_STORE.DIR '${STORE_DIR}'"
        [[ $(echo "$STORE_DIR" | tr -cd / | wc -c) -ge 2 ]] || die "OUTPUT_STORE.DIR '${STORE_DIR}' is too close to /"
    fi
    [[ "$EXPID" =~ ^[a-z0-9]{4}$ ]] || die "bad expid '${EXPID}'"
    [[ "$HPCROOTDIR" =~ ^/[A-Za-z0-9_./-]+/${EXPID}$ && "$HPCROOTDIR" != *..* ]] || die "unexpected HPCROOTDIR '${HPCROOTDIR}'"
    [[ $(echo "$HPCROOTDIR" | tr -cd / | wc -c) -ge 4 ]] || die "HPCROOTDIR '${HPCROOTDIR}' is too close to /"
    [[ -d "$ROOTDIR/conf" ]] || die "Autosubmit experiment folder '${ROOTDIR}' not found"
}

store_init() {
    store_check_paths
    REMOTE="${HPCUSER}@${HPCHOST}"
    MARKER="${ROOTDIR}/OUTPUT_STORE.json"

    # One STORE or ARCHIVE job of this experiment at a time: they share the store folder and the marker.
    exec 9>"${ROOTDIR}/.output_store.lock"
    flock 9

    # One ssh connection for every command of the job.
    local cm_dir="${HOME}/.ssh/cm"
    mkdir -p "$cm_dir" && chmod 700 "$cm_dir"
    SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=6
         -o ControlMaster=auto -o "ControlPath=${cm_dir}/archive-${EXPID}" -o ControlPersist=120)
    RSH="${SSH[*]}"
}

on_hpc() { "${SSH[@]}" "$REMOTE" "$@"; }

marker_get() {  # marker_get KEY: value from OUTPUT_STORE.json, empty if missing
    [[ -f "$MARKER" ]] || return 0
    python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2]) or "")' "$MARKER" "$1"
}

marker_set() {  # marker_set STATUS DIR
    python3 - "$MARKER" "$1" "$2" "$EXPID" "${REMOTE}:${HPCROOTDIR}" <<'EOF'
import datetime, json, os, sys
path, status, dest, expid, source = sys.argv[1:]
rec = {"expid": expid, "status": status, "dir": dest, "source": source,
       "updated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
if status == "done":
    rec["outputs"] = dest + "/outputs"      # readers switch to the store only when everything is there
with open(path + ".tmp", "w") as fh:
    json.dump(rec, fh, indent=2)
os.replace(path + ".tmp", path)
EOF
}

store_dest() {  # sets DEST: the experiment's store folder, recorded in the marker, or a new one
    DEST=$(marker_get dir)
    if [[ -z "$DEST" ]]; then
        local stamp n=1
        stamp=$(python3 -c 'import datetime; print(datetime.datetime.now().isoformat(timespec="seconds").replace("-", "").replace(":", ""))')
        DEST="${STORE_DIR}/${EXPID}_${stamp}"
        while [[ -e "$DEST" ]]; do DEST="${STORE_DIR}/${EXPID}_${stamp}_${n}"; n=$((n + 1)); done
    fi
    mkdir -p "${DEST}/outputs"
    if [[ ! -f "${STORE_DIR}/README" ]]; then
        cat > "${STORE_DIR}/README" <<'EOF'
AIUQ output store, filled by the STORE jobs (outputs of each start date, as soon as its simulations are done)
and the ARCHIVE job (everything left, once the experiment is finished). Autosubmit can reuse an expid, so each
experiment has its own folder <expid>_<time of the first copy>:

<expid>_<time>/outputs/<var>/<member>/*.nc   outputs (restart checkpoints in outputs/checkpoints)
<expid>_<time>/LOG_<expid>/                  job logs, configurations, COMPLETED flags, STAT files
<expid>_<time>/runscripts/ conf/ lib/        the code that ran
<expid>_<time>/autosubmit/                   the experiment's Autosubmit configuration
<expid>_<time>/archive.json                  source, time, files, bytes, git commit, stored start dates

A folder without LOG_<expid> is an experiment still running: only some start dates have arrived.
EOF
    fi
}

no_slurm_jobs() {  # no_slurm_jobs PREFIX: fail if a Slurm job whose name starts with PREFIX is queued or running
    local left
    left=$(on_hpc "squeue -h -u '${HPCUSER}' -O Name:200" | grep -c "^$1" || true)
    [[ "$left" -eq 0 ]] || die "${left} Slurm job(s) named $1* are still queued or running; nothing moved"
}
