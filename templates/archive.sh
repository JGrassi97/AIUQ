#!/bin/bash
#
# ARCHIVE: move a finished experiment from the HPC to the output store.
#
# Runs on the Autosubmit host (PLATFORM: local) after every simulation job has completed. The store is a folder
# on the Autosubmit host (OUTPUT_STORE.DIR, conf/general.yml or main.yml). Autosubmit can reuse an expid after
# `autosubmit delete`, so every archived experiment gets its own folder, named after the expid and the time of
# the first archive:
#
#   <OUTPUT_STORE.DIR>/<expid>_<YYYYmmddTHHMMSS>/
#       outputs/<var>/<member>/*.nc   the outputs (restart checkpoints included)
#       LOG_<expid>/                  every job log, configuration, COMPLETED flag and STAT file
#       runscripts/ conf/ lib/        the code that ran (as synchronised to the HPC)
#       autosubmit/                   the experiment's Autosubmit configuration (conf/*.yml)
#       archive.json                  source, time, files, bytes, git commit of the project
#
# The folder has the layout of the experiment folder on the HPC, so the same tools read both.
# <Autosubmit experiment>/OUTPUT_STORE.json (on this host) records the folder: the Monitor follows it, and a
# second run of this job (e.g. after the experiment is extended) completes the same folder.
# With DELETE_SOURCE the whole experiment folder on the HPC (<HPCROOTDIR>, initial conditions included) is
# removed, and only after a second rsync pass finds nothing left to copy.
#
# (No literal percent signs in this template: Autosubmit would read them as placeholders.)

set -euo pipefail

EXPID=%DEFAULT.EXPID%
ROOTDIR=%ROOTDIR%
PROJDEST=%PROJECT.PROJECT_DESTINATION%
HPCROOTDIR=%HPCROOTDIR%
HPCUSER=%HPCUSER%
HPCHOST=%HPCHOST%
ENABLED="%OUTPUT_STORE.ENABLED%"
STORE_DIR="%OUTPUT_STORE.DIR%"
DELETE_SOURCE="%OUTPUT_STORE.DELETE_SOURCE%"

is_true() { case "$(echo "$1" | tr '[:upper:]' '[:lower:]')" in true|yes|1) return 0 ;; *) return 1 ;; esac; }
die() { echo "ARCHIVE ERROR: $*" >&2; exit 1; }

if ! is_true "$ENABLED"; then
    echo "OUTPUT_STORE.ENABLED is '${ENABLED}': the experiment stays in ${HPCROOTDIR}."
    exit 0
fi

# ---- sanity checks on the paths (a wrong value here could copy into or delete the wrong place)
while [[ "$STORE_DIR" == */ ]]; do STORE_DIR="${STORE_DIR:0:${#STORE_DIR}-1}"; done
[[ "$STORE_DIR" == /* ]] || die "OUTPUT_STORE.DIR must be an absolute path, got '${STORE_DIR}'"
[[ "$STORE_DIR" != *..* && "$STORE_DIR" =~ ^/[A-Za-z0-9_./-]+$ ]] || die "unsafe OUTPUT_STORE.DIR '${STORE_DIR}'"
[[ $(echo "$STORE_DIR" | tr -cd / | wc -c) -ge 2 ]] || die "OUTPUT_STORE.DIR '${STORE_DIR}' is too close to /"
[[ "$EXPID" =~ ^[a-z0-9]{4}$ ]] || die "bad expid '${EXPID}'"
[[ "$HPCROOTDIR" =~ ^/[A-Za-z0-9_./-]+/${EXPID}$ && "$HPCROOTDIR" != *..* ]] || die "unexpected HPCROOTDIR '${HPCROOTDIR}'"
[[ $(echo "$HPCROOTDIR" | tr -cd / | wc -c) -ge 4 ]] || die "HPCROOTDIR '${HPCROOTDIR}' is too close to /"
[[ -d "$ROOTDIR/conf" ]] || die "Autosubmit experiment folder '${ROOTDIR}' not found"

REMOTE="${HPCUSER}@${HPCHOST}"
MARKER="${ROOTDIR}/OUTPUT_STORE.json"

# One ssh connection for every command below.
CM_DIR="${HOME}/.ssh/cm"
mkdir -p "$CM_DIR" && chmod 700 "$CM_DIR"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30 -o ServerAliveCountMax=6
     -o ControlMaster=auto -o "ControlPath=${CM_DIR}/archive-${EXPID}" -o ControlPersist=120)
RSH="${SSH[*]}"
on_hpc() { "${SSH[@]}" "$REMOTE" "$@"; }

marker_get() {  # marker_get KEY: value from OUTPUT_STORE.json, empty if missing
    [[ -f "$MARKER" ]] || return 0
    python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2]) or "")' "$MARKER" "$1"
}
marker_set() {  # marker_set STATUS DIR
    python3 - "$MARKER" "$1" "$2" "$EXPID" "${REMOTE}:${HPCROOTDIR}" <<'EOF'
import datetime, json, os, sys
path, status, dest, expid, source = sys.argv[1:]
rec = {"expid": expid, "status": status, "dir": dest, "outputs": dest + "/outputs", "source": source,
       "updated_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
with open(path + ".tmp", "w") as fh:
    json.dump(rec, fh, indent=2)
os.replace(path + ".tmp", path)
EOF
}

status=$(marker_get status)
DEST=$(marker_get dir)

if ! on_hpc "test -d '${HPCROOTDIR}'"; then
    [[ "$status" == done ]] && { echo "Already archived in ${DEST}; ${HPCROOTDIR} is gone from the HPC."; exit 0; }
    die "${REMOTE}:${HPCROOTDIR} does not exist"
fi
on_hpc "test -d '${HPCROOTDIR}/LOG_${EXPID}'" || die "${HPCROOTDIR} has no LOG_${EXPID}: not an Autosubmit experiment folder"

# ---- nothing of this experiment may still be queued or running on the HPC
left_jobs=$(on_hpc "squeue -h -u '${HPCUSER}' -O Name:200" | grep -c "^${EXPID}_" || true)
[[ "$left_jobs" -eq 0 ]] || die "${left_jobs} Slurm job(s) of ${EXPID} are still queued or running; nothing moved"

if [[ -z "$DEST" ]]; then
    stamp=$(python3 -c 'import datetime; print(datetime.datetime.now().isoformat(timespec="seconds").replace("-", "").replace(":", ""))')
    DEST="${STORE_DIR}/${EXPID}_${stamp}"
    n=1
    while [[ -e "$DEST" ]]; do DEST="${STORE_DIR}/${EXPID}_${stamp}_${n}"; n=$((n + 1)); done
fi
marker_set copying "$DEST"
echo "Archiving ${REMOTE}:${HPCROOTDIR} to ${DEST}"

mkdir -p "${DEST}/outputs" "${DEST}/LOG_${EXPID}" "${DEST}/autosubmit"
if [[ ! -f "${STORE_DIR}/README" ]]; then
    cat > "${STORE_DIR}/README" <<'EOF'
AIUQ output store, filled by the ARCHIVE job of each experiment (templates/archive.sh).
Autosubmit can reuse an expid, so each experiment has its own folder <expid>_<time of the first archive>:

<expid>_<time>/outputs/<var>/<member>/*.nc   outputs (restart checkpoints in outputs/checkpoints)
<expid>_<time>/LOG_<expid>/                  job logs, configurations, COMPLETED flags, STAT files
<expid>_<time>/runscripts/ conf/ lib/        the code that ran
<expid>_<time>/autosubmit/                   the experiment's Autosubmit configuration
<expid>_<time>/archive.json                  source, time, files, bytes, git commit
EOF
fi

# ---- 1. copy (pairs: source below HPCROOTDIR -> destination below DEST)
OUT_FILTERS=(--exclude=/tmp/ --exclude=/temp/ --exclude='*_temp.nc' --exclude='*_grid.nc')
copy() {  # copy SUBDIR [rsync options...]
    local sub=$1; shift
    on_hpc "test -d '${HPCROOTDIR}/${sub}'" || return 0
    mkdir -p "${DEST}/${sub}"
    rsync -a --partial "$@" -e "$RSH" "${REMOTE}:${HPCROOTDIR}/${sub}/" "${DEST}/${sub}/"
}
check() {  # check SUBDIR [rsync options...]: number of files that still differ
    local sub=$1; shift
    on_hpc "test -d '${HPCROOTDIR}/${sub}'" || { echo 0; return; }
    rsync -a --dry-run --itemize-changes "$@" -e "$RSH" "${REMOTE}:${HPCROOTDIR}/${sub}/" "${DEST}/${sub}/" | grep -c '^>f' || true
}
copy outputs "${OUT_FILTERS[@]}"
for sub in "LOG_${EXPID}" runscripts conf lib; do copy "$sub"; done
cp -p "${ROOTDIR}"/conf/*.yml "${DEST}/autosubmit/"

# ---- 2. verify: a second pass must find nothing left to copy (size and modification time)
for sub in outputs "LOG_${EXPID}" runscripts conf lib; do
    if [[ "$sub" == outputs ]]; then pending=$(check outputs "${OUT_FILTERS[@]}"); else pending=$(check "$sub"); fi
    [[ "$pending" -eq 0 ]] || die "${pending} file(s) of ${sub} differ between the HPC and ${DEST}; nothing deleted"
done
n_src=$(on_hpc "find '${HPCROOTDIR}/outputs' -type f -name '*.nc' ! -path '*/tmp/*' ! -path '*/temp/*' ! -name '*_temp.nc' ! -name '*_grid.nc' 2>/dev/null | wc -l")
n_dest=$(find "${DEST}/outputs" -type f -name '*.nc' | wc -l)
[[ "$n_dest" -ge "$n_src" ]] || die "only ${n_dest} of ${n_src} output files in ${DEST}/outputs; nothing deleted"

# ---- 3. record what was archived (manifest in the store, then the pointer for the Monitor)
GIT_DIR_="${ROOTDIR}/proj/${PROJDEST}"
python3 - "$DEST" "$EXPID" "${REMOTE}:${HPCROOTDIR}" "$GIT_DIR_" "$DELETE_SOURCE" <<'EOF'
import datetime, json, os, socket, subprocess, sys
dest, expid, source, git_dir, delete = sys.argv[1:]

def git(*args):
    try:
        return subprocess.run(["git", "-C", git_dir, *args], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""

files = size = 0
for d, _, fns in os.walk(os.path.join(dest, "outputs")):
    for fn in fns:
        if fn.endswith(".nc") and "/checkpoints" not in d:
            files += 1
            size += os.path.getsize(os.path.join(d, fn))
path = os.path.join(dest, "archive.json")
old = {}
if os.path.exists(path):
    with open(path) as fh:
        old = json.load(fh)
rec = {"expid": expid, "name": os.path.basename(dest), "source": source, "store_host": socket.getfqdn(),
       "dir": dest, "outputs": dest + "/outputs", "files": files, "bytes": size,
       "source_deleted": delete.lower() in ("true", "yes", "1"),
       "git": {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
               "remote": git("config", "--get", "remote.origin.url"), "dirty": bool(git("status", "--porcelain"))},
       "first_archived_at": old.get("first_archived_at") or old.get("archived_at")
                            or datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
       "archived_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
with open(path, "w") as fh:
    json.dump(rec, fh, indent=2)
EOF
marker_set done "$DEST"
echo "Store manifest:"; cat "${DEST}/archive.json"; echo

# ---- 4. remove the experiment folder from the HPC
if is_true "$DELETE_SOURCE"; then
    on_hpc "rm -rf -- '${HPCROOTDIR}'"
    on_hpc "test ! -e '${HPCROOTDIR}'" || die "could not remove ${REMOTE}:${HPCROOTDIR}"
    echo "Removed ${REMOTE}:${HPCROOTDIR}."
else
    echo "OUTPUT_STORE.DELETE_SOURCE is '${DELETE_SOURCE}': ${HPCROOTDIR} is kept on the HPC."
fi
echo "ARCHIVE done: ${n_dest} output files in ${DEST}"
