#!/bin/bash
#
# ARCHIVE: move what is left of a finished experiment from the HPC to the output store.
#
# Runs on the Autosubmit host (PLATFORM: local) once, after every STORE job (one per start date,
# templates/store.sh), which have already moved the outputs of their start dates as soon as these were done.
# This job copies everything still on the HPC: outputs that no STORE job moved (restart checkpoints, or every
# output when the experiment has no STORE jobs), the logs and the code. The store is a folder on the Autosubmit
# host (OUTPUT_STORE.DIR, conf/general.yml or main.yml). Autosubmit can reuse an expid after
# `autosubmit delete`, so every archived experiment gets its own folder, named after the expid and the time of
# the first copy:
#
#   <OUTPUT_STORE.DIR>/<expid>_<YYYYmmddTHHMMSS>/
#       outputs/<var>/<member>/*.nc   the outputs (restart checkpoints included)
#       LOG_<expid>/                  every job log, configuration, COMPLETED flag and STAT file
#       runscripts/ conf/ lib/        the code that ran (as synchronised to the HPC)
#       autosubmit/                   the experiment's Autosubmit configuration (conf/*.yml)
#       archive.json                  source, time, files, bytes, git commit of the project, stored start dates
#
# The folder has the layout of the experiment folder on the HPC, so the same tools read both.
# <Autosubmit experiment>/OUTPUT_STORE.json (on this host) records the folder: the Monitor follows it once it
# says "done", and a second run of this job (e.g. after the experiment is extended) completes the same folder.
# With DELETE_SOURCE the whole experiment folder on the HPC (<HPCROOTDIR>, initial conditions included) is
# removed, and only after a second rsync pass finds nothing left to copy. Shared code: lib/output_store.sh.
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

STORE_JOB="ARCHIVE"
LIB="${ROOTDIR}/proj/${PROJDEST}/lib/output_store.sh"
[[ -f "$LIB" ]] || { echo "ARCHIVE ERROR: ${LIB} not found" >&2; exit 1; }
source "$LIB"

if ! is_true "$ENABLED"; then
    echo "OUTPUT_STORE.ENABLED is '${ENABLED}': the experiment stays in ${HPCROOTDIR}."
    exit 0
fi
store_init

status=$(marker_get status)
DEST=$(marker_get dir)

if ! on_hpc "test -d '${HPCROOTDIR}'"; then
    [[ "$status" == done ]] && { echo "Already archived in ${DEST}; ${HPCROOTDIR} is gone from the HPC."; exit 0; }
    die "${REMOTE}:${HPCROOTDIR} does not exist"
fi
on_hpc "test -d '${HPCROOTDIR}/LOG_${EXPID}'" || die "${HPCROOTDIR} has no LOG_${EXPID}: not an Autosubmit experiment folder"

# ---- nothing of this experiment may still be queued or running on the HPC
no_slurm_jobs "${EXPID}_"

store_dest                                   # the folder of the STORE jobs, or a new one
marker_set copying "$DEST"
echo "Archiving ${REMOTE}:${HPCROOTDIR} to ${DEST}"
mkdir -p "${DEST}/outputs" "${DEST}/LOG_${EXPID}" "${DEST}/autosubmit"

# ---- 1. copy (pairs: source below HPCROOTDIR -> destination below DEST)
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
       "status": "done", "stored_dates": old.get("stored_dates") or {},
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
