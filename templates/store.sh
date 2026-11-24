#!/bin/bash
#
# STORE: move the outputs of one start date to the output store as soon as its jobs are done.
#
# Runs on the Autosubmit host (PLATFORM: local), once per start date, after every SIM (and POSTPROCESS, CLEAN)
# job of that date. Together with the limit on how far PREPARE_ICS runs ahead (PREPARE_ICS of a date waits for
# the SIM jobs two dates earlier, conf/jobs/<platform>/models/), it keeps the HPC folder small: only the start
# dates being simulated have initial conditions and outputs there.
#
#   1. The jobs that write final outputs list them in <HPCROOTDIR>/store_pending/<SDATE>/<job>.txt
#      (runscripts/AIUQst_lib/output.py, record_output). Exactly those files are copied to
#      <OUTPUT_STORE.DIR>/<expid>_<time>/outputs/, checked with a second rsync pass, and only then removed from
#      the HPC. Files of other start dates, restart checkpoints and work files stay where they are.
#   2. The initial conditions of the start date (ics/<ICS>/<date>, ics_temp/<model>/<date>) are removed.
#   3. <OUTPUT_STORE.DIR>/<expid>_<time>/archive.json records the start dates already stored; OUTPUT_STORE.json
#      says "partial", so the readers keep using the HPC folder until the ARCHIVE job has moved the rest.
#
# With OUTPUT_STORE.ENABLED false the outputs stay on the HPC (step 2 still runs). Running this job again is
# harmless: what is already stored is not copied again. Shared code: lib/output_store.sh.
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
SDATE=%SDATE%
ICS_DATE=%CHUNK_START_DATE%
ICS_NAME=%MODEL.ICS%
MODEL_NAME=%MODEL.NAME%

STORE_JOB="STORE ${SDATE}"
LIB="${ROOTDIR}/proj/${PROJDEST}/lib/output_store.sh"
[[ -f "$LIB" ]] || { echo "STORE ERROR: ${LIB} not found" >&2; exit 1; }
source "$LIB"

[[ "$SDATE" =~ ^[0-9]{8}([0-9]{2})?$ ]] || die "bad start date '${SDATE}'"
[[ "$ICS_DATE" =~ ^[0-9]{8}([0-9]{2})?$ ]] || die "bad initial-conditions date '${ICS_DATE}'"
[[ "$ICS_NAME" =~ ^[A-Za-z0-9_-]+$ && "$MODEL_NAME" =~ ^[A-Za-z0-9_-]+$ ]] || die "bad ICS '${ICS_NAME}' or model '${MODEL_NAME}'"
store_init

if ! on_hpc "test -d '${HPCROOTDIR}'"; then
    echo "${REMOTE}:${HPCROOTDIR} does not exist (already archived?): nothing to store for ${SDATE}."
    exit 0
fi
on_hpc "test -d '${HPCROOTDIR}/LOG_${EXPID}'" || die "${HPCROOTDIR} has no LOG_${EXPID}: not an Autosubmit experiment folder"
no_slurm_jobs "${EXPID}_${SDATE}_"

PENDING="${HPCROOTDIR}/store_pending/${SDATE}"

# ---- 1. outputs of this start date
if is_true "$ENABLED"; then
    LIST=$(mktemp)
    trap 'rm -f "$LIST"' EXIT
    # The recorded files that exist, relative to outputs/, once each (a job run twice lists its files twice).
    on_hpc "python3 - '${HPCROOTDIR}' '${SDATE}'" > "$LIST" <<'EOF'
import glob, os, sys
root, sdate = sys.argv[1:]
outputs = os.path.join(root, "outputs") + os.sep
seen = set()
for record in sorted(glob.glob(os.path.join(root, "store_pending", sdate, "*.txt"))):
    with open(record) as fh:
        for line in fh:
            path = os.path.normpath(line.strip())
            if not path.startswith(outputs) or path in seen or path.endswith((".writing", "_temp.nc", "_grid.nc")):
                continue
            seen.add(path)
            if os.path.isfile(path):
                print(path[len(outputs):])
EOF
    n_files=$(grep -c . "$LIST" || true)
    if [[ "$n_files" -gt 0 ]]; then
        status=$(marker_get status)
        [[ -z "$status" || "$status" == partial ]] || die "OUTPUT_STORE.json says '${status}': the ARCHIVE job is running or done"
        store_dest
        marker_set partial "$DEST"
        echo "Storing ${n_files} output file(s) of ${SDATE}: ${REMOTE}:${HPCROOTDIR}/outputs -> ${DEST}/outputs"

        rsync -a --partial --files-from="$LIST" -e "$RSH" "${REMOTE}:${HPCROOTDIR}/outputs/" "${DEST}/outputs/"
        # A second pass must find nothing left to copy (size and modification time), and every file must be there.
        pending=$(rsync -a --dry-run --itemize-changes --files-from="$LIST" -e "$RSH" \
                      "${REMOTE}:${HPCROOTDIR}/outputs/" "${DEST}/outputs/" | grep -c '^>f' || true)
        [[ "$pending" -eq 0 ]] || die "${pending} file(s) differ between the HPC and ${DEST}/outputs; nothing removed"
        while IFS= read -r rel; do
            [[ -f "${DEST}/outputs/${rel}" ]] || die "${DEST}/outputs/${rel} is missing; nothing removed"
        done < "$LIST"

        python3 - "$DEST" "$EXPID" "${REMOTE}:${HPCROOTDIR}" "$SDATE" "$n_files" <<'EOF'
import datetime, json, os, sys
dest, expid, source, sdate, n_files = sys.argv[1:]
now = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
path = os.path.join(dest, "archive.json")
rec = {}
if os.path.exists(path):
    with open(path) as fh:
        rec = json.load(fh)
rec.update({"expid": expid, "name": os.path.basename(dest), "source": source, "dir": dest,
            "outputs": dest + "/outputs"})
rec.setdefault("status", "partial")
rec.setdefault("first_archived_at", now)
stored = rec.get("stored_dates") or {}
before = stored.get(sdate, {}).get("files", 0)
stored[sdate] = {"files": before + int(n_files), "stored_at": now}
rec["stored_dates"] = dict(sorted(stored.items()))
with open(path + ".tmp", "w") as fh:
    json.dump(rec, fh, indent=2)
os.replace(path + ".tmp", path)
EOF
        # Copied and checked: remove them from the HPC (paths relative to outputs/, checked above).
        on_hpc "cd '${HPCROOTDIR}/outputs' && xargs -r -d '\n' rm -f --" < "$LIST"
        echo "Stored ${n_files} file(s) of ${SDATE} in ${DEST}/outputs and removed them from the HPC."
    else
        echo "No recorded outputs of ${SDATE} on the HPC (none written, or already stored)."
    fi
    on_hpc "rm -rf -- '${PENDING}'"
else
    echo "OUTPUT_STORE.ENABLED is '${ENABLED}': the outputs of ${SDATE} stay in ${HPCROOTDIR}/outputs."
fi

# ---- 2. initial conditions of this start date: every simulation that needs them is done
on_hpc "rm -rf -- '${HPCROOTDIR}/ics/${ICS_NAME}/${ICS_DATE}' '${HPCROOTDIR}/ics_temp/${MODEL_NAME}/${SDATE}'"
echo "Removed the initial conditions of ${SDATE} (ics/${ICS_NAME}/${ICS_DATE}, ics_temp/${MODEL_NAME}/${SDATE})."
echo "STORE ${SDATE} done"
