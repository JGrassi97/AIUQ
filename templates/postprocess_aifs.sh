#!/bin/bash

HPCROOTDIR=%HPCROOTDIR%
EXPID=%DEFAULT.EXPID%
JOBNAME=%JOBNAME%
CARDSDIR="${HPCROOTDIR}/conf/cards/models"
CHUNK_START_DATE=%CHUNK_START_DATE%
CHUNK_END_DATE=%CHUNK_END_DATE%
STD_VERSION="v100"
STD_DICT_CARD="${HPCROOTDIR}/conf/AIUQ-st/${STD_VERSION}/variables.yml"
JOBNAME_WITHOUT_EXPID=$(echo "${JOBNAME}" | sed 's/^[^_]*_//')

CHUNK_START_DATE=%CHUNK_START_DATE%
CHUNK_END_DATE=%CHUNK_END_DATE%

JOBNAME_WITHOUT_EXPID=$(echo "${JOBNAME}" | sed 's/^[^_]*_//')

logs_dir="${HPCROOTDIR}/LOG_${EXPID}"
configfile="${logs_dir}/config_${JOBNAME_WITHOUT_EXPID}"
PLATFORM_NAME=%PLATFORM.NAME%

OUTPUT_PATH="%HPCROOTDIR%/outputs"
GRID_FILE=%PATHS.SUPPORT_FOLDER%/aifs_grid.txt

# Use CHUNK_* if available, otherwise fall back to START_TIME/END_TIME
CHUNK_START_DATE=${CHUNK_START_DATE:-$START_TIME}
CHUNK_END_DATE=${CHUNK_END_DATE:-$END_TIME}

MEMBER=%MEMBER%

# Variable list may come in a broken, unquoted form (e.g. RAW_VARS=['t', 'u', 'v'])
# Build a robust variable string and extract variable names.
RAW_VARS_STR="${VARS:-${OUT_VARS:-%EXPERIMENT.OUT_VARS%}}"

# If RAW_VARS_STR is empty and Autosubmit injected tokens unquoted, reconstruct from script arguments
if [ -z "$RAW_VARS_STR" ] && [ $# -gt 0 ]; then
    RAW_VARS_STR="$*"
fi

# Extract variable names (letters/digits/underscore) from any representation
# Examples handled:
#   "['t', 'u', 'v']"   -> t u v
#   "[t, u, v]"         -> t u v
#   "t,u,v"             -> t u v
#   "[t, u, v]" tokens  -> t u v
vars=$(printf '%s' "$RAW_VARS_STR" | sed -E 's/[^A-Za-z0-9_]+/ /g')

# Load required modules on MareNostrum5
if [ "$PLATFORM_NAME" = "MARENOSTRUM5" ]; then
    module load EB/apps EB/install CDO/2.2.2-gompi-2023b
    ml singularity
fi

# The final files are written in the AIFS image (runscripts/finalize_aifs_output.py)
source %HPCROOTDIR%/lib/functions.sh
SIF_PATH=%PATHS.SIF_FOLDER%/$(aifs_sif_name "%MODEL.CHECKPOINT_NAME%")

# Dates: input may be YYYYMMDD; files are YYYY-MM-DD
start_date=$(echo "$CHUNK_START_DATE" | sed 's/^\([0-9]\{4\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)$/\1-\2-\3/')
end_date=$(echo "$CHUNK_END_DATE"   | sed 's/^\([0-9]\{4\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)$/\1-\2-\3/')

# Output grid from EXPERIMENT.OUT_RES: the regular latitude-longitude grids of the other models
# (latitudes -90..90, longitudes from 0), or the native N320 grid for "original".
OUT_RES="%EXPERIMENT.OUT_RES%"
case "$OUT_RES" in
    0.25)     target_grid=r1440x721 ;;
    0.5)      target_grid=r720x361 ;;
    1)        target_grid=r360x181 ;;
    1.5)      target_grid=r240x121 ;;
    2)        target_grid=r180x91 ;;
    original) target_grid="" ;;
    *)        echo "ERROR: unsupported OUT_RES '$OUT_RES' (original / 0.25 / 0.5 / 1 / 1.5 / 2)"; exit 1 ;;
esac

# For every variable: SIM output on the model grid (_temp.nc) -> cdo file with the grid description, regridded
# unless OUT_RES is original (_grid.nc) -> final file in the AIUQ output convention (.nc). The intermediate
# files end in _temp.nc / _grid.nc, which every reader skips, and are removed once the final file is complete.
finalize_args=()
for var in $vars; do
    base="${OUTPUT_PATH}/${var}/${MEMBER}/out-${start_date}-${end_date}-${MEMBER}-${var}"
    infile="${base}_temp.nc"
    nativef="${base}_native_grid.nc"
    gridf="${base}_grid.nc"
    outf="${base}.nc"

    if [ ! -f "${infile}" ]; then
        # SIM skips requested variables that AIFS does not produce; a rerun finds finished variables done.
        if [ -f "${outf}" ]; then
            echo "${outf} is already final"
        else
            echo "WARNING: no SIM output for '${var}' (${infile}), skipped"
        fi
        continue
    fi

    if [ -n "$target_grid" ]; then
        cdo -P 8 -setgrid,${GRID_FILE} "${infile}" "${nativef}" || exit 1
        cdo -P 8 -f nc4 remapdis,${target_grid} "${nativef}" "${gridf}" || exit 1
        rm -f "${nativef}"
    else
        cdo -P 8 -setgrid,${GRID_FILE} "${infile}" "${gridf}" || exit 1
    fi
    finalize_args+=(--file "${var}" "${gridf}" "${infile}" "${outf}")
done

if [ ${#finalize_args[@]} -gt 0 ]; then
    singularity exec \
        --bind $HPCROOTDIR \
        --env HPCROOTDIR=$HPCROOTDIR \
        ${SIF_PATH} \
        python3 $HPCROOTDIR/runscripts/finalize_aifs_output.py -c $configfile "${finalize_args[@]}" || exit 1
fi
