#!/bin/bash

HPCROOTDIR=%HPCROOTDIR%
EXPID=%DEFAULT.EXPID%
JOBNAME=%JOBNAME%

SIF_PATH=%PATHS.SIF_FOLDER%/image_anemoi.sif

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
RAW_VARS=%EXPERIMENT.OUT_VARS%

# Load required modules on MareNostrum5
if [ "$PLATFORM_NAME" = "MARENOSTRUM5" ]; then
    module load EB/apps EB/install CDO/2.2.2-gompi-2023b
fi

# Parse variable list string like "['t', 'u', 'v']" into a whitespace-separated list
vars=$(printf '%s' "$RAW_VARS" | tr -d "[]'\"" | tr ',' ' ')

# Process exactly the expected files (no directory scanning)
for var in $vars; do
    infile="${OUTPUT_PATH}/${var}/out-${CHUNK_START_DATE}-${CHUNK_END_DATE}-${MEMBER}-${var}_temp.nc"
    gridf="${OUTPUT_PATH}/${var}/out-${CHUNK_START_DATE}-${CHUNK_END_DATE}-${MEMBER}-${var}_grid.nc"
    outf="${OUTPUT_PATH}/${var}/out-${CHUNK_START_DATE}-${CHUNK_END_DATE}-${MEMBER}-${var}.nc"

    # Set grid and remap to target resolution
    cdo -P 8 -setgrid,${GRID_FILE} "${infile}" "${gridf}"
    rm -f "${infile}"
    cdo -P 8 -f nc4 remapdis,r360x181 "${gridf}" "${outf}"
    rm -f "${gridf}"
done