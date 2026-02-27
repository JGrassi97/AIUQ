#!/bin/bash

HPCROOTDIR=%HPCROOTDIR%
EXPID=%DEFAULT.EXPID%
JOBNAME=%JOBNAME%

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
fi

# Dates: input may be YYYYMMDD; files are YYYY-MM-DD
start_date=$(echo "$CHUNK_START_DATE" | sed 's/^\([0-9]\{4\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)$/\1-\2-\3/')
end_date=$(echo "$CHUNK_END_DATE"   | sed 's/^\([0-9]\{4\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)$/\1-\2-\3/')

for var in $vars; do
    infile="${OUTPUT_PATH}/${var}/out-${start_date}-${end_date}-${MEMBER}-${var}_temp.nc"
    gridf="${OUTPUT_PATH}/${var}/out-${start_date}-${end_date}-${MEMBER}-${var}_grid.nc"
    outf="${OUTPUT_PATH}/${var}/out-${start_date}-${end_date}-${MEMBER}-${var}.nc"

    cdo -P 8 -setgrid,${GRID_FILE} "${infile}" "${gridf}"
    rm -f "${infile}"
    cdo -P 8 -f nc4 remapdis,r360x181 "${gridf}" "${outf}"
    rm -f "${gridf}"
done