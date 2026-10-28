#!/bin/bash

HPCROOTDIR=%HPCROOTDIR%
EXPID=%DEFAULT.EXPID%
JOBNAME=%JOBNAME%

SIF_PATH=%PATHS.SIF_FOLDER%/image_neuralgcm.sif
AIUQ_ROOT=$(dirname "$(dirname "$SIF_PATH")")

JOBNAME_WITHOUT_EXPID=$(echo "$JOBNAME" | sed 's/^[^_]*_//')

logs_dir="${HPCROOTDIR}/LOG_${EXPID}"
configfile="${logs_dir}/config_${JOBNAME_WITHOUT_EXPID}"

PLATFORM_NAME=%PLATFORM.NAME%

if [ "$PLATFORM_NAME" = "MARENOSTRUM5" ]; then
    ml singularity
fi

singularity exec --nv \
    --bind "$HPCROOTDIR:$HPCROOTDIR" \
    --bind "$AIUQ_ROOT:$AIUQ_ROOT:ro" \
    --env HPCROOTDIR="$HPCROOTDIR" \
    --env configfile="$configfile" \
    "$SIF_PATH" \
    python3 "$HPCROOTDIR/runscripts/sim_neuralgcm.py" \
        -c "$configfile"