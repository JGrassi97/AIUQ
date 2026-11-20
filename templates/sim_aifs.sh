#!/bin/bash

HPCROOTDIR=%HPCROOTDIR%
EXPID=%DEFAULT.EXPID%
JOBNAME=%JOBNAME%

source %HPCROOTDIR%/lib/functions.sh
SIF_PATH=%PATHS.SIF_FOLDER%/$(aifs_sif_name "%MODEL.CHECKPOINT_NAME%")
AIUQ_ROOT=$(dirname "$(dirname "$SIF_PATH")")

JOBNAME_WITHOUT_EXPID=$(echo ${JOBNAME} | sed 's/^[^_]*_//')

logs_dir=${HPCROOTDIR}/LOG_${EXPID}
configfile=$logs_dir/config_${JOBNAME_WITHOUT_EXPID}
PLATFORM_NAME=%PLATFORM.NAME%

# Load Singularity module only on MareNostrum5
if [ "$PLATFORM_NAME" = "MARENOSTRUM5" ]; then
    ml singularity
fi

EARTHKIT_CACHE=${HPCROOTDIR}/earthkit_cache
export EARTHKIT_REGRID_CACHE=${EARTHKIT_CACHE}
# Chunks of the anemoi mapper: more chunks, less GPU memory (AIFS ENS 2.0 is larger than 1.0).
export ANEMOI_INFERENCE_NUM_CHUNKS=${ANEMOI_INFERENCE_NUM_CHUNKS:-16}

singularity exec --nv \
    --bind $HPCROOTDIR \
    --bind "$AIUQ_ROOT:$AIUQ_ROOT:ro" \
    --env HPCROOTDIR=$HPCROOTDIR \
    --env configfile=$configfile \
    --env ANEMOI_INFERENCE_NUM_CHUNKS=$ANEMOI_INFERENCE_NUM_CHUNKS \
    ${SIF_PATH} \
    python3 $HPCROOTDIR/runscripts/sim_aifs.py -c $configfile