#!/bin/bash

HPCROOTDIR=%HPCROOTDIR%
EXPID=%DEFAULT.EXPID%
JOBNAME=%JOBNAME%

SIF_PATH=%PATHS.SIF_FOLDER%/image_anemoi.sif

JOBNAME_WITHOUT_EXPID=$(echo ${JOBNAME} | sed 's/^[^_]*_//')

logs_dir=${HPCROOTDIR}/LOG_${EXPID}
configfile=$logs_dir/config_${JOBNAME_WITHOUT_EXPID}
PLATFORM_NAME=%PLATFORM.NAME%

EARTHKIT_CACHE=${HPCROOTDIR}/earthkit_cache
export EARTHKIT_REGRID_CACHE=EARTHKIT_CACHE

# Load Singularity module only on MareNostrum5
if [ "$PLATFORM_NAME" = "MARENOSTRUM5" ]; then
    ml singularity
fi

singularity exec --nv \
    --bind $HPCROOTDIR \
    --env HPCROOTDIR=$HPCROOTDIR \
    --env configfile=$configfile \
    ${SIF_PATH} \
    python3 $HPCROOTDIR/runscripts/warmup_aifs.py -c $configfile