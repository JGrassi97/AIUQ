#!/bin/bash

HPCROOTDIR=%HPCROOTDIR%
EXPID=%DEFAULT.EXPID%
JOBNAME=%JOBNAME%

SIF_PATH=%PATHS.SIF_FOLDER%/image_neuralgcm.sif

JOBNAME_WITHOUT_EXPID=$(echo ${JOBNAME} | sed 's/^[^_]*_//')

logs_dir=${HPCROOTDIR}/LOG_${EXPID}
configfile=$logs_dir/config_${JOBNAME_WITHOUT_EXPID}

BIND_PATHS="$HPCROOTDIR"

# Load Singularity module only on MareNostrum5
if [ "$PLATFORM_NAME" = "MARENOSTRUM5" ]; then
    ml singularity
fi

singularity exec --nv \
    --bind ${BIND_PATHS} \
    --env HPCROOTDIR=$HPCROOTDIR \
    --env configfile=$configfile \
    ${SIF_PATH} \
    python3 $HPCROOTDIR/runscripts/sim_neuralgcm.py -c $configfile