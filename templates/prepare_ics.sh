#!/bin/bash

# Input variables (should be passed via env or set before calling the script)
HPCROOTDIR=%HPCROOTDIR%
CHUNK_START_DATE=%CHUNK_START_DATE%
CHUNK_END_DATE=%CHUNK_END_DATE%

MODEL_NAME=%MODEL.NAME%
MODEL_CHECKPOINT_NAME=%MODEL.CHECKPOINT_NAME%
INI_DATA_PATH=%DIRS.INI_DATA_PATH%

IC_SOURCE=%MODEL.ICS%

JOBNAME=%JOBNAME%
EXPID=%DEFAULT.EXPID%
GSV_CONTAINER=%GSV.CONTAINER%

FDB_HOME=%DIRS.FDB_PATH%

PLATFORM_NAME=%PLATFORM.NAME%
LOCAL_ICS=%MODEL.USE_LOCAL_ICS%

# Derived paths
JOBNAME_WITHOUT_EXPID=$(echo ${JOBNAME} | sed 's/^[^_]*_//')
LOGS_DIR=${HPCROOTDIR}/LOG_${EXPID}
CONFIGFILE=$LOGS_DIR/config_${JOBNAME_WITHOUT_EXPID}
REQUESTS_DIR=${HPCROOTDIR}/requests

DATA_PATH="${INI_DATA_PATH}/${MODEL_NAME}/${MODEL_CHECKPOINT_NAME}/${CHUNK_START_DATE}"


# Load Singularity module only on MareNostrum5
if [ "$PLATFORM_NAME" = "MARENOSTRUM5" ]; then
    ml singularity
fi

LIBDIR=${HPCROOTDIR}/lib

source ${LIBDIR}/functions.sh


if [ "$IC_SOURCE" = "era5" ]; then
    SIF_PATH=%PATHS.SIF_FOLDER%/image_era.sif
    prepare_ics_era5 $HPCROOTDIR $LOGS_DIR $CONFIGFILE $SIF_PATH

elif [ "$IC_SOURCE" = "eerie" ]; then
    SIF_PATH=%PATHS.SIF_FOLDER%/image_eerie.sif
    if [ "$LOCAL_ICS" = "true" ]; then
        prepare_ics_eerie_local $HPCROOTDIR $LOGS_DIR $CONFIGFILE $SIF_PATH
    fi

    if [ "$LOCAL_ICS" = "false" ]; then
        prepare_ics_eerie_mars $HPCROOTDIR $LOGS_DIR $CONFIGFILE $SIF_PATH
    fi
    
else
    echo "Invalid IC source specified. Please use 'fdb' / 'era5' / 'eerie'."
    exit 1
fi