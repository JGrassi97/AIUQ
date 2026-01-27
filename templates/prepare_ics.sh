#!/bin/bash

# Input variables (should be passed via env or set before calling the script)
HPCROOTDIR=%HPCROOTDIR%
CHUNK_START_DATE=%CHUNK_START_DATE%
CHUNK_END_DATE=%CHUNK_END_DATE%

MODEL_NAME=%MODEL.NAME%
MODEL_CHECKPOINT_NAME=%MODEL.CHECKPOINT_NAME%
INI_DATA_PATH=%DIRS.INI_DATA_PATH%

REGENERATE_ICS=%MODEL.REGENERATE_ICS%
IC_SOURCE=%MODEL.ICS%

JOBNAME=%JOBNAME%
EXPID=%DEFAULT.EXPID%
GSV_CONTAINER=%GSV.CONTAINER%

FDB_HOME=%DIRS.FDB_PATH%

# Derived paths
JOBNAME_WITHOUT_EXPID=$(echo ${JOBNAME} | sed 's/^[^_]*_//')
LOGS_DIR=${HPCROOTDIR}/LOG_${EXPID}
CONFIGFILE=$LOGS_DIR/config_${JOBNAME_WITHOUT_EXPID}
REQUESTS_DIR=${HPCROOTDIR}/requests

DATA_PATH="${INI_DATA_PATH}/${MODEL_NAME}/${MODEL_CHECKPOINT_NAME}/${CHUNK_START_DATE}"

# Load Singularity module
# ml singularity

LIBDIR=${HPCROOTDIR}/lib

source ${LIBDIR}/functions.sh

if [ "$REGENERATE_ICS" = "false" ]; then
    echo "ICs already exist for this chunk at $DATA_PATH. Skipping."
else
    if [ "$IC_SOURCE" = "fdb" ]; then
        prepare_ics_fdb $HPCROOTDIR $FDB_HOME $REQUESTS_DIR $CHUNK_START_DATE $DATA_PATH $GSV_CONTAINER $CHUNK_END_DATE
    elif [ "$IC_SOURCE" = "era5" ]; then
        SIF_PATH=%PATHS.SIF_FOLDER%/image_era.sif
        prepare_ics_era5 $HPCROOTDIR $LOGS_DIR $CONFIGFILE $SIF_PATH
    elif [ "$IC_SOURCE" = "eerie" ]; then
        SIF_PATH=%PATHS.SIF_FOLDER%/image_eerie.sif
        prepare_ics_eerie $HPCROOTDIR $LOGS_DIR $CONFIGFILE $SIF_PATH
    else
        echo "Invalid IC source specified. Please use 'fdb' / 'era5' / 'eerie'."
        exit 1
    fi
fi