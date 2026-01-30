#!/bin/bash

INI_DATA_PATH="%DIRS.INI_DATA_PATH%/%CHUNK_START_DATE%"
ICS_TEMP_DIR="%HPCROOTDIR%/ics_temp/%MODEL.NAME%/%SDATE%/ics/"

# Delete the folder if it exists
if [ -d "${INI_DATA_PATH}" ]; then
    rm -rf "${INI_DATA_PATH}"
fi

if [ -d "${ICS_TEMP_DIR}" ]; then
    rm -rf "${ICS_TEMP_DIR}"
fi