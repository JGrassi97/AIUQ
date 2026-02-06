#!/bin/bash

HPCROOTDIR=%HPCROOTDIR%
EXPID=%DEFAULT.EXPID%
JOBNAME=%JOBNAME%

SIF_PATH=%PATHS.SIF_FOLDER%/image_anemoi.sif

JOBNAME_WITHOUT_EXPID=$(echo ${JOBNAME} | sed 's/^[^_]*_//')

logs_dir=${HPCROOTDIR}/LOG_${EXPID}
configfile=$logs_dir/config_${JOBNAME_WITHOUT_EXPID}
PLATFORM_NAME=%PLATFORM.NAME%

OUTPUT_PATH=%HPCROOTDIR%/outputs
GRID_FILE=%PATHS.SUPPORT_FOLDER%/aifs_grid.txt

# Load Singularity module only on MareNostrum5
if [ "$PLATFORM_NAME" = "MARENOSTRUM5" ]; then
     module load  EB/apps EB/install CDO/2.2.2-gompi-2023b
fi

for f in $(find ${OUTPUT_PATH} -type f -name "*_temp.nc"); do
    gridf=${f%_temp.nc}_grid.nc
    outf=${f%_temp.nc}.nc

    cdo -P 8 -setgrid,${GRID_FILE} ${f} ${gridf}
    rm -f ${f} 
    cdo -P 8 -f nc4 remapnn,r360x181 ${gridf} ${outf}
    rm -f ${gridf}
done