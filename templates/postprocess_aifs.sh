#!/bin/bash
set -euo pipefail

HPCROOTDIR=%HPCROOTDIR%
EXPID=%DEFAULT.EXPID%
JOBNAME=%JOBNAME%
CHUNK_START_DATE=%CHUNK_START_DATE%
CHUNK_END_DATE=%CHUNK_END_DATE%
MEMBER=%MEMBER%
PLATFORM_NAME=%PLATFORM.NAME%

JOBNAME_WITHOUT_EXPID=$(echo "${JOBNAME}" | sed 's/^[^_]*_//')
configfile="${HPCROOTDIR}/LOG_${EXPID}/config_${JOBNAME_WITHOUT_EXPID}"
OUTPUT_PATH="${HPCROOTDIR}/outputs"
GRID_FILE=%PATHS.SUPPORT_FOLDER%/aifs_grid.txt
SIF_PATH=%PATHS.SIF_FOLDER%/image_anemoi.sif
AIUQ_ROOT=$(dirname "$(dirname "$SIF_PATH")")

if [ "$PLATFORM_NAME" = "MARENOSTRUM5" ]; then
    module load EB/apps EB/install CDO/2.2.2-gompi-2023b
    ml singularity
fi

start_date=$(echo "$CHUNK_START_DATE" | sed 's/^\([0-9]\{4\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)$/\1-\2-\3/')
end_date=$(echo "$CHUNK_END_DATE" | sed 's/^\([0-9]\{4\}\)\([0-9]\{2\}\)\([0-9]\{2\}\)$/\1-\2-\3/')

# Intermediate files already use canonical names, including when OUT_VARS
# contains long names. Limit discovery to this job's member and date interval.
shopt -s nullglob
inputs=("${OUTPUT_PATH}"/*/"${MEMBER}"/out-"${start_date}"-"${end_date}"-"${MEMBER}"-*_temp.nc)
if [ ${#inputs[@]} -eq 0 ]; then
    echo "No AIFS intermediate outputs found for this job" >&2
    exit 1
fi

# Retain the existing 1-degree AIFS default for OUT_RES=original.
case "%EXPERIMENT.OUT_RES%" in
    0.25) target_grid=r1440x721 ;;
    0.5) target_grid=r720x361 ;;
    1|original) target_grid=r360x181 ;;
    1.5) target_grid=r240x121 ;;
    2) target_grid=r180x91 ;;
    *) echo "Unsupported AIFS output resolution" >&2; exit 1 ;;
esac

for infile in "${inputs[@]}"; do
    base="${infile%_temp.nc}"
    gridf="${base}_grid.nc"
    remapped="${base}_remapped.nc"
    outf="${base}.nc"
    cdo -P 8 -setgrid,"${GRID_FILE}" "${infile}" "${gridf}"
    cdo -P 8 -f nc4 remapdis,"${target_grid}" "${gridf}" "${remapped}"
    singularity exec \
        --bind "$HPCROOTDIR:$HPCROOTDIR" \
        --bind "$AIUQ_ROOT:$AIUQ_ROOT:ro" \
        "$SIF_PATH" \
        python3 "$HPCROOTDIR/runscripts/finalize_aifs.py" \
        --config "$configfile" --input "$remapped" \
        --reference "$infile" --output "$outf"
    # Keep all intermediate files if remapping or standardization fails.
    rm -f "$infile" "$gridf" "$remapped"
done
