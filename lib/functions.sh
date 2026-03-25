#!/bin/bash

# Function to prepare ICs from ERA5 source
prepare_ics_era5() {
    local hpc_rootdir="$1"
    local logs_dir="$2"
    local configfile="$3"
    local sif_path="$4"

    echo "Downloading ERA5 data..."
    singularity exec \
        --nv \
        --bind "$hpc_rootdir","$logs_dir" \
        --env HPCROOTDIR="$hpc_rootdir" \
        --env configfile="$configfile" \
        "$sif_path" \
        python3 "$hpc_rootdir/runscripts/download_era5.py" --config "$configfile"
}


# Function to prepare ICs from EERIE source
# a) From mars
prepare_ics_eerie_mars() {
    local hpc_rootdir="$1"
    local logs_dir="$2"
    local configfile="$3"
    local sif_path="$4"

    echo "Downloading EERIE data..."
    singularity exec \
        --nv \
        --bind "$hpc_rootdir","$logs_dir" \
        --env HPCROOTDIR="$hpc_rootdir" \
        --env configfile="$configfile" \
        "$sif_path" \
        python3 "$hpc_rootdir/runscripts/download_eerie.py" --config "$configfile"
}

# b) From local archive
prepare_ics_eerie_local() {
    local hpc_rootdir="$1"
    local logs_dir="$2"
    local configfile="$3"
    local sif_path="$4"

    echo "Downloading EERIE data..."
    singularity exec \
        --nv \
        --bind "$hpc_rootdir","$logs_dir" \
        --env HPCROOTDIR="$hpc_rootdir" \
        --env configfile="$configfile" \
        "$sif_path" \
        python3 "$hpc_rootdir/runscripts/retrieve_eerie_local.py" --config "$configfile"
}