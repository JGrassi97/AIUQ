#!/bin/bash

retrieve_surface_data() {
    local DATE=$1
    cat <<EOF > mars_sfc
retrieve,
  class=d1,
  dataset=climate-dt,
  activity=${ACTIVITY},
  experiment=${EXPERIMENT},
  generation=1,
  model=${MODEL},
  realization=1,
  resolution=${RESOLUTION},
  expver=${EXPVER},
  type=fc,
  stream=clte,
  levtype=sfc,
  target=aifs_sfc.grib,
  param=${VAR2D},
  date=${DATE},
  time=${TIME}
EOF

    ${BINDIR}/fdb-read --raw mars_sfc aifs_sfc.grib
    rm mars_sfc
}

retrieve_tcw_data() {
    local DATE=$1
    cat <<EOF > mars_tcw
retrieve,
  class=d1,
  dataset=climate-dt,
  activity=${ACTIVITY},
  experiment=${EXPERIMENT},
  generation=1,
  model=${MODEL},
  realization=1,
  resolution=${RESOLUTION},
  expver=${EXPVER},
  type=fc,
  stream=clte,
  levtype=sfc,
  param=137,
  date=${DATE},
  time=${TIME},
  fieldset=tcwv
retrieve,
  param=78,
  fieldset=tclw
retrieve,
  param=79,
  fieldset=tciw
compute,
   formula  = "tcwv+tclw+tciw",
   target   = "tcw"
EOF

    ${BINDIR}/fdb-read --raw mars_tcw aifs_tcw_tmp.grib
    ${BINDIR}/grib_set -s shortName=tcw aifs_tcw_tmp.grib aifs_tcw.grib
    rm mars_tcw aifs_tcw_tmp.grib
}

retrieve_pressure_level_data() {
    local DATE=$1
    cat <<EOF > mars_pl
retrieve,
  class=d1,
  dataset=climate-dt,
  activity=${ACTIVITY},
  experiment=${EXPERIMENT},
  generation=1,
  model=${MODEL},
  realization=1,
  resolution=${RESOLUTION},
  expver=${EXPVER},
  type=fc,
  stream=clte,
  param=${VAR3D},
  levtype=pl,
  levelist=${PLEVELS},
  date=${DATE},
  time=${TIME}
EOF

    ${BINDIR}/fdb-read --raw mars_pl aifs_pl.grib
    rm mars_pl
}

combine_and_remap() {
    local DATE=$1
    ${BINDIR}/grib_copy -B'level:i asc' aifs_sfc.grib aifs_tcw.grib aifs_pl.grib combined_${SOURCESTREAM}.grib
    rm aifs_sfc.grib aifs_tcw.grib aifs_pl.grib

    TARGETGRID_NUMERICAL=$(echo ${TARGETGRID} | sed 's/[^0-9]*//g')
    GRIDTYPE=$(echo ${TARGETGRID} | cut -c1)

    if [ ${GRIDTYPE} == "N" ]; then
        GRIDTYPE="reduced"
    elif [ ${GRIDTYPE} == "O" ]; then
        GRIDTYPE="octahedral"
    fi

    case ${GRIDTYPE} in
      "reduced" | "octahedral")
      ${BINDIR}/mir --${GRIDTYPE}=${TARGETGRID_NUMERICAL} combined_${SOURCESTREAM}.grib aifs_combined_${TARGETGRID}.grib
      ;;
      *)
      echo "Unknown GRIDTYPE: ${GRIDTYPE}"
      exit 1
      ;;
    esac

    mv combined_${SOURCESTREAM}.grib aimodels-climate-dt-${ACTIVITY}-${EXPERIMENT}-${DATE}-${TIME1}-${TIME2}.grib2
}

add_subgrid_info() {
    local DATE=$1
    ${BINDIR}/grib_set -s date=${DATE},time=${TIME1},subCentre=${SUBCENTRE} ${SCRDIR}/aifs_lsm_slor_sdor_z.${TARGETGRID}.grib2 aifs_external.${TARGETGRID}_1.grib
    ${BINDIR}/grib_set -s date=${DATE},time=${TIME2},subCentre=${SUBCENTRE} ${SCRDIR}/aifs_lsm_slor_sdor_z.${TARGETGRID}.grib2 aifs_external.${TARGETGRID}_2.grib
}

create_final_output() {
    local DATE=$1
    ${BINDIR}/grib_copy -B'level:i asc' aifs_combined_${TARGETGRID}.grib aifs_external.${TARGETGRID}_1.grib aifs_external.${TARGETGRID}_2.grib aifs-climate-dt-${ACTIVITY}-${EXPERIMENT}-${DATE}-${TIME1}-${TIME2}.grib2
    rm aifs_external.${TARGETGRID}_1.grib aifs_external.${TARGETGRID}_2.grib aifs_combined_${TARGETGRID}.grib

    ${BINDIR}/grib_set -s edition=1 aifs-climate-dt-${ACTIVITY}-${EXPERIMENT}-${DATE}-${TIME1}-${TIME2}.grib2 aifs-climate-dt-${ACTIVITY}-${EXPERIMENT}-${DATE}-${TIME1}-${TIME2}.grib1
    rm aifs-climate-dt-${ACTIVITY}-${EXPERIMENT}-${DATE}-${TIME1}-${TIME2}.grib2
}

# Function to check if ICs already exist and skip if so
check_existing_ics() {
    local data_path="$1"
    local regenerate="$2"

    if [ "$regenerate" == "true" ]; then
        echo "Regeneration of ICs is requested. Removing old ICs."
        rm -rf "$data_path"
    fi

    if [ -d "$data_path" ] && [ "${regenerate,,}" != "true" ]; then
        echo "ICs already exist for this chunk at $data_path. Skipping."
        create_ics=false
    else
        echo "ICs do not exist or regeneration is requested. Proceeding to create ICs."
        create_ics=true
    fi

    echo $create_ics
}

# Function to prepare ICs from FDB source
prepare_ics_fdb() {
    local hpc_rootdir="$1"
    local fdb_home="$2"
    local requests_dir="$3"
    local start_date="$4"
    local data_path="$5"
    local gsv_container="$6"
    local end_date="$7"

    mkdir -p "$requests_dir"
    mkdir -p "$data_path"

    echo "Generating FDB requests..."
    python3 "$hpc_rootdir/runscripts/build_requests.py" \
        --general "$hpc_rootdir/conf/ics/general_request.yaml" \
        --model "$hpc_rootdir/conf/models/neuralgcm/variables.yaml" \
        --output "$requests_dir" \
        --startdate "$start_date" \
        --enddate "$end_date"

    echo "Retrieving data from FDB..."
    singularity exec \
        --env FDB_HOME="$fdb_home" \
        --env HPCROOTDIR="$hpc_rootdir" \
        --bind "$fdb_home","$data_path","$hpc_rootdir/runscripts","$requests_dir","$PWD","$hpc_rootdir/conf" \
        "$gsv_container" \
        bash -c "python3 $hpc_rootdir/runscripts/retrieve.py \
        --requests $requests_dir \
        --output $data_path \
        --translator $hpc_rootdir/conf/models/neuralgcm/translator.yaml"
}

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


prepare_ics_eerie() {
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