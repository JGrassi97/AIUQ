#!/bin/bash
#
# This step is in charge of syncing the workflow project with the remote platform

set -xuve

# HEADER

HPCROOTDIR=${1:-%HPCROOTDIR%}
ROOTDIR=${2:-%ROOTDIR%}
HPCUSER=${3:-%HPCUSER%}
HPCHOST=${4:-%HPCHOST%}
PROJDEST=${5:-%PROJECT.PROJECT_DESTINATION%}

# END_HEADER

#####################################################
# Synchronizes file or directory to remote
# Globals:
# Arguments:
#   Remote user
#   Target host
#   Source file or directory
#   Target directory
#####################################################
function rsync_to_remote() {
    echo "rsyncing the dir to the target platform"
    USR=$1
    HOST=$2
    SOURCE=$3
    DIR=$4

    rsync -avp "${SOURCE}" "${USR}"@"${HOST}":"${DIR}"/
}

# MAIN code

cd "${ROOTDIR}"/proj

# If the tar was already sent, we assume that we can update only the changed files in the project
# The workflow will send the tarball again if the workflow starts over from LOCAL_SETUP
# if [ ! -f flag_tarball_sent ]; then
#     rsync_to_remote "${HPCUSER}" "${HPCHOST}" "${PROJDEST}".tar.gz "${HPCROOTDIR}"
#     rm "${PROJDEST}".tar.gz
#     touch flag_tarball_sent
# else
rsync_to_remote "${HPCUSER}" "${HPCHOST}" "${PROJDEST}"/lib "${HPCROOTDIR}"
rsync_to_remote "${HPCUSER}" "${HPCHOST}" "${PROJDEST}"/runscripts "${HPCROOTDIR}"
rsync_to_remote "${HPCUSER}" "${HPCHOST}" "${PROJDEST}"/conf "${HPCROOTDIR}"
#fi
