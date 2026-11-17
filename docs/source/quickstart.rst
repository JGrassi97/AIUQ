Quickstart
==========

Create an AIUQ experiment with Autosubmit and connect it to your HPC resources.

Before you begin
----------------

* Install Autosubmit on your local machine using the
  `Autosubmit installation guide <https://autosubmit.readthedocs.io/en/master/>`_.
* Set up access to your HPC account and the matching platform configuration. The
  inference jobs need Slurm, Singularity/Apptainer and GPUs; MareNostrum 5 and FELIPE
  are configured in ``conf/platforms.yml``.
* Prepare model checkpoints, container images and supporting data on the HPC.
* Choose an available combination in :doc:`models`.

Prepare supporting files
------------------------

Use a shared support directory accessible from the compute nodes. Set
``PATHS.SUPPORT_FOLDER`` to the **AIUQ directory itself**, as shown below.

.. code-block:: text

   <SUPPORT_FOLDER>/
   ├── models/
   │   ├── neuralgcm/       # *.pkl checkpoints
   │   ├── aifs/            # *.ckpt checkpoints
   │   └── ms-aurora/       # *.ckpt checkpoints
   ├── sif/                 # image_era.sif, image_eerie.sif (also ecmwf-an),
   │                        # image_neuralgcm.sif, image_anemoi.sif (AIFS), image_aurora.sif
   ├── static/              # AIUQ-static.nc, aurora-0.25-static.pickle,
   │                        # aurora-0.25-v1.5-static.pickle
   ├── climatology/         # AIUQ-climatology.nc
   ├── amip-forcings/       # <FORCING_VERSION>/, AMIP experiments only
   └── aifs_grid.txt        # AIFS N320 grid, used by the AIFS POSTPROCESS job

Checkpoints are resolved as ``<SUPPORT_FOLDER>/models/<MODEL.NAME>/<CHECKPOINT_NAME>``,
so file names must match :doc:`models`. ``remote_set/get_models.txt`` downloads every
checkpoint and the Aurora static files, and ``remote_set/image_aurora.def`` is the
recipe of the Aurora image. The remaining runtime paths are defined in
``templates/config.yml`` and the job templates.

.. important::

   The repository does not yet install the container images, the static and
   climatology files or the AMIP forcings automatically. Make sure they are in place
   before submitting jobs. The AIFS POSTPROCESS job also needs
   `CDO <https://code.mpimet.mpg.de/projects/cdo>`_ on the HPC.

Create the experiment
---------------------

The example targets MareNostrum 5; use ``--HPC FELIPE`` on FELIPE:

.. code-block:: bash

   autosubmit expid \
     --description "AIUQ" \
     --HPC MareNostrum5ACC \
     --minimal_configuration \
     --git_as_conf conf/bootstrap/ \
     --git_repo https://github.com/JGrassi97/AIUQ.git \
     --git_branch main

Keep the experiment ID returned by Autosubmit. Create ``<EXPID>/conf/main.yml``
using the example in :doc:`configuration`, and replace the support directory and
HPC user with your own values, then run ``autosubmit create <EXPID>`` and
``autosubmit run <EXPID>``. The :doc:`dashboard` can do the same from a web page.

Data access
-----------

.. list-table:: Initial-condition access
   :header-rows: 1
   :widths: 20 35 45

   * - Source
     - Retrieval
     - Credentials
   * - ERA5
     - Google Cloud Storage
     - No credentials required for the documented retrieval workflow.
   * - EERIE
     - MARS / ECMWF Web API
     - ECMWF Web API credentials required.
   * - ECMWF analysis (``ecmwf-an``)
     - ECMWF Open Data (data.ecmwf.int, Google Cloud mirror)
     - No credentials required. HTTPS access from the platform running ``PREPARE_ICS``.

See the `ECMWF Web API guide <https://www.ecmwf.int/en/computing/software/ecmwf-web-api>`_
for credential setup. Never store credentials in a committed experiment file.

Next, review :doc:`configuration` for output controls and AMIP forcing settings.
To follow running experiments, see :doc:`dashboard`.
