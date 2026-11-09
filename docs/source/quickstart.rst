Quickstart
==========

Create an AIUQ experiment with Autosubmit and connect it to your HPC resources.

Before you begin
----------------

* Install Autosubmit on your local machine using the
  `Autosubmit installation guide <https://autosubmit.readthedocs.io/en/master/>`_.
* Set up access to your HPC account and the matching platform configuration.
* Prepare model checkpoints, container images and supporting data on the HPC.
* Choose an available combination in :doc:`models`.

Prepare supporting files
------------------------

Use a shared support directory accessible from the compute nodes. Set
``PATHS.SUPPORT_FOLDER`` to the **AIUQ directory itself**, as shown below.

.. code-block:: text

   <supporting_folder>/AIUQ/
   ├── models/          # Model checkpoints, grouped by model name
   ├── sif/             # Container images
   ├── static/          # Static data for first-level fallback
   ├── climatology/     # Climatological data for second-level fallback
   └── amip-forcings/   # Prescribed forcings for AMIP experiments

For example, AIFS checkpoints are resolved as
``<SUPPORT_FOLDER>/models/aifs/<CHECKPOINT_NAME>``. Container paths and the
remaining runtime paths are defined in ``templates/config.yml`` and the job templates.

.. important::

   Supporting-file provisioning is still being documented. The repository does
   not provide a complete automated installation of all checkpoints, images and
   datasets. Make sure these resources are available before submitting jobs.

Create the experiment
---------------------

The repository's example targets MareNostrum5ACC:

.. code-block:: bash

   autosubmit expid \
     --description "AIUQ" \
     --HPC MareNostrum5ACC \
     --minimal_configuration \
     --git_as_conf conf/bootstrap/ \
     --git_repo https://gitlab.earth.bsc.es/ces/AIUQ.git \
     --git_branch main

Keep the experiment ID returned by Autosubmit. Create ``<EXPID>/conf/main.yml``
using the example in :doc:`configuration`, and replace the support directory and
HPC user with your own values. Check your Autosubmit platform settings before
creating and running the experiment with the commands for your installed version.

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

See the `ECMWF Web API guide <https://www.ecmwf.int/en/computing/software/ecmwf-web-api>`_
for credential setup. Never store credentials in a committed experiment file.

Next, review :doc:`configuration` for output controls and AMIP forcing settings.
