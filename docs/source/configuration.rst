Experiment configuration
========================

The experiment's ``conf/main.yml`` connects a model, initial conditions and HPC
paths. Autosubmit expands these values into the job templates.

Example: AIFS hindcast
----------------------

This example uses AIFS-single with ERA5, a combination marked available in the
support matrix. Replace the HPC path and user code before use.

.. code-block:: yaml

   MODEL:
     NAME: aifs
     CHECKPOINT_NAME: aifs-single-mse-1.1.ckpt
     ICS: era5

   EXPERIMENT:
     RUN_TYPE: hindcast
     MEMBERS: "1 2"
     DATELIST: 20100101
     CHUNKSIZEUNIT: day
     CHUNKSIZE: 2
     NUMCHUNKS: 1
     CALENDAR: standard
     OUT_VARS: [t]
     OUT_FREQ: daily
     OUT_RES: "1"
     OUT_LEVS: [1000, 850, 700, 500, 250, 100, 50, 10]
     FORCING_VERSION: ESA-CCI-v3.0

   PATHS:
     SUPPORT_FOLDER: /path/to/AIUQ
     SIF_FOLDER: "%PATHS.SUPPORT_FOLDER%/sif"

   PLATFORM:
     NAME: MARENOSTRUM5
     USER_CODE: your_hpc_user

Model and initial conditions
----------------------------

``MODEL.NAME``
   Model identifier used by the job configurations: ``aifs``, ``neuralgcm`` or
   ``ms-aurora``. ACE appears in the support matrix as an integration in development.
``MODEL.CHECKPOINT_NAME``
   Checkpoint filename. It must match the checkpoint installed under
   ``<SUPPORT_FOLDER>/models/<MODEL.NAME>/``.
``MODEL.ICS``
   Initial-condition source: ``era5`` or ``eerie``. Availability varies by checkpoint;
   see :doc:`models`.

Experiment controls
-------------------

``RUN_TYPE``
   ``hindcast`` or ``amip``, depending on model support.
``MEMBERS``
   Space-separated ensemble member identifiers. Deterministic models produce the
   same result from identical initial conditions; initial-condition perturbation
   methods are still under development.
``DATELIST``
   Initial date or date list interpreted by Autosubmit.
``CHUNKSIZEUNIT``, ``CHUNKSIZE``, ``NUMCHUNKS``
   Units, duration and number of chunks used to split the simulation.
``CALENDAR``
   Experiment calendar; the example uses ``standard``.

Output controls
---------------

.. list-table:: Postprocessing settings
   :header-rows: 1
   :widths: 25 35 40

   * - Setting
     - Values
     - Purpose
   * - ``OUT_VARS``
     - List, for example ``[t]``
     - Variable names from the AIUQ-st standard.
   * - ``OUT_FREQ``
     - ``original`` / ``daily``
     - Output time frequency.
   * - ``OUT_RES``
     - ``original`` / ``0.25`` / ``0.5`` / ``1`` / ``1.5`` / ``2``
     - Target horizontal resolution in degrees.
   * - ``OUT_LEVS``
     - List of pressure levels in hPa, or ``original``
     - Pressure-level selection.

Output support depends on the model's postprocessing implementation. Consult its
job configuration and templates when extending these settings to a new model.

AMIP forcings
-------------

For a supported AMIP model, set ``RUN_TYPE: amip`` and choose
``FORCING_VERSION: ESA-CCI-v3.0`` or ``FORCING_VERSION: AIMIP``.
The forcing path is resolved to
``<SUPPORT_FOLDER>/amip-forcings/<FORCING_VERSION>/``. This setting selects prescribed
forcings for AMIP runs; it does not turn a hindcast-only model into an AMIP model.

HPC platform
------------

``PLATFORM.NAME`` selects the matching configuration under ``conf/jobs/``.
The repository includes ``MARENOSTRUM5`` and ``FELIPE`` job directories. Set
``PLATFORM.USER_CODE`` to your HPC user and review ``conf/platforms.yml`` for
site-specific resources and access settings.
