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
   Initial-condition source: ``era5``, ``eerie`` or ``ecmwf-an``. Availability varies
   by checkpoint; see :doc:`models`.

Latest ECMWF analysis (``ecmwf-an``)
------------------------------------

``ecmwf-an`` initialises the models from the newest ECMWF IFS analysis published in
`ECMWF Open Data <https://www.ecmwf.int/en/forecasts/datasets/open-data>`_
(the IFS HRES state at step 0, 0.25°, CC BY 4.0). No ECMWF account is needed.

* **Cycle selection.** ``PREPARE_ICS`` picks the newest cycle *T* for which both
  *T*-6 h and *T* are published, searching from 18 UTC of the start date back to
  ``max_lookback_hours`` (card ``conf/cards/ics/ecmwf-an_card.yml``) before 00 UTC.
  Set ``DATELIST`` to today's date to forecast from the latest analysis; a past date
  within the Open Data archive (0.25° files from February 2024, Google Cloud mirror)
  gives that day's 18 UTC analysis. The chosen cycle is stored in the ``analysis_time``
  attribute of the initial conditions, and the model outputs take their valid times
  from it, which may precede the ``DATELIST`` date by up to the lookback.
* **Fields not in Open Data.** ``slt``, ``lsm``, ``sdor``, ``slor`` and ``orog`` come
  from the static file and ``clwc``/``ciwc`` from the ERA5 climatology, as for the other
  sources. ``sst`` is the skin temperature over open sea (271.46 K under ice) and
  ``ci`` is 1 where the IFS ocean model has sea ice (binary, so it slightly
  overestimates the ice edge).
* **Levels.** Open Data publishes 10–1000 hPa; the remaining standard levels are
  interpolated in log-pressure, and 1–7 hPa are extrapolated (temperature following
  the US Standard Atmosphere profile anchored at 10 hPa, hydrostatic geopotential,
  constant winds and humidity).

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
     - Variable names from the AIUQ-st standard. Variables the model does not
       produce (for example ``t2m`` with NeuralGCM) are skipped with a warning.
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
