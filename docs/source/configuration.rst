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
   Model identifier used by the job configurations: ``aifs``, ``neuralgcm``,
   ``ms-aurora`` or ``weathernext``. ACE appears in the support matrix as an integration in development.
``MODEL.CHECKPOINT_NAME``
   Checkpoint filename. It must match the checkpoint installed under
   ``<SUPPORT_FOLDER>/models/<MODEL.NAME>/``. For ``weathernext`` it is a folder
   (``WeatherNext2_2025``, ``WeatherNextCyclones_Mini_2024``) holding one weights
   file per seed.
``MODEL.ICS``
   Initial-condition source: ``era5``, ``eerie``, ``ecmwf-an`` or ``ec-earth3``.
   Availability varies by checkpoint; see :doc:`models`.
``MODEL.ICS_EXPERIMENT``, ``MODEL.ICS_MEMBER``
   ``ec-earth3`` only, optional: CMIP6 experiment (``historical``, ``ssp119`` …
   ``ssp585``) and member of EC-Earth3. By default dates up to 2014 use
   ``historical`` and later dates the scenario of ``conf/cards/ics/ec-earth3_card.yml``.

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

EC-Earth3 CMIP6 simulations (``ec-earth3``)
-------------------------------------------

``ec-earth3`` initialises the models from the state of the EC-Earth3 CMIP6 simulations
(member ``r4i1p1f1``) at 00 and 06 UTC of the start date, read from ESGF over OPeNDAP
without credentials. It is a climate-model state, not an analysis.

* **Upper air.** EC-Earth3 publishes instantaneous 6-hourly fields on 7 levels
  (925–50 hPa) and daily means on 19 levels (1000–1 hPa). The profiles are the daily
  means plus the 6-hourly departure from them, interpolated in log-pressure between
  the 6-hourly levels, held below 925 hPa and tapered to zero above 50 hPa.
* **Means.** Daily and 6-hourly means (cloud cover, snow, soil moisture, sea ice, SST,
  100 m wind, vertical velocity) are interpolated in time to 00/06 UTC.
* **Derived.** ``sp`` (``msl`` reduced to the static orography), ``d2m``, ``tcwv``,
  ``tcw``, ``skt``, ``sot`` and the cloud layers; static fields and ``clwc``/``ciwc``
  come from the support files. No ocean-wave fields (AIFS ENS v2 is not supported).

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
     - AIUQ-st short names (``variables.yml`` and ``outputs.yml``). Each checkpoint
       writes the names listed in its model card (see :doc:`models`); others
       (for example ``t2m`` with NeuralGCM) are skipped with a warning.
   * - ``OUT_FREQ``
     - ``original`` / ``daily``
     - Output time frequency.
   * - ``OUT_RES``
     - ``original`` / ``0.25`` / ``0.5`` / ``1`` / ``1.5`` / ``2``
     - Target horizontal resolution in degrees: a regular grid with latitudes −90…90
       and longitudes from 0. ``original`` keeps the model grid (N320 for AIFS).
   * - ``OUT_LEVS``
     - List of pressure levels in hPa, or ``original``
     - Pressure-level selection.

Every model writes the same CF-compliant NetCDF files, described in :doc:`outputs`.

Output compression
------------------

``EXPERIMENT.OUT_COMPRESSION`` chooses how the values are stored; the default,
``lossless``, is in ``conf/general.yml``.

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Value
     - Storage
   * - ``none``
     - As the model gives them, not compressed.
   * - ``lossless``
     - ``float32``, compressed, nothing rounded. Default.
   * - ``bits``
     - As ``lossless``, rounded to ``KEEPBITS`` significant bits (default 12).
   * - ``integer``
     - Whole degrees, m/s, hPa, metres, millimetres and percent; the other variables
       as ``bits``. Whole millimetres set light rain to zero.
   * - ``mixed``
     - As ``integer`` with precipitation at 0.1 mm: for operational forecasts.
   * - ``custom``
     - A rule per variable, in ``OUT_COMPRESSION_OPTS.CUSTOM``.

.. code-block:: yaml

   EXPERIMENT:
     OUT_COMPRESSION: mixed
     OUT_COMPRESSION_OPTS:       # optional
       KEEPBITS: 12              # 4-23
       CHUNKS: point             # point | map | [valid times, latitudes, longitudes]
       COMPLEVEL: 1              # zlib 1-9
       CUSTOM:                   # only with OUT_COMPRESSION: custom; a list of texts
         - "t2m: {step: 0.5, offset: 273.15}"
         - "default: {keepbits: 12}"

The steps, the errors, the chunks and the command that compresses existing outputs
are in :doc:`outputs`. A wrong value (an unknown preset, bits outside 4-23, a step
of zero, ``custom`` without rules) stops the ``SIM`` job at its start with a message
that names it. One more setting says who produced the data:

.. code-block:: yaml

   OUTPUT_FORMAT:
     INSTITUTION: ""     # written as the global attribute "institution"; left out when empty

Output store
------------

On FELIPE the ``ARCHIVE`` job runs once, on the Autosubmit host, after every
simulation job (and, for AIFS, every ``POSTPROCESS`` and ``CLEAN`` job). It copies
the experiment folder of the HPC to ``<OUTPUT_STORE.DIR>/<expid>_<YYYYmmddTHHMMSS>/``
(Autosubmit can reuse an expid, so the time of the first archive makes the name
unique): ``outputs/`` (restart checkpoints included, temporary files excluded),
the whole ``LOG_<expid>/``, ``runscripts/``, ``conf/`` and ``lib/``, the experiment's
Autosubmit configuration in ``autosubmit/`` and an ``archive.json`` record (source,
time, files, bytes, git commit). It records the folder in
``~/autosubmit/<expid>/OUTPUT_STORE.json``, which the Monitor follows, and then
removes ``<HPCROOTDIR>`` from the HPC (initial conditions included).

.. list-table:: ``OUTPUT_STORE`` settings (defaults in ``conf/general.yml``)
   :header-rows: 1
   :widths: 25 35 40

   * - Setting
     - Default
     - Purpose
   * - ``DIR``
     - ``/data/users/<PLATFORM.USER_CODE>/AIUQ-outputs``
     - Store folder on the Autosubmit host. Set it per experiment in ``main.yml``
       or in the Launcher.
   * - ``DELETE_SOURCE``
     - ``true``
     - Remove ``<HPCROOTDIR>`` once a second rsync pass finds nothing left to
       copy. ``false`` keeps the HPC folder.
   * - ``ENABLED``
     - ``true``
     - ``false`` leaves everything on the HPC (the job finishes without doing anything).

The job refuses to move anything while Slurm still lists jobs of the experiment.
Running it again completes the same folder.

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

Running AIUQ at a new site
--------------------------

Nothing in the repository is tied to a person. Everything that belongs to you or to
your machine is set in one of three places:

.. list-table:: Where site settings live
   :header-rows: 1
   :widths: 30 70

   * - Place
     - What it holds
   * - The experiment's ``conf/main.yml``
     - ``PLATFORM.USER_CODE`` (your HPC user), ``PLATFORM.NAME``,
       ``PATHS.SUPPORT_FOLDER`` and, where the platform archives its outputs,
       ``OUTPUT_STORE.DIR``. The platforms and the default output store take the user
       from ``PLATFORM.USER_CODE``.
   * - ``conf/platforms.yml``
     - The machines: host, scheduler, queue or partition, scratch folder and project
       of each platform. ``FELIPE`` and MareNostrum 5 are examples that work as they are
       for users of those machines; the MareNostrum 5 ``PROJECT`` is the allocation the
       framework was developed on, replace it with yours.
   * - ``conf/jobs/<PLATFORM.NAME>/``
     - One file per model and per initial-condition source: which platform each job
       runs on, its resources, wall-clock limits and dependencies.

To add a machine:

1. Fork the repository, so that Autosubmit can clone your configuration.
2. Add the platforms of your machine to ``conf/platforms.yml`` (a login platform of
   type ``ps`` for the light jobs and a Slurm platform for the GPU jobs). Keep
   ``USER: "%PLATFORM.USER_CODE%"`` so that no user name is committed.
3. Copy ``conf/jobs/FELIPE`` to ``conf/jobs/<NAME>``, point the jobs to your platforms
   and adjust partitions, memory and wall-clock limits. Remove the ``ARCHIVE`` job if
   you do not want an output store.
4. Prepare the support folder on the HPC (:doc:`quickstart`).
5. Create the experiment with ``--HPC <your platform>`` and ``--git_repo <your fork>``,
   and set ``PLATFORM.NAME: <NAME>`` in ``main.yml``.

The :doc:`dashboard` and the agent skill (``skill/``) read their own site settings
from files in your home folder; neither has a built-in host or user.
