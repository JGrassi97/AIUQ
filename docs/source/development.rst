Developer guide
===============

AIUQ separates data retrieval, model inference and orchestration so that model
and initial-condition integrations can evolve independently.

The AIUQ-st convention
----------------------

Initial conditions and model inference use a shared, versioned convention called
**AIUQ-st**. This avoids hardcoding every combination of model-specific variable
names, grids and pressure levels.

The standard definitions in this repository live under ``conf/AIUQ-st/``.
Runtime templates currently select ``STD_VERSION: v100``; the earlier versions
(``v010``, ``v000``) are kept as reference and are not used by the templates.

.. list-table:: AIUQ-st v100
   :header-rows: 1
   :widths: 30 70

   * - Property
     - Convention
   * - Storage
     - Zarr (``.zarr``)
   * - Variable naming
     - ECMWF parameter conventions, defined in ``variables.yml``.
   * - Coordinates
     - Latitude, longitude, time and level.
   * - Horizontal grid
     - Regular latitude–longitude grid at 0.25° × 0.25°.
   * - Latitude range
     - −90° to 90° N.
   * - Longitude range
     - 0° to 360° E.

Pressure-level definitions are stored alongside the variables in
``conf/AIUQ-st/v100/pressure_levels.yml``. Keep integrations independent of a
single standard version wherever possible.

Every variable has a ``type``: ``sfc`` (single level), ``pl`` (pressure levels),
``soil`` (soil layers) and, since v100, ``wave`` (ocean-wave fields, defined over
sea only). v100 adds the eleven wave fields used by AIFS ENS v2 (significant wave
height, mean period and direction, drag coefficient, model bathymetry and the
heights per period band ``h1012`` … ``h2530``). They are flagged ``optional: true``:
an initial-condition source that cannot provide them (EERIE) still produces valid
initial conditions, and a model that needs them fails at ``SAVE_ICS`` with the list
of the missing fields. ERA5 provides them from ARCO ERA5 (the period bands are
derived from the wave partitions), the ECMWF analysis from the ``wave`` stream of
ECMWF Open Data.

Output-only variables (diagnostics, accumulations, mean rates and fluxes such as
``tp``, ``ssrd``, ``i10fg``, ``avg_tprate``) are defined in
``conf/AIUQ-st/v100/outputs.yml``, apart from ``variables.yml`` so that the
initial-condition sources are never asked for them. ``accumulated: true`` marks
totals over a period, which daily output sums instead of averaging. A model card
maps them to model names in its ``outputs`` block; an entry can be a mapping with
``name``, ``scale`` (factor to the AIUQ-st units), ``window`` (the period, e.g.
``6 h ending at valid_time``: it becomes the time bounds and ``cell_methods``, written
to the file attributes) and ``deaccumulate`` (the model gives a running total). The
``checkpoints`` block lists, per checkpoint file, the ``outputs`` it writes and
optionally its ``levels``; the Launcher and ``SIM`` follow it
(``AIUQst_lib.variables.resolve_outputs``). ``tests/test_output_variables.py``
checks the lists against what each checkpoint produces.

A variable of the standard can also carry ``pack``, the step it is stored with when
the experiment asks for integers (``pack: {mixed: 1, integer: 1, offset: 273.15}``,
in the units of the variable; see :doc:`outputs`). It is the one table behind
``AIUQst_lib/compression.py``, the Launcher and the documentation: give a new
variable a ``pack`` only when a step in the unit people read loses nothing that
matters; without it the variable keeps its significant bits.

AIFS checkpoints are not hard-coded: ``runscripts/ics_aifs.py`` reads the input
fields a checkpoint needs from its metadata and takes them from the standard through
the model card, which lists the union of the variables and levels of all AIFS
checkpoints. Per-checkpoint conventions (fields masked over sea, container image)
are in the ``checkpoints`` block of ``conf/cards/models/aifs_card.yml``.

Repository map
--------------

.. code-block:: text

   conf/
   ├── AIUQ-st/         # Versioned variable and pressure-level definitions
   ├── bootstrap/       # Autosubmit configuration entry point
   ├── cards/           # Model and initial-condition metadata
   └── jobs/            # Platform / model / initial-condition jobs
   runscripts/          # Retrieval, inference and postprocessing scripts
   runscripts/AIUQst_lib/ # Shared standard utilities
   templates/           # HPC shell templates and runtime configuration
   lib/                 # Shell functions shared by the templates
   remote_set/          # Checkpoint downloads and container recipes
   tests/               # pytest suite (no HPC or network needed)
   tools/dashboard/     # Monitor and Launcher web dashboard
   docs/                # Sphinx documentation and support matrix

Add an integration
------------------

1. Review the selected AIUQ-st version and its variable and level definitions.
2. For a new initial-condition source, implement retrieval and storage in that
   convention, using the existing retrieval scripts and cards as references.
3. For a new model, define its metadata card and inference adapter against the
   shared convention.
4. Add the platform-specific job configuration and the required shell templates.
5. Update ``docs/support-matrix.yml`` with the checkpoint, supported modes and
   actual integration status. The documentation table updates at build time; the
   README table is regenerated with
   ``python docs/generate_table.py docs/support-matrix.yml README.md``
   (``tests/test_support_matrix.py`` fails while they differ).

Build the documentation
-----------------------

From the repository root, in your Python environment:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   python -m sphinx -W --keep-going -b html docs/source docs/_build/html

Open ``docs/_build/html/index.html`` to preview the result. Styling lives in
``docs/source/_static/aiuq.css``. Read the Docs uses the same source and dependencies
through the root ``.readthedocs.yaml`` configuration.

Run the tests
-------------

.. code-block:: bash

   python -m pip install pytest numpy xarray pyyaml
   python -m pytest tests

Release a version
-----------------

1. Run the ICS × checkpoint tests on the HPC and record the results in
   ``docs/support-matrix.yml``; regenerate the README table.
2. Set the version in ``docs/source/conf.py`` (``release``), ``docs/source/index.rst``
   and ``README.md``, and add an entry to ``CHANGELOG.md``.
3. Merge into ``main`` and tag it ``vX.Y.Z``.
