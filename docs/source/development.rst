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
Runtime templates currently select ``STD_VERSION: v010``.

.. list-table:: AIUQ-st v010
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
``conf/AIUQ-st/v010/pressure_levels.yml``. Keep integrations independent of a
single standard version wherever possible.

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
   actual integration status. The documentation table updates at build time.

Build the documentation
-----------------------

From the repository root, in your Python environment:

.. code-block:: bash

   python -m pip install -r docs/requirements.txt
   python -m sphinx -W --keep-going -b html docs/source docs/_build/html

Open ``docs/_build/html/index.html`` to preview the result. Styling lives in
``docs/source/_static/aiuq.css``. Read the Docs uses the same source and dependencies
through the root ``readthedocs.yml`` configuration.
