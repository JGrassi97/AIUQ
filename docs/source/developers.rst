Developers guide
================

Architecture principles
-----------------------

AIUQ decouples:

- initial condition retrieval and normalization,
- model-specific inference logic,
- postprocessing and output conversion.

This separation avoids hardcoded pairwise logic for each IC/model combination.

AIUQ-st standard
----------------

The project uses a versioned internal standard called **AIUQ-st**.

- ICs are saved and restored according to AIUQ-st.
- Inference scripts consume AIUQ-st formatted data.
- Standard versions are maintained in ``conf/AIUQ-st/``.

Current reference version
-------------------------

**AIUQ-st v010**

- Storage format: ``.zarr``
- Coordinates: ``latitude``, ``longitude``, ``time``, ``level``
- Grid: regular lat-lon, 0.25 x 0.25 degrees
- Naming conventions: ECMWF Parameter DB aliases

Developer workflow
------------------

When adding a new IC provider or model backend:

1. Map variables to AIUQ-st names.
2. Normalize horizontal/vertical coordinates.
3. Ensure pressure-level compatibility and fallback behavior.
4. Keep model adapters isolated from retrieval code.
5. Update docs and job templates under ``conf/jobs`` and ``templates``.

Recommended files to inspect
----------------------------

- ``runscripts/AIUQst_lib/cards.py``
- ``runscripts/AIUQst_lib/variables.py``
- ``runscripts/AIUQst_lib/pressure_levels.py``
- ``conf/cards/``
- ``conf/jobs/``