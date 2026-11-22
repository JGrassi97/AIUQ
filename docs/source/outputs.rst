Output format
=============

Every model writes its outputs in the same format: NetCDF-4 files that follow the
`CF conventions <https://cfconventions.org>`_ (CF-1.8), with the names and units of
the AIUQ-st standard. A file holds one variable of one ensemble member:

.. code-block:: text

   <experiment>/outputs/<variable>/<member>/out-<first day>-<last day>-<member>-<variable>.nc

The format is written by one module for all models
(``runscripts/AIUQst_lib/output.py``) and has a version, stored in the global
attribute ``aiuq_output_format`` (currently ``1.0``). The version changes when a
reader would have to change.

Layout of a file
----------------

.. code-block:: text

   dimensions:  valid_time, [level,] latitude, longitude[, bnds]

   <variable>(valid_time, [level,] latitude, longitude)   float32, or integers with a step
   valid_time(valid_time)           hours since 1970-01-01 00:00:00
   valid_time_bnds(valid_time, 2)   totals and means only: the period of each value
   time                             initial time of the forecast
   step(valid_time)                 hours since the initial time (not in daily output)
   number                           ensemble member
   level(level)                     pressure in hPa
   latitude(latitude)               degrees north
   longitude(longitude)             degrees east
   height                           fields at a fixed height only, in metres

.. list-table:: Coordinates
   :header-rows: 1
   :widths: 18 30 52

   * - Variable
     - CF ``standard_name``
     - Rule
   * - ``valid_time``
     - ``time``
     - The time a value is valid at, always increasing. For a total or a mean over a
       period it is the end of the period; in daily output it is the day (00 UTC).
   * - ``time``
     - ``forecast_reference_time``
     - Initial time of the forecast: the time of the initial conditions the model
       starts from.
   * - ``step``
     - ``forecast_period``
     - ``valid_time`` − ``time`` in hours.
   * - ``number``
     - ``realization``
     - Member, as in ``EXPERIMENT.MEMBERS`` (for stochastic checkpoints, the seed).
   * - ``level``
     - ``air_pressure``
     - hPa, from the surface upwards: 1000, 850, 500 …
   * - ``latitude``
     - ``latitude``
     - From south to north, −90 … 90 on the regular grids.
   * - ``longitude``
     - ``longitude``
     - From 0 eastwards, 0 … 360.
   * - ``height``
     - ``height``
     - 2 m for ``t2m`` and ``d2m``, 10 m for ``u10``, ``v10`` and ``i10fg``, 100 m for
       ``u100`` and ``v100``.

With ``OUT_RES: original`` AIFS keeps its reduced Gaussian grid, which has no
latitude and longitude axes: the horizontal dimension is ``values`` and
``latitude(values)`` and ``longitude(values)`` give the position of each point.

Variables
---------

A variable keeps its AIUQ-st short name (``t``, ``z``, ``t2m``, ``tp`` …), the name
used by ECMWF. CF does not prescribe variable names; what identifies a field is its
``standard_name``. Each variable carries:

``standard_name``
   The CF standard name, where the CF table has one for the field (49 of the 62
   variables, for example ``air_temperature``, ``geopotential``,
   ``lwe_thickness_of_precipitation_amount``). Fields without one have ``long_name``
   only, which CF allows.
``long_name`` and ``units``
   From ``conf/AIUQ-st/<version>/variables.yml`` and ``outputs.yml``. The units are
   the ECMWF ones in CF notation: ``K``, ``m2 s-2`` (``z`` is the geopotential:
   divide by 9.80665 for metres), ``m s-1``, ``kg kg-1``, ``Pa``, ``m`` for
   precipitation, ``1`` for fractions.
``cell_methods``
   Only for values that describe a period, together with ``valid_time_bnds``:

   .. list-table::
      :header-rows: 1
      :widths: 34 28 38

      * - Kind of value
        - ``cell_methods``
        - Period (``valid_time_bnds``)
      * - Instantaneous (``t``, ``msl`` …)
        - none
        - none
      * - Total (``tp``, ``sf``, ``ssrd`` …)
        - ``valid_time: sum``
        - The 6 hours (the output step) ending at ``valid_time``
      * - Mean rate or flux (``avg_tprate`` …)
        - ``valid_time: mean``
        - The period the model averages over, ending at ``valid_time`` (Aurora 1.5:
          the last hour of each step)
      * - Daily output, instantaneous fields
        - ``valid_time: mean (interval: 6 hours)``
        - The day, 00 to 24 UTC
      * - Daily output, totals
        - ``valid_time: sum``
        - The day: the totals of the periods that end within it
``GRIB_paramId``
   The ECMWF parameter number of the field.

Missing values are ``NaN`` (for example the wave fields over land).

Global attributes
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Attribute
     - Content
   * - ``Conventions``
     - ``CF-1.8``
   * - ``title``, ``source``, ``history``, ``references``
     - What the file is, the model and checkpoint that produced it, when it was
       written, the repository.
   * - ``institution``
     - ``OUTPUT_FORMAT.INSTITUTION`` of the experiment; left out when empty.
   * - ``aiuq_output_format``
     - Version of this format.
   * - ``aiuq_standard``
     - Version of AIUQ-st, e.g. ``AIUQ-st v100``.
   * - ``aiuq_model``, ``aiuq_checkpoint``
     - ``MODEL.NAME`` and ``MODEL.CHECKPOINT_NAME``.
   * - ``aiuq_initial_conditions``
     - ``MODEL.ICS``; for ``ec-earth3`` also ``aiuq_ics_experiment`` and
       ``aiuq_ics_member``.
   * - ``aiuq_run_type``, ``aiuq_forcing``
     - ``hindcast`` or ``amip``, and the AMIP forcing.
   * - ``aiuq_experiment``, ``aiuq_member``
     - Autosubmit experiment and ensemble member.
   * - ``aiuq_commit``, ``aiuq_branch``
     - The code that ran.
   * - ``aiuq_compression``
     - How the values are stored (below).

Storage and compression
-----------------------

How the values are stored is chosen per experiment with ``EXPERIMENT.OUT_COMPRESSION``
in ``main.yml`` (in the Launcher: "Output compression"). Whatever the choice, a reader
gets floating-point values in the units of the variable: xarray and netCDF4 decode
the files by themselves.

.. list-table:: Presets
   :header-rows: 1
   :widths: 14 50 36

   * - Preset
     - How the values are stored
     - Meant for
   * - ``none``
     - As the model gives them (``float64`` for NeuralGCM and Aurora, ``float32`` for
       AIFS), not compressed, not chunked.
     - Debugging.
   * - ``lossless``
     - ``float32``, the precision the models compute in; zlib with shuffle, chunked.
       Nothing is rounded.
     - Research, verification, hindcasts. **The default.**
   * - ``bits``
     - As ``lossless``, with the values rounded to ``KEEPBITS`` significant bits
       (netCDF ``BitRound``; 12 by default: relative error at most 2.5 · 10⁻⁴,
       0.07 K on a temperature of 280 K).
     - A compact research archive.
   * - ``integer``
     - Integers with a step (CF ``scale_factor`` / ``add_offset``) for the variables
       that have one in the table below; the others as ``bits``. The error is at most
       half a step.
     - The smallest files. Precipitation in whole millimetres sets to zero about a
       third of the grid points with rain in 6 hours.
   * - ``mixed``
     - As ``integer``, with precipitation at 0.1 mm.
     - Operational forecasts. The Launcher proposes it for ``ICS: ecmwf-an``.
   * - ``custom``
     - A rule per variable: a step, a number of bits, or lossless.
     - Special cases.

.. list-table:: Steps of ``mixed`` and ``integer`` (``pack`` in the AIUQ-st standard)
   :header-rows: 1
   :widths: 40 22 22 16

   * - Variables
     - ``mixed``
     - ``integer``
     - Stored in
   * - ``t2m`` ``d2m`` ``t`` ``skt`` ``stl1`` ``stl2`` ``sst``
     - 1 K
     - 1 K
     - whole °C (offset 273.15)
   * - ``u10`` ``v10`` ``u100`` ``v100`` ``u`` ``v`` ``i10fg``
     - 1 m s⁻¹
     - 1 m s⁻¹
     - m s⁻¹
   * - ``msl`` ``sp``
     - 100 Pa
     - 100 Pa
     - hPa
   * - ``z``
     - 9.80665 m² s⁻²
     - 9.80665 m² s⁻²
     - metres
   * - ``tp`` ``cp`` ``sf`` ``ro``
     - 10⁻⁴ m
     - 10⁻³ m
     - 0.1 mm / 1 mm
   * - ``avg_tprate`` ``avg_tsrwe``
     - 0.1 mm in 6 h
     - 1 mm in 6 h
     - the same
   * - ``tcc`` ``lcc`` ``mcc`` ``hcc`` ``ci`` ``fscov``
     - 0.01
     - 0.01
     - percent
   * - ``tcw`` ``tcwv``
     - 1 kg m⁻²
     - 1 kg m⁻²
     - mm
   * - everything else (``q``, ``clwc``, ``ciwc``, ``w``, soil water, snow depth,
       radiation, the other fluxes, waves)
     - ``KEEPBITS`` bits
     - ``KEEPBITS`` bits
     - ``float32``

Integers take 16 bits when every value of the file fits, otherwise 32; a missing
value is −32767 (or −2147483647).

.. code-block:: yaml

   EXPERIMENT:
     OUT_COMPRESSION: mixed        # none | lossless | bits | integer | mixed | custom
     OUT_COMPRESSION_OPTS:         # optional
       KEEPBITS: 12                # 4-23: for 'bits', and for the variables without a step
       CHUNKS: point               # point | map | [valid times, latitudes, longitudes]
       COMPLEVEL: 1                # zlib level, 1-9
       CUSTOM:                     # only for 'custom': one text per variable
         - "t2m: {step: 0.5, offset: 273.15}"
         - "tp: {step: 0.0001}"
         - "q: {keepbits: 10}"
         - "z: {mode: lossless}"
         - "default: {keepbits: 12}"

The rules of ``custom`` are a list of texts and not a mapping, because Autosubmit
hands the jobs the single values of the configuration only: a nested mapping would
not reach them, and the job stops saying so rather than writing with other rules.
A step and an offset are in the units of the variable.

Chunks: ``point`` is 4 valid times × 45 latitudes × 90 longitudes (one level per
chunk), read in milliseconds for a time series at a point or a small map, with files
about 5 % larger; ``map`` is one global map per chunk, fastest for whole maps and
about half a second for a point. On the AIFS model grid the chunks run along
``values``. zlib level 1 compresses as much as higher levels on these fields.

Every file says how it is stored in the global attribute ``aiuq_compression`` (JSON:
preset, bits, chunks and, for its variable, the step or the bits). Before a file gets
its final name it is read back and compared with what was computed: same missing
values, and an error within what its storage allows (none, half a step, or the
rounding of the bits). A file appears under its final name only when it is complete:
it is written as ``<name>.writing`` and renamed at the end. AIFS also leaves
intermediate files ending in ``_temp.nc`` and ``_grid.nc`` while it runs.

Compressing existing outputs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``runscripts/compress_outputs.py`` stores outputs that already exist, in a running
folder or in an output store, with any of the schemes:

.. code-block:: bash

   python runscripts/compress_outputs.py /data/store/a01e_20261121T101500 --scheme mixed --procs 16
   python runscripts/compress_outputs.py out-...-t2m.nc --scheme custom --custom "t2m: {step: 0.5, offset: 273.15}"

Each file is rewritten beside the original, verified as above (coordinates included)
and only then renamed over it; restart checkpoints and intermediate files are never
touched. The command can be repeated: files already stored as asked are skipped,
including those compressed by the scripts used before this command existed (their
``aiuq_compression`` is a text: ``float32, BitRound keepbits=12, ...`` or
``AIUQ mixed integer packing v1 (...)``). A file stored with less precision than
asked is left as it is, because a rewrite cannot bring the precision back
(``--force`` rewrites it anyway). ``--dry`` writes and verifies without replacing.

Checking a file
---------------

``tests/test_output_format.py`` writes files as each model produces them and checks
every rule on this page. With the
`IOOS compliance checker <https://github.com/ioos/compliance-checker>`_ installed
it also checks them against CF-1.8:

.. code-block:: bash

   pip install compliance-checker
   compliance-checker -t cf:1.8 -c strict out-2026-11-21-2026-12-01-1-t.nc

The checker reports no error, with every compression preset. It gives two warnings
that are accepted: the attribute ``_QuantizeBitRoundNumberOfSignificantBits``, which
the netCDF library writes for values rounded to bits and whose name starts with an
underscore, and, on the AIFS
model grid only, the order of the dimensions, because one horizontal dimension has
no latitude and longitude axes.

Reading the files
-----------------

.. code-block:: python

   import xarray as xr

   t = xr.open_dataset("out-2026-11-21-2026-12-01-1-t.nc")["t"]
   t.sel(level=850).sel(latitude=45.07, longitude=7.69, method="nearest")

   # all members of an experiment along a new dimension
   members = xr.open_mfdataset("outputs/t/*/*.nc", combine="nested", concat_dim="number")

Files written before this format (AIUQ 1.0) differ between models: AIFS names the
horizontal dimensions ``lat`` and ``lon`` and has no initial time, NeuralGCM stores
longitude before latitude, and the variables have no ``standard_name``. The Monitor
and the other tools of the ecosystem read both.
