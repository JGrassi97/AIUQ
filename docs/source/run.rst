Run the framework
=================

Create an Autosubmit experiment
-------------------------------

Use ``conf/bootstrap`` as the minimal experiment configuration source:

.. code-block:: bash

	 autosubmit expid \
		 --description "AIUQ" \
		 --HPC MareNostrum5ACC \
		 --minimal_configuration \
		 --git_as_conf conf/bootstrap/ \
		 --git_repo https://gitlab.earth.bsc.es/ces/AIUQ.git \
		 --git_branch main

After creating the experiment, edit ``<EXPID>/conf/main.yml``.

Minimal ``main.yml`` example
----------------------------

.. code-block:: yaml

	 MODEL:
		 NAME: aifs                         # aifs / neuralgcm / aurora
		 CHECKPOINT_NAME: aifs-single-mse-1.1.ckpt
		 ICS: eerie                         # eerie / era5
		 USE_LOCAL_ICS: false

	 EXPERIMENT:
		 RUN_TYPE: "hindcast"              # hindcast / amip
		 MEMBERS: "1 2"
		 CHUNKSIZEUNIT: day
		 DATELIST: 20100101
		 CHUNKSIZE: 2
		 NUMCHUNKS: 1
		 CALENDAR: standard

		 OUT_VARS:
			 - t
		 OUT_FREQ: daily                    # original / daily
		 OUT_RES: "1"                      # original / 0.25 / 0.5 / 1 / 1.5 / 2
		 OUT_LEVS: [1000, 850, 700, 500, 250, 100, 50, 10]

		 FORCING_VERSION: ESA-CCI-v3.0      # ESA-CCI-v3.0 / AIMIP

	 PATHS:
		 SUPPORT_FOLDER: /path/to/AIUQ
		 SIF_FOLDER: "%PATHS.SUPPORT_FOLDER%/sif"

	 PLATFORM:
		 NAME: MARENOSTRUM5                 # FELIPE / MARENOSTRUM5
		 USER_CODE: your_hpc_user

Notes on run modes
------------------

- ``hindcast``: forecast from IC snapshots.
- ``amip``: forecast with prescribed AMIP forcings.

For AMIP mode, make sure the selected forcing version exists in your support folder.

Submit and monitor
------------------

After configuration, use standard Autosubmit commands to run and monitor the experiment:

.. code-block:: bash

	 autosubmit run <EXPID>
	 autosubmit monitor <EXPID>