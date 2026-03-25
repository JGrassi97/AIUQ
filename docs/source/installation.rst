Installation
============

Prerequisites
-------------

To use AIUQ you need:

- Autosubmit installed on your local machine.
- Access to an HPC platform (currently configured: MARENOSTRUM5, FELIPE).
- A support directory on the HPC with model checkpoints and runtime assets.

For Autosubmit installation, follow the official guide:
`Autosubmit documentation <https://autosubmit.readthedocs.io/en/master/>`_.

Local setup
-----------

Clone the repository:

.. code-block:: bash

	git clone https://gitlab.earth.bsc.es/ces/AIUQ.git
	cd AIUQ

The configuration templates and platform job definitions are under:

- ``conf/bootstrap/``
- ``conf/jobs/``
- ``templates/``

HPC support folder
------------------

AIUQ requires a support folder in the target HPC filesystem.
The structure should look like:

.. code-block:: text

	<support_folder>
	└── AIUQ
		 ├── models
		 ├── sif
		 ├── static
		 ├── climatology
		 └── amip-forcings

Where:

- ``models``: AI model checkpoints.
- ``sif``: Singularity images used by jobs.
- ``static``: static fields for fallback level 1.
- ``climatology``: climatology data for fallback level 2.
- ``amip-forcings``: AMIP forcing datasets.

Credentials
-----------

Some initial-condition providers require external credentials:

- ERA5 via Google Cloud Store: no credentials needed.
- EERIE via MARS: ECMWF Web API credentials required.

ECMWF API docs: `ECMWF Web API <https://www.ecmwf.int/en/computing/software/ecmwf-web-api>`_.

Building this documentation locally
-----------------------------------

Install doc dependencies:

.. code-block:: bash

	pip install -r docs/requirements.txt

Build HTML:

.. code-block:: bash

	sphinx-build -b html docs/source docs/build