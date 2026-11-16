Monitoring dashboard
====================

A read-only web dashboard follows AIUQ experiments run with Autosubmit. Its code is
not yet part of the repository; it is kept locally under ``tools/dashboard``. It runs
on the Autosubmit host as a ``systemd --user`` service and listens on
``127.0.0.1:8765`` only. It never launches, stops or modifies runs: it reads
Autosubmit's job-list pickles, databases (opened read-only), YAML configuration and
logs, and queries ``squeue`` and ``nvidia-smi`` on the HPC host.

Open it
-------

From your laptop, open an ssh tunnel to the Autosubmit host and browse to
http://localhost:8765:

.. code-block:: bash

   ssh -N -L 8765:localhost:8765 <autosubmit-host>

The selected experiment is part of the URL (``?exp=<expid>``), so links can be shared.

What it shows
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 45 25

   * - Panel
     - Source
     - Kind
   * - Experiments, active/stopped badge, setup card
     - ``conf/main.yml``, ``conf/minimal.yml``, ``autosubmit.db``, live ``autosubmit run`` process
     - measured / config
   * - Job counts and status grid (date × section/member/chunk)
     - ``pkl/job_list_<expid>.pkl``, read with Autosubmit's own Python
     - measured
   * - Timeline (Gantt) and durations
     - ``<job>_STAT_<n>`` files on the HPC host, falling back to the job-data database
     - measured
   * - Time to finish
     - jobs left × median duration ÷ observed parallelism, per section
     - estimate
   * - Initial conditions
     - key lines of the ``PREPARE_ICS`` output (requests, MARS cost, errors)
     - parsed logs
   * - SLURM and GPU
     - ``squeue``, ``sinfo``, ``nvidia-smi`` on the HPC host; GPU history kept 7 days
     - snapshot
   * - Running and failed jobs
     - tail of stdout/stderr; stderr of failed jobs from the last ``Traceback``
     - measured
   * - Outputs: map, point time series, area average
     - ``outputs/<var>/<member>/*.nc``, read on the HPC host
     - measured
   * - HPC performance
     - waiting, queued and running time per step, parallelism, failed attempts
     - measured; findings are heuristics

How data is collected
---------------------

A single background thread refreshes the data every 45 s by default
(``AIUQ_REFRESH``). Local files are re-read only when they change, and the HPC host is
reached with **one** ssh call per cycle through a persistent ControlMaster, streaming a
standard-library probe over stdin, so nothing is installed on the HPC host. Idle
experiments are refreshed every 600 s (``AIUQ_SLOW_REFRESH``). HTTP requests never wait
on ssh, except map previews, which are cached.

Install and manage
------------------

On the Autosubmit host, from a copy of ``tools/dashboard``:

.. code-block:: bash

   bash deploy/install.sh                        # create/update env, install and start the service
   systemctl --user status aiuq-dashboard        # state
   systemctl --user restart aiuq-dashboard       # after updating the code
   journalctl --user -u aiuq-dashboard -f        # logs
   curl -s localhost:8765/api/health             # 200 when fresh, 503 when stale

``install.sh`` creates the conda environment ``aiuq-dash`` (Python, FastAPI, Uvicorn,
PyYAML), downloads the front-end libraries, installs the systemd unit, enables linger
and restarts the service. Settings go in ``~/.config/aiuq-dashboard.env``.

Limits
------

* Only experiments whose jobs run on the configured remote host are probed remotely.
* GPU figures cover the whole node; other users' processes cannot be attributed to a job.
* The time-to-finish estimate ignores failed jobs and anything blocked behind them.

See the dashboard's own README for the full list of panels, the API and the code layout.
