Dashboard
=========

``tools/dashboard`` is a web dashboard for AIUQ experiments run with Autosubmit. It is
made of two apps, switched in the header:

* **Autosubmit Monitor** is read-only. It reads Autosubmit's job-list pickles,
  databases (opened read-only), YAML configuration and logs, and queries ``squeue``
  and ``nvidia-smi`` on the HPC host.
* **AIUQ Launcher** creates experiments, starts and stops their runs and deletes
  them, always through Autosubmit commands (see `Launcher`_). It is the only part
  that writes.

The dashboard runs on the Autosubmit host as a ``systemd --user`` service and listens
on a private Unix socket, ``~/.aiuq-dashboard/dash.sock`` (folder 0700, socket 0600),
so only your user can reach it.

Open it
-------

From your laptop, forward a local port to the socket on the Autosubmit host (the
socket path must be absolute) and browse to http://localhost:8765:

.. code-block:: bash

   ssh -N -L 8765:/home/<user>/.aiuq-dashboard/dash.sock <autosubmit-host>

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
     - ``outputs/<var>/<member>/*.nc``, read on the HPC host; for archived experiments from the output
       store: once ``~/autosubmit/<expid>/OUTPUT_STORE.json`` says the ARCHIVE job is done, the whole experiment
       (logs and outputs) is read from ``<store>/<expid>_<time>/``
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

Launcher
--------

A five-step form (model and checkpoint, initial conditions and run type, members and
start dates, outputs) builds ``conf/main.yml`` from the selected Git branch: models and
initial conditions with a job configuration for the platform, model cards and the
AIUQ-st standard. Checkpoints are checked on the HPC and the support matrix is shown
next to each choice.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Action
     - What runs on the Autosubmit host
   * - Create
     - ``autosubmit expid``, writes ``<expid>/conf/main.yml`` (never overwrites), ``autosubmit create``. Nothing is submitted.
   * - Launch / resume
     - ``autosubmit run <expid>`` in its own ``systemd-run --user`` unit, so restarting the dashboard does not stop it.
   * - Stop
     - ``autosubmit stop`` (optionally cancelling queued and running jobs).
   * - Delete
     - Refused while the experiment runs or has jobs in the queue. After the expid is typed in:
       ``autosubmit delete`` and removal of the experiment's working folder on the HPC, which must be
       ``<scratch root>/<project>/<user>/<expid>``, owned by you, with no symlinks.

Every action needs a confirmation in the page and is appended to
``~/.local/share/aiuq-dashboard/launcher.log``. Write requests must carry the
``X-AIUQ-Launcher: 1`` header and come from ``localhost``.

Install and manage
------------------

On the Autosubmit host, from a copy of ``tools/dashboard`` (for example ``rsync``-ed to
``~/aiuq-dashboard``):

.. code-block:: bash

   bash deploy/install.sh                        # create/update env, install and start the service
   systemctl --user status aiuq-dashboard        # state
   systemctl --user restart aiuq-dashboard       # after updating the code
   journalctl --user -u aiuq-dashboard -f        # logs
   curl -s --unix-socket ~/.aiuq-dashboard/dash.sock http://localhost/api/health   # 200 when fresh

``install.sh`` creates the conda environment ``aiuq-dash`` (Python, FastAPI, Uvicorn,
PyYAML), downloads the front-end libraries, installs the systemd unit, enables linger
and restarts the service. Settings go in ``~/.config/aiuq-dashboard.env``.

Limits
------

* Only experiments whose jobs run on the configured remote host are probed remotely.
* GPU figures cover the whole node; other users' processes cannot be attributed to a job.
* The time-to-finish estimate ignores failed jobs and anything blocked behind them.

See ``tools/dashboard/README.md`` for the full list of panels, the settings, the API and the code layout.
