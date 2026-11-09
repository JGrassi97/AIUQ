Models & support
================

Choose a checkpoint and initial-condition source before preparing an experiment.
The matrix below is generated directly from ``docs/support-matrix.yml`` so the
published documentation follows the repository's recorded support status.

.. note::

   **Available** indicates a supported combination in this repository.
   **In development** indicates work in progress. **Not implemented** indicates
   that the integration is absent. These statuses describe AIUQ integration,
   rather than the capabilities of the upstream model.

Checkpoint compatibility
------------------------

.. aiuq-support-matrix::

Simulation modes
----------------

* **NeuralGCM:** AMIP and hindcast workflows, with deterministic and stochastic
  checkpoints at different resolutions.
* **AIFS:** hindcast workflows, with single-model and ensemble checkpoints.
* **MS Aurora:** hindcast workflows; availability depends on checkpoint and
  initial conditions.
* **ACE:** AMIP integration in development; check the matrix before planning a run.

Ensembles and uncertainty
-------------------------

Stochastic models can generate different realizations from identical initial
conditions. Deterministic models require perturbed initial conditions to produce
an ensemble with spread. AIUQ's initial-condition perturbation methods are still
under development.

Use :doc:`configuration` to select members, checkpoint and run type.
