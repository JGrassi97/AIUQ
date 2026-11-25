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
* **WeatherNext 2** (Google DeepMind, FGN): hindcast workflows, ensemble only.
  Members are split evenly over the independently trained models of a checkpoint
  (4 seeds for ``WeatherNext2_2025``): member *m* runs seed ((*m* − 1) mod 4) + 1
  with its own noise, drawn anew at every 6-hour step. The operational checkpoint
  was fine-tuned on HRES analyses, so ``ecmwf-an`` is the intended initial
  condition. Integration in progress.

  .. warning::

     ``WeatherNext2_2025`` (0.25°) does not run on FELIPE: one forecast step does
     not fit in the 46 GB of an NVIDIA L40S (upstream recommends an 80 GB GPU).
     ``WeatherNextCyclones_Mini_2024`` (1°) runs there.
* **ACE:** AMIP integration in development; check the matrix before planning a run.

Output variables
----------------

What a run can write depends on the checkpoint. Each model card
(``conf/cards/models/<model>_card.yml``) lists it per checkpoint in its
``checkpoints`` block, as AIUQ-st short names; the Launcher offers only those, and
``SIM`` skips any other requested name with a warning. The lists come from the
checkpoints themselves (AIFS: ``Checkpoint.variable_categories()``; Aurora: the
model classes of the ``microsoft-aurora`` package; NeuralGCM: the fields returned
by ``data_to_xarray``) and agree with the upstream documentation.

.. list-table:: Outputs per checkpoint
   :header-rows: 1
   :widths: 24 76

   * - Checkpoint
     - Outputs (AIUQ-st short names)
   * - AIFS Single 1.1
     - ``z t u v w q`` on 13 levels (50–1000 hPa); ``t2m d2m u10 v10 msl sp skt
       tcw swvl1 swvl2 stl1 stl2``; diagnostics ``u100 v100 tcc lcc mcc hcc`` and
       the 6 h totals ``tp cp sf ro ssrd strd``.
   * - AIFS ENS 1.0
     - As AIFS Single 1.1 without ``swvl1 swvl2``.
   * - AIFS ENS 2.0
     - As AIFS Single 1.1 on 14 levels (10 hPa added, not for ``q``), plus ``sd``,
       ``fscov`` (snow cover fraction) and the wave fields ``swh mwd mwp cdww
       h1012 h1214 h1417 h1721 h2125 h2530``.
   * - Aurora 0.25 pretrained / fine-tuned
     - ``z t u v q`` on 13 levels; ``t2m u10 v10 msl``.
   * - Aurora 1.5 / 1.5 Ensemble
     - ``z t u v q`` on 13 levels; ``t2m d2m u10 v10 u100 v100 msl sp skt stl1
       swvl1 sd ci tcwv tcc lcc mcc hcc``; output-only ``i10fg blh`` and the hourly
       fields as mean rates and fluxes ``avg_tprate avg_tsrwe avg_sdswrf avg_tnlwrf
       avg_sduvrf``.
   * - WeatherNext 2 (``WeatherNext2_2025``, 0.25°)
     - ``z t u v w q`` on 13 levels (50–1000 hPa); ``t2m msl u10 v10 u100 v100
       sst``; output-only ``tp`` (6 h total). Sea surface temperature is NaN over
       land, as in the input.
   * - WeatherNext Cyclones Mini (``WeatherNextCyclones_Mini_2024``, 1°)
     - As WeatherNext 2 without ``u100 v100``.
   * - NeuralGCM (deterministic, ``stochastic_1_4_deg``)
     - ``z t u v q clwc ciwc`` on 37 levels (1–1000 hPa). No surface field: sea
       surface temperature and sea ice are forcings.
   * - NeuralGCM ``stochastic_precip_2_8_deg`` / ``stochastic_evap_2_8_deg``
     - As above, plus ``tp`` and ``avg_ie`` (evaporation rate).

Accumulations and mean rates refer to different periods, written to the
``aiuq_period`` and ``cell_methods`` attributes of each file:

* **AIFS** ``tp cp sf ro`` (m) and ``ssrd strd`` (J m⁻²) are totals over the 6 h
  model step ending at the valid time. With ``OUT_FREQ: daily`` they are summed
  over the periods ending in the day (00–24 UTC).
* **WeatherNext** ``tp`` (m) is the total over the 6 h model step ending at the
  valid time, as for AIFS. It is written as predicted and can be slightly
  negative (a few tenths of a millimetre).
* **NeuralGCM** returns precipitation as a running total since the start of the
  run; AIUQ writes ``tp`` as the total over each output step. The ``evap``
  checkpoint diagnoses precipitation from the water budget, which can be slightly
  negative. ``avg_ie`` is the rate of the last 1 h model step (kg m⁻² s⁻¹,
  negative for evaporation, as in ERA5).
* **Aurora 1.5** predicts 1-hour totals (``scaled_tp_1h``, ``ssrd_1h`` …) for the
  hour ending at the valid time. AIUQ writes them as mean rates (kg m⁻² s⁻¹) and
  fluxes (W m⁻²) over that hour: with 6-hourly steps they describe the last hour
  of each step, not the whole step. Their units are not documented upstream and
  were inferred from the package's normalisation statistics (m and J m⁻²).

Ensembles and uncertainty
-------------------------

Stochastic models can generate different realizations from identical initial
conditions. Deterministic models require perturbed initial conditions to produce
an ensemble with spread. AIUQ's initial-condition perturbation methods are still
under development.

Use :doc:`configuration` to select members, checkpoint and run type.
