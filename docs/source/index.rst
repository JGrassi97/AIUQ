AIUQ documentation
==================

.. raw:: html

   <section class="hero" aria-label="Introducing AIUQ">
     <div class="eyebrow">AI WEATHER &amp; CLIMATE · v1.0.0</div>
     <h2>Quantifying uncertainty<br><span>in data-driven atmospheric prediction.</span></h2>
     <p>AIUQ initialises machine-learning weather and climate models from reanalyses, operational analyses and climate simulations harmonised to a common standard, and samples forecast uncertainty through stochastic models and ensembles of initial conditions, in hindcast and AMIP experiments run with Autosubmit.</p>
     <div class="hero-actions">
       <a class="action-primary" href="quickstart.html">Start an experiment <span aria-hidden="true">↗</span></a>
       <a class="action-secondary" href="models.html">Explore model support <span aria-hidden="true">→</span></a>
     </div>
     <div class="hero-tags"><span>Hindcasts</span><span>AMIP simulations</span><span>Ensemble workflows</span></div>
   </section>

AIUQ brings AI forecasting models into a shared workflow for uncertainty
quantification. Use deterministic or stochastic models, choose initial conditions
and configure experiments through YAML on your HPC platform.

.. raw:: html

   <div class="section-label">YOUR NEXT STEP</div>
   <div class="guide-grid">
     <a class="guide-card" href="quickstart.html"><span class="card-number">01 / GET STARTED</span><h3>From setup to experiment</h3><p>Prepare the supporting files and create your first Autosubmit experiment.</p><span class="card-link">Quickstart →</span></a>
     <a class="guide-card" href="configuration.html"><span class="card-number">02 / CONFIGURE</span><h3>Make it your own</h3><p>Choose a model, initial conditions, ensemble members and output settings.</p><span class="card-link">Configuration →</span></a>
     <a class="guide-card" href="models.html"><span class="card-number">03 / EXPLORE</span><h3>Find your model</h3><p>Compare checkpoints and check availability for ERA5, EERIE and ECMWF-analysis initial conditions.</p><span class="card-link">Models &amp; support →</span></a>
     <a class="guide-card" href="development.html"><span class="card-number">04 / EXTEND</span><h3>Build on a shared standard</h3><p>Understand AIUQ-st, model cards and the structure of the framework.</p><span class="card-link">Developer guide →</span></a>
   </div>

Forecasts in practice
---------------------

Three model families, one starting point. AIFS ENS 1.0, Aurora 1.5 ENS and
stochastic NeuralGCM (1.4°) were started from the ECMWF analysis of
**19 November 2026** (NeuralGCM: 18 UTC on 18 November) and run for 10 days with
10 members each, with outputs at 0.25° every 6 hours (experiments a00v, a00w and a00x).

.. container:: forecast-grid single

   .. figure:: ../images/forecast_maps_light.png
      :alt: Temperature and geopotential height at 850 hPa over Europe forecast by AIFS ENS, Aurora 1.5 ENS and NeuralGCM, valid at 00 UTC on 24 November 2026.

      **Same day, three models** · temperature (shading) and geopotential height
      (contours) at 850 hPa, valid at 00 UTC on 24 November 2026, member 1

   .. figure:: ../images/ensemble_plume_light.png
      :alt: Ensemble plumes of 850 hPa temperature over Turin for AIFS ENS, Aurora 1.5 ENS and NeuralGCM.

      **Ensemble plume** · 850 hPa temperature over Turin: every member, ensemble
      mean and member range

   .. figure:: ../images/ensemble_spread_light.png
      :alt: Global maps of the ensemble spread of 850 hPa temperature for AIFS ENS, Aurora 1.5 ENS and NeuralGCM, valid at 00 UTC on 27 November 2026.

      **Ensemble spread** · standard deviation of 850 hPa temperature across the
      10 members, valid at 00 UTC on 27 November 2026

From initial conditions to outputs
----------------------------------

.. raw:: html

   <ol class="workflow">
     <li><span>01</span><strong>Prepare</strong><p>Retrieve initial conditions and apply fallback data where needed.</p></li>
     <li><span>02</span><strong>Simulate</strong><p>Run model inference through HPC job templates and containers.</p></li>
     <li><span>03</span><strong>Postprocess</strong><p>Configure output variables, pressure levels, frequency and resolution.</p></li>
   </ol>

.. note::

   **AMIP experiments are available.** The project includes support for AIMIP
   and ESA-CCI-v3.0 prescribed forcings. Check :doc:`models` before choosing a
   model and initial-condition combination; some integrations are still in development.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Documentation

   quickstart
   configuration
   models
   outputs
   dashboard
   development
