AIUQ documentation
==================

.. raw:: html

   <section class="hero" aria-label="Introducing AIUQ">
     <div class="eyebrow">AI WEATHER &amp; CLIMATE · v0.2.0</div>
     <h2>Explore uncertainty.<br><span>Advance prediction.</span></h2>
     <p>One framework for AI weather and climate experiments. Connect initial conditions, run models and orchestrate ensembles with Autosubmit.</p>
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
     <a class="guide-card" href="models.html"><span class="card-number">03 / EXPLORE</span><h3>Find your model</h3><p>Compare checkpoints and check availability for ERA5 and EERIE initial conditions.</p><span class="card-link">Models &amp; support →</span></a>
     <a class="guide-card" href="development.html"><span class="card-number">04 / EXTEND</span><h3>Build on a shared standard</h3><p>Understand AIUQ-st, model cards and the structure of the framework.</p><span class="card-link">Developer guide →</span></a>
   </div>

Forecasts in practice
---------------------

Two model families, a shared starting point. These forecasts for **21 January
1988** were initialized on **1 January 1988** with ERA5 reanalysis data.

.. container:: forecast-grid

   .. figure:: ../neuralgcm_forecast.png
      :alt: NeuralGCM forecast for 21 January 1988 at 1.4 degree resolution.

      **NeuralGCM** · 1.4° resolution

   .. figure:: ../aifs_forecast.png
      :alt: AIFS ensemble forecast for 21 January 1988 at 0.25 degree resolution.

      **AIFS-ens** · 0.25° resolution

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
   development
