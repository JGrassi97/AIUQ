AIUQ Documentation
==================

AIUQ (Artificial Intelligence weather forecasting models for Uncertainty Quantification)
is a framework to run AI-based weather and climate models with Autosubmit.

The current implementation supports:

- multiple AI models (NeuralGCM, AIFS)
- multiple initial-condition providers (ERA5, EERIE)
- deterministic and stochastic workflows
- hindcast and AMIP run modes

This documentation focuses on practical usage, platform setup, and extension guidelines.

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   installation
   run

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide

   developers
   expand
