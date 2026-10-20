Context-Aware Predictive Storage Tiering
========================================

This documentation describes the implemented modules. See the repository README
for installation and command-line usage, and ``IMPLEMENTATION_PLAN.md`` for
workflows that still need end-to-end integration and validation.

Core modules
------------

.. automodule:: src.tiering.ingestion.dataset_generator
   :members:

.. automodule:: src.tiering.ingestion.workload_generator
   :members:

.. automodule:: src.tiering.workload.feature_extractor
   :members:

.. automodule:: src.tiering.baselines.lru
   :members:

.. automodule:: src.tiering.baselines.lfu
   :members:

.. automodule:: src.tiering.models.feature_fusion
   :members:

.. automodule:: src.tiering.migration.controller
   :members:

.. automodule:: src.tiering.simulation.simulator
   :members:

.. automodule:: src.tiering.experiments.framework
   :members:
