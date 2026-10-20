# Implementation Plan: Stabilize and Complete the Tiering Project

**Status:** In progress  
**Baseline checked:** 2026-10-21  
**Baseline test result:** 264 passed, 3 failed, 3,641 warnings

This plan replaces the “all complete” impression from `.kiro/specs/context-aware-predictive-tiering/tasks.md` with a test-backed path to a working, reproducible project. Although the Kiro tasks are all checked, several promised end-to-end deliverables are still absent and the checklist needs to be reconciled with verified code and outputs.

**Progress on 2026-10-21:** Fixed the three known failures; added package metadata, a README, a CLI quick start, initial semantic/fusion/migration/simulation/metrics/leakage modules, six experiment runner types, basic JSON/Markdown/HTML reporting, Sphinx source docs, and focused component tests. The CLI smoke run wrote a 20-file dataset and 593 events. The full suite passed (**277 passed, no warnings**) and `compileall` completed. Full end-to-end experiment orchestration, visualization validation, API HTML build, model/semantic pipeline integration, and broad requirement-by-requirement validation remain open.

### Verified status

- **Phase 1:** Complete. The LFU occupancy bug, LRU migration decision reporting, and seasonal event sampling were corrected.
- **Phase 2:** Complete. Focused regression tests and the full suite pass in Python 3.13.
- **Phase 3:** Partial. TF-IDF, optional embeddings, vector search with NumPy fallback, and deterministic RAG fallback exist. Integration into a single trained semantic/workload/future classifier pipeline and all specified semantic acceptance checks remain.
- **Phase 4:** Partial. Feature fusion, migration policy, and a SimPy simulator exist with component tests. Capacity-aware migration and a complete simulation workload/model integration need validation.
- **Phase 5:** Partial. Metrics, leakage checks, six runner classes, and basic reports exist. Full YAML-driven reproducibility, parallel execution, comprehensive visualization, all ablations, generated comparison reports, and final checkpoint evidence remain.
- **Setup/docs:** Partial. `pyproject.toml`, README, CLI, and Sphinx source docs exist. Verify clean editable installation and build the API docs before marking complete.

## Completion criteria

- All tests pass in the documented supported Python environment.
- The three known regressions have focused regression coverage and correct behavior, not just relaxed assertions.
- The implemented features and experiment outputs match the requirements/design and the Kiro plan.
- A new user can install the project, run its documented quick start and experiments, and find generated reports in documented locations.
- Test commands, expected runtime, and remaining limitations are documented; no unexplained test failures remain.

## Phase 1 — Fix the confirmed test failures

### 1. LFU capacity overflow and demotion

**Failure:** `tests/test_lfu_baseline.py::TestCapacityManagement::test_capacity_overflow_cascading_demotion` expects 3.0 units to remain in HOT after a second 3.0-unit file competes for a 5.0-unit HOT tier; observed HOT occupancy is 0.0.

- Trace the tier assignment and capacity reconciliation flow in `src/tiering/baselines/lfu.py`, including how multiple files are selected for promotion and how a tier is emptied when capacity is exceeded.
- Define and document the invariant: tier occupancy must not exceed capacity, and reconciliation should demote the lowest-priority eligible file(s) while retaining the higher-frequency HOT resident.
- Correct the promotion/demotion ordering and ensure occupancy is updated consistently after each move.
- Add or refine tests for overflow by one file, cascading demotion across tiers, ties, and a file larger than the tier capacity. Keep expectations aligned with the chosen policy.
- Run the LFU tests and the full suite.

**Done when:** HOT occupancy is 3.0 in the reported scenario, the displaced file lands in the correct lower tier, capacities remain respected, and the LFU regression tests pass.

### 2. LRU demotion decisions

**Failure:** `tests/test_lru_baseline.py::TestLRUDemotion::test_demote_least_recently_used_from_hot` confirms `file1` moves to WARM and `file2` stays HOT, but the returned decisions contain no decision with `source_tier == Tier.HOT`.

- Inspect `on_decision_interval` in `src/tiering/baselines/lru.py` and the `MigrationDecision` schema.
- Make the returned decision records describe every actual tier movement, including the original/source and destination tiers, file identity, and any required reason/cost fields.
- Ensure decisions are emitted from the same state transition that updates tier placement so reported actions cannot disagree with actual placement.
- Add tests for no-op intervals, a single demotion, and multiple demotions, checking both final placements and decision contents.

**Done when:** the reported LRU demotion has a HOT source tier and WARM destination, and no-op intervals do not fabricate decisions.

### 3. Seasonal event coverage across the requested interval

**Failure:** `tests/test_workload_generator.py::TestEventGeneration::test_generate_seasonal_events_has_period` generates events only through about 13.2 hours for a 48-hour interval; it expects events beyond 24 hours.

- Inspect `_generate_seasonal_events` in `src/tiering/ingestion/workload_generator.py`, with focus on rate calculation, periodic intensity, random sampling, and interval boundaries.
- Preserve the intended seasonal distribution while ensuring event generation considers every period in `[start_time, end_time]` rather than effectively truncating after an early high-activity segment.
- Keep timestamps within the requested interval and in chronological order. Ensure the seed continues to make results reproducible.
- Add deterministic tests for a multi-period interval, one-period interval, low amplitude/frequency, and boundary handling. Assert period coverage using a robust invariant rather than a fragile exact event timestamp.

**Done when:** a 48-hour, two-period request can produce events in both periods under the seeded fixture; timestamps remain valid and reproducible.

## Phase 2 — Re-run and record a clean test baseline

- Run the three focused test modules first, then the complete suite using the supported Python 3.13 environment.
- Record the exact command, Python/pytest versions, pass/fail counts, and warnings in this plan or a test report.
- If failures appear outside the three known cases, investigate and fix them rather than marking the plan complete with a partial run.
- Keep test outputs and generated artifacts out of source directories unless the project deliberately versions them.

**Done when:** the full suite passes and the result can be reproduced from a clean checkout/environment.

## Phase 3 — Audit Kiro tasks against actual deliverables

The Kiro checklist is fully checked, but a repository scan showed empty or near-empty `experiments/` subdirectories, only package initializers under `src/tiering/semantic/` and `src/tiering/simulation/`, and no README or dependency manifest among the discovered project files. Verify these observations and compare each claimed deliverable to the source and requirements before implementing scope.

- Build a traceability table from `.kiro/specs/context-aware-predictive-tiering/requirements.md` and `design.md` to concrete modules, tests, runnable commands, and outputs.
- Check the planned semantic retrieval/RAG path, simulation, migration controller, feature fusion/integration, experiment runners, ablation analysis, metrics, visualization/reporting, and data-leakage validation for actual implementations and end-to-end coverage.
- For each gap, implement the smallest complete feature matching the requirement and add an appropriate test. If an item is intentionally deferred, update the Kiro task state and document it as a limitation instead of leaving it marked complete.
- Verify that all six experiment types described by the Kiro tasks can be run and produce their promised reports; add missing runners or report generation where needed.

**Done when:** every required capability maps to working code and a test or executable acceptance check; no task is considered complete solely because its checkbox is checked.

## Phase 4 — Make the project installable and usable

- Add or verify a dependency manifest and a supported Python version declaration, including optional heavyweight semantic/ML dependencies where appropriate.
- Add a README with setup, quick start, test command, experiment commands, configuration guidance, output locations, and interpretation notes.
- Ensure documented commands work from the repository root and have clear error messages when optional dependencies are missing.
- Confirm result directories are populated by a small reproducible smoke run; keep large/generated artifacts out of version control unless required.

**Done when:** a fresh environment can install the project and reproduce a documented smoke experiment and report.

## Phase 5 — Reduce maintenance noise and final validation

- Replace deprecated Pydantic v2 APIs/configuration with supported equivalents, beginning with the high-volume `.dict()`/`.json()` use in dataset and workload serialization and class-based model `Config` in `src/tiering/ingestion/schemas.py`.
- Re-run the full test suite after the migration and compare warning counts; do not hide warnings with broad filters.
- Run final end-to-end experiments, reproducibility checks with at least two seeds, and verify result/report contents.
- Update `.kiro/specs/context-aware-predictive-tiering/tasks.md` only after evidence for each corresponding checkpoint is available.

**Done when:** the complete suite and documented end-to-end checks pass, deprecation warnings introduced by project code are resolved or explicitly tracked, and the Kiro status reflects verified work.

## Suggested execution order

1. Fix and test LFU, LRU, and seasonal event generation.
2. Run all tests and capture a fresh baseline.
3. Audit requirements and Kiro tasks against implementation; fill confirmed gaps.
4. Add install/setup and usage documentation, then verify commands in a clean environment.
5. Address deprecations and perform final full-suite and experiment validation.

