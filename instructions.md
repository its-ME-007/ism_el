# Project Guide and Completion Plan

## Project description

This repository prototypes predictive storage tiering. At a cutoff time, it uses document metadata, past access patterns, and text to estimate demand over the next 24 hours and recommend HOT, WARM, or COLD. The experiment compares four matched feature sets:

| Run | Inputs | Purpose |
|---|---|---|
| A | Metadata + past workload | Product baseline |
| B | TF-IDF document text | Semantic-only diagnostic |
| C | Metadata + workload + TF-IDF | Test whether text adds signal |
| D | C + related-document context | Test authorized, past-only retrieval features |

The repository currently runs A/B/C/D training and a SimPy comparison on synthetic data. The single-document MVP serves C. The synthetic trace generator assigns access patterns independently of content, so these runs validate the pipeline and do not show that semantics improves real storage placement.

## Setup

Python 3.10 or newer is required. In PowerShell, from the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Install optional semantic dependencies only when using Sentence Transformer embeddings:

```powershell
python -m pip install -e ".[semantic]"
```

## Run the prototype

Train A/B/C/D, evaluate the models, and run the SimPy comparison:

```powershell
python -m src.tiering.experiments.product_ablation
```

Outputs:

- `results/models/product_ablation_synthetic.joblib`
- `results/metrics/product_ablation_synthetic.json`
- `results/datasets/product_cutoff_examples.jsonl`

Make a one-document prediction with the combined C model:

```powershell
python -m src.tiering.prediction.mvp examples/mvp_input.json
```

The input shape is illustrated in `examples/mvp_input.json`. Supply ISO 8601 timestamps with a timezone. The `access_events` list must not contain events after `cutoff`; the current text must not have `modified_at` after that cutoff. The predictor returns a tier, per-tier probabilities, expected next-day access count, and probability of any next-day access.

The predictor does not yet serve D. D requires a permission-filtered corpus and neighbor event histories, rather than only the target document.

## Run tests

In this managed Windows workspace, pytest temp files need to be placed inside the writable project folder:

```powershell
$env:TEMP = Join-Path (Get-Location) '.pytest-temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
$base = Join-Path $env:TEMP 'system-test'
python -m pytest -q --basetemp $base --tb=short
```

Latest result: **282 passed**. The full result and interpretation are in `docs/TEST_REPORT_MVP.md`; raw output is `results/metrics/pytest_full_system.txt`.

## Architecture map

- `src/tiering/ingestion/`: schemas, data generation, cleaning, and validation.
- `src/tiering/workload/`: past access feature extraction.
- `src/tiering/semantic/`: TF-IDF, embeddings, vector search, and cutoff/ACL-filtered retrieval features.
- `src/tiering/models/` and `src/tiering/prediction/`: model training, tier classifier, and C prediction entry point.
- `src/tiering/experiments/`: A/B/C/D experiments and metrics.
- `src/tiering/migration/`, `src/tiering/baselines/`, `src/tiering/simulation/`: migration decisions, comparison policies, and SimPy evaluation.
- `data/`: downloaded and processed data.
- `results/`: model artifacts, metrics, and cutoff examples.
- `docs/`: data contract, architecture, feasibility review, and test reports.

## Implementation plan to complete the architecture

### 1. Create a realistic, cutoff-versioned input bundle

Define a repeatable input format for document versions, access events, and ACLs. Each document version needs an effective timestamp and immutable version ID or content hash. Each access event needs a document ID, timestamp, operation, and source provenance. Preserve missing data as missing; do not infer zero accesses from incomplete logs.

**Done when:** the example builder can reconstruct exactly which content and metadata were visible at every cutoff and can create targets only from later events.

### 2. Give D eligible, authorized neighbors

Add fixtures or a data adapter with explicit `metadata.access_groups`. Extend tests to cover shared and disjoint groups, absent groups, self-exclusion, later document versions, and events exactly before/after the cutoff. Report retrieval coverage and zero-neighbor rate beside D's metrics. Do not treat a synthetic retrieval run as a real ACL evaluation.

**Done when:** nonzero eligible neighbors occur in a controlled fixture and all access and time filters are tested.

### 3. Serve D from the MVP

Add an inference interface that accepts the target document plus an authorized corpus snapshot and past event history. Use the exact saved TF-IDF/vectorizer and retrieval settings from training. Return neighbor count/similarity summaries, D's prediction, and a trace of which feature groups were used; never return unauthorized neighbor content.

**Done when:** C and D can be selected at inference and a matching training/inference feature test proves their feature order and values agree.

### 4. Validate on real data

Connect a real file-access trace and versioned document metadata/text, with licenses, access coverage, ACL semantics, and missingness documented. Keep the public Wikipedia sample as a web-attention proxy only. Use chronological splits plus unseen-file holdout; fit vocabulary, transforms, and thresholds on training/validation only.

**Done when:** A/B/C/D are scored on identical real examples and the report includes coverage, target prevalence, leakage checks, and uncertainty.

### 5. Make placement decisions cost-aware

Calibrate access forecasts and define tier choices using actual storage price, retrieval latency, capacity, migration bandwidth/cost, and policy constraints. Compare the ML policy with LRU, LFU, age, and all-cold baselines on identical traces. Use sensitivity analysis for uncertain prices and migration assumptions.

**Done when:** savings and latency tradeoffs are calculated from measured or explicitly sourced system values and migration decisions respect capacity and policy limits.

### 6. Package and operate the MVP

After data and evaluation are credible, expose prediction through a small service or interface, add model/data version metadata, input validation, structured logs, and a rollback path. Keep migration in recommendation-only mode until an operator can review decisions and outcomes.

**Done when:** the application can be installed, configured, operated against approved data, monitored, and rolled back with documented ownership.

## Current gaps to keep visible

- Training and simulation data are synthetic; semantic benefit is unproven.
- Synthetic documents have no ACL groups, so D currently has zero eligible neighbors.
- D trains and runs through simulation, but the single-document MVP serves C only.
- The predictor accepts one supplied text version and rejects a modification after cutoff; it does not fetch historical versions itself.
- There is no enterprise event-log adapter, production service/UI, cost calibration, or deployment workflow.

Use `docs/PRODUCT_DATA_CONTRACT.md` for row-level feature/target rules, `docs/DATASET_FEASIBILITY.md` for public-data limits, and `docs/TEST_REPORT_MVP.md` for current verification results.
