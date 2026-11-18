# Project Guide and Completion Plan

## Project description

This repository prototypes predictive storage tiering. At a cutoff time, it uses document metadata, past access patterns, and text to estimate demand over the next 24 hours and recommend HOT, WARM, or COLD. The experiment compares four matched feature sets:

| Run | Inputs | Purpose |
|---|---|---|
| A | Metadata + past workload | Product baseline |
| B | TF-IDF document text | Semantic-only diagnostic |
| C | Metadata + workload + TF-IDF | Test whether text adds signal |
| D | C + related-document context | Test authorized, past-only retrieval features |

A/B/C/D run on synthetic data (a pipeline/null baseline) and on the real Enron email corpus. The MVP serves C and D from a versioned model registry with logging, a drift monitor and rollback. **Current result on real data:** text (C) and authorized retrieval (D) do not beat metadata plus workload (A). C ranks next-day access slightly worse than A (ROC-AUC -0.013, 95% CI [-0.026, -0.001]); D vs C is +0.003 [-0.002, 0.008]. All learned models beat a persistence forecast. Details and caveats: `docs/TEST_REPORT_MVP.md`.

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

Serve retrieval run D by passing a corpus snapshot in the input JSON and selecting the saved D model:

```powershell
python -m src.tiering.prediction.mvp examples/mvp_input_d.json --run D_combined_retrieval
```

Each corpus record needs a unique `document_id`, cutoff-safe `created_at` and `modified_at`, text, optional `metadata.access_groups`, and optional past `access_events`. The query document must not also appear in `corpus`. Retrieval returns aggregate counts and similarity only; it does not return neighbor text. The example returns one authorized neighbor. Missing or nonmatching access groups yield no neighbor.

## Run on real data (Enron)

The A/B/C/D experiment also runs on the public Enron email corpus (FERC public domain), which supplies real document text, real timestamps, and real participant-based access groups — giving run D genuine, content-correlated authorized neighbors that synthetic data cannot. See `docs/DATASET_FEASIBILITY.md` for the mapping, the access-event proxy, and caveats.

Download the archive once (443 MB; see `data/raw/enron/SOURCE.md` for the checksum), then parse whole mailboxes. Selecting mailboxes is better than taking the first N files, which in the tarball are alphabetical by mailbox:

```powershell
curl.exe -L -C - -o data/raw/enron_mail.tar.gz https://www.cs.cmu.edu/~enron/enron_mail_20150507.tar.gz
tiering fetch-enron --source data/raw/enron_mail.tar.gz --owners dasovich-j,kean-s,shapiro-r --output data/processed/enron_govaffairs.jsonl
```

Then run the matched A/B/C/D experiment:

```powershell
tiering ablate --enron data/processed/enron_govaffairs.jsonl
```

Outputs are `results/models/product_ablation_enron.joblib`, `results/metrics/product_ablation_enron.json`, and `results/datasets/enron_cutoff_examples.jsonl`. Emails are one-shot, so access events are a **thread re-engagement proxy**: a later email in the same conversation (same normalized subject, a shared participant) within `--horizon-days` (default 14). `--proxy participant` restores the legacy any-shared-participant proxy, kept only for comparison. `--densify K` synthesizes extra events only for sparse documents; the default keeps every event real. The manifest records `data_source`, `target_prevalence`, `leakage_checks`, bootstrap `uncertainty`, `cost_evaluation`, and `simpy_evaluation`.

## Serve, monitor and roll back

```powershell
tiering models promote results/models/product_ablation_enron.joblib --note "Enron Government Affairs"
tiering predict examples/mvp_input.json
tiering predict examples/mvp_input_d.json --run D_combined_retrieval
tiering monitor
tiering models rollback
```

See `docs/OPERATIONS.md` for configuration, logging, drift alerts, rollback and ownership.

## Run tests

In this managed Windows workspace, pytest temp files need to be placed inside the writable project folder:

```powershell
$env:TEMP = Join-Path (Get-Location) '.pytest-temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
$base = Join-Path $env:TEMP 'system-test'
python -m pytest -q --basetemp $base --tb=short
```

Latest result: **299 passed**. The full result and interpretation are in `docs/TEST_REPORT_MVP.md`; raw output is `results/metrics/pytest_full_system.txt`.

## Architecture map

- `src/tiering/ingestion/`: schemas, data generation, cleaning, validation, and the Enron real-data fetcher/adapter (`fetch_enron.py`, `enron_adapter.py`).
- `src/tiering/workload/`: past access feature extraction.
- `src/tiering/semantic/`: TF-IDF, embeddings, vector search, and cutoff/ACL-filtered retrieval features (reference function plus the vectorized `RetrievalIndex`).
- `src/tiering/models/` and `src/tiering/prediction/`: model training, tier classifier, C/D prediction, and the model registry, prediction log and drift monitor (`registry.py`).
- `src/tiering/experiments/`: A/B/C/D experiments, metrics, bootstrap uncertainty, and cost-aware placement (`cost_evaluation.py`).
- `src/tiering/migration/`, `src/tiering/baselines/`, `src/tiering/simulation/`: migration decisions, comparison policies, and SimPy evaluation.
- `data/`: downloaded and processed data.
- `results/`: model artifacts, metrics, and cutoff examples.
- `configs/pricing_aws_s3_us_east_1.yaml`: sourced storage prices for the cost evaluation.
- `docs/`: data contract, feasibility review, operations guide, and test reports.

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

## Plan status (2026-11-18)

| Step | Status |
|---|---|
| 1. Cutoff-versioned input bundle | Done for single-version documents: text visibility is per-row `modified_at <= cutoff`, and targets come only from later events. Multi-version text history is not modeled. |
| 2. Authorized neighbors for D | Done: Enron access groups; fixtures cover shared, disjoint and absent groups, self-exclusion, later versions, and events at the cutoff. |
| 3. Serve D from the MVP | Done: `tiering predict --run D_combined_retrieval`, with a training/inference parity test. |
| 4. Validate on real data | Done on one corpus: Enron thread-proxy target with coverage, prevalence, leakage checks and bootstrap CIs. |
| 5. Cost-aware placement | Done with sourced S3 list prices, a HOT capacity limit, transition and minimum-duration charges, LRU/LFU/age/persistence baselines, and a sensitivity grid. |
| 6. Package and operate | Done for a single-process, recommendation-only MVP: CLI, versioned registry, rollback, prediction log, drift monitor, `docs/OPERATIONS.md`. Owner roles are unassigned. |

## Current gaps to keep visible

- The real-data target is a thread re-engagement **proxy** on email, not native file accesses. Only one mailbox set (Government Affairs) and one horizon (14 days) have been run.
- Only TF-IDF (256 terms) has been evaluated as the semantic representation.
- With full S3 transition and minimum-duration fees, no adaptive policy beats keeping young documents in Standard. Tiering value would lie in older documents, which the proxy cannot score.
- No HTTP service, authentication, live ACL/catalog integration, or enterprise event-log adapter.
- Owner roles in `docs/OPERATIONS.md` must be assigned before use against approved data.

Use `docs/PRODUCT_DATA_CONTRACT.md` for row-level feature/target rules, `docs/DATASET_FEASIBILITY.md` for public-data limits, and `docs/TEST_REPORT_MVP.md` for current verification results.
