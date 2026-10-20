# Context-Aware Predictive Storage Tiering

Research code for comparing conventional and workload-based storage tiering policies using reproducible synthetic files and access traces.

## Requirements

- Python 3.10 or newer
- Core dependencies are listed in `pyproject.toml`
- Optional semantic model dependencies are installed with the `semantic` extra

## Install

From the repository root:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` and install with `python -m pip install -e '.[dev]'`.

Install optional sentence-transformer and FAISS dependencies when using semantic retrieval:

```powershell
python -m pip install -e ".[semantic]"
```

## Run tests

```powershell
python -m pytest -q
```

On restricted Windows environments where the system temporary directory is not writable, set `TEMP` and `TMP` to a writable directory before running pytest.

## Generate a small reproducible input set

```powershell
python -m src.tiering.cli generate --num-files 100 --duration-hours 24
```

This writes `data/generated/files.jsonl` and `data/generated/access_traces.jsonl`. The seed and workload distributions come from `configs/default.yaml`; command-line size and duration overrides keep the quick start manageable.

## Available code

- Synthetic file and access trace generation: `src/tiering/ingestion/`
- Configuration validation and reproducible seed helpers: `src/tiering/config/`
- LRU, LFU, age-based, and workload-ML baselines: `src/tiering/baselines/`
- Workload feature extraction and seasonality detection: `src/tiering/workload/`
- Future access prediction and tier classification: `src/tiering/prediction/` and `src/tiering/models/`
- Semantic TF-IDF, optional sentence-transformer embeddings, similarity search, and deterministic RAG fallback: `src/tiering/semantic/`
- Feature fusion, migration hysteresis/cooldown, and SimPy access simulation: `src/tiering/models/feature_fusion.py`, `src/tiering/migration/`, and `src/tiering/simulation/`
- Metrics and temporal leakage checks: `src/tiering/metrics.py` and `src/tiering/validation/`

The checked-in default configuration is `configs/default.yaml`. The project plan and requirement trace are in `.kiro/specs/context-aware-predictive-tiering/` and `IMPLEMENTATION_PLAN.md`.

## Current limitations

The project is still being completed against the full Kiro design. The six experiment runner types, basic orchestration, and Markdown/JSON summaries are present, but they do not yet form a validated end-to-end experiment pipeline. Model training and semantic features still need integrated runners; visualizations, comprehensive API documentation, and final comparison reports remain to be implemented and validated. Do not interpret the checked Kiro boxes as evidence that these workflows have passed end-to-end validation. Sentence Transformer embedding generation requires the optional `semantic` extra and a locally available model.

## Outputs

The `data/` and `results/` folders hold generated datasets, models, metrics, logs, and plots. Dataset and trace generators write JSONL outputs; experiment JSON and Markdown summaries can be written through `ExperimentFramework`. Visualization plots are pending.
