# Operating the Tiering MVP

The MVP is a **recommendation-only** service. It returns a HOT/WARM/COLD tier,
tier probabilities and an expected next-24h access count for one document. It
never moves data. An operator or a separate, approved job acts on the
recommendations.

## Install

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
tiering --help
```

## Configure

| Setting | Where | Default |
|---|---|---|
| Model registry | `--registry` | `results/models` (`registry.json` + `versions/`) |
| Prediction log | `--log` | `results/logs/predictions.jsonl` |
| Feature run | `--run` | `C_combined`; `D_combined_retrieval` needs a `corpus` snapshot |
| Storage prices | `configs/pricing_aws_s3_us_east_1.yaml` | S3 list prices, us-east-1, checked 2026-11-18 |

Input contract: `docs/PRODUCT_DATA_CONTRACT.md` and `examples/mvp_input*.json`.
All timestamps need a timezone. Inputs carrying events or text versions
after `cutoff` are rejected.

## Train, promote and serve

```powershell
tiering fetch-enron --source data/raw/enron_mail.tar.gz --owners dasovich-j,kean-s,shapiro-r --output data/processed/enron_govaffairs.jsonl
tiering ablate --enron data/processed/enron_govaffairs.jsonl
tiering models promote results/models/product_ablation_enron.joblib --note "first real-data model"
tiering predict examples/mvp_input.json
tiering predict examples/mvp_input_d_enron.json --run D_combined_retrieval
```

`promote` copies the bundle to `versions/<name>-<sha12>.joblib`. That copy is
immutable and content-addressed, and the registry points at it. Every
prediction records the `model_version` that served it.

## Monitor

```powershell
tiering monitor --last 1000 --threshold 0.2
```

The monitor reports served tier shares, mean any-access probability and the
model versions seen. It raises `drift_alert` when any tier's served share
differs from the active model's training share by more than the threshold.
Labels arrive 24h late, so this check catches input/output shifts, not
accuracy loss. To check accuracy, join the log to access counts observed in
the following 24h and recompute the metrics in
`src/tiering/experiments/product_ablation.py` (`_summarize`).

The log holds scores and retrieval aggregates only. It contains no document
text and no neighbor identities.

## Roll back

```powershell
tiering models rollback
tiering models list
```

`rollback` reactivates the previously active version and appends a
`rollback` entry to `registry.json`. Nothing is deleted. Registry writes are
atomic (write to a temp file, then replace).

## Ownership

These roles must be assigned before the service runs against approved data.
The repository cannot assign them.

| Responsibility | Role | Assigned to |
|---|---|---|
| Approves data sources, ACL snapshots and retention | Data owner | _unassigned_ |
| Trains, promotes and rolls back models; reviews drift alerts | Model owner | _unassigned_ |
| Acts on recommendations; owns migration jobs | Storage operator | _unassigned_ |

## Known limits

* The models are trained on a research corpus whose access events are a
  re-engagement proxy (see `docs/DATASET_FEASIBILITY.md`). Retrain on your own
  access logs before acting on recommendations.
* D is served only against a caller-supplied, already-authorized corpus
  snapshot. There is no live catalog or ACL integration.
* Single process and a file-based registry. There is no HTTP service,
  authentication or concurrent-writer locking.
