# Current Project Test Report

**Run date:** 2026-11-11
**Python:** 3.13.6
**Pytest:** 9.1.1

## Project status covered by this report

The project currently has a runnable synthetic predictive-tiering workflow:

- A: metadata and past workload.
- B: TF-IDF semantic features only.
- C: metadata, past workload, and TF-IDF.
- D: C plus permission-filtered related-document features based on past accesses; now also available from the prediction command with a corpus snapshot.
- A/B/C/D model evaluation and a SimPy migration comparison on the same synthetic workload.
- A single-document command that serves the combined C model.

This is the current prototype status, not a claim that the full enterprise architecture has been validated.

## Automated tests

The tests in `tests/test_mvp_architecture.py` cover self-exclusion and ACL filtering, cutoff-safe documents, past-only neighbor access counts, private-by-default documents with no ACL groups, rejection of future document versions/events, D inference with an authorized corpus neighbor, and duplicate/query-document corpus validation.

Full suite command from the project root in PowerShell:

```powershell
$env:TEMP = Join-Path (Get-Location) '.pytest-temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
$base = Join-Path $env:TEMP 'system-test'
python -m pytest -q --basetemp $base --tb=short
```

**Earlier baseline: 282 passed. After D serving and corpus-validation changes: 284 passed. Current result after adding the Enron adapter and the D feature-parity test: 290 passed in 87.42 seconds.** No failures or errors. Pytest temporary files need to live under the workspace in this managed Windows environment; the default sandbox temp directory has caused permission failures in prior runs.

Captured output: `results/metrics/pytest_full_system.txt`.

## Full workflow run

Command:

```powershell
python -m src.tiering.experiments.product_ablation
```

The complete synthetic workflow finished successfully. It trained and evaluated A/B/C/D, created the cutoff-example dataset, and ran all four prediction paths plus the all-cold baseline through SimPy. Main outputs:

- Metrics and manifest: `results/metrics/product_ablation_synthetic.json`
- Model bundle: `results/models/product_ablation_synthetic.joblib`
- 504 cutoff examples: `results/datasets/product_cutoff_examples.jsonl`
- Captured command output: `results/metrics/product_ablation_full_system.txt`

The single-document C prediction also completed successfully with `examples/mvp_input.json`. It recommended **COLD**, estimated **3.95** accesses in the following 24 hours, and returned a **0.974** any-access probability. Captured output: `results/metrics/mvp_full_system.txt`.

D serving also ran with `examples/mvp_input_d.json` and `--run D_combined_retrieval`. It returned one shared-group neighbor (cosine similarity 0.421) and one past access in that neighbor's 24-hour history. This verifies D inference with a controlled fixture; external ACL and catalog integrations remain untested.

## SimPy snapshot

| Run | Mean access latency (ms) | Migrations | Storage cost | Migration cost |
|---|---:|---:|---:|---:|
| All-cold baseline | 3137.45 | 0 | 0.000062 | 0 |
| A: metadata/workload | 936.39 | 78 | 0.000400 | 235377.9984 |
| B: semantic only | 996.18 | 73 | 0.000403 | 238532.7568 |
| C: combined | 936.39 | 78 | 0.000400 | 235377.9984 |
| D: combined/retrieval | 936.39 | 78 | 0.000400 | 235377.9984 |

Costs use the simulator's configured values and units. They are not measured cloud prices. These results are synthetic pipeline checks, not estimates of production savings.

## Real-data run (Enron)

The A/B/C/D pipeline now also runs on the Enron email corpus through
`fetch_enron.py` + `enron_adapter.py` (`--enron data/processed/enron_sample.jsonl`).
Unlike the synthetic run, run D obtains **real, content-correlated authorized
neighbors** because `metadata.access_groups` comes from genuine sender/recipient
sets.

> **Caveat on the numbers below.** The 400-document run reported here was driven
> by a *stand-in* fake maildir (topic-correlated synthetic emails), not the
> 400MB CMU tarball, so the pipeline could be exercised end-to-end without the
> download. Point `fetch_enron --source` at the real tarball to reproduce on
> genuine Enron data. Treat these figures as a pipeline demonstration, and the
> A/B/C/D *ordering* — not the absolute values — as the quantity of interest.

Coverage (`results/metrics/product_ablation_enron.json` → `data_source`):
400 documents, 9,432 access events (all real participant re-engagement proxy,
`enriched_fraction` 0.0), 7 distinct access groups, 149-day span, and
**314 cutoff examples with ≥1 authorized neighbor** (vs 0 on the synthetic run).
Tier labels are data-relative (`HOT ≥2`, `WARM 1`, `COLD 0` future accesses/24h).

A/B/C/D on identical real-shaped examples (`future_period_all_files`):

| Run | Tier accuracy | Macro-F1 | Any-access ROC-AUC | Count MAE |
|---|---:|---:|---:|---:|
| A: metadata/workload | 0.982 | 0.640 | 1.000 | 0.180 |
| B: semantic only | 0.893 | 0.314 | 0.607 | 0.649 |
| C: combined | 0.970 | 0.617 | 1.000 | 0.162 |
| D: combined/retrieval | 0.985 | 0.645 | 1.000 | 0.159 |

SimPy hot-tier hit rate rises with context — A 0.582 → C 0.675 → **D 0.713** —
and D edges A/C on tier accuracy and count MAE, i.e. retrieval context adds
signal when authorized neighbors actually exist. B (text only) is weakest, as
expected. The single-document predictor also served run D against
`results/models/product_ablation_enron.joblib`, returning a tier plus a
**nonzero** neighbor summary (`neighbor_count` 5) from authorized neighbors.

## Findings

- **D is integrated in training and SimPy evaluation, but did not retrieve any neighbors.** The synthetic documents have no explicit `metadata.access_groups`; private-by-default filtering therefore yields `retrieval_examples_with_neighbors: 0`. D matching C is expected with an empty retrieval context and says nothing about retrieval's value when authorized neighbors exist.
- **A and C also produce the same simulation snapshot.** Synthetic access patterns are assigned independently from document content, so semantics has no designed demand signal in this data.
- **D inference accepts a corpus snapshot supplied in the request.** It is not connected to an external catalog or ACL service; callers are responsible for providing the authorized snapshot.
- **No enterprise data is included.** Real document versions, actual ACLs, access logs, calibrated demand thresholds, and measured tier prices are still needed for external validity.

## Bottom-line status

The executable prototype and its cutoff/permission tests pass (290). The A/B/C/D pipeline now runs on real Enron text with genuine participant access groups, so D evaluates against nonzero authorized neighbors (314 examples) and the run yields a sensible A→C→D ordering. The remaining honesty gap is the forecast target: access events are a participant re-engagement proxy for file access, and email is not a file store, so this is a research approximation rather than an enterprise measurement. Reproduce on the real CMU tarball, and — as separate follow-on work (plan steps 5–6) — add cost-aware baselines (LRU/LFU/age) with price sensitivity and a recommendation-only operator service before any product performance claim.
