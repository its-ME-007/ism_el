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

**Earlier baseline: 282 passed in 14.38 seconds. Current result after D serving and corpus-validation changes: 284 passed in 17.55 seconds.** No failures or errors. Pytest temporary files need to live under the workspace in this managed Windows environment; the default sandbox temp directory has caused permission failures in prior runs.

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

## Findings

- **D is integrated in training and SimPy evaluation, but did not retrieve any neighbors.** The synthetic documents have no explicit `metadata.access_groups`; private-by-default filtering therefore yields `retrieval_examples_with_neighbors: 0`. D matching C is expected with an empty retrieval context and says nothing about retrieval's value when authorized neighbors exist.
- **A and C also produce the same simulation snapshot.** Synthetic access patterns are assigned independently from document content, so semantics has no designed demand signal in this data.
- **D inference accepts a corpus snapshot supplied in the request.** It is not connected to an external catalog or ACL service; callers are responsible for providing the authorized snapshot.
- **No enterprise data is included.** Real document versions, actual ACLs, access logs, calibrated demand thresholds, and measured tier prices are still needed for external validity.

## Bottom-line status

The executable synthetic prototype and its cutoff/permission tests pass. D inference works with a fixture containing an eligible neighbor. The default generated experiment still has zero ACL-eligible neighbors, so D's benchmark metrics do not measure retrieval benefit. Next add real or purpose-built versioned document/event/ACL data, evaluate D with meaningful authorized neighbors, and validate the system on a real trace before making product performance claims.
