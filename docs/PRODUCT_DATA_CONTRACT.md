# Product Experiment Data Contract

This contract defines the input needed to test whether document meaning adds useful signal to storage-tier decisions. It is intentionally independent of any single corpus format.

## Source records

### Documents (`documents.jsonl` or equivalent)

One current or versioned record per document:

| Field | Required | Meaning |
|---|---:|---|
| `document_id` | Yes | Stable source identifier; never used as a predictive feature |
| `text` | Recommended | Document body available to the system at the prediction cutoff |
| `created_at` | Recommended | Creation time, for document age |
| `modified_at` | Recommended | Latest version time; must be cutoff-safe |
| `size_bytes` | Recommended | File size available at the cutoff |
| `file_type` | Optional | Extension or normalized file type |
| `category` | Optional | Known document category, if actually available to the product |
| `department` | Optional | Owning department, if available and permitted |
| `confidentiality` | Optional | Access-control classification; never expose it through retrieval filters incorrectly |
| `source_id` | Yes for research | Dataset/source provenance |

Preserve source fields and add normalized fields separately. Do not fabricate enterprise metadata from a public text corpus. Keep missing values explicit. Hash document text for semantic-cache keys and record the encoder/vectorizer version.

### Access events (`access_events.jsonl` or equivalent)

One record per observed access:

| Field | Required | Meaning |
|---|---:|---|
| `document_id` | Yes | Must link to the document record |
| `timestamp` | Yes | Event time, normalized to UTC |
| `operation` | Recommended | Read/write or the source's available operation |
| `bytes_transferred` | Optional | Transferred bytes, if measured |

An event source must state its coverage interval, sampling or filtering rules, license, and known selection bias. A page-view count is an access-demand proxy, not a file-system access event.

## Prediction examples and target

Create one row per document and cutoff timestamp. The minimum target is:

- `future_access_count_24h`: number of events in `(cutoff, cutoff + 24 hours]`.
- `future_any_access_24h`: 1 when that count is nonzero, otherwise 0.
- Optional `operational_tier_24h`: a declared mapping from future counts to HOT/WARM/COLD for simulation experiments only.

Store identifiers and target columns for tracing and scoring, but exclude them from model features. Never call the operational tier an observed ground-truth storage tier unless the source actually supplies one.

### Feature cutoff rules

Allowed at cutoff `t`:

- Metadata values whose effective timestamp is at or before `t`.
- Text from the document version available by `t`.
- Access history with timestamps `<= t`, summarized over fixed lookback windows.
- Retrieved-neighbor identity/similarity and neighbor access summaries computed from documents and events available by `t`, with permission filters applied.

Forbidden:

- Any event after `t`, future-window counts, future access labels, or metadata updated after `t`.
- Document ID, raw row order, split membership, or any derived encoding that reveals target/split identity.
- Retrieval of the query document itself or use of a neighbor's future accesses.

## Matched model runs

All runs must use the same example IDs, cutoffs, target, temporal partitions, model family, and tuning budget:

| Run | Predictive inputs |
|---|---|
| A | Metadata + historical workload |
| B | Document text representation only |
| C | Metadata + historical workload + document text representation |
| D | Run C + authorized retrieval context computed from past-only data |

The first semantic representation can be frozen Sentence Transformer vectors. TF-IDF is a lexical baseline. They are alternative text representations; don't concatenate both into the first comparison unless that is a separately named ablation. RAG generation is not needed: D can use auditable numeric neighbor features.

Fit TF-IDF vocabulary, dimensionality reduction, imputation, scaling, and model parameters on training data only. Cache text features by content hash plus representation version. Empty text gets a missing-content indicator and a documented zero/fallback representation.

## Evaluation split and metrics

Use chronological train/validation/test periods with label windows fully contained in each period. Add a file-identity holdout and report it separately from a known-file future-period test. Do not allow test-period or held-out-file information to influence fitting, feature selection, retrieval configuration, or threshold selection.

Report future-count MAE/RMSE; any-access PR-AUC, ROC-AUC and Brier score; prevalence and calibration; and the declared tier mapping's per-class precision/recall/F1, macro-F1 and confusion matrix. Compare each model policy on the same SimPy access traces, tier capacities, costs, and migration controller. Report latency (mean and p95), hot-tier hit rate, storage cost, migration cost, and migration count separately.

## Public-data candidate and limitations

Current findings and the recommended bounded pilot are recorded in [DATASET_FEASIBILITY.md](DATASET_FEASIBILITY.md). In brief, [vtasca/wikipedia-pageviews](https://huggingface.co/datasets/vtasca/wikipedia-pageviews) is small and has daily counts for the top 100 articles, but missing article-days are censored and cannot be treated as zero. The FineWiki plus raw Wikimedia time-series join remains unvalidated; the latter is roughly 640 GB and its documented IDs do not establish a direct page-ID match. [NeuML's Wikipedia snapshot](https://huggingface.co/datasets/NeuML/wikipedia-20260401) bundles text with aggregated per-page views but does not provide an established future-window target.

Even with a successful join, Wikimedia views measure public web attention, not enterprise file-access demand. They can test an association between text semantics and public attention, but cannot establish that semantic features improve real enterprise storage placement. Use synthetic enterprise workload traces for controlled simulation and label conclusions accordingly.

## Per-run manifest

Save a machine-readable manifest beside each model/report containing:

- source names, versions/revisions, license references, field mapping, row counts, and access/event coverage;
- exact document-text policy, preprocessing, content hash method, and representation model/version;
- prediction cutoff schedule, lookbacks, future target window and tier mapping;
- train/validation/test date ranges and document-ID split hashes;
- random seed, model family/parameters, feature names by group, and tuning budget;
- package versions, metrics, model artifact paths, simulation costs/capacities, and known limitations.

## Current prototype artifact

The current synthetic A/B/C/D run writes row-level cutoff examples to
`results/datasets/product_cutoff_examples.jsonl` and records its manifest in
`results/metrics/product_ablation_synthetic.json`. The export contains 504 rows:
288 train, 96 validation, 96 known-file test, and 24 unseen-file test. Each row
stores `metadata_workload`, `retrieval_context`, and `semantic_tfidf` separately under `features`, while
future count, any-access, and operational-tier targets are under `targets`.
The TF-IDF vectorizer is fit only on training file identities. `document_id` and
`example_id` are traceability fields and are not part of either feature group.

This is an executable pipeline prototype, not evidence that semantics predicts
enterprise demand: synthetic access patterns are assigned independently of file
content. D uses TF-IDF similarity, excludes the query document, requires shared explicit `metadata.access_groups`, checks candidate creation/modification times, and uses only past neighbor accesses. The synthetic corpus has no access groups, so D has zero eligible neighbors; its metrics only verify pipeline integration. Static text is excluded if modified after the earliest training cutoff. The single-document predictor refuses text modified after the supplied cutoff, since it cannot reconstruct historical versions. Real text version history, real ACL integration, and real access logs remain necessary for enterprise evaluation.
