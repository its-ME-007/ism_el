# Current Project Test Report

**Run date:** 2026-11-18
**Python:** 3.13.6
**Pytest:** 9.1.1

## Status in one paragraph

The A/B/C/D pipeline now runs end to end on the real Enron email corpus, and
the MVP serves C and D from a versioned, rollback-capable model registry.
**On real data, document text (C) and authorized retrieval context (D) do not
improve on metadata and workload features alone (A).** C is slightly *worse*
than A at ranking next-day access (ROC-AUC -0.013, 95% CI [-0.026, -0.001]).
D is indistinguishable from C (+0.003, CI [-0.002, 0.008]). Text alone (B) is
at chance (ROC-AUC 0.474). All learned models beat a last-24h persistence
forecast at ranking (A +0.131, CI [0.094, 0.169]). Under full Amazon S3 fees,
keeping documents in their first two weeks in Standard is cheapest. Without
per-move fees, A cuts cost by 34.6% versus the best static tier (1 MB
objects). The target is a thread re-engagement proxy, not a file-access log.

## Automated tests

```powershell
$env:TEMP = Join-Path (Get-Location) '.pytest-temp'
$env:TMP = $env:TEMP
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null
python -m pytest -q --basetemp (Join-Path $env:TEMP 'system-test') --tb=short
```

**299 passed** (290 at the previous report). Output:
`results/metrics/pytest_full_system.txt`. The 9 new tests are in
`tests/test_real_data_validity.py` and `tests/test_registry.py`:

- the thread proxy counts replies, but not unrelated mail between the same people;
- messages dated outside 1998-2002 are dropped;
- weekly split label windows do not overlap;
- documents older than the proxy horizon are not scored;
- text counts only when its version is visible at the row's cutoff;
- the vectorized `RetrievalIndex` matches the reference retrieval function on
  randomized fixtures with ties, including that a neighbor event exactly at
  the cutoff is excluded;
- the document-clustered bootstrap reports zero difference for identical
  predictions;
- registry promote, rollback, prediction log (no text) and the drift monitor
  with its minimum-sample guard work.

`cost_evaluation.py` and `enron_adapter.py` also carry `__main__` self-checks.

## Fixes that changed earlier results

The 2026-11-14 Enron numbers (ROC-AUC 1.000, "D 0.713 hot-hit rate") are
superseded. They came from a stand-in fake maildir and these defects:

| Defect | Effect | Fix |
|---|---|---|
| TF-IDF zeroed for any document created after the *earliest* training cutoff | ~80% of Enron rows had empty text in training; serving used real text (train/serve skew) | Text is used when `modified_at <= row cutoff`, in training, retrieval and SimPy alike |
| Documents older than the 14-day proxy horizon were scored | Their target is 0 by construction, so ROC-AUC was trivially 1.0 | `max_sample_age_days = horizon_days` |
| Proxy = any later email sharing *any* participant | The mailbox owner is on nearly every email, so the target measured owner activity | Thread proxy: same normalized subject plus a shared participant |
| `all_test` rows never received retrieval features | D's headline "all files" score had an empty retrieval context | All splits enriched |
| SimPy evaluation started at synthetic day 26 for Enron | Wrong window | Uses the data's first test day; bounded to 28 days |
| Duplicate copies of one email under different Message-IDs | Copies could straddle the train/held-out split | Content-key dedupe |
| Neighbor access counts included events at the cutoff instant | One email could feed a feature and the label | Strictly before the cutoff, matching workload features |
| Tier head trained unweighted on 97% COLD labels | Always predicted COLD (macro-F1 0.329) | Balanced class weights on real data; the unweighted report is kept at `results/metrics/product_ablation_enron_unweighted_tier.json` |

The synthetic rerun is unchanged where it should be. SimPy latency is
936.4 ms for A/C/D and 996.2 ms for B, against 3137.5 ms all-cold, with
78/73 migrations: the same as the previous report.

## Real-data experiment (Enron)

**Data:** `enron_mail_20150507.tar.gz` (CMU; checksum in
`data/raw/enron/SOURCE.md`), Government Affairs mailboxes `dasovich-j`,
`kean-s` and `shapiro-r`. The fetcher parsed 59,656 records, leaving 23,946
unique emails from 1998-09 to 2002-09 after date clipping and dedupe. The
proxy produced 18,199 thread re-engagement events, all real
(`enriched_fraction` 0.0). 7,309 documents were re-engaged at least once.

```powershell
tiering fetch-enron --source data/raw/enron_mail.tar.gz --owners dasovich-j,kean-s,shapiro-r --output data/processed/enron_govaffairs.jsonl
tiering ablate --enron data/processed/enron_govaffairs.jsonl
```

Full manifest: `results/metrics/product_ablation_enron.json`. Runtime is
about 48 minutes on this laptop.

**Splits and prevalence** (weekly cutoffs; rows = document x cutoff; only
documents at most 14 days old):

| Split | Cutoffs | Rows | Documents | Any access next 24h | HOT / WARM / COLD |
|---|---:|---:|---:|---:|---|
| Train | 43 | 21,193 | 10,826 | 2.1% | 0.6% / 1.5% / 97.9% |
| Validation | 9 | 4,028 | 2,385 | 3.6% | 1.1% / 2.5% / 96.4% |
| Test, all files | 6 | 6,089 | 3,509 | 2.6% | 0.6% / 2.0% / 97.4% |
| Test, unseen files | 6 | 1,228 | 719 | 2.3% | 0.5% / 1.8% / 97.7% |

Tier labels are data-relative: HOT is 2 or more accesses in 24h, WARM is 1,
COLD is 0. 30,806 rows had at least one authorized retrieval neighbor.

**Leakage checks** (the run fails if any is false): train label windows end
before validation, and validation windows end before test; train and
held-out file IDs are disjoint; TF-IDF is fit only on training documents
visible at the earliest cutoff; workload and neighbor features use events
strictly before the cutoff. Events exactly at a cutoff: 0.

### A/B/C/D on identical test rows

These numbers are document-clustered bootstrap estimates with 95% CIs, from
1,000 resamples over the test (all files) rows. P is the last-24h
persistence forecast.

| Run | Tier macro-F1 | Any-access ROC-AUC | Avg. precision | Count MAE |
|---|---|---|---|---|
| P persistence | 0.396 [0.372, 0.420] | 0.756 [0.714, 0.797] | 0.167 [0.121, 0.231] | 0.135 |
| A metadata + workload | 0.382 [0.364, 0.401] | **0.887** [0.862, 0.908] | **0.177** [0.139, 0.226] | 0.067 |
| B text only | 0.333 [0.322, 0.347] | 0.474 [0.418, 0.526] | 0.058 [0.033, 0.104] | 0.059 |
| C A + text | **0.404** [0.380, 0.427] | 0.874 [0.848, 0.898] | 0.143 [0.113, 0.184] | 0.063 |
| D C + retrieval | 0.383 [0.363, 0.404] | 0.877 [0.851, 0.900] | 0.154 [0.122, 0.196] | 0.064 |

Paired differences, as mean [95% CI]:

| Comparison | Macro-F1 | ROC-AUC | Avg. precision |
|---|---|---|---|
| A - P | -0.015 [-0.038, 0.009] | +0.131 [0.094, 0.169] | +0.006 [-0.056, 0.062] |
| C - A | +0.022 [-0.002, 0.044] | -0.013 [-0.026, -0.001] | -0.033 [-0.060, -0.004] |
| D - C | -0.021 [-0.038, -0.003] | +0.003 [-0.002, 0.008] | +0.010 [-0.002, 0.023] |
| D - A | +0.001 [-0.021, 0.024] | -0.010 [-0.022, 0.002] | -0.023 [-0.052, 0.008] |
| B - A | -0.049 [-0.072, -0.026] | -0.413 [-0.471, -0.353] | -0.119 [-0.164, -0.069] |

On unseen files the pattern is the same and wider: C - A ROC-AUC is -0.012
[-0.035, 0.011], and D - C is -0.005 [-0.014, 0.003].

Count MAE is not a useful separator here. Near-zero predictions score well
when 97% of targets are 0; B has the lowest MAE while its ranking is at
chance.

**Reading:** past workload and age carry the predictable signal. Adding
TF-IDF text costs a little ranking quality, probably because 256 sparse text
features dilute 21k training rows with about 2% positives. Retrieval
recovers about a quarter of that loss, and its CI includes zero. The small
C-over-A macro-F1 gain is borderline and reverses for D. This is a null
result for "semantics improves tiering" on this corpus and target, not
evidence that semantics cannot help elsewhere.

### Cost-aware placement

Implemented in `src/tiering/experiments/cost_evaluation.py`, with prices in
`configs/pricing_aws_s3_us_east_1.yaml`: S3 Standard / Standard-IA / Glacier
Instant Retrieval, us-east-1, checked 2026-11-18. The cost model charges
billable size (128 KB minimum on IA/GIR), GET and retrieval fees, lifecycle
transition and restore fees, and early-deletion charges under the 30/90-day
minimums. A HOT capacity limit applies as a policy limit. Every policy is
scored on the same 6,089 test rows from realized next-24h accesses. The
grid covers object size {actual, 1 MB, 10 MB} x retrieval fee {0.5x, 1x, 2x}
x HOT capacity {100%, 25%, 10%} x fee model {full S3, storage + access
only}. AWS documents millisecond first-byte latency for all three classes,
so the trade-off here is cost. "Accesses served from HOT" is the latency
proxy.

Saving versus the best static tier (1x retrieval fee):

| Setting | A | C | D | LRU | LFU | Age | Persistence | Oracle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 MB, no capacity limit, full S3 fees | -1514% | -1361% | -1374% | -1711% | -1656% | -612% | -1541% | -1461% |
| 1 MB, no capacity limit, storage + access only | **+34.6%** | +31.9% | +33.1% | -8.3% | -11.4% | +16.2% | -24.3% | +79.0% |
| 1 MB, HOT ≤ 10%, storage + access only | **+29.1%** | +26.9% | +28.8% | -5.4% | -11.8% | +10.8% | -21.1% | +79.5% |
| Actual email size, HOT ≤ 25%, storage + access only | +11.2% | +10.7% | +10.4% | -134.6% | -133.1% | **+17.6%** | -3.8% | +34.7% |

**Reading:**

- With full S3 fees, every adaptive policy loses to keeping these young
  documents in Standard. That includes the myopic oracle that knows the
  realized counts. One transition plus the 90-day Glacier IR minimum costs
  more than weeks of storage savings on a 1 MB object.
- Real tiering value would sit in the long tail of older documents, which
  this target cannot score. There, an age rule decides placement anyway.
- Without per-move fees, the learned forecasters beat LRU, LFU, age and
  persistence. A is best, D is within about 1.5 points of A, and text and
  retrieval add nothing.
- Emails smaller than 128 KB are billed at the minimum on IA/GIR, so at
  actual sizes the age rule wins.

### SimPy (first 28 test days, 2,875 documents, 2,138 accesses)

| Run | Mean latency (ms) | HOT hit rate | Migrations |
|---|---:|---:|---:|
| All-cold | 106.2 | 0.0% | 0 |
| A | 95.6 | 1.1% | 218 |
| B | 103.7 | 0.4% | 25 |
| C | 98.7 | 0.7% | 167 |
| D | 99.2 | 1.2% | 147 |

Hit rates are low because most proxy accesses occur within a day or two of
an email's arrival. The controller needs two consistent daily predictions
(hysteresis) before it migrates, so most promotions land after the burst.
The simulator uses configured tier latencies, not measured ones. The cost
evaluation above is the primary placement result.

## Production-shaped synthetic store (coupling sweep)

`src/tiering/ingestion/production_synth.py` generates a file store with
projects, project lifecycles, reactivation bursts, heavy-tailed popularity,
age decay, a working-hours cycle, MB-GB sizes, project topic text and
project/department ACL groups. `coupling` is the share of each file's demand
that follows its project's shared schedule; `coupling=0` is the negative
control. 50 projects, 1,811 files, 120 days, about 217k accesses; seed 42;
200 bootstrap resamples; one seed per coupling.

```powershell
python -m src.tiering.experiments.product_ablation --prod-synth --coupling 0.5
```

Any-access ROC-AUC, all files:

| Coupling | P | A | C | D | C-A | D-C |
|---|---:|---:|---:|---:|---:|---:|
| 0 (control) | 0.854 | 0.909 | 0.907 | 0.907 | -0.001 | -0.001 |
| 0.5 | 0.819 | 0.861 | 0.866 | 0.868 | +0.004 | +0.002 |
| 0.8 | 0.821 | 0.867 | 0.870 | 0.867 | +0.003 | -0.003 |

The control behaves: no C or D gain at coupling 0. With coupling, gains are
consistent across resamples but small (tier macro-F1 +0.02 to +0.04). Retrieval
finds the right neighbors (93.5% of top-5 neighbors share the query's project
at coupling 0.8), yet neighbor heat barely correlates with the next-24h target
(0.04, against 0.45 for the file's own 24h count). When a project's files
move together, each file's own history already carries the project signal, so
per-file neighbor features are redundant. Retrieval context can only add
information where neighbors lead the file (new files, lagged spread) or where
the policy acts on the group, not as extra columns in a per-file model.

## Semantic graph tiering (group prefetch)

`src/tiering/semantic/graph.py` builds a file graph: an edge joins two files
that share an ACL group, weighted by TF-IDF cosine (>= 0.3, up to 50 edges per
node), with neighbors filtered to files that exist at access time.
`GraphPrefetchPolicy` promotes an accessed file to HOT, pre-promotes its top-5
neighbors, and demotes idle files HOT -> WARM (after the HOT idle timeout) ->
COLD (after 168 h). Same policy with `k=0` is plain recency. A random-neighbor
graph over the same ACL pools is the control for "semantics" vs "same people".
Same 28-day window as the ablation SimPy run, all tiers start COLD.

```powershell
python -m src.tiering.experiments.graph_tiering --coupling 0.0 0.5 0.8
```

Coupling 0.8 (storage cost in configured units; first-touch = the file's first
access in the window):

| Policy (HOT idle) | HOT hit | First touch from HOT | Mean latency (ms) | Storage | Prefetch precision |
|---|---:|---:|---:|---:|---:|
| Recency 24 h | 0.946 | 0.000 | 3,723 | 0.888 | - |
| Recency 336 h | 0.978 | 0.000 | 3,223 | 1.194 | - |
| Random group 6 h | 0.936 | 0.464 | 2,347 | 1.518 | 0.17 |
| Semantic graph 6 h | 0.949 | 0.526 | 2,063 | 1.260 | 0.26 |
| Semantic graph 24 h | 0.985 | 0.623 | 1,762 | 1.429 | 0.25 |

- Group prefetch turns 50-70% of first touches into HOT hits; recency can never
  serve a first touch from HOT. Mean latency roughly halves, because cold
  first reads of large files dominate latency. Recency plateaus near 0.98 hit
  rate and about 3.2 s mean latency even at a 336 h timeout.
- It costs storage: about +40-60% over recency at the same timeout.
- Semantic edges beat random edges over the same ACL groups at every timeout:
  about 15-20% less storage, 5-12% lower latency, prefetch precision 0.25-0.29
  vs 0.17-0.20. At coupling 0 the semantic advantage narrows (equal precision
  from 24 h up) but storage stays lower.
- Limits: synthetic trace, one seed, configured tier latencies and costs, and
  the storage charge uses end-of-interval placement.

### With a HOT capacity limit

`GraphPrefetchPolicy(hot_capacity_bytes=...)`: a read file evicts HOT files to
WARM until it fits (unread prefetches first, then least recently touched); a
prefetch only takes free HOT space, otherwise it lands in WARM. HOT idle 72 h.
Total stored bytes 217 GB.

| HOT cap | Coupling | Hit: recency / graph / random | Latency ms: recency / graph / random |
|---|---|---|---|
| 2% | 0.8 | 0.259 / **0.310** / 0.272 | 10,921 / **9,230** / 9,428 |
| 5% | 0.8 | 0.476 / **0.517** / 0.487 | 8,464 / **6,841** / 6,985 |
| 10% | 0.8 | 0.694 / **0.702** / 0.685 | 6,225 / **4,755** / 5,000 |
| 20% | 0.8 | **0.882** / 0.876 / 0.862 | 4,357 / **2,810** / 3,150 |
| 5% | 0 (control) | 0.544 / 0.533 / **0.545** | 7,739 / 6,583 / **6,618** |
| 20% | 0 (control) | **0.901** / 0.875 / 0.885 | 3,769 / **2,672** / 2,772 |

- Prefetch cuts mean latency 15-35% at every cap, mostly by moving cold
  neighbors to WARM (10x the COLD throughput) when HOT is full.
- With demand coupled to content, the semantic graph also raises HOT hit rate
  at tight caps (2-10%) and beats random ACL-group neighbors on hit rate,
  latency (2-11% lower), storage (about 12% lower) and prefetch precision
  (about 2x). In the control, semantic and random neighbors tie.
- Costs: storage about +30-40% over recency (WARM placements) and 3-6x the
  evictions, i.e. migration churn that real transition fees would charge for.
- The first version let prefetches evict read files (LRU saw both as just
  touched); evicting unread prefetches first fixed a hit-rate loss.

## Tiered store MVP (`tiering store`)

`src/tiering/store.py`: `TieredStore` keeps bytes in `hot/`, `warm/`, `cold/`
directories (each can point elsewhere, e.g. COLD on a NAS), an atomically
written `index.json`, and `access_log.jsonl`. Put and get go through
`GraphPrefetchPolicy`; `tick` demotes idle files. The semantic graph updates
incrementally on put and refits TF-IDF after 25% corpus growth. File ids are
validated (no path traversal); an index left stale by a crash mid-move is
repaired on read.

```powershell
tiering store --root data/store --hot-capacity-gb 50 put report.pdf --groups finance
tiering store --root data/store get report.pdf --out copy.pdf
tiering store --root data/store tick
tiering store --root data/store status
```

End-to-end replay through the real on-disk store (`python -m
src.tiering.experiments.store_benchmark --coupling 0.8`): 7-day window,
pre-window files put at creation time, payloads scaled 1/10,000, latency
modelled from real sizes.

| Run | HOT hit | First touch from HOT | Modelled latency (ms) | get / put / tick (ms) | Cap respected | Misplaced |
|---|---:|---:|---:|---|---|---:|
| Recency | 0.947 | 0.128 | 3,935 | 22 / 9 / 5 | - | 0 |
| Graph k=5 | 0.984 | 0.672 | 1,696 | 24 / 15 / 6 | - | 0 |
| Recency, 5% cap | 0.461 | 0.023 | 8,627 | 39 / 31 / 15 | yes | 0 |
| Graph k=5, 5% cap | 0.508 | 0.076 | 6,707 | 20 / 25 / 9 | yes | 0 |

The store reproduces the simulator's direction. Throughput is the limit: the
full `index.json` rewrite per operation costs 20-40 ms with about 1,800 files.
Move the index to SQLite before larger corpora or concurrent writers.

## MVP verification

```powershell
tiering models promote results/models/product_ablation_enron.joblib --note "Enron Government Affairs"
tiering predict examples/mvp_input_d_enron.json --run D_combined_retrieval
tiering predict examples/mvp_input_d_enron.json
tiering monitor
tiering models rollback
```

- **Registry:** `models promote` created `product_ablation_enron-2ae56038a9a3`.
  `rollback` returned to the previous version, and re-promotion
  reactivated the Enron model.
- **D:** HOT 0.101 / WARM 0.332 / COLD 0.568 (COLD), any-access 0.064, with
  one authorized neighbor (cosine 0.490, 2 past accesses in 24h). The
  same-topic document with a different access group was excluded.
- **C:** WARM 0.526, any-access 0.075.
- **Logging and monitoring:** both predictions were logged with their model
  version. `monitor` compared served shares to D's validation tier mix and
  withheld the alarm below 50 predictions.
- **Vocabulary:** `examples/mvp_input_d.json` still finds its neighbor
  (0.421) under the synthetic model. Under the Enron model its corpus texts
  share no vocabulary with the 256-term TF-IDF, so they embed to zero and no
  neighbor is returned. This is expected; use the Enron-style example with
  the Enron model.

## Limits that still apply

- The target is a thread re-engagement proxy on email, not file-access logs.
  Results are for this corpus and three related mailboxes. A second mailbox
  set or horizon has not been run.
- Only TF-IDF (256 terms) is evaluated. Sentence embeddings are an optional
  dependency and are not tested.
- The MVP is recommendation-only, single process, with a file-based
  registry. It has no HTTP service, authentication or live ACL/catalog
  connection. Owner roles in `docs/OPERATIONS.md` are unassigned.
- Prices are list prices, re-verify before any cost claim. The cost
  decisions are myopic.
