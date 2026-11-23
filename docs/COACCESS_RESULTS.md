# Co-access Prefetch: Primary Results

**Status:** 2026-11-21, branch `codex/serve-retrieval-run-d`. These are the paper's primary results. The abstract is in `PAPER_ABSTRACT.md`, and the research gap and references are in `RELATED_WORK.md`.

## What was tested

Tiering policies on a Wikipedia pageview store: 1,680 articles in 28 categories, with each category used as an access-control group. Hourly pageviews over 8 days were thinned to about 55k events. Day 1 is warm-up and days 2–8 (about 48.7k reads) are evaluated. Latency is modelled from `TIER_CONFIGS` in `src/tiering/experiments/product_ablation.py`. One thinning seed (42).

- `recency`: promote a file on read, demote it when idle.
- `random_k5`: also prefetch 5 random files from the same group (the control).
- `coaccess_k5`: prefetch files that are usually read within 1 h after this one, learned online (`src/tiering/coaccess.py`).
- `coaccess_fb_k5`: co-access, with random same-group files filling the gap when there are too few learned links.
- `coaccess_k5_c10` / `_c30`, `coaccess_fb_k5_c30`: links must reach confidence 0.10 or 0.30.
- `coaccess_be_k5`: break-even admission in I/O-ms (`GraphPrefetchPolicy(tier_configs=...)`). A prefetch must expect to save more read time than a wasted move pair would cost.

Usage units (`usage_units_io_ms`) = total read latency + migration cost, both in ms of device time. Storage cost is reported separately.

## Results (5% HOT cap)

| Policy | Latency | Usage | Migrations | Prefetch precision |
|---|---|---|---|---|
| recency | 20.84 ms | 2.34M | 74k | – |
| random_k5 | 18.78 | 3.69M | 207k | 0.11 |
| coaccess_k5 | 20.51 | 3.05M | 138k | 0.25 |
| coaccess_fb_k5 | **18.38** | 3.15M | 154k | 0.23 |
| coaccess_be_k5 | 20.51 | 2.71M | 106k | 0.34 |

- Co-access with fallback has the lowest latency at the 5%, 10% and 20% caps (2–5% faster than random) and ties random at 2%. It needs 22–33% fewer migrations than random.
- Break-even admission keeps co-access latency and cuts migrations by 23% at the 5% cap (21–31% across caps). It has the lowest usage of all prefetch policies at every cap.
- Under a cap, recency still uses the fewest usage units. With no cap, prefetching uses fewer than recency (0.38M against 0.44M at 72 h idle).
- The admission confidence is miscalibrated: the rule needs about 0.67 to admit, but measured precision is about 0.34.
- Semantic (TF-IDF) neighbours did no better than random same-group neighbours (`graph_tiering_wikipedia.json`). That is why semantics was dropped.
- Removing text from the store made it about 10x faster and smaller, with identical placement.

**Limitations:** one seed and one dataset, simulated latency. The on-disk store was checked for correctness on 1 day only (cap respected, 0 misplaced files); that result is not in `results/metrics/`.

## Files

- `results/metrics/graph_tiering_wikipedia_coaccess.json`: primary results, all 9 policies × 5 idle timeouts and 4 HOT caps.
- `results/metrics/graph_tiering_wikipedia.json`: earlier semantic vs random run.
- `results/figures/*.pdf|png`: paper figures (latency, migrations, usage and precision against HOT cap, plus the latency/migration trade-off at 5%). Regenerate with `python -m src.tiering.experiments.plot_coaccess` (needs the `visualization` extra). In the latency figure, the co-access and break-even lines overlap because their latency is identical.
- `docs/coaccess_results.html`: a one-page summary of the results to show in a browser.
- Code: `src/tiering/coaccess.py`, `src/tiering/semantic/graph.py` (`GraphPrefetchPolicy`), `src/tiering/experiments/graph_tiering.py`, `src/tiering/store.py`.

## Reproduce

The Wikipedia data is not in git (`data/raw/*` is ignored). Fetch it first. The fetch downloads hourly pageview dumps from dumps.wikimedia.org and takes a while.

```powershell
python -m src.tiering.ingestion.wikipedia_store fetch --start 2026-11-08 --days 8
python -m src.tiering.experiments.graph_tiering --wikipedia            # about 20 min, writes the JSON above
python -m src.tiering.experiments.graph_tiering --wikipedia --semantic  # adds the TF-IDF baseline; slow, about 250 MB more memory
python -m pytest -q
```

On Windows, if pytest fails with `PermissionError: [WinError 5]` on its temp folder, add `--basetemp=<a writable folder>`.

## Open items

1. Run 3–5 seeds and 2–3 different weeks, and report confidence intervals.
2. Calibrate confidence: the learning window is 1 h, but usefulness is measured over 24 h.
3. Count HOT eviction churn in the break-even rule.
4. Verify the unverified references in `RELATED_WORK.md`.
