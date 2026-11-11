# Wikipedia page-view and article-text sample

- Page-view sample date: 2026/10/28.
- Page views: `vtasca/wikipedia-pageviews` on Hugging Face, sourced from Wikimedia page-view data.
- Current article titles, page IDs, revision timestamps and up to 1,200 characters of introduction text: English Wikipedia Action API, retrieved 2026-10-31.
- Historical page text: English Wikipedia Action API rendered revision content, latest revision strictly before 2026-10-28 00:00 UTC, fetched for the 100 sampled page IDs.
- Article content is available under Wikipedia contributor licenses (CC BY-SA and GFDL; see each article history/footer). Page-view data is CC0.
- This sample contains only the 100 top-ranked articles for one day. Title coverage was 100/100, with no ambiguous normalized titles. It is for join/pipeline feasibility, not a representative demand sample.
- Historical retrieval produced a pre-cutoff revision and text for 38/100 pages (38%). The remaining 62 had no revision before the cutoff and therefore no semantic text available at prediction time. The usable subset is selection-biased; it is not a representative dataset for training or evaluating the full pageview sample.
- Detailed results: `results/metrics/wikipedia_top100_join_report.json` measures current-text timing; `results/metrics/wikipedia_top100_asof_join_report.json` measures the historical snapshot. Historical text is stored in `data/processed/wikipedia_top100_asof.jsonl`.
