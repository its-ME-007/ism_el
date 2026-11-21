# Project notes for Claude Code

- Current research line: co-access prefetch for tiered storage. The primary results, how to reproduce them and the open items are in `docs/COACCESS_RESULTS.md`. Read it before changing tiering code or results.
- Paper drafts: `docs/PAPER_ABSTRACT.md` and `docs/RELATED_WORK.md`. References marked unverified must not be cited until checked.
- Result JSONs in `results/metrics/` are tracked. Models, datasets and `data/raw/*` are not; the Wikipedia data must be fetched first.
- Tests: `python -m pytest -q`. On Windows, add `--basetemp=<writable dir>` if the temp folder is blocked.
