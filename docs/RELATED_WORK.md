# Related Work and Research Gap (draft)

**Status:** draft, 2026-11-21. This list comes from a quick three-way scan (WHY / HOW / WHAT), not a systematic search. Every entry carries a verification status. Do not cite an entry marked **unverified** until its full bibliographic record has been checked against the source (for example with `/ars-citation-check`).

## Research gap

Correlation-based prefetching is well studied: block-sequence mining (C-Miner), access sequences combined with process and user attributes (FARMER), correlated-object clustering on object stores (Pacaca, Sheepdog), and ML-guided admission and prefetching for flash caches (Baleen). Learned placement in hybrid storage (Sibyl) and cost-gated tier moves (xTier) are also covered.

The scan found no work that combines all four of the following:

1. Cross-object prefetching from links learned online from read timing, restricted to files that share an access-control group.
2. A prefetch decision weighed against the cost of a wasted move (here, a break-even rule in I/O-ms usage units).
3. Tiering decisions at an hour-to-day time scale rather than a millisecond-to-second cache scale.
4. A control that tests whether content similarity adds anything over the access group alone (here, random same-group neighbours).

This is a lead from a quick scan, not an established gap. Before claiming novelty, check FAST, USENIX ATC and SoCC proceedings from 2022 to 2026.

**How this work differs:** on a Wikipedia pageview benchmark, text-similarity neighbours did no better than random same-group neighbours. Links learned from read timing picked better neighbours (prefetch precision about 0.23–0.25 against 0.11 for random). Under a capped HOT tier, recency-only tiering still used the fewest usage units. See `PAPER_ABSTRACT.md` for the figures.

## References

| # | Reference | Status | Link |
|---|---|---|---|
| 1 | Li, Z., Chen, Z., Srinivasan, S. M., & Zhou, Y. (2004). C-Miner: Mining block correlations in storage systems. *Proceedings of the 3rd USENIX Conference on File and Storage Technologies (FAST '04)*. | Paper retrieved; author list from memory, check it | [PDF](https://usenix.org/event/fast04/tech/full_papers/li/li.pdf) |
| 2 | Xia, P., Feng, D., Jiang, H., Tian, L., & Wang, F. (2008). FARMER: A novel approach to file access correlation mining and evaluation reference model for optimizing peta-scale file system performance. *Proceedings of HPDC '08*. | Paper retrieved; authors and full title from memory, check them | [PDF](https://ranger.uta.edu/~jiang/publication/Conferences/2008/2008-HPDC-FARMER.pdf) |
| 3 | Wong, D. L.-K., Wu, H., Molder, C., et al. (2024). Baleen: ML admission & prefetching for flash caches. *22nd USENIX Conference on File and Storage Technologies (FAST '24)*, 347–371. | Verified | [USENIX](https://www.usenix.org/conference/fast24/presentation/wong) |
| 4 | Singh, G., et al. (2022). Sibyl: Adaptive and extensible data placement in hybrid storage systems using online reinforcement learning. *Proceedings of the 49th International Symposium on Computer Architecture (ISCA '22)*, 320–336. https://doi.org/10.1145/3470496.3527442 | Verified | [arXiv](https://arxiv.org/abs/2205.07394) |
| 5 | Pacaca: correlated-object mining and prefetching on cloud storage. *MASCOTS 2018*. | **Unverified**: authors and exact title unknown | [Paper](https://www.csc.lsu.edu/~bhou/publish/mascots18) |
| 6 | Correlation-aware object prefetching for object-based storage. *ICA3PP 2010*. https://doi.org/10.1007/978-3-642-13119-6_21 | **Unverified**: title and authors unknown; DOI from a link | [Unpaywall](https://unpaywall.org/10.1007%2F978-3-642-13119-6_21) |
| 7 | Learned I/O pattern discovery for prefetching on the Sheepdog object store (2020). | **Unverified**: title and authors unknown | [PDF](https://par.nsf.gov/servlets/purl/10290647) |
| 8 | xTier: Intelligent tiering for CXL-enabled memory (2026). arXiv:2609.27266. | **Unverified**: seen only in search snippets | [arXiv](https://arxiv.org/pdf/2609.27266) |
| 9 | TIDAL: Recovering temporal phase for cloud block storage placement from LLM-derived semantics (2026). arXiv:2605.18066. | **Unverified**: seen only in search snippets | [arXiv](https://arxiv.org/pdf/2605.18066) |
| 10 | Griffioen, J., & Appleton, R. (1994). Reducing file system latency using a predictive approach. *USENIX Summer Technical Conference*. | **Unverified**: from memory, not retrieved | – |
| 11 | Kroeger, T. M., & Long, D. D. E. Prediction of file system actions from prior events (PPM-based predictor). | **Unverified**: from memory; year and venue not checked | – |
| 12 | Cao, P., Felten, E. W., Karlin, A. R., & Li, K. (1995). A study of integrated prefetching and caching strategies. *ACM SIGMETRICS '95*. | **Unverified**: from memory, not retrieved | – |
| 13 | Patterson, R. H., Gibson, G. A., Ginting, E., Stodolsky, D., & Zelenka, J. (1995). Informed prefetching and caching (TIP). *ACM SOSP '95*. | **Unverified**: from memory, not retrieved | – |

Pricing context, if a cloud-cost version is written later: [AWS S3 Glacier storage classes](https://docs.aws.amazon.com/AmazonS3/latest/userguide/glacier-storage-classes.html).
