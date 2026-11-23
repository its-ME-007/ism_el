# Paper Abstract (draft)

**Status:** draft, 2026-11-21. Standard regime (EN 150–250 words, zh-TW 300–500 characters, 5–7 keywords). Numbers come from `python -m src.tiering.experiments.graph_tiering --wikipedia` (one seed, modelled latency). The research gap and references are kept separately in `RELATED_WORK.md`, because the abstract carries no citations.


### English Abstract

Tiered storage keeps recently read files on fast media and demotes the rest, so the first read of a cold file is slow. Prefetching related files can hide that delay, but wrong guesses cost extra data movement. This study tests whether prefetch links learned from read timing outperform links learned from text content. A link from file A to file B is learned online when B is read shortly after A within the same access-control group; files with too few links are filled in with random same-group neighbours, and a break-even rule admits a prefetch only when its expected saving exceeds the cost of a wasted move, both in input/output (I/O) milliseconds. We replayed about 55,000 pageview events from eight days of 1,680 Wikipedia articles in 28 categories, in a trace-driven simulator, not on a deployed file server. Reads cost 1, 10 or 100 ms on HOT, WARM or COLD tiers plus transfer at 1,000, 100 or 10 Mbit/s. Text-similarity neighbours did no better than random same-group neighbours. With HOT capped at 5% of stored bytes, mean modelled read latency was 18.38 ms for co-access links with random fill, against 18.78 ms for random neighbours and 20.84 ms for recency-only tiering. Under the same cap, break-even admission cut migrations by 23% relative to co-access without it and used the least device time among prefetch policies, yet recency-only tiering still used less. Results come from one random seed and modelled latency, so they motivate, but do not settle, calibrated cost-aware prefetching.

**Keywords**: tiered storage, prefetching, co-access, access-control groups, cost-aware admission, Wikipedia pageviews

---

### Chinese Abstract

分層儲存將近期讀取的檔案留在高速媒體、其餘降級至較慢層，因此冷檔案的首次讀取延遲偏高。預取相關檔案可緩解此問題，但誤判會增加搬移。本研究檢驗：由讀取時序學得的預取關聯，是否優於由文字內容學得的關聯。同一存取權限群組內，若檔案B常在檔案A之後不久被讀取，即線上建立A至B的預取連結；連結不足者以同群組隨機鄰居補足；損益平衡規則僅在預期節省的讀取時間大於一次白做搬移的成本時才預取（皆以I/O毫秒計）。我們以1,680篇維基百科文章（分屬28個分類）為對象，以追蹤驅動模擬器（非實際檔案伺服器）重播八天內抽樣的約55,000筆瀏覽事件；HOT、WARM、COLD層每次讀取計1、10、100毫秒，另加傳輸時間（每秒1,000、100、10百萬位元）。文字相似度鄰居未優於同群組隨機鄰居。當HOT層上限為儲存量5%時，平均模擬讀取延遲為：共同存取連結加隨機補足18.38毫秒、隨機鄰居18.78毫秒、僅依最近讀取的分層20.84毫秒。在同一上限下，損益平衡規則使搬移次數較未採用者減少23%，並為各預取策略中裝置時間最低；然而僅依最近讀取的分層用量仍較少。結果僅來自單一隨機種子與模擬延遲，屬初步證據。

**Keywords**: 分層儲存、預取、共同存取、存取權限群組、成本感知預取、維基百科瀏覽量
