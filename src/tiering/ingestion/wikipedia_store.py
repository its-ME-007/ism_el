"""Wikipedia as a real-demand document store for the tiering benchmarks.

- Articles: members of seed categories; each seed category is an ACL-like
  access group (an article can sit in several).
- Content: wikitext of the last revision *before* the window start, so edits
  made during the window (often driven by the news that drives views) never
  leak into the text. Object size is that revision's byte size.
- Demand: Wikimedia hourly pageview dumps (complete per-article counts,
  en + en.m, user and bot traffic not separated in dumps), streamed and
  filtered to the sample; only the per-hour filtered counts are cached.
- Events: hourly counts Poisson-thinned by ``thin`` so the simulators stay
  tractable; relative timing and burst shape are preserved. Redirect views are
  not credited to their target.

    python -m src.tiering.ingestion.wikipedia_store fetch --start 2026-11-08 --days 8
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import gzip
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen

import numpy as np

from src.tiering.ingestion.enron_adapter import EnronBundle
from src.tiering.ingestion.fetch_wikipedia_asof import USER_AGENT, api_get
from src.tiering.ingestion.schemas import AccessEvent, Category, FileDocument

DUMPS = "https://dumps.wikimedia.org/other/pageviews"
CACHE = Path("data/raw/wikipedia/store")
SEED_CATEGORIES = [
    "2026 United States elections", "2026 in American politics",
    "National Football League teams", "National Basketball Association teams",
    "Premier League clubs", "Formula One drivers", "Tennis players",
    "Large language models", "Artificial intelligence companies",
    "Cryptocurrencies", "Companies in the Dow Jones Industrial Average",
    "Smartphones", "Video game consoles", "2026 films", "2026 video games",
    "American television series", "Netflix original programming",
    "Taylor Swift songs", "Chemical elements", "Planets of the Solar System",
    "Programming languages", "Infectious diseases", "Member states of the European Union",
    "Capitals in Europe", "Nobel laureates in Physics", "Presidents of the United States",
    "Hurricanes in the United States", "Volcanoes of Iceland",
]


def _members(category: str, limit: int) -> list[dict]:
    data = api_get({"action": "query", "list": "categorymembers",
                    "cmtitle": f"Category:{category}", "cmnamespace": 0,
                    "cmlimit": limit, "format": "json"})
    return data.get("query", {}).get("categorymembers", [])


def _revision_before(page_id: int, cutoff: datetime) -> dict | None:
    data = api_get({"action": "query", "pageids": page_id, "prop": "revisions",
                    "rvstart": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"), "rvdir": "older",
                    "rvlimit": 1, "rvprop": "ids|timestamp|size|content",
                    "rvslots": "main", "format": "json", "formatversion": 2})
    pages = data.get("query", {}).get("pages", [])
    revisions = pages[0].get("revisions", []) if pages else []
    return revisions[0] if revisions else None


def _plain(wikitext: str) -> str:
    text = re.sub(r"<ref[^>]*?/>|<ref.*?</ref>", " ", wikitext, flags=re.S)
    while True:  # strip templates innermost-first until none are left
        stripped = re.sub(r"\{\{[^{}]*\}\}", " ", text)
        if stripped == text:
            break
        text = stripped
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", text)
    text = re.sub(r"<[^>]+>|'{2,}|={2,}|\[https?://\S+|\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def fetch_members(per_category: int = 120, min_members: int = 10, pause: float = 1.0) -> dict:
    """Seed-category membership (page id -> title, groups); cached."""
    path = CACHE / "members.json"
    if path.exists():
        return {int(k): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}
    groups: dict[int, dict] = {}
    for category in SEED_CATEGORIES:
        members = _members(category, per_category)
        time.sleep(pause)
        if len(members) < min_members:  # members are mostly subcategories
            print(f"skip {category}: {len(members)} articles", flush=True)
            continue
        print(f"{category}: {len(members)} articles", flush=True)
        for member in members:
            entry = groups.setdefault(member["pageid"], {"title": member["title"], "groups": []})
            entry["groups"].append(category)
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(groups), encoding="utf-8")
    return groups


def fetch_sample(start: datetime, per_category: int = 120, pause: float = 0.5) -> Path:
    """Fetch each member's pre-window text; resumable."""
    path = CACHE / "articles.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(line)["page_id"]
                for line in path.read_text(encoding="utf-8").splitlines()}
    groups = fetch_members(per_category)
    with path.open("a", encoding="utf-8") as out:
        for i, (page_id, entry) in enumerate(groups.items()):
            if page_id in done:
                continue
            revision = _revision_before(page_id, start)
            time.sleep(pause)
            if not revision:
                continue  # page did not exist before the window
            out.flush() if i % 50 == 0 else None
            out.write(json.dumps({
                "page_id": page_id, "title": entry["title"], "groups": entry["groups"],
                "revision_id": revision["revid"], "timestamp": revision["timestamp"],
                "size": revision["size"],
                "text": _plain(revision["slots"]["main"]["content"])[:20000],
            }, ensure_ascii=False) + "\n")
            if i % 200 == 0:
                print(f"text {i}/{len(groups)}", flush=True)
    return path


def _hour_counts(hour: datetime, titles: set[str]) -> dict[str, int]:
    cache = CACHE / "hours" / f"{hour:%Y%m%d-%H}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    url = f"{DUMPS}/{hour:%Y}/{hour:%Y-%m}/pageviews-{hour:%Y%m%d-%H}0000.gz"
    counts: dict[str, int] = {}
    for attempt in range(4):
        try:
            with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=120) as resp:
                with gzip.GzipFile(fileobj=resp) as stream:
                    for raw in stream:
                        if not (raw.startswith(b"en ") or raw.startswith(b"en.m ")):
                            continue
                        parts = raw.split(b" ")
                        title = parts[1].decode("utf-8", "replace")
                        if title in titles:
                            counts[title] = counts.get(title, 0) + int(parts[2])
            break
        except OSError as exc:  # network hiccup: retry with backoff
            if attempt == 3:
                raise
            print(f"{hour:%Y%m%d-%H}: {exc}; retrying", flush=True)
            counts = {}
            time.sleep(20 * (attempt + 1))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(counts), encoding="utf-8")
    return counts


def fetch_pageviews(start: datetime, days: int, workers: int = 3) -> None:
    titles = {entry["title"].replace(" ", "_") for entry in fetch_members().values()}
    hours = [start + timedelta(hours=h) for h in range(days * 24)]
    # Wikimedia asks dump users to keep to a few parallel connections.
    with ThreadPoolExecutor(max_workers=min(workers, 3)) as pool:
        for i, _ in enumerate(pool.map(lambda h: _hour_counts(h, titles), hours)):
            if i % 12 == 0:
                print(f"hours {i + 1}/{len(hours)}", flush=True)


def load_wikipedia_bundle(start: datetime, days: int, thin: float | None = None,
                          target_events: int = 300_000, seed: int = 42,
                          with_text: bool = False) -> EnronBundle:
    """``with_text=False`` (default) leaves ``content`` empty: the co-access
    pipeline never reads it and the text is most of the bundle's memory."""
    rows = []
    with (CACHE / "articles.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not with_text:
                row["text"] = ""
            rows.append(row)
    hours = [start + timedelta(hours=h) for h in range(days * 24)]
    missing = [h for h in hours if not (CACHE / "hours" / f"{h:%Y%m%d-%H}.json").exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} hours not fetched (first {missing[0]:%Y%m%d-%H}); "
                                "run the pageviews step first")
    counts = {h: _hour_counts(h, set()) for h in hours}
    by_title = {row["title"].replace(" ", "_"): row for row in rows}
    total = sum(n for c in counts.values() for t, n in c.items() if t in by_title)
    thin = thin or min(1.0, target_events / max(total, 1))
    rng = np.random.default_rng(seed)
    files = [FileDocument(
        file_id=f"w{row['page_id']}", filename=row["title"], category=Category.RESEARCH,
        content=row["text"], size_bytes=int(row["size"]),
        created_at=datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")),
        modified_at=datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00")),
        metadata={"access_groups": row["groups"], "file_type": "document",
                  "title": row["title"]},
    ) for row in rows]
    traces = []
    for hour, hour_counts in counts.items():
        for title, n in hour_counts.items():
            row = by_title.get(title)
            if row is None:
                continue
            for offset in rng.random(rng.binomial(n, thin)):
                traces.append(AccessEvent(timestamp=hour + timedelta(hours=float(offset)),
                                          file_id=f"w{row['page_id']}", operation="read"))
    traces.sort(key=lambda e: e.timestamp)
    stats = {"generator": "wikipedia_store", "articles": len(files),
             "seed_categories": len(SEED_CATEGORIES), "hours": len(hours),
             "pageviews_in_sample": total, "thinning": thin,
             "access_events_total": len(traces),
             "window": [start.isoformat(), (start + timedelta(days=days)).isoformat()]}
    return EnronBundle(
        files, traces, start, [0], [0], [1], None, stats, suffix="wikipedia",
        label="Wikipedia articles (pre-window text, seed categories as ACL groups, "
              "hourly pageview dumps)",
        note=("Real hourly demand from Wikimedia pageview dumps, Poisson-thinned. "
              "Web readership, not enterprise file access."))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["fetch", "sample", "pageviews"])
    parser.add_argument("--start", default="2026-11-08")
    parser.add_argument("--days", type=int, default=8)
    parser.add_argument("--per-category", type=int, default=120)
    args = parser.parse_args()
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    if args.step in {"fetch", "sample"}:
        fetch_sample(start, args.per_category)
    if args.step in {"fetch", "pageviews"}:
        fetch_pageviews(start, args.days)


if __name__ == "__main__":
    main()
