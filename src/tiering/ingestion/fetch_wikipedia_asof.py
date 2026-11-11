"""Fetch Wikipedia page text from the latest revision before a cutoff date.

This is intended for small, bounded research samples, not bulk dumps. It uses
the public MediaWiki Action API and pauses between requests.

Example::

    python -m src.tiering.ingestion.fetch_wikipedia_asof \
        --documents data/processed/wikipedia_top100_text.jsonl \
        --cutoff 2026-10-28 --output data/processed/wikipedia_top100_asof.jsonl
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, time, timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import time as time_module
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "ISM-TieringResearch/0.1 (bounded research sample; contact: project maintainer)"


class TextExtractor(HTMLParser):
    """Extract visible text from MediaWiki-rendered HTML."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        classes = dict(attrs).get("class", "")
        if tag in {"script", "style", "sup"} or "reference" in classes.split():
            self.skip_depth += 1
        elif tag in {"p", "li", "h1", "h2", "h3", "h4", "br", "div"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "sup"} and self.skip_depth:
            self.skip_depth -= 1
        elif tag in {"p", "li", "h1", "h2", "h3", "h4", "div"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)

    def text(self) -> str:
        value = "".join(self.parts)
        value = re.sub(r"[ \t\xa0]+", " ", value)
        value = re.sub(r" *\n *", "\n", value)
        value = re.sub(r"\n{3,}", "\n\n", value)
        return value.strip()


def api_get(params: dict) -> dict:
    url = API + "?" + urlencode(params)
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(6):
        try:
            with urlopen(request, timeout=45) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            if exc.code not in {429, 503} or attempt == 5:
                raise
            retry_after = exc.headers.get("Retry-After")
            try:
                delay = max(5.0, float(retry_after)) if retry_after else 15.0 * (attempt + 1)
            except ValueError:
                delay = 15.0 * (attempt + 1)
            print(f"Wikimedia returned HTTP {exc.code}; retrying after {delay:g}s")
            time_module.sleep(delay)
    raise RuntimeError("Wikimedia API retry loop ended unexpectedly")


def revisions_before(page_ids: list[int], cutoff: datetime) -> dict[int, dict]:
    data = api_get({
        "action": "query", "pageids": "|".join(map(str, page_ids)), "prop": "revisions",
        "rvstart": cutoff.strftime("%Y-%m-%dT%H:%M:%SZ"), "rvdir": "older",
        "rvlimit": 1, "rvprop": "ids|timestamp", "format": "json",
    })
    result = {}
    for key, page in data.get("query", {}).get("pages", {}).items():
        revisions = page.get("revisions", [])
        if revisions:
            result[int(key)] = revisions[0]
    return result


def rendered_text(revision_id: int) -> str:
    data = api_get({
        "action": "parse", "oldid": revision_id, "prop": "text",
        "format": "json", "formatversion": 2,
    })
    parser = TextExtractor()
    parser.feed(data.get("parse", {}).get("text", ""))
    return parser.text()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documents", type=Path, required=True,
                        help="JSONL containing page_id and title")
    parser.add_argument("--cutoff", type=date.fromisoformat, required=True,
                        help="Use the latest revision strictly before this date")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional maximum number of pages; 0 means all")
    parser.add_argument("--pause", type=float, default=1.5,
                        help="Seconds between API requests (default: 1.5)")
    args = parser.parse_args()

    cutoff = datetime.combine(args.cutoff, time.min, tzinfo=timezone.utc)
    pages = []
    with args.documents.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                pages.append(json.loads(line))
    if args.limit:
        pages = pages[:args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    prior = {}
    if args.output.exists():
        with args.output.open(encoding="utf-8") as existing:
            for line in existing:
                if line.strip():
                    row = json.loads(line)
                    if row.get("status") == "ok":
                        prior[row.get("page_id")] = row
    need_revisions = [int(page["page_id"]) for page in pages
                      if page.get("page_id") not in prior]
    revisions = {}
    for start in range(0, len(need_revisions), 50):
        revisions.update(revisions_before(need_revisions[start:start + 50], cutoff))
        time_module.sleep(args.pause)
    successes = 0
    errors = 0
    with args.output.open("w", encoding="utf-8", newline="\n") as output:
        for index, page in enumerate(pages, start=1):
            if page.get("page_id") in prior:
                output.write(json.dumps(prior[page["page_id"]], ensure_ascii=False) + "\n")
                successes += 1
                continue
            record = {
                "page_id": page.get("page_id"), "title": page.get("title"),
                "url": page.get("url"), "cutoff_exclusive": cutoff.isoformat(),
                "source": "English Wikipedia Action API historical revision",
            }
            try:
                revision = revisions.get(int(page["page_id"]))
                if revision:
                    modified = datetime.fromisoformat(revision["timestamp"].replace("Z", "+00:00"))
                    if modified >= cutoff:
                        raise ValueError("API returned a revision at or after the exclusive cutoff")
                    time_module.sleep(args.pause)
                    record.update({
                        "revision_id": revision["revid"],
                        "date_modified": revision["timestamp"],
                        "text": rendered_text(int(revision["revid"])),
                        "status": "ok",
                    })
                    successes += 1
                else:
                    record["status"] = "no_revision_before_cutoff"
            except (HTTPError, URLError, TimeoutError, KeyError, ValueError) as exc:
                record.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                errors += 1
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            if index % 10 == 0 or index == len(pages):
                print(f"Processed {index}/{len(pages)}; successful={successes}, errors={errors}")
            time_module.sleep(args.pause)


if __name__ == "__main__":
    main()
