"""Measure title-based overlap between daily Wikipedia views and article text.

Example::

    python -m src.tiering.ingestion.validate_wikipedia_join \
        --pageviews data/raw/wikipedia/pageviews.csv \
        --documents data/raw/wikipedia/finewiki-en.parquet \
        --output results/metrics/wikipedia_join_report.json

This utility only reports join feasibility. It does not interpret missing
top-100 page-view rows as zero accesses or create training examples.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
import unicodedata
from urllib.parse import unquote


def normalize_title(value: str) -> str:
    """Normalize obvious title encoding differences without fuzzy matching."""
    decoded = unquote(str(value or "")).replace("_", " ")
    normalized = unicodedata.normalize("NFKC", decoded).casefold()
    return " ".join(normalized.split())


def iter_documents(path: Path):
    """Yield title/page ID from JSONL or Parquet without loading full text."""
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                title = record.get("title")
                if title:
                    yield (str(title), record.get("page_id", record.get("id")),
                           record.get("date_modified", record.get("modified_at")),
                           record.get("status"))
        return

    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise RuntimeError(
                "Reading Parquet requires pyarrow; install with `pip install -e '.[data]'`."
            ) from exc
        parquet = pq.ParquetFile(path)
        available = set(parquet.schema.names)
        if "title" not in available:
            raise ValueError("Document Parquet must have a 'title' column")
        id_column = "page_id" if "page_id" in available else (
            "id" if "id" in available else None
        )
        modified_column = "date_modified" if "date_modified" in available else None
        columns = ["title"] + ([id_column] if id_column else []) + (
            [modified_column] if modified_column else []
        )
        for batch in parquet.iter_batches(columns=columns, batch_size=8192):
            titles = batch.column(batch.schema.get_field_index("title")).to_pylist()
            ids = (batch.column(batch.schema.get_field_index(id_column)).to_pylist()
                   if id_column else [None] * len(titles))
            modified = (batch.column(batch.schema.get_field_index(modified_column)).to_pylist()
                        if modified_column else [None] * len(titles))
            yield from zip((str(title or "") for title in titles), ids, modified,
                           [None] * len(titles))
        return

    raise ValueError("Documents input must be .jsonl, .ndjson, or .parquet")


def validate_join(pageviews_path: Path, documents_path: Path) -> dict:
    """Summarize exact-normalized title join coverage and ambiguity."""
    documents_by_title = defaultdict(list)
    document_rows = 0
    for title, page_id, date_modified, document_status in iter_documents(documents_path):
        document_rows += 1
        key = normalize_title(title)
        if key:
            documents_by_title[key].append({
                "title": title, "page_id": page_id, "date_modified": date_modified,
                "status": document_status,
            })

    total_view_rows = 0
    unique_view_titles = set()
    matched_rows = 0
    unmatched_rows = 0
    ambiguous_rows = 0
    matched_titles = set()
    ambiguous_titles = set()
    unmatched_examples = []
    time_safe_rows = 0
    same_day_revision_rows = 0
    later_revision_rows = 0
    missing_revision_rows = 0
    no_pre_cutoff_revision_rows = 0
    with pageviews_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"article", "date", "views"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Pageviews CSV must contain article, date, and views columns"
            )
        for row in reader:
            total_view_rows += 1
            title = row.get("article", "")
            key = normalize_title(title)
            unique_view_titles.add(key)
            matches = documents_by_title.get(key, [])
            if len(matches) == 1:
                matched_rows += 1
                matched_titles.add(key)
                modified = matches[0].get("date_modified")
                if matches[0].get("status") == "no_revision_before_cutoff":
                    no_pre_cutoff_revision_rows += 1
                elif not modified:
                    missing_revision_rows += 1
                else:
                    modified_date = datetime.fromisoformat(
                        str(modified).replace("Z", "+00:00")
                    ).date()
                    target_date = date.fromisoformat(row["date"].replace("/", "-"))
                    if modified_date < target_date:
                        time_safe_rows += 1
                    elif modified_date == target_date:
                        same_day_revision_rows += 1
                    else:
                        later_revision_rows += 1
            elif len(matches) > 1:
                ambiguous_rows += 1
                ambiguous_titles.add(key)
            else:
                unmatched_rows += 1
                if len(unmatched_examples) < 25:
                    unmatched_examples.append(title)

    unique_count = len(unique_view_titles)
    return {
        "sources": {
            "pageviews_csv": str(pageviews_path),
            "documents": str(documents_path),
            "join_method": "Unicode NFKC + casefold + URL decode + underscore-to-space + whitespace collapse",
        },
        "document_rows_scanned": document_rows,
        "unique_pageview_titles": unique_count,
        "pageview_observation_rows": total_view_rows,
        "unique_title_matches": len(matched_titles),
        "unique_title_ambiguous": len(ambiguous_titles),
        "unique_title_unmatched": unique_count - len(matched_titles) - len(ambiguous_titles),
        "observation_rows_uniquely_matched": matched_rows,
        "observation_rows_ambiguous": ambiguous_rows,
        "observation_rows_unmatched": unmatched_rows,
        "unique_title_match_rate": len(matched_titles) / unique_count if unique_count else 0.0,
        "observation_row_unique_match_rate": matched_rows / total_view_rows if total_view_rows else 0.0,
        "matched_observations_with_revision_timestamp": (
            matched_rows - missing_revision_rows - no_pre_cutoff_revision_rows
        ),
        "text_revision_before_target_date_rows": time_safe_rows,
        "text_revision_same_day_ambiguous_rows": same_day_revision_rows,
        "text_revision_after_target_date_rows": later_revision_rows,
        "text_revision_missing_rows": missing_revision_rows,
        "no_pre_cutoff_revision_rows": no_pre_cutoff_revision_rows,
        "time_safe_text_rate_among_matched_rows": (
            time_safe_rows / matched_rows if matched_rows else 0.0
        ),
        "unmatched_title_examples": unmatched_examples,
        "interpretation": (
            "This is title-join coverage only. The source records only top-ranked pages; "
            "an absent article-day is unknown/censored, not zero views. Title matching "
            "does not guarantee historical text exists before the target. For a daily target "
            "on date D, a text revision dated before D is considered available at the prior-day "
            "cutoff; same-day revisions are ambiguous and pages with no earlier revision have "
            "no time-safe semantic text for this observation."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pageviews", type=Path, required=True,
                        help="Local vtasca-style CSV with article/date/views")
    parser.add_argument("--documents", type=Path, required=True,
                        help="English FineWiki Parquet or normalized JSONL")
    parser.add_argument("--output", type=Path,
                        default=Path("results/metrics/wikipedia_join_report.json"))
    args = parser.parse_args()
    report = validate_join(args.pageviews, args.documents)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
