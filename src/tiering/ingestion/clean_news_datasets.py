"""Normalize downloaded AG News and Reuters-21578 source files to JSONL.

This preparation step keeps the original wording and labels. It only removes
SGML/HTML markup, decodes entities, normalizes Unicode/whitespace, and skips
records without usable text. Model-specific lowercasing, stopword removal, and
stemming are intentionally left to the feature extraction pipeline.
"""

from __future__ import annotations

import argparse
from collections import Counter
from html import unescape
import json
import re
import tarfile
import unicodedata
from pathlib import Path
from typing import Any


_TAG_RE = re.compile(r"<[^>]*>")
_REUTERS_RE = re.compile(r"<REUTERS\b([^>]*)>(.*?)</REUTERS\s*>", re.I | re.S)
_ATTRIBUTE_RE = re.compile(r"([A-Z_]+)=\"([^\"]*)\"", re.I)
_TOPIC_RE = re.compile(r"<D\b[^>]*>(.*?)</D\s*>", re.I | re.S)
_AG_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}


def clean_text(value: Any) -> str:
    """Decode markup and normalize a text value without changing its wording."""
    text = unescape(str(value or ""))
    text = _TAG_RE.sub(" ", text)
    text = unicodedata.normalize("NFC", text)
    text = "".join(
        " " if unicodedata.category(char).startswith("C") else char
        for char in text
    )
    return " ".join(text.split()).strip()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_ag_news(raw_dir: Path, output_path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "Reading AG News Parquet requires PyArrow. Install with "
            "`python -m pip install -e '.[data]'`."
        ) from exc

    rows: list[dict[str, Any]] = []
    skipped_empty = 0
    source_files = sorted(raw_dir.glob("*.parquet"))
    if not source_files:
        raise FileNotFoundError(f"No AG News Parquet files found in {raw_dir}")

    for source_file in source_files:
        split = "test" if "test" in source_file.name.lower() else "train"
        table = parquet.read_table(source_file)
        for index, record in enumerate(table.to_pylist()):
            text = clean_text(record.get("text"))
            if not text:
                skipped_empty += 1
                continue
            label_id = int(record["label"])
            if label_id not in _AG_LABELS:
                raise ValueError(f"Unknown AG News label {label_id} in {source_file}")
            rows.append({
                "document_id": f"ag_news:{split}:{index}",
                "text": text,
                "labels": [_AG_LABELS[label_id]],
                "split": split,
                "source": "ag_news",
            })

    _write_jsonl(output_path, rows)
    return {
        "records_written": len(rows),
        "records_skipped_empty_text": skipped_empty,
        "splits": dict(sorted(Counter(row["split"] for row in rows).items())),
        "labels": dict(sorted(Counter(row["labels"][0] for row in rows).items())),
        "output": str(output_path),
    }


def clean_20_newsgroups(raw_dir: Path, output_path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError(
            "Reading 20 Newsgroups Parquet requires PyArrow. Install with "
            "`python -m pip install -e '.[data]'`."
        ) from exc

    rows: list[dict[str, Any]] = []
    skipped_empty = 0
    source_files = sorted(raw_dir.glob("*.parquet"))
    if not source_files:
        raise FileNotFoundError(f"No 20 Newsgroups Parquet files found in {raw_dir}")

    for source_file in source_files:
        filename = source_file.name.lower()
        split = "validation" if "val" in filename else (
            "test" if "test" in filename else "train"
        )
        for index, record in enumerate(parquet.read_table(source_file).to_pylist()):
            text = clean_text(record.get("text"))
            if not text:
                skipped_empty += 1
                continue
            rows.append({
                "document_id": f"20_newsgroups:{split}:{index}",
                "text": text,
                "labels": [str(record["label_text"])],
                "split": split,
                "source": "20_newsgroups",
            })

    _write_jsonl(output_path, rows)
    return {
        "records_written": len(rows),
        "records_skipped_empty_text": skipped_empty,
        "splits": dict(sorted(Counter(row["split"] for row in rows).items())),
        "labels": dict(sorted(Counter(row["labels"][0] for row in rows).items())),
        "output": str(output_path),
    }


def _field(block: str, name: str) -> str:
    tag = re.escape(name)
    match = re.search(r"<" + tag + r"\b[^>]*>(.*?)</" + tag + r"\s*>",
                      block, re.I | re.S)
    return match.group(1) if match else ""


def clean_reuters(archive_path: Path, output_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    skipped_empty = 0
    seen_ids: set[str] = set()

    with tarfile.open(archive_path, "r:gz") as archive:
        sgml_files = sorted(
            (member for member in archive.getmembers()
             if re.fullmatch(r"reut2-\d{3}\.sgm", Path(member.name).name, re.I)),
            key=lambda member: member.name,
        )
        if not sgml_files:
            raise ValueError(f"No Reuters SGML files found in {archive_path}")

        for member in sgml_files:
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            content = extracted.read().decode("latin-1")
            for attributes_raw, block in _REUTERS_RE.findall(content):
                attributes = {key.upper(): value for key, value in
                              _ATTRIBUTE_RE.findall(attributes_raw)}
                newid = attributes.get("NEWID")
                if not newid:
                    continue
                if newid in seen_ids:
                    raise ValueError(f"Duplicate Reuters NEWID {newid}")
                seen_ids.add(newid)

                text_block = _field(block, "TEXT") or block
                title = clean_text(_field(text_block, "TITLE"))
                body = clean_text(_field(text_block, "BODY"))
                text = clean_text(" ".join(part for part in (title, body) if part))
                if not text:
                    skipped_empty += 1
                    continue

                labels = sorted({clean_text(label).lower()
                                 for label in _TOPIC_RE.findall(_field(block, "TOPICS"))
                                 if clean_text(label)})
                split = attributes.get("LEWISSPLIT", "NOT-USED").lower()
                if split not in {"train", "test"}:
                    split = "unused"
                rows.append({
                    "document_id": f"reuters21578:{newid}",
                    "text": text,
                    "labels": labels,
                    "split": split,
                    "source": "reuters21578",
                })

    _write_jsonl(output_path, rows)
    return {
        "records_written": len(rows),
        "records_skipped_empty_text": skipped_empty,
        "splits": dict(sorted(Counter(row["split"] for row in rows).items())),
        "records_without_topics": sum(not row["labels"] for row in rows),
        "unique_topic_labels": len({label for row in rows for label in row["labels"]}),
        "output": str(output_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ag-news-dir", type=Path,
                        default=Path("data/raw/ag_news"))
    parser.add_argument("--reuters-archive", type=Path,
                        default=Path("data/raw/reuters21578/reuters21578.tar.gz"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/processed"))
    args = parser.parse_args()

    report = {
        "cleaning": "markup/entity/control-character/whitespace normalization; no stopword removal or stemming",
        "20_newsgroups": clean_20_newsgroups(
            Path("data/raw/20_newsgroups"), args.output_dir / "20_newsgroups.jsonl"
        ),
        "ag_news": clean_ag_news(args.ag_news_dir, args.output_dir / "ag_news.jsonl"),
        "reuters21578": clean_reuters(args.reuters_archive,
                                     args.output_dir / "reuters21578.jsonl"),
    }
    report_path = args.output_dir / "cleaning_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
