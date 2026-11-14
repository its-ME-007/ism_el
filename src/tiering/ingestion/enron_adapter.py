"""Turn parsed Enron emails into the canonical (files, traces) bundle.

The A/B/C/D experiment consumes exactly two in-memory collections:
``files: list[FileDocument]`` and ``traces: list[AccessEvent]``. This adapter
builds both from the JSONL produced by :mod:`fetch_enron`, so the entire
feature / TF-IDF / retrieval / SimPy / metrics pipeline is reused unchanged.

Real signals (no fabrication):
  * ``content``            -> subject + body
  * ``created_at``/``modified_at`` -> the email ``Date`` (emails are immutable)
  * ``metadata.access_groups`` -> participant addresses. Emails that share a
    participant share a group, giving run D *real, content-correlated*
    authorized neighbors -- the thing synthetic data could never provide.

The one honest gap: emails are one-shot, so there is no native repeated
file-access log. Access events are a **participant re-engagement proxy** -- a
document is "accessed" at the timestamp of every later email that shares a
participant with it, within a horizon. Optional ``densify`` synthesizes extra
events only for documents below a floor, counted separately so the report can
state the real-vs-enriched share. ``densify=0`` (default) keeps every access
event real.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from src.tiering.ingestion.schemas import AccessEvent, Category, FileDocument


@dataclass
class EnronBundle:
    files: list[FileDocument]
    traces: list[AccessEvent]
    start: datetime
    train_days: list[int]
    validation_day: int
    test_day: int
    stats: dict = field(default_factory=dict)


def _parse(value: str) -> datetime:
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _documents(records: Iterable[dict]) -> list[FileDocument]:
    """One FileDocument per unique email, carrying participants as access_groups."""
    seen: set[str] = set()
    files: list[FileDocument] = []
    for record in records:
        file_id = str(record.get("message_id") or record.get("path", "")).strip()
        participants = [str(group).lower() for group in record.get("participants", [])
                        if str(group).strip()]
        body = str(record.get("body", "") or "")
        if not file_id or file_id in seen or not participants or not body.strip():
            continue
        seen.add(file_id)
        when = _parse(record["date"])
        subject = str(record.get("subject", "") or "")
        content = f"{subject}\n{body}".strip()
        files.append(FileDocument(
            file_id=file_id,
            filename=(subject[:80] or file_id),
            category=Category.OPERATIONS,  # emails have no file category; neutral default
            content=content,
            size_bytes=len(body.encode("utf-8")),
            created_at=when,
            modified_at=when,
            metadata={
                "access_groups": participants,
                "file_type": "document",
                "department": str(record.get("owner", "unknown")) or "unknown",
                "confidentiality": "internal",
                "indexed": False,
                "compressed": False,
                "owner": str(record.get("owner", "unknown")),
                "participant_count": len(participants),
            },
        ))
    return files


def _proxy_traces(files: list[FileDocument], horizon_days: int,
                  max_accesses_per_doc: int, densify: int,
                  seed: int) -> tuple[list[AccessEvent], int, int]:
    """Participant re-engagement access events (+ optional sparse densify)."""
    activity: dict[str, list[datetime]] = defaultdict(list)
    for file in files:
        for group in file.metadata["access_groups"]:
            activity[group].append(file.created_at)
    for group in activity:
        activity[group].sort()

    rng = np.random.default_rng(seed)
    traces: list[AccessEvent] = []
    real = enriched = 0
    horizon = timedelta(days=horizon_days)
    for file in files:
        t0 = file.created_at
        window_end = t0 + horizon
        times: set[datetime] = set()
        for group in file.metadata["access_groups"]:
            # ponytail: naive per-group scan, O(docs * activity). Fine for a
            # bounded research sample; bisect by time window if scaled to the
            # full 500k-email corpus.
            for when in activity[group]:
                if when <= t0:
                    continue
                if when > window_end:
                    break
                times.add(when)
        ordered = sorted(times)[:max_accesses_per_doc]
        for when in ordered:
            traces.append(AccessEvent(timestamp=when, file_id=file.file_id,
                                      operation="read"))
        real += len(ordered)
        if densify and len(ordered) < densify:
            for _ in range(densify - len(ordered)):
                offset = float(rng.uniform(0, horizon_days * 24.0))
                traces.append(AccessEvent(
                    timestamp=t0 + timedelta(hours=offset),
                    file_id=file.file_id, operation="read"))
                enriched += 1
    traces.sort(key=lambda event: event.timestamp)
    return traces, real, enriched


def _split_days(files: list[FileDocument], start: datetime) -> tuple[list[int], int, int]:
    """Chronological split as day offsets at percentiles of document dates."""
    doc_days = sorted(max(0, (file.created_at - start).days) for file in files)
    pct = lambda q: int(np.percentile(doc_days, q))
    train = sorted({pct(20), pct(40), pct(60)})
    validation = pct(75)
    test = pct(85)
    # Keep the splits strictly increasing even on tiny/degenerate samples.
    validation = max(validation, train[-1] + 1)
    test = max(test, validation + 1)
    return train, validation, test


def load_enron_bundle(records: Iterable[dict], *, horizon_days: int = 14,
                      max_accesses_per_doc: int = 50, densify: int = 0,
                      seed: int = 7) -> EnronBundle:
    """Build the (files, traces, start, split) bundle from parsed email records."""
    files = _documents(records)
    if not files:
        raise ValueError("No usable Enron records (need id, participants, body, date)")
    earliest = min(file.created_at for file in files)
    start = earliest.replace(hour=0, minute=0, second=0, microsecond=0)
    traces, real, enriched = _proxy_traces(
        files, horizon_days, max_accesses_per_doc, densify, seed)
    train_days, validation_day, test_day = _split_days(files, start)
    stats = {
        "documents": len(files),
        "access_events_total": len(traces),
        "access_events_real_proxy": real,
        "access_events_enriched": enriched,
        "enriched_fraction": round(enriched / len(traces), 4) if traces else 0.0,
        "distinct_access_groups": len({group for file in files
                                       for group in file.metadata["access_groups"]}),
        "span_days": max((file.created_at for file in files)).toordinal()
        - start.toordinal(),
        "proxy": ("participant re-engagement within "
                  f"{horizon_days}d; densify floor={densify}"),
    }
    return EnronBundle(files, traces, start, train_days, validation_day,
                       test_day, stats)


def load_enron_jsonl(path: Path, **kwargs) -> EnronBundle:
    with Path(path).open(encoding="utf-8") as source:
        records = [json.loads(line) for line in source if line.strip()]
    return load_enron_bundle(records, **kwargs)


def _demo() -> None:
    """Self-check: shared participants must yield a nonzero authorized neighbor."""
    from src.tiering.semantic.retrieval_context import retrieval_context_features

    base = datetime(2001, 3, 1, tzinfo=timezone.utc)
    records = [
        {"message_id": "m1", "date": base.isoformat(), "subject": "budget",
         "body": "quarterly budget revenue forecast numbers",
         "participants": ["a@enron.com", "b@enron.com"], "owner": "a"},
        {"message_id": "m2", "date": (base + timedelta(days=1)).isoformat(),
         "subject": "budget re", "body": "revenue budget forecast quarterly review",
         "participants": ["a@enron.com", "c@enron.com"], "owner": "a"},
        {"message_id": "m3", "date": (base + timedelta(days=2)).isoformat(),
         "subject": "legal", "body": "contract clause review terms litigation",
         "participants": ["z@enron.com"], "owner": "z"},
    ]
    bundle = load_enron_bundle(records, horizon_days=30)
    assert len(bundle.files) == 3
    assert bundle.stats["access_events_enriched"] == 0
    # m1 and m2 share participant a@enron.com -> m2 is an authorized neighbor of m1.
    from src.tiering.semantic.tfidf_extractor import TFIDFExtractor
    tfidf = TFIDFExtractor({"max_features": 16, "ngram_range": (1, 1)})
    tfidf.fit(bundle.files)
    vectors = {f.file_id: tfidf.transform([f.content])[0] for f in bundle.files}
    events = defaultdict(list)
    for event in bundle.traces:
        events[event.file_id].append(event)
    cutoff = base + timedelta(days=5)
    query = next(f for f in bundle.files if f.file_id == "m1")
    context = retrieval_context_features(query, bundle.files, vectors, events, cutoff)
    assert context["neighbor_count"] >= 1, context
    # The lone legal email shares no participant -> never an eligible neighbor.
    legal = next(f for f in bundle.files if f.file_id == "m3")
    assert retrieval_context_features(legal, bundle.files, vectors, events,
                                      cutoff)["neighbor_count"] == 0
    print("enron_adapter demo OK:", bundle.stats)


if __name__ == "__main__":
    _demo()
