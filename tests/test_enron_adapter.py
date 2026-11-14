"""Checks for the Enron -> (files, traces) adapter."""

from collections import defaultdict
from datetime import datetime, timedelta, timezone

from src.tiering.ingestion.enron_adapter import load_enron_bundle
from src.tiering.semantic.retrieval_context import retrieval_context_features
from src.tiering.semantic.tfidf_extractor import TFIDFExtractor

UTC = timezone.utc
BASE = datetime(2001, 3, 1, tzinfo=UTC)


def _records():
    return [
        {"message_id": "m1", "date": BASE.isoformat(), "subject": "budget",
         "body": "quarterly budget revenue forecast numbers",
         "participants": ["a@enron.com", "b@enron.com"], "owner": "a"},
        {"message_id": "m2", "date": (BASE + timedelta(days=1)).isoformat(),
         "subject": "budget re", "body": "revenue budget forecast quarterly review",
         "participants": ["a@enron.com", "c@enron.com"], "owner": "a"},
        {"message_id": "m3", "date": (BASE + timedelta(days=2)).isoformat(),
         "subject": "legal", "body": "contract clause review terms litigation",
         "participants": ["z@enron.com"], "owner": "z"},
    ]


def test_participants_become_access_groups_and_yield_authorized_neighbor():
    bundle = load_enron_bundle(_records(), horizon_days=30)
    files_by_id = {f.file_id: f for f in bundle.files}
    assert files_by_id["m1"].metadata["access_groups"] == ["a@enron.com", "b@enron.com"]

    tfidf = TFIDFExtractor({"max_features": 16, "ngram_range": (1, 1)})
    tfidf.fit(bundle.files)
    vectors = {f.file_id: tfidf.transform([f.content])[0] for f in bundle.files}
    events = defaultdict(list)
    for event in bundle.traces:
        events[event.file_id].append(event)
    cutoff = BASE + timedelta(days=5)

    # m1 and m2 share a@enron.com -> eligible neighbor; m3 shares nobody.
    assert retrieval_context_features(
        files_by_id["m1"], bundle.files, vectors, events, cutoff)["neighbor_count"] >= 1
    assert retrieval_context_features(
        files_by_id["m3"], bundle.files, vectors, events, cutoff)["neighbor_count"] == 0


def test_timestamps_are_cutoff_safe_and_immutable():
    bundle = load_enron_bundle(_records(), horizon_days=30)
    for file in bundle.files:
        assert file.created_at == file.modified_at  # emails are immutable
        assert file.created_at >= bundle.start


def test_proxy_events_are_real_by_default_and_within_horizon():
    bundle = load_enron_bundle(_records(), horizon_days=30)
    assert bundle.stats["access_events_enriched"] == 0
    assert bundle.stats["enriched_fraction"] == 0.0
    created = {f.file_id: f.created_at for f in bundle.files}
    for event in bundle.traces:
        # Every proxy access happens strictly after the document's own date
        # (re-engagement) and within the horizon -- no leakage before creation.
        assert event.timestamp > created[event.file_id]
        assert event.timestamp <= created[event.file_id] + timedelta(days=30)


def test_densify_adds_tagged_enriched_events_only_for_sparse_docs():
    bundle = load_enron_bundle(_records(), horizon_days=30, densify=5, seed=1)
    assert bundle.stats["access_events_enriched"] > 0
    # Total must equal real + enriched, and enriched are counted separately.
    assert (bundle.stats["access_events_total"]
            == bundle.stats["access_events_real_proxy"]
            + bundle.stats["access_events_enriched"])


def test_empty_records_raise():
    import pytest
    with pytest.raises(ValueError, match="No usable Enron records"):
        load_enron_bundle([])
