from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from src.tiering.ingestion.schemas import AccessEvent
from src.tiering.prediction.mvp import predict_document
from src.tiering.semantic.retrieval_context import retrieval_context_features


UTC = timezone.utc
CUTOFF = datetime(2025, 1, 10, tzinfo=UTC)


def document(file_id, groups, *, created=CUTOFF - timedelta(days=10),
             modified=CUTOFF - timedelta(days=1)):
    return SimpleNamespace(
        file_id=file_id,
        metadata={"access_groups": groups},
        created_at=created,
        modified_at=modified,
    )


def test_retrieval_excludes_self_and_unauthorized_neighbors():
    query = document("query", ["finance"])
    authorized = document("authorized", ["finance", "team-a"])
    unauthorized = document("unauthorized", ["legal"])
    vectors = {
        "query": np.array([1.0, 0.0]),
        "authorized": np.array([1.0, 0.0]),
        "unauthorized": np.array([1.0, 0.0]),
    }
    context = retrieval_context_features(
        query, [query, authorized, unauthorized], vectors, {}, CUTOFF, top_k=5
    )
    assert context["neighbor_count"] == 1
    assert context["top_similarity"] == pytest.approx(1.0)


def test_retrieval_excludes_future_content_and_counts_only_past_accesses():
    query = document("query", ["shared"])
    past_doc = document("past", ["shared"])
    future_doc = document("future", ["shared"], modified=CUTOFF + timedelta(seconds=1))
    events = {
        "past": [
            AccessEvent(timestamp=CUTOFF - timedelta(hours=1), file_id="past", operation="read"),
            AccessEvent(timestamp=CUTOFF + timedelta(seconds=1), file_id="past", operation="read"),
        ],
    }
    vectors = {key: np.array([0.0, 1.0]) for key in ("query", "past", "future")}
    context = retrieval_context_features(
        query, [query, past_doc, future_doc], vectors, events, CUTOFF, top_k=5
    )
    assert context["neighbor_count"] == 1
    assert context["neighbor_mean_accesses_24h"] == 1


def test_documents_without_acl_groups_are_private():
    query = document("query", [])
    candidate = document("candidate", ["public"])
    context = retrieval_context_features(
        query, [query, candidate], {"query": np.array([1.0]),
                                    "candidate": np.array([1.0])}, {}, CUTOFF
    )
    assert context["neighbor_count"] == 0


def test_mvp_rejects_text_modified_after_prediction_cutoff():
    payload = {
        "cutoff": "2025-01-10T00:00:00Z",
        "created_at": "2025-01-01T00:00:00Z",
        "modified_at": "2025-01-11T00:00:00Z",
    }
    with pytest.raises(ValueError, match="versioned text snapshot"):
        predict_document(payload, model_path=None)


def test_mvp_rejects_future_access_events():
    payload = {
        "cutoff": "2025-01-10T00:00:00Z",
        "created_at": "2025-01-01T00:00:00Z",
        "modified_at": "2025-01-02T00:00:00Z",
        "access_events": [{"timestamp": "2025-01-10T00:00:01Z"}],
    }
    with pytest.raises(ValueError, match="after cutoff"):
        predict_document(payload, model_path=None)
