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


def test_d_predictor_uses_authorized_corpus_context(monkeypatch):
    class Vectorizer:
        def transform(self, texts):
            return np.ones((len(texts), 1), dtype=np.float32)

    class TierModel:
        def predict_proba(self, matrix):
            return np.array([[0.1, 0.2, 0.7]])

    class DemandModel:
        def predict(self, matrix):
            return np.array([2.0])

    class AccessModel:
        def predict_proba(self, matrix):
            return np.array([[0.25, 0.75]])

    names = ["semantic_tfidf_0", "retrieval_neighbor_count",
             "retrieval_top_similarity", "retrieval_mean_similarity",
             "retrieval_neighbor_mean_accesses_24h",
             "retrieval_neighbor_mean_accesses_7d"]
    bundle = {"tfidf": Vectorizer(), "runs": {
        "D_combined_retrieval": {
            "tier_model": TierModel(), "demand_model": DemandModel(),
            "access_model": AccessModel(), "feature_names": names,
        }
    }}
    monkeypatch.setattr("src.tiering.prediction.mvp.joblib.load", lambda _: bundle)
    payload = {
        "document_id": "query", "text": "shared project text",
        "created_at": "2025-01-01T00:00:00Z",
        "modified_at": "2025-01-02T00:00:00Z",
        "metadata": {"access_groups": ["team-x"]},
        "cutoff": "2025-01-10T00:00:00Z", "access_events": [],
        "corpus": [{
            "document_id": "neighbor", "text": "shared project text",
            "created_at": "2025-01-01T00:00:00Z",
            "modified_at": "2025-01-02T00:00:00Z",
            "metadata": {"access_groups": ["team-x"]},
            "access_events": [{"timestamp": "2025-01-09T23:00:00Z"}],
        }],
    }
    result = predict_document(payload, model_path=None, run_name="D_combined_retrieval")
    assert result["model"] == "D_combined_retrieval"
    assert result["retrieval"]["neighbor_count"] == 1
    assert result["retrieval"]["neighbor_mean_accesses_24h"] == 1


def test_d_predictor_rejects_duplicate_or_query_document_in_corpus():
    payload = {
        "document_id": "query",
        "cutoff": "2025-01-10T00:00:00Z",
        "created_at": "2025-01-01T00:00:00Z",
        "corpus": [
            {"document_id": "neighbor"},
            {"document_id": "neighbor"},
        ],
    }
    with pytest.raises(ValueError, match="duplicate"):
        predict_document(payload, model_path=None, run_name="D_combined_retrieval")

    payload["corpus"] = [{"document_id": "query"}]
    with pytest.raises(ValueError, match="query document"):
        predict_document(payload, model_path=None, run_name="D_combined_retrieval")


def test_d_inference_feature_row_matches_training_order_and_values(monkeypatch):
    """The row mvp feeds the model must equal a row built from the training helpers.

    This guards against the hand-rebuilt metadata/workload/semantic/retrieval
    logic in mvp.py drifting from the experiment's _feature_dict + retrieval.
    """
    from src.tiering.experiments.train_synthetic_tiering_models import _feature_dict
    from src.tiering.semantic.tfidf_extractor import TFIDFExtractor
    from src.tiering.workload.feature_extractor import WorkloadFeatureExtractor

    cutoff = datetime(2025, 1, 10, tzinfo=UTC)
    created = datetime(2025, 1, 1, tzinfo=UTC)
    modified = datetime(2025, 1, 2, tzinfo=UTC)
    query_text = "finance budget report quarterly revenue forecast"
    neighbor_text = "finance revenue budget planning quarterly report"

    tfidf = TFIDFExtractor({"max_features": 8, "ngram_range": (1, 1)})
    tfidf.fit([SimpleNamespace(content=query_text),
               SimpleNamespace(content=neighbor_text),
               SimpleNamespace(content="legal contract review terms clause")])
    dim = len(tfidf.transform([""])[0])

    # Chronological order: mvp.py sorts events before extraction, and real
    # traces arrive in time order, so build the expected side the same way.
    query_events = [
        AccessEvent(timestamp=cutoff - timedelta(days=2), file_id="query", operation="write"),
        AccessEvent(timestamp=cutoff - timedelta(hours=3), file_id="query", operation="read"),
    ]
    neighbor_events = [
        AccessEvent(timestamp=cutoff - timedelta(hours=5), file_id="neighbor", operation="read"),
    ]
    query_doc = SimpleNamespace(
        file_id="query", filename="q.txt", category="Finance", content=query_text,
        metadata={"access_groups": ["team-x"], "file_type": "document",
                  "department": "Finance", "confidentiality": "internal",
                  "indexed": True, "compressed": False},
        size_bytes=1234, created_at=created, modified_at=modified)
    neighbor_doc = SimpleNamespace(
        file_id="neighbor", filename="n.txt", category="Finance", content=neighbor_text,
        metadata={"access_groups": ["team-x"]},
        size_bytes=900, created_at=created, modified_at=modified)

    # Training-side row: metadata/workload (sorted) + retrieval + semantic, same
    # order product_ablation assembles for D.
    md = _feature_dict(query_doc, query_events, cutoff, WorkloadFeatureExtractor(normalize=False))
    md.pop("content_chars", None)
    retrieval_names = ["retrieval_neighbor_count", "retrieval_top_similarity",
                       "retrieval_mean_similarity", "retrieval_neighbor_mean_accesses_24h",
                       "retrieval_neighbor_mean_accesses_7d"]
    semantic_names = [f"semantic_tfidf_{i}" for i in range(dim)]
    feature_names = sorted(md) + retrieval_names + semantic_names

    semantic = tfidf.transform([query_text])[0]
    vectors = {"query": semantic, "neighbor": tfidf.transform([neighbor_text])[0]}
    retrieval = retrieval_context_features(
        query_doc, [query_doc, neighbor_doc], vectors,
        {"query": query_events, "neighbor": neighbor_events}, cutoff, top_k=5)
    expected = {
        **md,
        **{f"semantic_tfidf_{i}": float(v) for i, v in enumerate(semantic)},
        **{f"retrieval_{k}": float(v) for k, v in retrieval.items()},
    }
    expected_row = np.asarray([expected.get(name, 0.0) for name in feature_names],
                              dtype=np.float32)

    captured = {}

    class TierModel:
        def predict_proba(self, matrix):
            captured["row"] = np.asarray(matrix)
            return np.array([[0.1, 0.2, 0.7]])

    class DemandModel:
        def predict(self, matrix):
            return np.array([1.0])

    class AccessModel:
        classes_ = [0, 1]

        def predict_proba(self, matrix):
            return np.array([[0.4, 0.6]])

    bundle = {"tfidf": tfidf, "runs": {"D_combined_retrieval": {
        "tier_model": TierModel(), "demand_model": DemandModel(),
        "access_model": AccessModel(), "feature_names": feature_names}}}
    monkeypatch.setattr("src.tiering.prediction.mvp.joblib.load", lambda _: bundle)

    def event_json(event):
        return {"timestamp": event.timestamp.isoformat(), "operation": event.operation}

    payload = {
        "document_id": "query", "filename": "q.txt", "category": "Finance",
        "text": query_text, "size_bytes": 1234,
        "created_at": created.isoformat(), "modified_at": modified.isoformat(),
        "metadata": query_doc.metadata, "cutoff": cutoff.isoformat(),
        "access_events": [event_json(e) for e in query_events],
        "corpus": [{
            "document_id": "neighbor", "text": neighbor_text,
            "created_at": created.isoformat(), "modified_at": modified.isoformat(),
            "metadata": {"access_groups": ["team-x"]},
            "access_events": [event_json(e) for e in neighbor_events],
        }],
    }
    predict_document(payload, model_path=None, run_name="D_combined_retrieval")
    assert captured["row"].shape == (1, len(feature_names))
    assert np.allclose(captured["row"][0], expected_row, atol=1e-6)
