"""Single-document prediction using the existing combined prototype model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

from src.tiering.ingestion.schemas import AccessEvent, Category
from src.tiering.models.tier_classifier import TierClassifier
from src.tiering.semantic.tfidf_extractor import TFIDFExtractor
from src.tiering.semantic.retrieval_context import retrieval_context_features
from src.tiering.workload.feature_extractor import WorkloadFeatureExtractor


TIER_NAMES = {0: "HOT", 1: "WARM", 2: "COLD"}
WORKLOAD_NAMES = [
    "access_count_1h", "access_count_24h", "access_count_7d",
    "access_count_30d", "last_access_hours", "time_since_creation_hours",
    "mean_inter_arrival_hours", "std_inter_arrival_hours",
    "read_write_ratio", "periodicity_score",
]


def _parse_datetime(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError(f"Timestamp must include a timezone: {value}")
    return result.astimezone(timezone.utc)


def predict_document(payload: dict, model_path: Path) -> dict:
    """Return demand and tier predictions from a document record and access history."""
    cutoff = _parse_datetime(payload["cutoff"])
    created_at = _parse_datetime(payload["created_at"])
    modified_at = _parse_datetime(payload.get("modified_at", payload["created_at"]))
    if created_at > cutoff:
        raise ValueError("created_at is after cutoff; document was not available for prediction")
    if modified_at > cutoff:
        raise ValueError(
            "modified_at is after cutoff; provide a versioned text snapshot available at cutoff"
        )
    document_id = str(payload.get("document_id", "unidentified"))
    content = str(payload.get("text", "") or "")
    metadata = dict(payload.get("metadata", {}))
    file_record = SimpleNamespace(
        file_id=document_id,
        filename=str(payload.get("filename", "")),
        category=str(payload.get("category", "unknown")),
        content=content,
        metadata=metadata,
        size_bytes=int(payload.get("size_bytes", 0)),
        created_at=created_at,
        modified_at=modified_at,
    )
    events = [AccessEvent(
        timestamp=_parse_datetime(event["timestamp"]),
        file_id=document_id,
        operation=event.get("operation", "read"),
        size_bytes=event.get("size_bytes"),
    ) for event in payload.get("access_events", [])]
    events.sort(key=lambda event: event.timestamp)
    if any(event.timestamp > cutoff for event in events):
        raise ValueError("access_events contains data after cutoff")

    bundle = joblib.load(model_path)
    combined = bundle["runs"]["C_combined"]
    tier_model: TierClassifier = combined["tier_model"]
    tfidf: TFIDFExtractor = bundle["tfidf"]

    history = WorkloadFeatureExtractor(normalize=False).extract(
        events, cutoff, created_at
    )
    workload_values = history.to_vector().astype(float).tolist()
    workload_values[4] = min(workload_values[4], 24 * 30)
    workload_values[5] = min(workload_values[5], 24 * 365 * 10)
    workload_values[8] = min(workload_values[8], 100.0)
    features = {
        f"workload_{name}": value
        for name, value in zip(WORKLOAD_NAMES, workload_values)
    }

    vocabularies = {
        "category": [item.value for item in Category],
        "file_type": ["document", "archive", "database", "log"],
        "department": ["IT", "Finance", "HR", "Operations"],
        "confidentiality": ["public", "internal", "confidential", "restricted"],
    }
    observed = {
        "category": str(payload.get("category", "unknown")),
        "file_type": str(metadata.get("file_type", "unknown")),
        "department": str(metadata.get("department", "unknown")),
        "confidentiality": str(metadata.get("confidentiality", "unknown")),
    }
    age_hours = max(0.0, (cutoff - created_at).total_seconds() / 3600)
    modified_age_hours = max(0.0, (cutoff - modified_at).total_seconds() / 3600)
    features.update({
        "size_log_bytes": float(np.log1p(max(0, file_record.size_bytes))),
        "created_age_days": age_hours / 24,
        "modified_age_days": modified_age_hours / 24,
        "indexed": float(bool(metadata.get("indexed", False))),
        "compressed": float(bool(metadata.get("compressed", False))),
    })
    for field, choices in vocabularies.items():
        for choice in choices:
            features[f"{field}={choice}"] = float(observed[field] == choice)

    semantic = tfidf.transform([content])[0].astype(float).tolist()
    for index, value in enumerate(semantic):
        features[f"semantic_tfidf_{index}"] = value
    feature_names = combined["feature_names"]
    row = np.asarray([[features.get(name, 0.0) for name in feature_names]], dtype=np.float32)
    tier_probabilities = tier_model.predict_proba(row)[0]
    tier_index = int(np.argmax(tier_probabilities))
    demand_count = max(0.0, float(combined["demand_model"].predict(row)[0]))
    access_probability = float(combined["access_model"].predict_proba(row)[0][1])
    return {
        "document_id": document_id,
        "cutoff": cutoff.isoformat(),
        "model": "C_combined_synthetic_tfidf",
        "predicted_tier": TIER_NAMES[tier_index],
        "tier_probabilities": {
            TIER_NAMES[index]: float(probability)
            for index, probability in enumerate(tier_probabilities)
        },
        "estimated_accesses_next_24h": demand_count,
        "probability_any_access_next_24h": access_probability,
        "feature_groups": ["metadata_workload", "tfidf_semantics"],
        "limitations": [
            "Model was trained on synthetic traces; this output is a prototype recommendation.",
            "Synthetic access patterns were assigned independently of document meaning.",
        ],
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON input record")
    parser.add_argument("--model", type=Path,
                        default=Path("results/models/product_ablation_synthetic.joblib"))
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = predict_document(payload, args.model)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
