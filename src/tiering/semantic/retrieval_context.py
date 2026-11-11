"""Cutoff-safe, permission-filtered numeric retrieval features."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Mapping, Sequence

import numpy as np


def _groups(document) -> set[str]:
    metadata = getattr(document, "metadata", {}) or {}
    value = metadata.get("access_groups", [])
    if isinstance(value, str):
        value = [value]
    return {str(group) for group in value if str(group)}


def retrieval_context_features(query, documents: Sequence, vectors: Mapping[str, np.ndarray],
                               events_by_file: Mapping, cutoff: datetime,
                               top_k: int = 5) -> dict[str, float]:
    """Aggregate similar authorized neighbors using content and accesses known at cutoff.

    Documents without explicit access_groups are private and cannot be retrieved
    as neighbors. A document is searchable only after its latest modification.
    Access summaries include events at or before cutoff only.
    """
    zero = {
        "neighbor_count": 0.0,
        "top_similarity": 0.0,
        "mean_similarity": 0.0,
        "neighbor_mean_accesses_24h": 0.0,
        "neighbor_mean_accesses_7d": 0.0,
    }
    query_groups = _groups(query)
    if not query_groups or top_k <= 0:
        return zero
    query_vector = np.asarray(vectors.get(query.file_id, []), dtype=np.float32)
    norm = float(np.linalg.norm(query_vector))
    if not norm:
        return zero
    query_vector = query_vector / norm
    candidates = []
    for document in documents:
        if document.file_id == query.file_id:
            continue
        if document.created_at > cutoff or document.modified_at > cutoff:
            continue
        if not (query_groups & _groups(document)):
            continue
        vector = np.asarray(vectors.get(document.file_id, []), dtype=np.float32)
        vector_norm = float(np.linalg.norm(vector))
        if not vector_norm or vector.shape != query_vector.shape:
            continue
        similarity = float(np.dot(query_vector, vector / vector_norm))
        candidates.append((similarity, document.file_id))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    neighbors = candidates[:top_k]
    if not neighbors:
        return zero

    recent_start = cutoff - timedelta(hours=24)
    week_start = cutoff - timedelta(days=7)
    day_counts, week_counts = [], []
    for _, file_id in neighbors:
        past = [event for event in events_by_file.get(file_id, [])
                if event.timestamp <= cutoff]
        day_counts.append(sum(recent_start < event.timestamp <= cutoff for event in past))
        week_counts.append(sum(week_start < event.timestamp <= cutoff for event in past))
    similarities = [score for score, _ in neighbors]
    return {
        "neighbor_count": float(len(neighbors)),
        "top_similarity": max(similarities),
        "mean_similarity": float(np.mean(similarities)),
        "neighbor_mean_accesses_24h": float(np.mean(day_counts)),
        "neighbor_mean_accesses_7d": float(np.mean(week_counts)),
    }
