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
    Access summaries include events strictly before cutoff only, matching the
    workload features; the label window starts at cutoff.
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
    if query.modified_at > cutoff:
        return zero  # query text version not yet visible at cutoff
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
                if event.timestamp < cutoff]
        day_counts.append(sum(recent_start <= event.timestamp < cutoff for event in past))
        week_counts.append(sum(week_start <= event.timestamp < cutoff for event in past))
    similarities = [score for score, _ in neighbors]
    return {
        "neighbor_count": float(len(neighbors)),
        "top_similarity": max(similarities),
        "mean_similarity": float(np.mean(similarities)),
        "neighbor_mean_accesses_24h": float(np.mean(day_counts)),
        "neighbor_mean_accesses_7d": float(np.mean(week_counts)),
    }


class RetrievalIndex:
    """Vectorized equivalent of :func:`retrieval_context_features` for one corpus.

    The per-call function scans every document in Python, which is
    O(samples x documents). Experiments score tens of thousands of cutoff rows,
    so this precomputes a normalized vector matrix, a group -> documents
    inverted index, and sorted per-document access times. Same permission,
    cutoff, self-exclusion and tie-break rules; ``tests`` check parity.
    """

    def __init__(self, documents: Sequence, vectors: Mapping[str, np.ndarray],
                 events_by_file: Mapping):
        self.vectors = vectors
        self.ids = [document.file_id for document in documents]
        self.position = {file_id: index for index, file_id in enumerate(self.ids)}
        id_order = sorted(range(len(self.ids)), key=lambda i: self.ids[i])
        self.id_rank = np.empty(len(self.ids), dtype=np.int64)
        self.id_rank[id_order] = np.arange(len(self.ids))
        self.created = np.asarray([d.created_at.timestamp() for d in documents])
        self.modified = np.asarray([d.modified_at.timestamp() for d in documents])
        rows = [np.asarray(vectors.get(i, []), dtype=np.float32) for i in self.ids]
        self.dim = max((len(row) for row in rows), default=0)
        self.matrix = np.zeros((len(rows), self.dim), dtype=np.float32)
        for index, row in enumerate(rows):
            norm = float(np.linalg.norm(row))
            if norm and row.shape == (self.dim,):
                self.matrix[index] = row / norm
        self.usable = np.linalg.norm(self.matrix, axis=1) > 0
        members: dict[str, list[int]] = {}
        for index, document in enumerate(documents):
            for group in _groups(document):
                members.setdefault(group, []).append(index)
        self.members = {group: np.asarray(rows, dtype=np.int64)
                        for group, rows in members.items()}
        self.event_times = {
            file_id: np.sort(np.asarray([e.timestamp.timestamp()
                                         for e in events_by_file.get(file_id, [])]))
            for file_id in self.ids
        }

    def features(self, query, cutoff: datetime, top_k: int = 5) -> dict[str, float]:
        zero = {"neighbor_count": 0.0, "top_similarity": 0.0, "mean_similarity": 0.0,
                "neighbor_mean_accesses_24h": 0.0, "neighbor_mean_accesses_7d": 0.0}
        groups = _groups(query)
        if not groups or top_k <= 0 or query.modified_at > cutoff:
            return zero
        vector = np.asarray(self.vectors.get(query.file_id, []), dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if not norm or vector.shape != (self.dim,):
            return zero
        vector = vector / norm
        pools = [self.members[group] for group in groups if group in self.members]
        if not pools:
            return zero
        candidates = np.unique(np.concatenate(pools))
        limit = cutoff.timestamp()
        keep = ((self.created[candidates] <= limit) & (self.modified[candidates] <= limit)
                & self.usable[candidates])
        candidates = candidates[keep]
        own = self.position.get(query.file_id)
        if own is not None:
            candidates = candidates[candidates != own]
        if not len(candidates):
            return zero
        similarities = self.matrix[candidates] @ vector
        order = np.lexsort((self.id_rank[candidates], -similarities))[:top_k]
        neighbors, scores = candidates[order], similarities[order]
        day, week = [], []
        for index in neighbors:
            times = self.event_times[self.ids[index]]
            upto = np.searchsorted(times, limit, side="left")
            day.append(upto - np.searchsorted(times, limit - 86400, side="left"))
            week.append(upto - np.searchsorted(times, limit - 7 * 86400, side="left"))
        return {
            "neighbor_count": float(len(neighbors)),
            "top_similarity": float(scores.max()),
            "mean_similarity": float(scores.mean()),
            "neighbor_mean_accesses_24h": float(np.mean(day)),
            "neighbor_mean_accesses_7d": float(np.mean(week)),
        }
