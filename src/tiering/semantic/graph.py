"""Lightweight semantic file graph and a group-aware tiering policy.

Nodes are files. An edge links two files that share at least one ACL group
(``metadata.access_groups``), weighted by cosine similarity of their text
vectors; edges below ``min_similarity`` are dropped. Each node keeps its
candidates sorted by similarity, and ``neighbors(file_id, now, k)`` returns the
top-k that already existed at ``now``, so the graph never exposes later files.

``GraphPrefetchPolicy`` plugs into ``SimPySimulator(config={"access_policy": ...})``:
an access promotes the file to HOT and pre-promotes its top-k neighbors; files
idle past a timeout demote HOT -> WARM -> COLD. With ``k=0`` it is plain
recency tiering, which isolates what the graph adds.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Mapping, Sequence

import numpy as np

from src.tiering.ingestion.schemas import Tier
from src.tiering.semantic.retrieval_context import _groups

_RANK = {Tier.HOT: 0, Tier.WARM: 1, Tier.COLD: 2}


class SemanticGraph:
    def __init__(self, files: Sequence, vectors: Mapping[str, np.ndarray],
                 min_similarity: float = 0.3, max_degree: int = 50):
        ids = [file.file_id for file in files]
        matrix = np.asarray([vectors[i] for i in ids], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
        self.created = {file.file_id: file.created_at for file in files}
        members: dict[str, list[int]] = {}
        for index, file in enumerate(files):
            for group in _groups(file):
                members.setdefault(group, []).append(index)
        self.edges: dict[str, list[tuple[str, float]]] = {}
        for index, file in enumerate(files):
            pools = [members[g] for g in _groups(file)]
            if not pools or not norms[index, 0]:
                self.edges[file.file_id] = []
                continue
            candidates = np.unique(np.concatenate(pools))
            candidates = candidates[candidates != index]
            sims = matrix[candidates] @ matrix[index]
            keep = sims >= min_similarity
            candidates, sims = candidates[keep], sims[keep]
            order = np.argsort(-sims, kind="stable")[:max_degree]
            self.edges[file.file_id] = [(ids[c], float(s))
                                        for c, s in zip(candidates[order], sims[order])]

    def neighbors(self, file_id: str, now: datetime, k: int) -> list[tuple[str, float]]:
        found = []
        for neighbor, sim in self.edges.get(file_id, []):
            if len(found) >= k:
                break
            if self.created[neighbor] <= now:
                found.append((neighbor, sim))
        return found

    def stats(self) -> dict:
        degrees = [len(v) for v in self.edges.values()]
        return {"nodes": len(degrees), "edges": int(sum(degrees)),
                "mean_degree": float(np.mean(degrees)) if degrees else 0.0,
                "isolated": int(sum(d == 0 for d in degrees))}


class GraphPrefetchPolicy:
    def __init__(self, graph: SemanticGraph | None, k: int = 5,
                 prefetch_tier: Tier = Tier.HOT, hot_idle_hours: float = 24.0,
                 warm_idle_hours: float = 168.0, useful_window_hours: float = 24.0):
        self.graph, self.k, self.prefetch_tier = graph, k, Tier(prefetch_tier)
        self.hot_idle = timedelta(hours=hot_idle_hours)
        self.warm_idle = timedelta(hours=warm_idle_hours)
        self.useful_window = timedelta(hours=useful_window_hours)
        self.last_touch: dict[str, datetime] = {}
        self.prefetched_at: dict[str, datetime] = {}
        self.seen: set[str] = set()
        self.counts = {"accesses": 0, "first_touch": 0, "first_touch_hot": 0,
                       "prefetches": 0, "useful_prefetches": 0}

    def on_access(self, file_id, now, tiers):
        serving = Tier(tiers[file_id])
        self.counts["accesses"] += 1
        if file_id not in self.seen:
            self.seen.add(file_id)
            self.counts["first_touch"] += 1
            self.counts["first_touch_hot"] += serving == Tier.HOT
        fetched = self.prefetched_at.pop(file_id, None)
        if fetched is not None and now - fetched <= self.useful_window:
            self.counts["useful_prefetches"] += 1
        self.last_touch[file_id] = now
        moves = [(file_id, Tier.HOT)]
        if self.graph is not None and self.k:
            for neighbor, _ in self.graph.neighbors(file_id, now, self.k):
                self.last_touch[neighbor] = now
                if _RANK[Tier(tiers[neighbor])] > _RANK[self.prefetch_tier]:
                    moves.append((neighbor, self.prefetch_tier))
                    self.prefetched_at[neighbor] = now
                    self.counts["prefetches"] += 1
        return moves

    def on_tick(self, now, tiers):
        moves = []
        for file_id, touched in self.last_touch.items():
            tier, idle = Tier(tiers[file_id]), now - touched
            if tier == Tier.HOT and idle > self.hot_idle:
                moves.append((file_id, Tier.WARM))
            elif tier == Tier.WARM and idle > self.warm_idle:
                moves.append((file_id, Tier.COLD))
        return moves

    def report(self) -> dict:
        c = self.counts
        return {**c,
                "first_touch_hot_rate": c["first_touch_hot"] / c["first_touch"]
                if c["first_touch"] else 0.0,
                "prefetch_precision": c["useful_prefetches"] / c["prefetches"]
                if c["prefetches"] else 0.0}
