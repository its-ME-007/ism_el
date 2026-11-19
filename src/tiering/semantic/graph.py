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

import bisect
from datetime import datetime, timedelta
from typing import Mapping, Sequence

import numpy as np

from src.tiering.ingestion.schemas import Tier
from src.tiering.semantic.retrieval_context import _groups

_RANK = {Tier.HOT: 0, Tier.WARM: 1, Tier.COLD: 2}


class SemanticGraph:
    def __init__(self, files: Sequence, vectors: Mapping[str, np.ndarray],
                 min_similarity: float = 0.3, max_degree: int = 50):
        self.min_similarity, self.max_degree = min_similarity, max_degree
        ids = [file.file_id for file in files]
        matrix = np.asarray([vectors[i] for i in ids], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
        self.unit = {file_id: matrix[i] for i, file_id in enumerate(ids)}
        self.created = {file.file_id: file.created_at for file in files}
        self.groups = {file.file_id: _groups(file) for file in files}
        members: dict[str, list[int]] = {}
        for index, file in enumerate(files):
            for group in self.groups[file.file_id]:
                members.setdefault(group, []).append(index)
        self.members = {g: [ids[i] for i in rows] for g, rows in members.items()}
        self.edges: dict[str, list[tuple[str, float]]] = {}
        for index, file in enumerate(files):
            pools = [members[g] for g in self.groups[file.file_id]]
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

    def add_node(self, file, vector: np.ndarray):
        """Insert or replace one file without rebuilding: O(ACL-group size)."""
        file_id = file.file_id
        if file_id in self.unit:
            self.remove_node(file_id)
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        vector = vector / norm if norm else vector
        groups = _groups(file)
        candidates = sorted(set().union(*(self.members.get(g, []) for g in groups)))
        self.unit[file_id], self.created[file_id], self.groups[file_id] = (
            vector, file.created_at, groups)
        for group in groups:
            self.members.setdefault(group, []).append(file_id)
        self.edges[file_id] = []
        if not candidates or not norm:
            return
        sims = np.stack([self.unit[c] for c in candidates]) @ vector
        pairs = sorted(((c, float(s)) for c, s in zip(candidates, sims)
                        if s >= self.min_similarity), key=lambda p: -p[1])
        self.edges[file_id] = pairs[:self.max_degree]
        for other, sim in pairs:
            bisect.insort(self.edges[other], (file_id, sim), key=lambda p: -p[1])
            del self.edges[other][self.max_degree:]

    def remove_node(self, file_id: str):
        groups = self.groups.pop(file_id, set())
        for group in groups:
            self.members[group].remove(file_id)
        for other in set().union(*(self.members.get(g, []) for g in groups)):
            self.edges[other] = [p for p in self.edges[other] if p[0] != file_id]
        for table in (self.unit, self.created, self.edges):
            table.pop(file_id, None)

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
    """Recency tiering plus semantic-neighbor prefetch, with an optional HOT cap.

    With ``hot_capacity_bytes`` set, a directly accessed file evicts the
    least-recently-touched HOT files (to WARM) until it fits; a file larger
    than the cap goes to WARM. Prefetched neighbors only take free HOT space,
    otherwise they land in WARM, so prefetch never evicts a file that was read.
    The policy tracks HOT occupancy from the moves it returns, so the caller
    must apply every move (the simulator and TieredStore both do).
    """

    def __init__(self, graph: SemanticGraph | None, k: int = 5,
                 prefetch_tier: Tier = Tier.HOT, hot_idle_hours: float = 24.0,
                 warm_idle_hours: float = 168.0, useful_window_hours: float = 24.0,
                 hot_capacity_bytes: int | None = None,
                 sizes: Mapping[str, int] | None = None):
        self.graph, self.k, self.prefetch_tier = graph, k, Tier(prefetch_tier)
        self.hot_idle = timedelta(hours=hot_idle_hours)
        self.warm_idle = timedelta(hours=warm_idle_hours)
        self.useful_window = timedelta(hours=useful_window_hours)
        self.capacity = hot_capacity_bytes
        self.sizes = dict(sizes or {})
        self.hot: dict[str, int] | None = None  # HOT view, synced on first call
        self.hot_bytes = self.peak_hot_bytes = 0
        self.last_touch: dict[str, datetime] = {}
        self.prefetched_at: dict[str, datetime] = {}
        self.seen: set[str] = set()
        self.counts = {"accesses": 0, "first_touch": 0, "first_touch_hot": 0,
                       "prefetches": 0, "prefetches_to_warm": 0,
                       "useful_prefetches": 0, "evictions": 0}

    def _sync(self, tiers):
        if self.hot is None:
            self.hot = {f: self.sizes.get(f, 0) for f, t in tiers.items()
                        if Tier(t) == Tier.HOT}
            self.hot_bytes = sum(self.hot.values())

    def _move(self, moves, file_id, tier):
        if tier == Tier.HOT:
            if file_id not in self.hot:
                self.hot[file_id] = self.sizes.get(file_id, 0)
                self.hot_bytes += self.hot[file_id]
        elif file_id in self.hot:
            self.hot_bytes -= self.hot.pop(file_id)
        self.peak_hot_bytes = max(self.peak_hot_bytes, self.hot_bytes)
        moves.append((file_id, tier))

    def _fits(self, size):
        return self.capacity is None or self.hot_bytes + size <= self.capacity

    def _admit(self, moves, file_id, now):
        if file_id in self.hot:
            return
        size = self.sizes.get(file_id, 0)
        if self.capacity is not None and size > self.capacity:
            self._move(moves, file_id, Tier.WARM)
            return
        # Prefetched files nobody has read yet go first, then least recently
        # touched; otherwise speculative prefetches push out files people read.
        # ponytail: O(|HOT|) min scan per eviction; use a heap if HOT holds 100k+ files.
        oldest = now - self.warm_idle * 100
        while not self._fits(size):
            victim = min(self.hot, key=lambda f: (f not in self.prefetched_at,
                                                  self.last_touch.get(f, oldest)))
            self._move(moves, victim, Tier.WARM)
            self.counts["evictions"] += 1
        self._move(moves, file_id, Tier.HOT)

    def on_access(self, file_id, now, tiers):
        self._sync(tiers)
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
        moves: list[tuple[str, Tier]] = []
        self._admit(moves, file_id, now)
        if self.graph is None or not self.k:
            return moves
        for neighbor, _ in self.graph.neighbors(file_id, now, self.k):
            self.last_touch[neighbor] = now
            pending = dict(moves)
            current = Tier(pending.get(neighbor, tiers[neighbor]))
            target = self.prefetch_tier
            if (target == Tier.HOT and neighbor not in self.hot
                    and not self._fits(self.sizes.get(neighbor, 0))):
                target = Tier.WARM
            if _RANK[current] > _RANK[target]:
                self._move(moves, neighbor, target)
                self.prefetched_at[neighbor] = now
                self.counts["prefetches"] += 1
                self.counts["prefetches_to_warm"] += target == Tier.WARM
        return moves

    def on_tick(self, now, tiers):
        self._sync(tiers)
        moves: list[tuple[str, Tier]] = []
        for file_id, touched in self.last_touch.items():
            tier, idle = Tier(tiers[file_id]), now - touched
            if tier == Tier.HOT and idle > self.hot_idle:
                self._move(moves, file_id, Tier.WARM)
            elif tier == Tier.WARM and idle > self.warm_idle:
                self._move(moves, file_id, Tier.COLD)
        return moves

    def report(self) -> dict:
        c = self.counts
        return {**c, "peak_hot_bytes": self.peak_hot_bytes,
                "hot_capacity_bytes": self.capacity,
                "first_touch_hot_rate": c["first_touch_hot"] / c["first_touch"]
                if c["first_touch"] else 0.0,
                "prefetch_precision": c["useful_prefetches"] / c["prefetches"]
                if c["prefetches"] else 0.0}
