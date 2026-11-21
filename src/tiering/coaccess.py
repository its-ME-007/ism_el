"""Co-access graph: prefetch links learned online from read timing.

An edge A -> B means "B tends to be read within ``lag`` after A". Only files
that share an ACL group (``access_groups``) can link, so a prefetch never moves
a file the reader's group could not see. For each edge we count how many of
A's reads were followed by a B read within the lag (``follows``), and estimate

    confidence(A -> B) = follows / (reads(A) + 1)

against B's base rate (B's reads per lag window over the observed span). A
neighbor is offered only when confidence beats the base rate (B really follows
A, it is not just popular), clears ``min_support`` follows, and reaches
``min_confidence``: the admission threshold, which trades prefetch hits against
the storage and migration cost of wrong prefetches.

Drop-in for ``GraphPrefetchPolicy``: same ``neighbors(file_id, now, k)`` and
``stats()`` as ``SemanticGraph``, plus ``observe`` (called on every access).
With ``fallback`` (e.g. a random same-group graph) files with too few learned
links are topped up from it, which covers files nobody has read yet.
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta
import heapq
from typing import Iterable, Mapping


class CoAccessGraph:
    def __init__(self, groups: Mapping[str, Iterable[str]], lag_hours: float = 1.0,
                 min_support: int = 2, min_confidence: float = 0.0,
                 fallback=None, max_recent: int = 200):
        self.groups = {f: frozenset(g) for f, g in groups.items()}
        self.lag = timedelta(hours=lag_hours)
        self.min_support, self.min_confidence = min_support, min_confidence
        self.fallback = fallback
        # ponytail: recent reads per group capped at max_recent; a burstier
        # group than that loses its oldest pairs inside one lag window.
        self.recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_recent))
        self.reads: dict[str, int] = defaultdict(int)
        self.follows: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.first_seen: datetime | None = None

    def set_groups(self, file_id: str, groups: Iterable[str]):
        self.groups[file_id] = frozenset(groups)

    def observe(self, file_id: str, now: datetime):
        """Record a read: credit every earlier same-group read within the lag."""
        self.first_seen = self.first_seen or now
        credited = set()
        for group in self.groups.get(file_id, ()):
            for when, earlier in self.recent[group]:
                key = (when, earlier)
                if earlier != file_id and now - when <= self.lag and key not in credited:
                    credited.add(key)
                    self.follows[earlier][file_id] += 1
            self.recent[group].append((now, file_id))
        self.reads[file_id] += 1

    def confidence(self, source: str, target: str) -> float:
        return self.follows.get(source, {}).get(target, 0) / (self.reads.get(source, 0) + 1)

    def neighbors(self, file_id: str, now: datetime, k: int) -> list[tuple[str, float]]:
        span = (now - self.first_seen) / self.lag if self.first_seen else 0.0
        reads = self.reads.get(file_id, 0) + 1

        def base(target):  # chance B is read in any one lag window anyway
            return self.reads[target] / span if span > 1 else 1.0
        candidates = [(target, n / reads) for target, n in self.follows.get(file_id, {}).items()
                      if n >= self.min_support and n / reads >= self.min_confidence]
        found = heapq.nlargest(k, ((t, c) for t, c in candidates if c > base(t)),
                               key=lambda p: p[1])
        if self.fallback is not None and len(found) < k:
            have = {target for target, _ in found} | {file_id}
            for target, score in self.fallback.neighbors(file_id, now, k + len(have)):
                if len(found) >= k:
                    break
                if target not in have:
                    found.append((target, 0.0))
        return found

    def stats(self) -> dict:
        edges = sum(len(t) for t in self.follows.values())
        strong = sum(n >= self.min_support for t in self.follows.values() for n in t.values())
        return {"nodes": len(self.groups), "files_read": len(self.reads),
                "edges": edges, "edges_min_support": strong,
                "lag_hours": self.lag.total_seconds() / 3600,
                "min_support": self.min_support, "min_confidence": self.min_confidence}
