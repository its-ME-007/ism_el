"""Tiered file store (MVP) on local tier directories.

Layout under ``root``:

- ``hot/``, ``warm/``, ``cold/``: file bytes, one file per id. Each tier
  directory can be pointed elsewhere (e.g. COLD on a NAS or HDD mount).
- ``index.json``: config plus per-file tier, size, ACL groups and
  timestamps (including the policy's last-touch time). No file content.
- ``access_log.jsonl``: every put/get with the tier that served it, for replay
  and retraining.

Placement is decided by ``GraphPrefetchPolicy`` over a ``CoAccessGraph``
learned from read timing (rebuilt from ``access_log.jsonl`` on open): a put or
get promotes the file and prefetches same-ACL-group files that usually get read
next; ``tick`` demotes idle files. Moves are
``shutil.move`` between tier directories (a rename on the same volume).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from src.tiering.coaccess import CoAccessGraph
from src.tiering.ingestion.schemas import Tier
from src.tiering.semantic.graph import GraphPrefetchPolicy

_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,199}")
DEFAULTS = {"hot_capacity_bytes": None, "k": 5, "lag_hours": 1.0, "min_support": 2,
            "min_confidence": 0.0, "hot_idle_hours": 24.0, "warm_idle_hours": 168.0,
            "tier_dirs": {}}


def _now(now=None):
    return now or datetime.now(timezone.utc)


def _check_id(file_id: str) -> str:
    # File ids become filenames: reject anything that could escape a tier dir.
    if not _ID.fullmatch(file_id or "") or ".." in file_id:
        raise ValueError(f"invalid file id {file_id!r}: use [A-Za-z0-9._-], no '..'")
    return file_id


def _write_atomic(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class TieredStore:
    """Single-process store. ponytail: no file locking and a full index.json
    rewrite per operation; move the index to SQLite past ~10k files or when
    several writers share a root."""

    def __init__(self, root, **overrides):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.log_path = self.root / "access_log.jsonl"
        if self.index_path.exists():
            self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        else:
            self.index = {"config": dict(DEFAULTS), "files": {}}
        for key, value in DEFAULTS.items():  # indexes from older versions
            self.index["config"].setdefault(key, value)
        for entry in self.files.values():
            entry.pop("text", None)  # older indexes stored content; never needed now
        self.index["config"].update({k: v for k, v in overrides.items() if v is not None})
        config = self.index["config"]
        self.dirs = {tier: Path(config["tier_dirs"].get(tier.value,
                                                        self.root / tier.value.lower()))
                     for tier in Tier}
        for directory in self.dirs.values():
            directory.mkdir(parents=True, exist_ok=True)
        self._policy = None
        self._save()

    @property
    def files(self) -> dict:
        return self.index["files"]

    # -- public API -------------------------------------------------------
    def put(self, file_id: str, data: bytes, groups=(),
            now: datetime | None = None) -> Tier:
        """Store or replace a file; a write counts as an access."""
        file_id, now = _check_id(file_id), _now(now)
        old = self.files.get(file_id)
        tier = Tier(old["tier"]) if old else Tier.COLD
        _write_atomic(self._path(file_id, tier), data)
        self.files[file_id] = {
            "tier": tier.value, "size": len(data), "groups": sorted(set(groups)),
            "modified_at": now.isoformat(),
            "created_at": old["created_at"] if old else now.isoformat(),
            "last_touch": old.get("last_touch") if old else None,
        }
        self._add_to_graph(file_id)
        return self._access(file_id, now, "write")

    def get(self, file_id: str, now: datetime | None = None) -> bytes:
        file_id, now = _check_id(file_id), _now(now)
        if file_id not in self.files:
            raise KeyError(file_id)
        data = self._locate(file_id).read_bytes()
        self._access(file_id, now, "read")
        return data

    def tick(self, now: datetime | None = None) -> int:
        """Demote idle files. Run periodically (cron / scheduler)."""
        now = _now(now)
        policy = self._get_policy()
        moves = policy.on_tick(now, self._tiers())
        self._apply(moves)
        self._save()
        return len(moves)

    def tier_of(self, file_id: str) -> Tier:
        return Tier(self.files[_check_id(file_id)]["tier"])

    def status(self) -> dict:
        summary = {tier.value: {"files": 0, "bytes": 0, "dir": str(self.dirs[tier])}
                   for tier in Tier}
        for entry in self.files.values():
            summary[entry["tier"]]["files"] += 1
            summary[entry["tier"]]["bytes"] += entry["size"]
        policy = self._policy
        return {"root": str(self.root), "config": self.index["config"], "tiers": summary,
                "graph": policy.graph.stats() if policy and policy.graph else None}

    def verify(self) -> list[str]:
        """Ids whose bytes are not in the tier the index records."""
        return [f for f in self.files if not self._path(f, self.tier_of(f)).exists()]

    # -- internals --------------------------------------------------------
    def _path(self, file_id, tier) -> Path:
        return self.dirs[Tier(tier)] / file_id

    def _locate(self, file_id) -> Path:
        expected = self._path(file_id, self.tier_of(file_id))
        if expected.exists():
            return expected
        # A crash between a move and the index save leaves the bytes in
        # another tier: repair the index instead of losing the file.
        for tier in Tier:
            if self._path(file_id, tier).exists():
                self.files[file_id]["tier"] = tier.value
                self._policy = None
                self._save()
                return self._path(file_id, tier)
        raise FileNotFoundError(f"{file_id} is indexed but missing from every tier")

    def _tiers(self) -> dict[str, Tier]:
        return {f: Tier(e["tier"]) for f, e in self.files.items()}

    def _get_policy(self) -> GraphPrefetchPolicy:
        if self._policy is not None:
            return self._policy
        config = self.index["config"]
        graph = None
        if config["k"]:
            graph = CoAccessGraph({f: e["groups"] for f, e in self.files.items()},
                                  lag_hours=config["lag_hours"],
                                  min_support=config["min_support"],
                                  min_confidence=config["min_confidence"])
            # ponytail: relearns from the whole log on open, O(log size); persist
            # the counts if the log grows past a few million lines.
            if self.log_path.exists():
                with self.log_path.open(encoding="utf-8") as log:
                    for line in log:
                        event = json.loads(line)
                        if event["file_id"] in self.files:
                            graph.observe(event["file_id"],
                                          datetime.fromisoformat(event["timestamp"]))
        policy = GraphPrefetchPolicy(
            graph, k=config["k"], hot_idle_hours=config["hot_idle_hours"],
            warm_idle_hours=config["warm_idle_hours"],
            hot_capacity_bytes=config["hot_capacity_bytes"],
            sizes={f: e["size"] for f, e in self.files.items()})
        policy.last_touch = {f: datetime.fromisoformat(e["last_touch"])
                             for f, e in self.files.items() if e.get("last_touch")}
        self._policy = policy
        return policy

    def _add_to_graph(self, file_id):
        """Tell the live policy about a new or replaced file."""
        policy = self._policy
        if policy is None:
            return
        entry = self.files[file_id]
        if policy.graph is not None:
            policy.graph.set_groups(file_id, entry["groups"])
        policy.sizes[file_id] = entry["size"]
        policy.hot = None  # resync HOT bytes (a replaced file may have changed size)

    def _access(self, file_id, now, operation) -> Tier:
        policy = self._get_policy()
        served = self.tier_of(file_id)
        self._apply(policy.on_access(file_id, now, self._tiers()))
        for touched, when in policy.last_touch.items():
            if touched in self.files:
                self.files[touched]["last_touch"] = when.isoformat()
        self._save()
        with self.log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"timestamp": now.isoformat(), "file_id": file_id,
                                  "operation": operation, "served_tier": served.value}) + "\n")
        return self.tier_of(file_id)

    def _apply(self, moves):
        for file_id, target in moves:
            source = self.tier_of(file_id)
            target = Tier(target)
            if source == target:
                continue
            shutil.move(str(self._path(file_id, source)), str(self._path(file_id, target)))
            self.files[file_id]["tier"] = target.value

    def _save(self):
        _write_atomic(self.index_path, json.dumps(self.index).encode("utf-8"))
