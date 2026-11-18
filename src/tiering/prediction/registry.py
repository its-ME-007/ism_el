"""Versioned model registry, prediction log, and drift monitor for the MVP.

Layout under ``root`` (default ``results/models``)::

    versions/<label>-<sha12>.joblib   immutable copies of promoted bundles
    registry.json                     {"active": id, "history": [...]}

``promote`` copies a trained bundle in and activates it; ``rollback`` reactivates
the previously active version. Predictions are appended to a JSONL log (no
document text) so ``monitor`` can compare served tier shares with the
training prevalence stored in the bundle.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

DEFAULT_ROOT = Path("results/models")
DEFAULT_LOG = Path("results/logs/predictions.jsonl")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _registry_path(root: Path) -> Path:
    return Path(root) / "registry.json"


def load_registry(root: Path = DEFAULT_ROOT) -> dict:
    path = _registry_path(root)
    if not path.exists():
        return {"active": None, "history": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _save(registry: dict, root: Path) -> None:
    path = _registry_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)  # atomic on the same filesystem


def promote(bundle_path: Path, root: Path = DEFAULT_ROOT, note: str = "") -> dict:
    """Copy ``bundle_path`` into the registry as an immutable version and activate it."""
    bundle_path = Path(bundle_path)
    digest = hashlib.sha256(bundle_path.read_bytes()).hexdigest()[:12]
    version_id = f"{bundle_path.stem}-{digest}"
    target = Path(root) / "versions" / f"{version_id}.joblib"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(bundle_path, target)
    registry = load_registry(root)
    registry["history"].append({"id": version_id, "path": str(target),
                                "source": str(bundle_path), "activated_at": _now(),
                                "action": "promote", "note": note})
    registry["active"] = version_id
    _save(registry, root)
    return registry["history"][-1]


def rollback(root: Path = DEFAULT_ROOT) -> dict:
    """Reactivate the version that was active before the current one."""
    registry = load_registry(root)
    active = registry["active"]
    previous = [entry["id"] for entry in registry["history"] if entry["id"] != active]
    if not previous:
        raise RuntimeError("No earlier version to roll back to")
    target = previous[-1]
    path = next(e["path"] for e in reversed(registry["history"]) if e["id"] == target)
    registry["history"].append({"id": target, "path": path, "activated_at": _now(),
                                "action": "rollback", "note": f"from {active}"})
    registry["active"] = target
    _save(registry, root)
    return registry["history"][-1]


def active_model(root: Path = DEFAULT_ROOT) -> tuple[str, Path]:
    registry = load_registry(root)
    if not registry["active"]:
        raise RuntimeError(f"No active model in {_registry_path(root)}; run "
                           "`tiering models promote <bundle.joblib>` first")
    entry = next(e for e in reversed(registry["history"]) if e["id"] == registry["active"])
    return registry["active"], Path(entry["path"])


def log_prediction(result: dict, model_id: str, log_path: Path = DEFAULT_LOG) -> None:
    """Append one prediction (scores and retrieval aggregates only, no text)."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"logged_at": _now(), "model_id": model_id,
              **{key: result.get(key) for key in (
                  "document_id", "cutoff", "model", "predicted_tier",
                  "tier_probabilities", "estimated_accesses_next_24h",
                  "probability_any_access_next_24h")},
              "neighbor_count": (result.get("retrieval") or {}).get("neighbor_count")}
    with log_path.open("a", encoding="utf-8") as destination:
        destination.write(json.dumps(record) + "\n")


def monitor(log_path: Path = DEFAULT_LOG, reference: dict | None = None,
            last: int = 1000, threshold: float = 0.2,
            min_predictions: int = 50) -> dict:
    """Summarize recent served predictions and flag tier-share drift.

    ``reference`` is either one tier-share dict or ``{run_name: tier_share}``
    (the bundle's ``served_tier_share_reference``: each run's predicted tier mix
    on validation). Drift is flagged when any tier's served share differs from
    the reference by more than ``threshold`` (absolute). Labels are not
    available at serving time, so this is an input/output shift alarm, not an
    accuracy check.
    """
    lines = Path(log_path).read_text(encoding="utf-8").splitlines() if \
        Path(log_path).exists() else []
    records = [json.loads(line) for line in lines[-last:] if line.strip()]
    tiers = Counter(record["predicted_tier"] for record in records)
    share = {tier: tiers.get(tier, 0) / len(records) if records else 0.0
             for tier in ("HOT", "WARM", "COLD")}
    report = {"predictions": len(records), "served_tier_share": share,
              "models_seen": sorted({record["model_id"] for record in records}),
              "mean_probability_any_access": (
                  sum(r["probability_any_access_next_24h"] for r in records) / len(records)
                  if records else None)}
    if reference and records and not {"HOT", "WARM", "COLD"} <= set(reference):
        run = Counter(record.get("model") for record in records).most_common(1)[0][0]
        reference = reference.get(run)
        report["reference_run"] = run
    if reference:
        gaps = {tier: share[tier] - reference.get(tier, 0.0) for tier in share}
        report["reference_tier_share"] = reference
        report["share_gap"] = gaps
        # Too few predictions make tier shares noise; report but do not alarm.
        report["drift_alert"] = (None if len(records) < min_predictions else
                                 any(abs(g) > threshold for g in gaps.values()))
    return report
