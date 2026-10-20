"""Reusable classification, latency, cost, and migration metrics."""

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)


class MetricsCollector:
    def classification(self, y_true, y_pred, labels=None):
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_per_class": precision.astype(float).tolist(),
            "recall_per_class": recall.astype(float).tolist(),
            "f1_per_class": f1.astype(float).tolist(),
            "support_per_class": support.astype(int).tolist(),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "report": classification_report(y_true, y_pred, labels=labels,
                                            output_dict=True, zero_division=0),
        }

    @staticmethod
    def latency(latencies_ms, tiers=None):
        values = np.asarray(latencies_ms, dtype=float)
        if values.size and (not np.all(np.isfinite(values)) or np.any(values < 0)):
            raise ValueError("Latency values must be finite and non-negative")
        result = {"average_ms": float(np.mean(values)) if values.size else 0.0,
                  "p95_ms": float(np.percentile(values, 95)) if values.size else 0.0,
                  "count": int(values.size)}
        if tiers is not None:
            result["by_tier"] = {}
            for tier in sorted(set(tiers)):
                tier_values = values[np.asarray(tiers) == tier]
                result["by_tier"][str(getattr(tier, "value", tier))] = {
                    "average_ms": float(np.mean(tier_values)) if tier_values.size else 0.0,
                    "count": int(tier_values.size),
                }
        return result

    @staticmethod
    def migrations(migration_times, total_accesses=None):
        ordered = sorted(migration_times)
        repeated = sum(1 for left, right in zip(ordered, ordered[1:])
                       if (right - left).total_seconds() < 3600)
        return {"total": len(ordered), "thrashing_rate": repeated / len(ordered) if ordered else 0.0,
                "migrations_per_access": (len(ordered) / total_accesses
                                          if total_accesses else 0.0)}

    @staticmethod
    def storage_cost(occupancy_gb, monthly_cost_per_gb, duration_hours):
        cost_by_tier = {str(key): float(value) * float(monthly_cost_per_gb.get(key, 0.0))
                        * float(duration_hours) / 730.0
                        for key, value in occupancy_gb.items()}
        return {"cost_by_tier": cost_by_tier, "total_cost": float(sum(cost_by_tier.values()))}

    @staticmethod
    def save(metrics, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"generated_at": datetime.now(timezone.utc).isoformat(), **metrics}
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path

