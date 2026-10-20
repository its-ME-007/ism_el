"""Confidence, hysteresis, and cooldown policy for tier migrations."""

from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple

from src.tiering.ingestion.schemas import MigrationCost, MigrationDecision, Tier, TierConfig


class MigrationController:
    def __init__(self, config=None, tier_configs=None):
        self.config = dict(config or {})
        self.confidence_threshold = float(self.config.get("confidence_threshold", 0.7))
        self.hysteresis_count = max(1, int(self.config.get("hysteresis_count", 3)))
        self.cooldown = timedelta(hours=float(self.config.get("cooldown_hours", 6.0)))
        self.tier_configs = tier_configs or self.config.get("tiers", {})
        self.file_sizes: Dict[str, int] = {}
        self._pending: Dict[str, Tuple[Tier, int]] = {}
        self._last_migration: Dict[str, datetime] = {}
        self.migration_log = []
        self.last_evaluation_migrated = False

    def evaluate(self, file_id, current_tier, predicted_tier, probability, timestamp):
        current = Tier(current_tier)
        predicted = Tier(predicted_tier)
        confidence = float(probability)
        self.last_evaluation_migrated = False
        rationale = ""
        pending_target, pending_count = self._pending.get(file_id, (predicted, 0))

        if confidence < self.confidence_threshold:
            self._pending.pop(file_id, None)
            rationale = f"confidence {confidence:.3f} below threshold {self.confidence_threshold:.3f}"
            target = current
        elif predicted == current:
            self._pending.pop(file_id, None)
            rationale = "predicted tier matches current tier"
            target = current
        else:
            count = pending_count + 1 if pending_target == predicted else 1
            self._pending[file_id] = (predicted, count)
            last = self._last_migration.get(file_id)
            if last and timestamp < last + self.cooldown:
                rationale = "migration cooldown is active"
                target = current
            elif count < self.hysteresis_count:
                rationale = f"awaiting sustained prediction ({count}/{self.hysteresis_count})"
                target = current
            else:
                rationale = f"migration approved after {count} consistent predictions"
                target = predicted
                self._last_migration[file_id] = timestamp
                self._pending.pop(file_id, None)
                self.last_evaluation_migrated = True

        decision = MigrationDecision(file_id=file_id, current_tier=current,
                                     recommended_tier=target, confidence=confidence,
                                     rationale=rationale)
        if self.last_evaluation_migrated:
            self.migration_log.append({"file_id": file_id, "timestamp": timestamp.isoformat(),
                                       "source_tier": current.value, "dest_tier": target.value,
                                       "confidence": confidence, "rationale": rationale})
        return decision

    def get_migration_cost(self, file_size, source_tier, dest_tier, file_id=""):
        source, dest = Tier(source_tier), Tier(dest_tier)
        source_config = self._tier_config(source)
        dest_config = self._tier_config(dest)
        size = max(0, int(file_size))
        throughput = max(float(source_config.get("throughput_mbps", 1.0)), 1e-9)
        transfer_ms = size * 8.0 / (throughput * 1_000_000.0) * 1000.0
        latency_ms = abs(float(dest_config.get("latency_ms", 0.0)) -
                         float(source_config.get("latency_ms", 0.0)))
        bandwidth_cost = size / (1024 ** 3) * float(self.config.get("bandwidth_cost_per_gb", 0.0))
        return MigrationCost(file_id=file_id, source_tier=source, dest_tier=dest,
                             latency_overhead_ms=latency_ms + transfer_ms,
                             bandwidth_cost=bandwidth_cost,
                             total_cost=latency_ms + transfer_ms + bandwidth_cost)

    def _tier_config(self, tier):
        value = self.tier_configs.get(tier, self.tier_configs.get(tier.value, {}))
        if isinstance(value, TierConfig):
            return value.model_dump()
        return value or {}

    def reset_file_state(self, file_id):
        self._pending.pop(file_id, None)
        self._last_migration.pop(file_id, None)
        self.file_sizes.pop(file_id, None)
