"""Discrete event simulation of storage access and tier migration."""

from collections import defaultdict
from datetime import timedelta
import time

import numpy as np
import simpy

from src.tiering.ingestion.schemas import (
    FileDocument, MigrationDecision, SimulationMetrics, SimulationResult,
    Tier, TierConfig,
)


class SimPySimulator:
    def __init__(self, config=None, tiers=None, migration_controller=None):
        self.config = dict(config or {})
        self.tiers = tiers or {}
        self.migration_controller = migration_controller
        self.feature_provider = self.config.get("feature_provider")

    def run(self, files, traces, classifier=None):
        started = time.perf_counter()
        files_by_id = {file.file_id: file for file in files}
        tiers = {file_id: Tier.COLD for file_id in files_by_id}
        history = defaultdict(list)
        events = sorted(traces, key=lambda event: event.timestamp)
        if not events:
            return SimulationResult(
                metrics=SimulationMetrics(total_accesses=0, total_migrations=0,
                                          avg_access_latency_ms=0, p95_access_latency_ms=0,
                                          total_storage_cost=0, cache_hit_rate=0),
                migrations=[], tier_distribution_timeline=[],
                execution_time_seconds=time.perf_counter() - started)

        env = simpy.Environment()
        origin = events[0].timestamp
        state = {"latencies": [], "migrations": [], "timeline": [], "hot": 0,
                 "storage_cost": 0.0, "last_seconds": 0.0}
        interval_seconds = float(self.config.get("decision_interval_hours", 1.0)) * 3600
        next_decision = interval_seconds

        def snapshot(seconds):
            nonlocal next_decision
            duration_hours = max(0.0, (seconds - state["last_seconds"]) / 3600.0)
            occupancy = defaultdict(float)
            for file_id, tier in tiers.items():
                occupancy[tier] += files_by_id[file_id].size_bytes / (1024 ** 3)
            for tier, amount in occupancy.items():
                cfg = self._tier_config(tier)
                state["storage_cost"] += amount * float(cfg.get("cost_per_gb_month", 0.0)) * duration_hours / 730.0
            total = sum(occupancy.values())
            state["timeline"].append({tier.value: amount / total if total else 0.0
                                       for tier, amount in occupancy.items()})
            state["last_seconds"] = seconds
            if classifier is None or self.migration_controller is None:
                next_decision += interval_seconds
                return
            for file_id, file in files_by_id.items():
                if not history[file_id] or self.feature_provider is None:
                    continue
                decision_time = origin + timedelta(seconds=seconds)
                features = np.asarray(self.feature_provider(file, history[file_id], decision_time), dtype=np.float32)
                probabilities = classifier.predict_proba(features.reshape(1, -1))[0]
                predicted_index = int(np.argmax(probabilities))
                predicted_tier = (Tier.HOT, Tier.WARM, Tier.COLD)[predicted_index]
                decision = self.migration_controller.evaluate(
                    file_id, tiers[file_id], predicted_tier, float(probabilities[predicted_index]),
                    decision_time)
                if self.migration_controller.last_evaluation_migrated:
                    source = tiers[file_id]
                    tiers[file_id] = decision.recommended_tier
                    state["migrations"].append(decision)
                    cost = self.migration_controller.get_migration_cost(
                        file.size_bytes, source, decision.recommended_tier)
                    state["storage_cost"] += cost.total_cost
            next_decision += interval_seconds

        def process():
            nonlocal next_decision
            for event in events:
                seconds = (event.timestamp - origin).total_seconds()
                yield env.timeout(max(0.0, seconds - env.now))
                while seconds >= next_decision:
                    snapshot(next_decision)
                file = files_by_id.get(event.file_id)
                if file is None:
                    continue
                tier = tiers[event.file_id]
                cfg = self._tier_config(tier)
                throughput = max(float(cfg.get("throughput_mbps", 1.0)), 1e-9)
                transfer_ms = (event.size_bytes or file.size_bytes) * 8 / (throughput * 1_000_000) * 1000
                state["latencies"].append(float(cfg.get("latency_ms", 0.0)) + transfer_ms)
                state["hot"] += tier == Tier.HOT
                history[event.file_id].append(event)
            snapshot((events[-1].timestamp - origin).total_seconds())

        env.process(process())
        env.run()
        latencies = state["latencies"]
        metrics = SimulationMetrics(
            total_accesses=len(latencies), total_migrations=len(state["migrations"]),
            avg_access_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
            p95_access_latency_ms=float(np.percentile(latencies, 95)) if latencies else 0.0,
            total_storage_cost=float(state["storage_cost"]),
            cache_hit_rate=float(state["hot"] / len(latencies)) if latencies else 0.0,
        )
        return SimulationResult(metrics=metrics, migrations=state["migrations"],
                                tier_distribution_timeline=state["timeline"],
                                execution_time_seconds=time.perf_counter() - started)

    def _tier_config(self, tier):
        config = self.tiers.get(tier, self.tiers.get(tier.value, {}))
        if isinstance(config, TierConfig):
            return config.model_dump()
        return config or {}
