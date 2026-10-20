"""Runnable experiment types used by the experiment framework."""

from copy import deepcopy
from datetime import timedelta
from typing import Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from src.tiering.ingestion.schemas import Tier
from src.tiering.models.feature_fusion import FeatureFusion
from src.tiering.models.tier_classifier import TierClassifier
from src.tiering.migration.controller import MigrationController


def _fit_and_score(X_train, y_train, X_test, y_test, model_config=None):
    model = TierClassifier(**(model_config or {}))
    model.fit(np.asarray(X_train), np.asarray(y_train))
    predictions = model.predict(np.asarray(X_test))
    return {"accuracy": float(accuracy_score(y_test, predictions)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
            "predictions": np.asarray(predictions).tolist(),
            "model_version": model.model_version}


class ConventionalBaselineExperiment:
    def run(self, files, traces, baselines, **kwargs):
        sizes = {file.file_id: file.size_bytes / (1024 ** 3) for file in files}
        results = {}
        for name, baseline in baselines.items():
            policy = baseline() if isinstance(baseline, type) else deepcopy(baseline)
            transitions = 0
            decisions = 0
            for event in sorted(traces, key=lambda item: item.timestamp):
                before = policy.get_tier(event.file_id)
                policy.on_access(event.file_id, event.timestamp, sizes[event.file_id])
                transitions += policy.get_tier(event.file_id) != before
                if hasattr(policy, "on_decision_interval"):
                    decisions += len(policy.on_decision_interval(event.timestamp))
            results[name] = {"tier_occupancy": {
                str(getattr(tier, "value", tier)): float(size)
                for tier, size in policy.get_tier_occupancy().items()},
                "access_transitions": int(transitions), "reported_decisions": int(decisions)}
        return {"experiment_type": "conventional_baseline_comparison", "baselines": results}


class WorkloadMLComparisonExperiment:
    def run(self, workload_train, y_train, workload_test, y_test, model_config=None, **kwargs):
        return {"experiment_type": "workload_ml_comparison",
                "workload_ml": _fit_and_score(workload_train, y_train,
                                               workload_test, y_test, model_config)}


class _AblationExperiment:
    name = "ablation"
    disabled = ()

    def run(self, workload_train, y_train, workload_test, y_test,
            semantic_train=None, semantic_test=None, future_train=None,
            future_test=None, model_config=None, **kwargs):
        fusion = FeatureFusion({"scaling": kwargs.get("scaling", "standard"),
                                "imputation_strategy": "zeros"})
        fusion.set_ablation(list(self.disabled))
        train = fusion.fit_transform(workload_train, semantic_train, future_train)
        test = fusion.transform(workload_test, semantic_test, future_test)
        return {"experiment_type": self.name,
                "disabled_features": list(self.disabled),
                "result": _fit_and_score(train, y_train, test, y_test, model_config)}


class SemanticAblationExperiment(_AblationExperiment):
    name = "semantic_ablation"
    disabled = ("semantic",)


class RAGAblationExperiment(_AblationExperiment):
    name = "rag_ablation"

    def run(self, rag_train=None, rag_test=None, semantic_train=None,
            semantic_test=None, **kwargs):
        if rag_train is None or rag_test is None:
            raise ValueError("RAG ablation requires rag_train and rag_test feature matrices")
        if semantic_train is not None:
            semantic_train = np.concatenate((np.asarray(semantic_train), np.asarray(rag_train)), axis=1)
            semantic_test = np.concatenate((np.asarray(semantic_test), np.asarray(rag_test)), axis=1)
        else:
            semantic_train, semantic_test = rag_train, rag_test
        return super().run(semantic_train=semantic_train, semantic_test=semantic_test,
                           **kwargs)


class FuturePredictionAblationExperiment(_AblationExperiment):
    name = "future_prediction_ablation"
    disabled = ("future",)


class MigrationPolicyStressTest:
    def run(self, predictions, policy_config=None, file_sizes=None, **kwargs):
        controller = MigrationController(policy_config or {})
        states = {}
        accepted = []
        for item in sorted(predictions, key=lambda value: value["timestamp"]):
            file_id = item["file_id"]
            current = Tier(item.get("current_tier", states.get(file_id, Tier.COLD)))
            predicted = Tier(item["predicted_tier"])
            decision = controller.evaluate(file_id, current, predicted,
                                           item["probability"], item["timestamp"])
            if controller.last_evaluation_migrated:
                states[file_id] = decision.recommended_tier
                accepted.append(decision.model_dump(mode="json"))
            else:
                states[file_id] = current
        return {"experiment_type": "migration_policy_stress_test",
                "evaluations": len(predictions), "accepted_migrations": accepted,
                "migration_count": len(accepted),
                "policy": {"confidence_threshold": controller.confidence_threshold,
                           "hysteresis_count": controller.hysteresis_count,
                           "cooldown_hours": controller.cooldown.total_seconds() / 3600}}


RUNNERS = {
    "conventional_baseline_comparison": ConventionalBaselineExperiment,
    "baseline_comparison": ConventionalBaselineExperiment,
    "workload_ml_comparison": WorkloadMLComparisonExperiment,
    "workload_ml": WorkloadMLComparisonExperiment,
    "semantic_ablation": SemanticAblationExperiment,
    "rag_ablation": RAGAblationExperiment,
    "future_prediction_ablation": FuturePredictionAblationExperiment,
    "migration_policy_stress_test": MigrationPolicyStressTest,
}
