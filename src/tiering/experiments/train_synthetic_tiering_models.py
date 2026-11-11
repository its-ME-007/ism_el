"""Train and evaluate the storage-tier models on reproducible synthetic traces.

Example::

    python -m src.tiering.experiments.train_synthetic_tiering_models

Features are computed strictly before each prediction timestamp. Future trace
events are used only to create the future-access and tier targets.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             f1_score, mean_squared_error, roc_auc_score,
                             average_precision_score)

from src.tiering.ingestion.dataset_generator import DatasetGenerator
from src.tiering.ingestion.schemas import Category, Tier
from src.tiering.ingestion.workload_generator import WorkloadGenerator
from src.tiering.migration.controller import MigrationController
from src.tiering.models.tier_classifier import TierClassifier
from src.tiering.prediction.future_predictor import FutureAccessPredictor
from src.tiering.simulation.simulator import SimPySimulator
from src.tiering.workload.feature_extractor import WorkloadFeatureExtractor


SEED = 42
FILE_COUNT = 120
DURATION_HOURS = 24 * 30
PREDICTION_WINDOW_HOURS = 24
TIER_NAMES = ["HOT", "WARM", "COLD"]


def _value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _metadata_feature_dict(file, reference_time: datetime) -> dict[str, float]:
    category = _value(file.category)
    metadata = file.metadata or {}
    created_age_days = max(0.0, (reference_time - file.created_at).total_seconds() / 86400)
    modified_age_days = max(0.0, (reference_time - file.modified_at).total_seconds() / 86400)
    features = {
        "size_log_bytes": float(np.log1p(file.size_bytes)),
        "content_chars": float(len(file.content or "")),
        "created_age_days": created_age_days,
        "modified_age_days": modified_age_days,
        "indexed": float(bool(metadata.get("indexed", False))),
        "compressed": float(bool(metadata.get("compressed", False))),
    }
    vocabularies = {
        "category": [item.value for item in Category],
        "file_type": ["document", "archive", "database", "log"],
        "department": ["IT", "Finance", "HR", "Operations"],
        "confidentiality": ["public", "internal", "confidential", "restricted"],
    }
    observed = {
        "category": category,
        "file_type": str(metadata.get("file_type", "unknown")),
        "department": str(metadata.get("department", "unknown")),
        "confidentiality": str(metadata.get("confidentiality", "unknown")),
    }
    for field_name, choices in vocabularies.items():
        for choice in choices:
            features[f"{field_name}={choice}"] = float(observed[field_name] == choice)
    return features


def _feature_dict(file, history, reference_time, extractor):
    workload = extractor.extract(history, reference_time, file.created_at)
    values = workload.to_vector().astype(np.float64)
    # The cold-start sentinel is sys.maxsize; cap it at the longest relevant
    # history window so scaling and the tree model receive a meaningful value.
    values[4] = min(values[4], 24 * 30)
    values[5] = min(values[5], 24 * 365 * 10)
    values[8] = min(values[8], 100.0)
    names = [
        "access_count_1h", "access_count_24h", "access_count_7d",
        "access_count_30d", "last_access_hours", "time_since_creation_hours",
        "mean_inter_arrival_hours", "std_inter_arrival_hours",
        "read_write_ratio", "periodicity_score",
    ]
    result = {f"workload_{name}": float(value) for name, value in zip(names, values)}
    result.update(_metadata_feature_dict(file, reference_time))
    return result


def _future_count(events, start: datetime) -> int:
    end = start + timedelta(hours=PREDICTION_WINDOW_HOURS)
    return sum(start <= event.timestamp < end for event in events)


def _target_tier(access_count: int) -> int:
    # Fixed operational cutoffs: >=50 accesses/day is HOT, >=5 is WARM,
    # otherwise COLD. These are simulator labels, not human-assigned truth.
    if access_count >= 50:
        return 0
    if access_count >= 5:
        return 1
    return 2


def _matrix(samples, feature_names):
    return np.asarray([[sample["features"].get(name, 0.0) for name in feature_names]
                       for sample in samples], dtype=np.float32)


def train_and_evaluate(
    output_models: Path = Path("results/models"),
    output_metrics: Path = Path("results/metrics"),
) -> dict[str, Any]:
    output_models.mkdir(parents=True, exist_ok=True)
    output_metrics.mkdir(parents=True, exist_ok=True)
    files = DatasetGenerator(seed=SEED).generate(FILE_COUNT)
    start = datetime(2025, 1, 1)
    workload_config = {
        "hot_ratio": 0.25, "warm_ratio": 0.30, "cold_ratio": 0.35,
        "seasonal_ratio": 0.10, "hot_frequency": 5.0,
        "warm_frequency": 1.0, "cold_frequency": 0.05,
        "seasonal_period_hours": 24.0, "seasonal_amplitude": 3.0,
        "read_write_ratio": 0.8,
    }
    # A fresh generator with the same seed reproduces the assigned patterns
    # while generating the access events.
    traces = WorkloadGenerator(workload_config, SEED).generate_traces(
        files, DURATION_HOURS, start
    )
    events_by_file = defaultdict(list)
    for event in traces:
        events_by_file[event.file_id].append(event)

    extractor = WorkloadFeatureExtractor(normalize=False)
    train_days = (5, 10, 15)
    validation_day = 20
    test_day = 26

    def make_samples(days):
        samples = []
        for day in days:
            reference_time = start + timedelta(days=day)
            for file in files:
                count = _future_count(events_by_file[file.file_id], reference_time)
                samples.append({
                    "file_id": file.file_id,
                    "reference_time": reference_time,
                    "features": _feature_dict(
                        file, events_by_file[file.file_id], reference_time, extractor
                    ),
                    "future_access": int(count > 0),
                    "tier": _target_tier(count),
                    "future_count": count,
                })
        return samples

    train_samples = make_samples(train_days)
    validation_samples = make_samples((validation_day,))
    test_samples = make_samples((test_day,))
    feature_names = sorted({name for sample in train_samples
                            for name in sample["features"]})
    X_train = _matrix(train_samples, feature_names)
    X_validation = _matrix(validation_samples, feature_names)
    X_test = _matrix(test_samples, feature_names)
    y_tier_train = np.asarray([sample["tier"] for sample in train_samples])
    y_tier_validation = np.asarray([sample["tier"] for sample in validation_samples])
    y_tier_test = np.asarray([sample["tier"] for sample in test_samples])
    y_access_train = np.asarray([sample["future_access"] for sample in train_samples])
    y_access_validation = np.asarray([sample["future_access"] for sample in validation_samples])
    y_access_test = np.asarray([sample["future_access"] for sample in test_samples])

    for name, labels in (("tier", y_tier_train), ("future access", y_access_train)):
        if len(np.unique(labels)) < 2:
            raise ValueError(f"Synthetic {name} training labels have fewer than two classes")

    tier_model = TierClassifier(n_estimators=100, max_depth=4,
                                learning_rate=0.1, random_state=SEED)
    tier_model.fit(X_train, y_tier_train, feature_names=feature_names)
    val_tier_pred = tier_model.predict(X_validation)
    val_tier_metrics = {
        "accuracy": float(accuracy_score(y_tier_validation, val_tier_pred)),
        "macro_f1": float(f1_score(y_tier_validation, val_tier_pred,
                                   average="macro", zero_division=0)),
    }

    future_model = FutureAccessPredictor(
        prediction_window_hours=PREDICTION_WINDOW_HOURS,
        n_estimators=100, max_depth=4, learning_rate=0.05,
        calibration_method="clip", random_state=SEED,
    )
    future_model.fit(X_train, y_access_train, feature_names=feature_names)
    val_access_scores = future_model.predict(X_validation)
    val_access_metrics = {
        "brier_score": float(mean_squared_error(y_access_validation, val_access_scores)),
        "roc_auc": float(roc_auc_score(y_access_validation, val_access_scores))
        if len(np.unique(y_access_validation)) == 2 else None,
    }

    # Refit on train plus validation time points after the fixed model settings
    # have been evaluated, then score once on the later held-out time point.
    final_train = train_samples + validation_samples
    X_final = _matrix(final_train, feature_names)
    y_tier_final = np.asarray([sample["tier"] for sample in final_train])
    y_access_final = np.asarray([sample["future_access"] for sample in final_train])
    tier_model = TierClassifier(n_estimators=100, max_depth=4,
                                learning_rate=0.1, random_state=SEED)
    tier_model.fit(X_final, y_tier_final, feature_names=feature_names)
    tier_pred = tier_model.predict(X_test)
    future_model = FutureAccessPredictor(
        prediction_window_hours=PREDICTION_WINDOW_HOURS,
        n_estimators=100, max_depth=4, learning_rate=0.05,
        calibration_method="clip", random_state=SEED,
    )
    future_model.fit(X_final, y_access_final, feature_names=feature_names)
    access_scores = future_model.predict(X_test)

    tier_metrics = {
        "accuracy": float(accuracy_score(y_tier_test, tier_pred)),
        "macro_f1": float(f1_score(y_tier_test, tier_pred,
                                   average="macro", zero_division=0)),
        "majority_class_accuracy_baseline": float(accuracy_score(
            y_tier_test,
            np.full_like(y_tier_test, int(np.bincount(y_tier_final, minlength=3).argmax())),
        )),
        "per_class": classification_report(
            y_tier_test, tier_pred, labels=[0, 1, 2], target_names=TIER_NAMES,
            output_dict=True, zero_division=0,
        ),
    }
    access_metrics = {
        "brier_score": float(mean_squared_error(y_access_test, access_scores)),
        "roc_auc": float(roc_auc_score(y_access_test, access_scores))
        if len(np.unique(y_access_test)) == 2 else None,
        "average_precision": float(average_precision_score(y_access_test, access_scores))
        if len(np.unique(y_access_test)) == 2 else None,
        "threshold_0_5_f1": float(f1_score(
            y_access_test, access_scores >= 0.5, zero_division=0
        )),
        "positive_rate": float(y_access_test.mean()),
        "always_positive_accuracy_baseline": float(y_access_test.mean()),
    }

    tier_configs = {
        Tier.HOT: {"latency_ms": 1.0, "cost_per_gb_month": 0.02,
                   "capacity_gb": 1000.0, "throughput_mbps": 1000.0},
        Tier.WARM: {"latency_ms": 10.0, "cost_per_gb_month": 0.01,
                    "capacity_gb": 10000.0, "throughput_mbps": 100.0},
        Tier.COLD: {"latency_ms": 100.0, "cost_per_gb_month": 0.001,
                    "capacity_gb": 100000.0, "throughput_mbps": 10.0},
    }

    def feature_provider(file, history, reference_time):
        feature_values = _feature_dict(file, history, reference_time, extractor)
        return np.asarray([feature_values.get(name, 0.0) for name in feature_names],
                          dtype=np.float32)

    baseline_simulator = SimPySimulator(
        config={"decision_interval_hours": 24.0}, tiers=tier_configs
    )
    baseline_simulation = baseline_simulator.run(files, traces)
    migration_policy = MigrationController({
        "confidence_threshold": 0.55,
        "hysteresis_count": 2,
        "cooldown_hours": 24.0,
        "tiers": tier_configs,
    })
    model_simulator = SimPySimulator(
        config={"decision_interval_hours": 24.0, "feature_provider": feature_provider},
        tiers=tier_configs, migration_controller=migration_policy,
    )
    model_simulation = model_simulator.run(files, traces, classifier=tier_model)
    simulation_metrics = {
        "all_cold_baseline": baseline_simulation.metrics.model_dump(mode="json"),
        "tier_classifier": model_simulation.metrics.model_dump(mode="json"),
        "tier_classifier_migrations": len(model_simulation.migrations),
    }

    model_path = output_models / "synthetic_tiering_models.joblib"
    joblib.dump({
        "tier_classifier": tier_model,
        "future_access_predictor": future_model,
        "feature_names": feature_names,
        "seed": SEED,
        "label_rules": {"HOT": ">=50 future accesses/24h",
                        "WARM": "5-49 future accesses/24h",
                        "COLD": "0-4 future accesses/24h"},
    }, model_path)
    report = {
        "dataset": "synthetic enterprise files + generated 30-day workload",
        "seed": SEED,
        "files": FILE_COUNT,
        "access_events": len(traces),
        "prediction_window_hours": PREDICTION_WINDOW_HOURS,
        "training_reference_days": list(train_days),
        "validation_reference_day": validation_day,
        "test_reference_day": test_day,
        "training_samples": len(train_samples),
        "validation_samples": len(validation_samples),
        "test_samples": len(test_samples),
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "evaluation_note": (
            "Synthetic temporal holdout: files repeat across earlier train snapshots and later test; "
            "labels are derived from generated future accesses. This does not measure generalization "
            "to unseen organizations or real access traces."
        ),
        "tier_labels_train": dict(zip(TIER_NAMES, np.bincount(y_tier_train, minlength=3).tolist())),
        "tier_labels_test": dict(zip(TIER_NAMES, np.bincount(y_tier_test, minlength=3).tolist())),
        "tier_label_rules": {"HOT": ">=50 future accesses/24h",
                             "WARM": "5-49 future accesses/24h",
                             "COLD": "0-4 future accesses/24h"},
        "validation": {"tier_classifier": val_tier_metrics,
                       "future_access": val_access_metrics},
        "test": {"tier_classifier": tier_metrics,
                 "future_access": access_metrics},
        "simpy_simulation": simulation_metrics,
        "model_path": str(model_path),
    }
    report_path = output_metrics / "synthetic_tiering_models.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    print(json.dumps(train_and_evaluate(), indent=2))


if __name__ == "__main__":
    main()
