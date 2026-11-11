"""Matched semantic ablation for the storage-tiering product target.

Run with::

    python -m src.tiering.experiments.product_ablation

The benchmark is deliberately synthetic: it checks that the product experiment
can compare metadata/workload features with and without document text under the
same examples and targets. Synthetic access patterns are currently assigned
independently of document content, so this is a pipeline check and a useful null
baseline, not evidence that semantics does or does not help enterprise tiering.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             average_precision_score, brier_score_loss,
                             confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor

from src.tiering.experiments.train_synthetic_tiering_models import (
    DURATION_HOURS,
    FILE_COUNT,
    PREDICTION_WINDOW_HOURS,
    SEED,
    TIER_NAMES,
    _feature_dict,
    _future_count,
    _target_tier,
    _value,
)
from src.tiering.ingestion.dataset_generator import DatasetGenerator
from src.tiering.ingestion.schemas import Tier
from src.tiering.ingestion.workload_generator import WorkloadGenerator
from src.tiering.migration.controller import MigrationController
from src.tiering.models.tier_classifier import TierClassifier
from src.tiering.semantic.tfidf_extractor import TFIDFExtractor
from src.tiering.semantic.retrieval_context import retrieval_context_features
from src.tiering.simulation.simulator import SimPySimulator
from src.tiering.workload.feature_extractor import WorkloadFeatureExtractor


TRAIN_DAYS = (5, 10, 15)
VALIDATION_DAY = 20
TEST_DAY = 26
TIER_CUTOFFS = {"HOT": ">=50 future accesses/24h",
                "WARM": "5-49 future accesses/24h",
                "COLD": "0-4 future accesses/24h"}


def _make_samples(files, events_by_file, days, file_ids, start, extractor):
    samples = []
    for day in days:
        cutoff = start + timedelta(days=day)
        for file in files:
            if file.file_id not in file_ids:
                continue
            future_count = _future_count(events_by_file[file.file_id], cutoff)
            feature_values = _feature_dict(
                file, events_by_file[file.file_id], cutoff, extractor
            )
            # Document length is not part of the semantic group and can be
            # represented by size_bytes only when the source provides it.
            feature_values.pop("content_chars", None)
            samples.append({
                "file_id": file.file_id,
                "cutoff": cutoff.isoformat(),
                "features": feature_values,
                "future_count": future_count,
                "future_access": int(future_count > 0),
                "tier": _target_tier(future_count),
            })
    return samples


def _matrix(samples, names, semantic_by_file=None):
    rows = []
    for sample in samples:
        row = [sample["features"].get(name, 0.0) for name in names]
        if semantic_by_file is not None:
            row.extend(semantic_by_file[sample["file_id"]].tolist())
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def _summarize(model, X, samples):
    y_count = np.asarray([row["future_count"] for row in samples], dtype=np.float32)
    y_access = np.asarray([row["future_access"] for row in samples], dtype=np.int64)
    y_tier = np.asarray([row["tier"] for row in samples], dtype=np.int64)
    count_prediction = np.maximum(0.0, model["demand_model"].predict(X))
    access_classes = list(model["access_model"].classes_)
    positive_column = access_classes.index(1)
    access_probability = model["access_model"].predict_proba(X)[:, positive_column]
    access_prediction = (access_probability >= 0.5).astype(np.int64)
    tier_probabilities = model["tier_model"].predict_proba(X)
    tier_prediction = model["tier_model"].predict(X)
    return {
        "examples": len(samples),
        "future_access_count": {
            "mae": float(mean_absolute_error(y_count, count_prediction)),
            "rmse": float(mean_squared_error(y_count, count_prediction) ** 0.5),
            "actual_mean": float(y_count.mean()) if len(y_count) else None,
            "predicted_mean": float(count_prediction.mean()) if len(y_count) else None,
        },
        "future_any_access": {
            "roc_auc": float(roc_auc_score(y_access, access_probability))
            if len(np.unique(y_access)) == 2 else None,
            "average_precision": float(average_precision_score(y_access, access_probability))
            if np.any(y_access == 1) else None,
            "brier_score": float(brier_score_loss(y_access, access_probability)),
            "positive_rate": float(y_access.mean()) if len(y_access) else None,
            "threshold_0_5_precision": float(precision_score(
                y_access, access_prediction, zero_division=0
            )),
            "threshold_0_5_recall": float(recall_score(
                y_access, access_prediction, zero_division=0
            )),
            "threshold_0_5_f1": float(f1_score(
                y_access, access_prediction, zero_division=0
            )),
        },
        "operational_tier": {
            "accuracy": float(accuracy_score(y_tier, tier_prediction)),
            "macro_f1": float(f1_score(y_tier, tier_prediction,
                                       average="macro", zero_division=0)),
            "per_class": classification_report(
                y_tier, tier_prediction, labels=[0, 1, 2],
                target_names=TIER_NAMES, output_dict=True, zero_division=0,
            ),
            "confusion_matrix": confusion_matrix(
                y_tier, tier_prediction, labels=[0, 1, 2]
            ).tolist(),
            "mean_brier_score": float(np.mean(
                (tier_probabilities - np.eye(3, dtype=np.float32)[y_tier]) ** 2
            )),
        },
    }


def _train_run(run_name, feature_names, train_samples, validation_samples,
               test_groups, semantic_train=None, semantic_validation=None,
               semantic_tests=None):
    use_semantics = semantic_train is not None
    X_train = _matrix(train_samples, feature_names, semantic_train)
    X_validation = _matrix(validation_samples, feature_names, semantic_validation)
    y_count = np.asarray([row["future_count"] for row in train_samples], dtype=np.float32)
    y_access = np.asarray([row["future_access"] for row in train_samples], dtype=np.int64)
    y_tier = np.asarray([row["tier"] for row in train_samples], dtype=np.int64)

    all_tiers = np.unique(y_tier)
    if len(all_tiers) < 3:
        raise ValueError(f"{run_name} training split does not contain all three tier labels")

    demand_model = XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        objective="reg:squarederror", random_state=SEED, n_jobs=1,
        verbosity=0,
    )
    demand_model.fit(X_train, y_count)
    if len(np.unique(y_access)) < 2:
        raise ValueError(f"{run_name} training split has only one future-access class")
    access_model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        objective="binary:logistic", eval_metric="logloss",
        random_state=SEED, n_jobs=1, verbosity=0,
    )
    access_model.fit(X_train, y_access)
    tier_model = TierClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=SEED
    )
    tier_model.fit(X_train, y_tier, feature_names=feature_names +
                   ([f"semantic_tfidf_{i}" for i in range(
                       X_train.shape[1] - len(feature_names))] if use_semantics else []))

    artifact = {
        "demand_model": demand_model,
        "access_model": access_model,
        "tier_model": tier_model,
        "feature_names": feature_names +
            ([f"semantic_tfidf_{i}" for i in range(
                X_train.shape[1] - len(feature_names))] if use_semantics else []),
        "feature_group": "metadata_workload_plus_tfidf" if use_semantics
                         else "metadata_workload",
    }
    validation_report = _summarize(artifact, X_validation, validation_samples)
    test_reports = {}
    for label, samples in test_groups.items():
        matrix = _matrix(samples, feature_names,
                         semantic_tests[label] if use_semantics else None)
        test_reports[label] = _summarize(artifact, matrix, samples)
    return artifact, validation_report, test_reports


def run_product_ablation(
    output_models: Path = Path("results/models"),
    output_metrics: Path = Path("results/metrics"),
    file_count: int = FILE_COUNT,
) -> dict[str, Any]:
    """Fit matched metadata/workload, semantic-only, and combined runs."""
    output_models.mkdir(parents=True, exist_ok=True)
    output_metrics.mkdir(parents=True, exist_ok=True)

    files = DatasetGenerator(seed=SEED).generate(file_count)
    start = datetime(2025, 1, 1)
    workload_config = {
        "hot_ratio": 0.25, "warm_ratio": 0.30, "cold_ratio": 0.35,
        "seasonal_ratio": 0.10, "hot_frequency": 5.0,
        "warm_frequency": 1.0, "cold_frequency": 0.05,
        "seasonal_period_hours": 24.0, "seasonal_amplitude": 3.0,
        "read_write_ratio": 0.8,
    }
    traces = WorkloadGenerator(workload_config, SEED).generate_traces(
        files, DURATION_HOURS, start
    )
    events_by_file = defaultdict(list)
    for event in traces:
        events_by_file[event.file_id].append(event)

    categories = [_value(file.category) for file in files]
    train_files, heldout_files = train_test_split(
        files, test_size=0.20, random_state=SEED, stratify=categories
    )
    train_ids = {file.file_id for file in train_files}
    heldout_ids = {file.file_id for file in heldout_files}
    all_ids = train_ids | heldout_ids

    workload_extractor = WorkloadFeatureExtractor(normalize=False)
    train_samples = _make_samples(
        files, events_by_file, TRAIN_DAYS, train_ids, start, workload_extractor
    )
    validation_samples = _make_samples(
        files, events_by_file, (VALIDATION_DAY,), train_ids, start, workload_extractor
    )
    known_test = _make_samples(
        files, events_by_file, (TEST_DAY,), train_ids, start, workload_extractor
    )
    unseen_test = _make_samples(
        files, events_by_file, (TEST_DAY,), heldout_ids, start, workload_extractor
    )
    all_test = _make_samples(
        files, events_by_file, (TEST_DAY,), all_ids, start, workload_extractor
    )
    test_groups = {
        "future_period_known_files": known_test,
        "future_period_unseen_files": unseen_test,
        "future_period_all_files": all_test,
    }

    metadata_feature_names = sorted({
        name for sample in train_samples for name in sample["features"]
    })
    earliest_training_cutoff = start + timedelta(days=min(TRAIN_DAYS))
    train_documents = [
        file for file in train_files
        if (file.content or "").strip()
        and file.created_at <= earliest_training_cutoff
        and file.modified_at <= earliest_training_cutoff
    ]
    tfidf = TFIDFExtractor({"max_features": 256, "ngram_range": (1, 2)})
    tfidf.fit(train_documents)
    semantic_by_file = {
        file.file_id: tfidf.transform([
            file.content if file.created_at <= earliest_training_cutoff
            and file.modified_at <= earliest_training_cutoff else ""
        ])[0] for file in files
    }
    retrieval_feature_names = [
        "retrieval_neighbor_count", "retrieval_top_similarity",
        "retrieval_mean_similarity", "retrieval_neighbor_mean_accesses_24h",
        "retrieval_neighbor_mean_accesses_7d",
    ]
    for samples in (train_samples, validation_samples, known_test, unseen_test):
        for sample in samples:
            document = next(file for file in files if file.file_id == sample["file_id"])
            context = retrieval_context_features(
                document, files, semantic_by_file, events_by_file,
                datetime.fromisoformat(sample["cutoff"]), top_k=5,
            )
            sample["features"].update({f"retrieval_{key}": value
                                       for key, value in context.items()})

    # Persist the exact cutoff-time rows used by the matched A/B/C experiment.
    # Identifiers and future labels stay outside the feature groups.
    example_path = Path("results/datasets/product_cutoff_examples.jsonl")
    example_path.parent.mkdir(parents=True, exist_ok=True)
    file_by_id = {file.file_id: file for file in files}
    split_samples = (
        [("train", train_samples, train_ids),
         ("validation", validation_samples, train_ids),
         ("test_known_file", known_test, train_ids),
         ("test_unseen_file", unseen_test, heldout_ids)]
    )
    example_count = 0
    with example_path.open("w", encoding="utf-8", newline="\n") as destination:
        for split_name, samples, split_ids in split_samples:
            for sample in samples:
                file = file_by_id[sample["file_id"]]
                semantic = semantic_by_file[file.file_id].astype(float).tolist()
                destination.write(json.dumps({
                    "example_id": f"{file.file_id}@{sample['cutoff']}",
                    "document_id": file.file_id,
                    "cutoff": sample["cutoff"],
                    "split": split_name,
                    "text_sha256": hashlib.sha256(
                        (file.content or "").encode("utf-8")
                    ).hexdigest(),
                    "semantic_representation": "training-fitted-tfidf-v1",
                    "features": {
                        "metadata_workload": {
                            key: value for key, value in sample["features"].items()
                            if not key.startswith("retrieval_")
                        },
                        "retrieval_context": {
                            key.removeprefix("retrieval_"): value
                            for key, value in sample["features"].items()
                            if key.startswith("retrieval_")
                        },
                        "semantic_tfidf": semantic,
                    },
                    "targets": {
                        "future_access_count_24h": sample["future_count"],
                        "future_any_access_24h": sample["future_access"],
                        "operational_tier_24h": TIER_NAMES[sample["tier"]],
                    },
                }, ensure_ascii=False) + "\n")
                example_count += 1
    semantic_train = {file_id: semantic_by_file[file_id] for file_id in train_ids}
    semantic_validation = {file_id: semantic_by_file[file_id] for file_id in train_ids}
    semantic_tests = {name: semantic_by_file for name in test_groups}

    # Isolate the text-only ablation from metadata and workload features.
    semantic_only_names = [f"semantic_tfidf_{i}" for i in range(
        len(next(iter(semantic_by_file.values())))
    )]
    semantic_only_train = [
        {**sample, "features": {name: float(value) for name, value in
                                 zip(semantic_only_names,
                                     semantic_by_file[sample["file_id"]])}}
        for sample in train_samples
    ]
    semantic_only_validation = [
        {**sample, "features": {name: float(value) for name, value in
                                 zip(semantic_only_names,
                                     semantic_by_file[sample["file_id"]])}}
        for sample in validation_samples
    ]
    semantic_only_tests = {
        name: [
            {**sample, "features": {key: float(value) for key, value in
                                     zip(semantic_only_names,
                                         semantic_by_file[sample["file_id"]])}}
            for sample in samples
        ]
        for name, samples in test_groups.items()
    }

    runs = {}
    artifacts = {}

    # A: baseline; B: text-only; C: combined; D: combined plus past-only retrieval.
    for (run_name, fit_features, fit_train, fit_validation, fit_tests,
         sem_train, sem_val, sem_tests) in (
        ("A_metadata_workload", metadata_feature_names, train_samples,
         validation_samples, test_groups, None, None, None),
        ("B_semantic_only", semantic_only_names, semantic_only_train,
         semantic_only_validation, semantic_only_tests, None, None, None),
        ("C_combined", metadata_feature_names, train_samples,
         validation_samples, test_groups, semantic_train,
         semantic_validation, semantic_tests),
        ("D_combined_retrieval", metadata_feature_names + retrieval_feature_names,
         train_samples, validation_samples, test_groups, semantic_train,
         semantic_validation, semantic_tests),
    ):
        artifact, val_metrics, test_metrics = _train_run(
            run_name, fit_features, fit_train, fit_validation, fit_tests,
            sem_train, sem_val, sem_tests,
        )
        runs[run_name] = {
            "feature_group": artifact["feature_group"],
            "feature_count": len(artifact["feature_names"]),
            "feature_names": artifact["feature_names"],
            "validation": val_metrics,
            "test": test_metrics,
        }
        artifacts[run_name] = artifact

    tier_configs = {
        Tier.HOT: {"latency_ms": 1.0, "cost_per_gb_month": 0.02,
                   "capacity_gb": 1000.0, "throughput_mbps": 1000.0},
        Tier.WARM: {"latency_ms": 10.0, "cost_per_gb_month": 0.01,
                    "capacity_gb": 10000.0, "throughput_mbps": 100.0},
        Tier.COLD: {"latency_ms": 100.0, "cost_per_gb_month": 0.001,
                    "capacity_gb": 100000.0, "throughput_mbps": 10.0},
    }
    evaluation_start = start + timedelta(days=TEST_DAY)
    simulation_reports = {}
    baseline_simulation = SimPySimulator(
        config={"decision_interval_hours": 24.0,
                "evaluation_start_time": evaluation_start},
        tiers=tier_configs,
    ).run(files, traces)
    simulation_reports["all_cold_baseline"] = (
        baseline_simulation.metrics.model_dump(mode="json")
    )

    def make_feature_provider(run_name, artifact):
        run_features = artifact["feature_names"]

        def provide(file, history, cutoff):
            if run_name == "B_semantic_only":
                values = {name: float(value) for name, value in
                          zip(semantic_only_names, semantic_by_file[file.file_id])}
            else:
                values = _feature_dict(file, history, cutoff, workload_extractor)
                values.pop("content_chars", None)
                if run_name in {"C_combined", "D_combined_retrieval"}:
                    values.update({name: float(value) for name, value in
                                   zip(semantic_only_names,
                                       semantic_by_file[file.file_id])})
                if run_name == "D_combined_retrieval":
                    context = retrieval_context_features(
                        file, files, semantic_by_file, events_by_file,
                        cutoff, top_k=5,
                    )
                    values.update({f"retrieval_{key}": float(value)
                                   for key, value in context.items()})
            return np.asarray([values.get(name, 0.0) for name in run_features],
                              dtype=np.float32)

        return provide

    for run_name, artifact in artifacts.items():
        policy = MigrationController({
            "confidence_threshold": 0.55,
            "hysteresis_count": 2,
            "cooldown_hours": 24.0,
            "tiers": tier_configs,
        })
        simulation = SimPySimulator(
            config={
                "decision_interval_hours": 24.0,
                "evaluation_start_time": evaluation_start,
                "feature_provider": make_feature_provider(run_name, artifact),
            },
            tiers=tier_configs,
            migration_controller=policy,
        ).run(files, traces, classifier=artifact["tier_model"])
        simulation_reports[run_name] = {
            **simulation.metrics.model_dump(mode="json"),
            "migration_count": len(simulation.migrations),
        }

    model_path = output_models / "product_ablation_synthetic.joblib"
    joblib.dump({
        "dataset": "synthetic enterprise files + generated 30-day workload",
        "seed": SEED,
        "train_file_ids": sorted(train_ids),
        "heldout_file_ids": sorted(heldout_ids),
        "tfidf": tfidf,
        "runs": artifacts,
        "prediction_window_hours": PREDICTION_WINDOW_HOURS,
        "tier_label_rules": TIER_CUTOFFS,
        "retrieval": {
            "top_k": 5,
            "permission_rule": "shared explicit metadata.access_groups; missing groups means no neighbors",
            "content_rule": "candidate created_at and modified_at must be at or before cutoff",
            "access_rule": "neighbor access statistics include events at or before cutoff only",
        },
    }, model_path)

    report = {
        "dataset": "synthetic enterprise files + generated 30-day workload",
        "seed": SEED,
        "files": len(files),
        "access_events": len(traces),
        "prediction_window_hours": PREDICTION_WINDOW_HOURS,
        "training_reference_days": list(TRAIN_DAYS),
        "validation_reference_day": VALIDATION_DAY,
        "test_reference_day": TEST_DAY,
        "file_split": {
            "training_file_count": len(train_ids),
            "heldout_file_count": len(heldout_ids),
            "method": "stratified by generated category; fixed seed",
        },
        "tier_label_rules": TIER_CUTOFFS,
        "runs": runs,
        "simpy_evaluation": {
            "evaluation_start": evaluation_start.isoformat(),
            "warmup_policy": (
                "All pre-cutoff events warm the workload history; pre-cutoff "
                "accesses, storage cost, and migrations are excluded from metrics."
            ),
            "same_access_trace_and_tier_costs": True,
            "results": simulation_reports,
        },
        "evaluation_note": (
            "Synthetic pipeline benchmark only. The generator assigns access patterns "
            "independently of document content, so semantics is not designed to predict "
            "future demand. The unseen-file score uses IDs excluded from model fitting; "
            "the known-file score evaluates a later period for training identities."
        ),
        "semantic_representation": {
            "type": "TF-IDF",
            "max_features": 256,
            "fit_documents": len(train_documents),
            "fit_scope": "training file identities only",
            "input_policy": "document content only; no filename/category/metadata",
        },
        "model_path": str(model_path),
        "cutoff_examples": {
            "path": str(example_path),
            "rows": example_count,
            "split_counts": {
                "train": len(train_samples),
                "validation": len(validation_samples),
                "test_known_file": len(known_test),
                "test_unseen_file": len(unseen_test),
            },
            "feature_groups": ["metadata_workload", "semantic_tfidf",
                               "retrieval_context"],
            "targets": ["future_access_count_24h", "future_any_access_24h",
                        "operational_tier_24h"],
            "leakage_note": (
                "Features use cutoff-time metadata/workload; TF-IDF is fit on train-file "
                "documents only, and unavailable later text is replaced with an empty "
                "representation. Retrieval uses shared explicit access_groups and past-only "
                "neighbor events. Future labels are stored separately under targets."
            ),
            "retrieval_neighbors_available": sum(
                int(sample["features"].get("retrieval_neighbor_count", 0) > 0)
                for samples in (train_samples, validation_samples, known_test, unseen_test)
                for sample in samples
            ),
        },
        "retrieval": {
            "top_k": 5,
            "permission_rule": "shared explicit metadata.access_groups; missing groups means no neighbors",
            "content_rule": "candidate created_at and modified_at must be at or before cutoff",
            "access_rule": "neighbor access statistics include events at or before cutoff only",
            "retrieval_examples_with_neighbors": sum(
                int(sample["features"].get("retrieval_neighbor_count", 0) > 0)
                for samples in (train_samples, validation_samples, known_test, unseen_test)
                for sample in samples
            ),
        },
    }
    report_path = output_metrics / "product_ablation_synthetic.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    print(json.dumps(run_product_ablation(), indent=2))


if __name__ == "__main__":
    main()
