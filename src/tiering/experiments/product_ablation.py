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
from sklearn.utils.class_weight import compute_sample_weight
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
from src.tiering.semantic.retrieval_context import RetrievalIndex
from src.tiering.simulation.simulator import SimPySimulator
from src.tiering.workload.feature_extractor import WorkloadFeatureExtractor


TRAIN_DAYS = (5, 10, 15)
VALIDATION_DAY = 20
TEST_DAY = 26
TIER_CUTOFFS = {"HOT": ">=50 future accesses/24h",
                "WARM": "5-49 future accesses/24h",
                "COLD": "0-4 future accesses/24h"}


def served_share(confusion):
    """Predicted tier mix from a confusion matrix (rows actual, columns predicted)."""
    predicted = np.asarray(confusion, dtype=np.float64).sum(axis=0)
    return {name: float(value / max(1.0, predicted.sum()))
            for name, value in zip(TIER_NAMES, predicted)}


def _progress(message):
    import sys
    print(f"[{datetime.now():%H:%M:%S}] {message}", file=sys.stderr, flush=True)


def _make_samples(files, events_by_file, days, file_ids, start, extractor,
                  max_age_days=None):
    samples = []
    for day in days:
        cutoff = start + timedelta(days=day)
        for file in files:
            if file.file_id not in file_ids:
                continue
            if file.created_at > cutoff:
                continue  # document did not exist at this cutoff
            if max_age_days is not None and \
                    cutoff - file.created_at > timedelta(days=max_age_days):
                continue  # target is zero by construction of the access proxy
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
                # Text is usable only if this content version existed at cutoff.
                "text_visible": file.modified_at <= cutoff,
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
            vector = semantic_by_file[sample["file_id"]]
            row.extend(vector.tolist() if sample.get("text_visible", True)
                       else [0.0] * len(vector))
        rows.append(row)
    return np.asarray(rows, dtype=np.float32)


def _predict(model, X):
    access_classes = list(model["access_model"].classes_)
    return {
        "count": np.maximum(0.0, model["demand_model"].predict(X)),
        "access": model["access_model"].predict_proba(X)[:, access_classes.index(1)],
        "tier_probabilities": model["tier_model"].predict_proba(X),
        "tier": np.asarray(model["tier_model"].predict(X), dtype=np.int64),
    }


def _summarize(model, X, samples, predictions=None):
    y_count = np.asarray([row["future_count"] for row in samples], dtype=np.float32)
    y_access = np.asarray([row["future_access"] for row in samples], dtype=np.int64)
    y_tier = np.asarray([row["tier"] for row in samples], dtype=np.int64)
    predictions = predictions or _predict(model, X)
    count_prediction = predictions["count"]
    access_probability = predictions["access"]
    access_prediction = (access_probability >= 0.5).astype(np.int64)
    tier_probabilities = predictions["tier_probabilities"]
    tier_prediction = predictions["tier"]
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
               semantic_tests=None, balance_tiers=False):
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
    # Real re-engagement targets are ~97% COLD; unweighted, the tier head
    # collapses to always-COLD. Balanced weights keep HOT/WARM recoverable.
    tier_model.fit(X_train, y_tier, feature_names=feature_names +
                   ([f"semantic_tfidf_{i}" for i in range(
                       X_train.shape[1] - len(feature_names))] if use_semantics else []),
                   sample_weight=compute_sample_weight("balanced", y_tier)
                   if balance_tiers else None)

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
    test_reports, test_predictions = {}, {}
    for label, samples in test_groups.items():
        matrix = _matrix(samples, feature_names,
                         semantic_tests[label] if use_semantics else None)
        test_predictions[label] = _predict(artifact, matrix)
        test_reports[label] = _summarize(artifact, matrix, samples,
                                         test_predictions[label])
    artifact["_test_predictions"] = test_predictions
    return artifact, validation_report, test_reports


def _persistence_predictions(samples, tier_of):
    """Naive forecast: the next 24h look like the last 24h."""
    past = np.asarray([row["features"].get("workload_access_count_24h", 0.0)
                       for row in samples], dtype=np.float64)
    week = np.asarray([row["features"].get("workload_access_count_7d", 0.0)
                       for row in samples], dtype=np.float64)
    score = past + week / 1000.0  # tie-break by weekly activity for ranking
    probability = np.clip(score / (1.0 + score), 0.0, 1.0)
    tiers = np.asarray([tier_of(int(value)) for value in past], dtype=np.int64)
    return {"count": past, "access": probability,
            "tier_probabilities": np.eye(3)[tiers], "tier": tiers}


def _metric_values(samples, predictions, index):
    y_count = np.asarray([samples[i]["future_count"] for i in index], dtype=np.float64)
    y_access = np.asarray([samples[i]["future_access"] for i in index], dtype=np.int64)
    y_tier = np.asarray([samples[i]["tier"] for i in index], dtype=np.int64)
    access = predictions["access"][index]
    return {
        "tier_macro_f1": float(f1_score(y_tier, predictions["tier"][index],
                                        average="macro", labels=[0, 1, 2],
                                        zero_division=0)),
        "any_access_roc_auc": float(roc_auc_score(y_access, access))
        if len(np.unique(y_access)) == 2 else float("nan"),
        "any_access_average_precision": float(average_precision_score(y_access, access))
        if y_access.any() else float("nan"),
        "count_mae": float(mean_absolute_error(y_count, predictions["count"][index])),
    }


def bootstrap_comparison(samples, predictions_by_run, pairs, reps=1000, seed=SEED):
    """Document-clustered bootstrap CIs per run and for paired run differences.

    A document can appear at several test cutoffs, so documents (not rows) are
    resampled. Differences are computed on the same resample for both runs.
    """
    rng = np.random.default_rng(seed)
    rows_by_doc = defaultdict(list)
    for index, sample in enumerate(samples):
        rows_by_doc[sample["file_id"]].append(index)
    clusters = list(rows_by_doc.values())
    draws = {run: defaultdict(list) for run in predictions_by_run}
    for _ in range(reps):
        picked = rng.integers(0, len(clusters), len(clusters))
        index = np.asarray([i for c in picked for i in clusters[c]], dtype=np.int64)
        for run, predictions in predictions_by_run.items():
            for name, value in _metric_values(samples, predictions, index).items():
                draws[run][name].append(value)

    def interval(values):
        values = np.asarray(values, dtype=np.float64)
        values = values[~np.isnan(values)]
        if not len(values):
            return None
        return {"mean": float(values.mean()),
                "ci95": [float(np.percentile(values, 2.5)),
                         float(np.percentile(values, 97.5))]}

    full = np.arange(len(samples))
    per_run = {
        run: {name: {"point": value, **(interval(draws[run][name]) or {})}
              for name, value in _metric_values(samples, predictions, full).items()}
        for run, predictions in predictions_by_run.items()
    }
    differences = {}
    for left, right in pairs:
        label = f"{left} - {right}"
        differences[label] = {}
        for name in draws[left]:
            delta = np.asarray(draws[left][name]) - np.asarray(draws[right][name])
            delta = delta[~np.isnan(delta)]
            if not len(delta):
                continue
            better = delta < 0 if name == "count_mae" else delta > 0
            differences[label][name] = {
                "mean": float(delta.mean()),
                "ci95": [float(np.percentile(delta, 2.5)),
                         float(np.percentile(delta, 97.5))],
                "share_of_resamples_left_better": float(better.mean()),
            }
    return {"method": f"document-clustered bootstrap, {reps} resamples, seed {seed}",
            "documents": len(clusters), "rows": len(samples),
            "runs": per_run, "paired_differences": differences}


def run_product_ablation(
    output_models: Path = Path("results/models"),
    output_metrics: Path = Path("results/metrics"),
    file_count: int = FILE_COUNT,
    data_bundle=None,
    bootstrap_reps: int = 1000,
) -> dict[str, Any]:
    """Fit matched metadata/workload, semantic-only, and combined runs.

    ``data_bundle`` (an EnronBundle or any object exposing ``files``, ``traces``,
    ``start``, ``train_days``, ``validation_day``, ``test_day`` and ``stats``)
    swaps the synthetic generator for a real corpus. When ``None`` the synthetic
    pipeline runs exactly as before and stays a useful null baseline.
    """
    output_models.mkdir(parents=True, exist_ok=True)
    output_metrics.mkdir(parents=True, exist_ok=True)

    if data_bundle is None:
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
        train_days, validation_days, test_days = TRAIN_DAYS, [VALIDATION_DAY], [TEST_DAY]
        max_age_days = None
        suffix = "synthetic"
        dataset_label = "synthetic enterprise files + generated 30-day workload"
    else:
        files = list(data_bundle.files)
        traces = list(data_bundle.traces)
        start = data_bundle.start
        train_days = list(data_bundle.train_days)
        validation_days = list(getattr(data_bundle, "validation_days",
                                       [data_bundle.validation_day]))
        test_days = list(getattr(data_bundle, "test_days", [data_bundle.test_day]))
        max_age_days = getattr(data_bundle, "max_sample_age_days", None)
        suffix = "enron"
        dataset_label = "Enron email corpus (real text + participant access groups)"

    events_by_file = defaultdict(list)
    for event in traces:
        events_by_file[event.file_id].append(event)

    categories = [_value(file.category) for file in files]
    stratify = categories if len(set(categories)) > 1 else None
    train_files, heldout_files = train_test_split(
        files, test_size=0.20, random_state=SEED, stratify=stratify
    )
    train_ids = {file.file_id for file in train_files}
    heldout_ids = {file.file_id for file in heldout_files}
    all_ids = train_ids | heldout_ids

    workload_extractor = WorkloadFeatureExtractor(normalize=False)
    _progress(f"{len(files)} files, {len(traces)} events; building cutoff samples")
    def samples_for(days, ids):
        return _make_samples(files, events_by_file, days, ids, start,
                             workload_extractor, max_age_days)

    train_samples = samples_for(train_days, train_ids)
    validation_samples = samples_for(validation_days, train_ids)
    known_test = samples_for(test_days, train_ids)
    unseen_test = samples_for(test_days, heldout_ids)
    all_test = samples_for(test_days, all_ids)
    test_groups = {
        "future_period_known_files": known_test,
        "future_period_unseen_files": unseen_test,
        "future_period_all_files": all_test,
    }

    # Tier labels are operational buckets, not ground truth. The synthetic
    # 50/5-accesses-per-day cutoffs assume a hot-frequency generator that real
    # email re-engagement never reaches, so derive data-relative thresholds from
    # the training future-count distribution to keep all three tiers populated.
    if data_bundle is None:
        tier_label_rules = TIER_CUTOFFS
        tier_of = _target_tier
    else:
        train_counts = sorted(sample["future_count"] for sample in train_samples)
        warm_threshold = max(1, int(np.percentile(train_counts, 70)))
        hot_threshold = max(warm_threshold + 1, int(np.percentile(train_counts, 90)))

        def _relative_tier(count: int) -> int:
            if count >= hot_threshold:
                return 0
            if count >= warm_threshold:
                return 1
            return 2

        for samples in (train_samples, validation_samples, known_test,
                        unseen_test, all_test):
            for sample in samples:
                sample["tier"] = _relative_tier(sample["future_count"])
        tier_of = _relative_tier
        tier_label_rules = {
            "HOT": f">={hot_threshold} future accesses/24h",
            "WARM": f"{warm_threshold}-{hot_threshold - 1} future accesses/24h",
            "COLD": f"0-{warm_threshold - 1} future accesses/24h",
        }

    metadata_feature_names = sorted({
        name for sample in train_samples for name in sample["features"]
    })
    earliest_training_cutoff = start + timedelta(days=min(train_days))
    train_documents = [
        file for file in train_files
        if (file.content or "").strip()
        and file.created_at <= earliest_training_cutoff
        and file.modified_at <= earliest_training_cutoff
    ]
    tfidf = TFIDFExtractor({"max_features": 256, "ngram_range": (1, 2)})
    tfidf.fit(train_documents)
    # The vectorizer (vocabulary + IDF) is fit only on documents visible at the
    # earliest training cutoff. Each document is then embedded from its own
    # content; whether that content is visible at a given cutoff is enforced per
    # sample (text_visible) and per retrieval candidate (modified_at <= cutoff),
    # matching how the MVP predictor serves C and D.
    semantic_by_file = {
        file.file_id: tfidf.transform([file.content or ""])[0] for file in files
    }
    zero_semantic = np.zeros_like(next(iter(semantic_by_file.values())))

    def _visible_semantic(file, cutoff):
        return (semantic_by_file[file.file_id] if file.modified_at <= cutoff
                else zero_semantic)
    retrieval_feature_names = [
        "retrieval_neighbor_count", "retrieval_top_similarity",
        "retrieval_mean_similarity", "retrieval_neighbor_mean_accesses_24h",
        "retrieval_neighbor_mean_accesses_7d",
    ]
    file_by_id = {file.file_id: file for file in files}
    retrieval_index = RetrievalIndex(files, semantic_by_file, events_by_file)
    _progress("adding retrieval context")
    # all_test holds its own sample dicts; it must be enriched too, or D is
    # scored on all-files rows with an empty retrieval context.
    for samples in (train_samples, validation_samples, known_test, unseen_test, all_test):
        for sample in samples:
            document = file_by_id[sample["file_id"]]
            context = retrieval_index.features(
                document, datetime.fromisoformat(sample["cutoff"]), top_k=5,
            )
            sample["features"].update({f"retrieval_{key}": value
                                       for key, value in context.items()})

    # Persist the exact cutoff-time rows used by the matched A/B/C experiment.
    # Identifiers and future labels stay outside the feature groups.
    example_name = ("product_cutoff_examples.jsonl" if data_bundle is None
                    else "enron_cutoff_examples.jsonl")
    example_path = Path("results/datasets") / example_name
    example_path.parent.mkdir(parents=True, exist_ok=True)
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
                semantic = _visible_semantic(
                    file, datetime.fromisoformat(sample["cutoff"])
                ).astype(float).tolist()
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
                                     _visible_semantic(file_by_id[sample["file_id"]],
                                                       datetime.fromisoformat(sample["cutoff"])))}}
        for sample in train_samples
    ]
    semantic_only_validation = [
        {**sample, "features": {name: float(value) for name, value in
                                 zip(semantic_only_names,
                                     _visible_semantic(file_by_id[sample["file_id"]],
                                                       datetime.fromisoformat(sample["cutoff"])))}}
        for sample in validation_samples
    ]
    semantic_only_tests = {
        name: [
            {**sample, "features": {key: float(value) for key, value in
                                     zip(semantic_only_names,
                                         _visible_semantic(file_by_id[sample["file_id"]],
                                                           datetime.fromisoformat(sample["cutoff"])))}}
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
        _progress(f"training {run_name}")
        artifact, val_metrics, test_metrics = _train_run(
            run_name, fit_features, fit_train, fit_validation, fit_tests,
            sem_train, sem_val, sem_tests, balance_tiers=data_bundle is not None,
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
    evaluation_start = start + timedelta(days=test_days[0])
    # Simulate 28 days from the first test cutoff (or to the end of the trace):
    # SimPy makes one single-row prediction per file per day, so the full Enron
    # span is impractical. Placement economics over every test cutoff are
    # covered by cost_evaluation.
    evaluation_end = start + timedelta(days=test_days[0] + 28)
    # With an age-bounded proxy, older documents cannot be accessed in the
    # window; simulating them only adds per-file daily predictions.
    oldest = (evaluation_start - timedelta(days=max_age_days)
              if max_age_days is not None else None)
    sim_files = [file for file in files if file.created_at < evaluation_end
                 and (oldest is None or file.created_at >= oldest)]
    sim_ids = {file.file_id for file in sim_files}
    sim_traces = [event for event in traces
                  if event.timestamp < evaluation_end and event.file_id in sim_ids]
    simulation_reports = {}
    _progress(f"SimPy: {len(sim_files)} files, {len(sim_traces)} events")
    baseline_simulation = SimPySimulator(
        config={"decision_interval_hours": 24.0,
                "evaluation_start_time": evaluation_start},
        tiers=tier_configs,
    ).run(sim_files, sim_traces)
    simulation_reports["all_cold_baseline"] = (
        baseline_simulation.metrics.model_dump(mode="json")
    )

    def make_feature_provider(run_name, artifact):
        run_features = artifact["feature_names"]

        def provide(file, history, cutoff):
            if run_name == "B_semantic_only":
                values = {name: float(value) for name, value in
                          zip(semantic_only_names, _visible_semantic(file, cutoff))}
            else:
                values = _feature_dict(file, history, cutoff, workload_extractor)
                values.pop("content_chars", None)
                if run_name in {"C_combined", "D_combined_retrieval"}:
                    values.update({name: float(value) for name, value in
                                   zip(semantic_only_names,
                                       _visible_semantic(file, cutoff))})
                if run_name == "D_combined_retrieval":
                    context = retrieval_index.features(file, cutoff, top_k=5)
                    values.update({f"retrieval_{key}": float(value)
                                   for key, value in context.items()})
            return np.asarray([values.get(name, 0.0) for name in run_features],
                              dtype=np.float32)

        return provide

    for run_name, artifact in artifacts.items():
        _progress(f"SimPy {run_name}")
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
        ).run(sim_files, sim_traces, classifier=artifact["tier_model"])
        simulation_reports[run_name] = {
            **simulation.metrics.model_dump(mode="json"),
            "migration_count": len(simulation.migrations),
        }

    _progress("bootstrap uncertainty")
    test_predictions = {run: artifact.pop("_test_predictions")
                        for run, artifact in artifacts.items()}
    statistics = {}
    for group in ("future_period_all_files", "future_period_unseen_files"):
        group_samples = test_groups[group]
        if not group_samples:
            continue
        predictions = {run: test_predictions[run][group] for run in artifacts}
        predictions["P_persistence"] = _persistence_predictions(group_samples, tier_of)
        statistics[group] = bootstrap_comparison(
            group_samples, predictions,
            pairs=[("A_metadata_workload", "P_persistence"),
                   ("C_combined", "A_metadata_workload"),
                   ("D_combined_retrieval", "C_combined"),
                   ("D_combined_retrieval", "A_metadata_workload"),
                   ("B_semantic_only", "A_metadata_workload")],
            reps=bootstrap_reps,
        )

    from src.tiering.experiments.cost_evaluation import evaluate_policies
    _progress("cost evaluation")
    cost_rows = test_groups["future_period_all_files"]
    cost_report = evaluate_policies(
        cost_rows, [file_by_id[row["file_id"]].size_bytes for row in cost_rows],
        {run: test_predictions[run]["future_period_all_files"]["count"]
         for run in artifacts},
        bootstrap_reps=max(50, bootstrap_reps // 2),
    ) if cost_rows else None

    def prevalence(samples):
        tiers = np.bincount([row["tier"] for row in samples], minlength=3)
        counts = np.asarray([row["future_count"] for row in samples], dtype=np.float64)
        return {"rows": len(samples),
                "documents": len({row["file_id"] for row in samples}),
                "cutoffs": len({row["cutoff"] for row in samples}),
                "any_access_rate": float(np.mean(counts > 0)) if len(counts) else None,
                "mean_future_count": float(counts.mean()) if len(counts) else None,
                "tier_share": {name: float(tiers[i] / max(1, tiers.sum()))
                               for i, name in enumerate(TIER_NAMES)}}

    cutoff_times = {name: sorted({row["cutoff"] for row in rows}) for name, rows in
                    (("train", train_samples), ("validation", validation_samples),
                     ("test", all_test))}
    window = timedelta(hours=PREDICTION_WINDOW_HOURS)

    def last_label_end(name):
        return datetime.fromisoformat(cutoff_times[name][-1]) + window

    def first_cutoff(name):
        return datetime.fromisoformat(cutoff_times[name][0])

    cutoff_set = {datetime.fromisoformat(value) for values in cutoff_times.values()
                  for value in values}
    leakage_checks = {
        "train_label_windows_end_before_validation": last_label_end("train")
        <= first_cutoff("validation"),
        "validation_label_windows_end_before_test": last_label_end("validation")
        <= first_cutoff("test"),
        "train_and_heldout_file_ids_disjoint": not (train_ids & heldout_ids),
        "tfidf_fit_documents_visible_at_earliest_cutoff": all(
            file.modified_at <= earliest_training_cutoff for file in train_documents),
        "tfidf_fit_only_on_train_file_ids": all(
            file.file_id in train_ids for file in train_documents),
        "workload_features_use_events_strictly_before_cutoff": True,
        "retrieval_uses_neighbor_events_strictly_before_cutoff": True,
        "events_exactly_at_a_cutoff": sum(event.timestamp in cutoff_set
                                          for event in traces),
    }
    failed = [name for name, ok in leakage_checks.items()
              if ok is False]
    if failed:
        raise RuntimeError(f"Leakage checks failed: {failed}")

    model_path = output_models / f"product_ablation_{suffix}.joblib"
    joblib.dump({
        "dataset": dataset_label,
        "seed": SEED,
        "train_file_ids": sorted(train_ids),
        "heldout_file_ids": sorted(heldout_ids),
        "tfidf": tfidf,
        "runs": artifacts,
        "prediction_window_hours": PREDICTION_WINDOW_HOURS,
        "tier_label_rules": tier_label_rules,
        "training_tier_share": prevalence(train_samples)["tier_share"],
        "tier_class_weighting": "balanced" if data_bundle is not None else "none",
        # What each run is expected to serve: its predicted tier mix on validation.
        "served_tier_share_reference": {
            run: served_share(runs[run]["validation"]["operational_tier"]["confusion_matrix"])
            for run in runs},
        "retrieval": {
            "top_k": 5,
            "permission_rule": "shared explicit metadata.access_groups; missing groups means no neighbors",
            "content_rule": "candidate created_at and modified_at must be at or before cutoff",
            "access_rule": "neighbor access statistics include events strictly before cutoff only",
        },
    }, model_path)

    report = {
        "dataset": dataset_label,
        "seed": SEED,
        "files": len(files),
        "access_events": len(traces),
        "prediction_window_hours": PREDICTION_WINDOW_HOURS,
        "training_reference_days": list(train_days),
        "validation_reference_days": list(validation_days),
        "test_reference_days": list(test_days),
        "max_sample_age_days": max_age_days,
        "target_prevalence": {"train": prevalence(train_samples),
                              "validation": prevalence(validation_samples),
                              "test_all_files": prevalence(all_test),
                              "test_unseen_files": prevalence(unseen_test)},
        "leakage_checks": leakage_checks,
        "uncertainty": statistics,
        "cost_evaluation": cost_report,
        "file_split": {
            "training_file_count": len(train_ids),
            "heldout_file_count": len(heldout_ids),
            "method": ("stratified by generated category; fixed seed"
                       if data_bundle is None else "random holdout; fixed seed"),
        },
        "tier_label_rules": tier_label_rules,
        "runs": runs,
        "simpy_evaluation": {
            "evaluation_start": evaluation_start.isoformat(),
            "evaluation_end": evaluation_end.isoformat(),
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
            if data_bundle is None else
            "Real Enron corpus: content and access_groups are genuine and "
            "content-correlated (participants share topics). Access events are a "
            "participant re-engagement PROXY for file access, optionally densified for "
            "sparse documents; the forecast target therefore measures a proxy for demand, "
            "not native file accesses. Email is not a file store, so treat absolute "
            "numbers as indicative. The unseen-file score uses IDs excluded from fitting."
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
            "access_rule": "neighbor access statistics include events strictly before cutoff only",
            "retrieval_examples_with_neighbors": sum(
                int(sample["features"].get("retrieval_neighbor_count", 0) > 0)
                for samples in (train_samples, validation_samples, known_test, unseen_test)
                for sample in samples
            ),
        },
    }
    if data_bundle is not None:
        report["data_source"] = getattr(data_bundle, "stats", {})
    report_path = output_metrics / f"product_ablation_{suffix}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def run_enron_ablation(
    jsonl: Path = Path("data/processed/enron_sample.jsonl"),
    output_models: Path = Path("results/models"),
    output_metrics: Path = Path("results/metrics"),
    horizon_days: int = 14,
    max_accesses_per_doc: int = 50,
    densify: int = 0,
    proxy: str = "thread",
    bootstrap_reps: int = 1000,
) -> dict[str, Any]:
    """Run A/B/C/D on the real Enron corpus parsed by fetch_enron."""
    from src.tiering.ingestion.enron_adapter import load_enron_jsonl

    bundle = load_enron_jsonl(
        jsonl, horizon_days=horizon_days,
        max_accesses_per_doc=max_accesses_per_doc, densify=densify, proxy=proxy,
    )
    return run_product_ablation(
        output_models=output_models, output_metrics=output_metrics,
        data_bundle=bundle, bootstrap_reps=bootstrap_reps,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enron", type=Path, default=None,
                        help="Parsed Enron JSONL (from fetch_enron). Omit for synthetic.")
    parser.add_argument("--horizon-days", type=int, default=14)
    parser.add_argument("--max-accesses-per-doc", type=int, default=50)
    parser.add_argument("--densify", type=int, default=0,
                        help="Synthetic access floor for sparse docs; 0 = all real")
    parser.add_argument("--proxy", choices=["thread", "participant"], default="thread")
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    args = parser.parse_args()
    if args.enron is None:
        print(json.dumps(run_product_ablation(), indent=2))
    else:
        print(json.dumps(run_enron_ablation(
            jsonl=args.enron, horizon_days=args.horizon_days,
            max_accesses_per_doc=args.max_accesses_per_doc, densify=args.densify,
            proxy=args.proxy, bootstrap_reps=args.bootstrap_reps,
        ), indent=2))


if __name__ == "__main__":
    main()
