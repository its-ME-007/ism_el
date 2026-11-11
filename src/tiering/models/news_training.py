"""Train TF-IDF text classifiers on the cleaned news corpora.

Examples::

    python -m src.tiering.models.news_training --dataset ag_news
    python -m src.tiering.models.news_training --dataset reuters21578 --model random_forest

Training and test splits are read from the cleaner's JSONL output. Reuters rows
without topic labels and its unused split are excluded from supervised fitting.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             hamming_loss)
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split

from src.tiering.semantic.tfidf_extractor import TFIDFExtractor


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {path}")
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _classifier(model_name: str, seed: int):
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=seed,
        )

    from xgboost import XGBClassifier

    base = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective="multi:softprob",
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=4,
        random_state=seed,
        verbosity=0,
    )
    return base


def _make_model(model_name: str, multilabel: bool, seed: int):
    if model_name == "xgboost" and multilabel:
        from xgboost import XGBClassifier

        binary_model = XGBClassifier(
            n_estimators=50,
            max_depth=4,
            learning_rate=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1,
            random_state=seed,
            verbosity=0,
        )
        return OneVsRestClassifier(binary_model, n_jobs=4)
    if model_name == "random_forest" and multilabel:
        return RandomForestClassifier(
            n_estimators=100, max_features="sqrt", n_jobs=4, random_state=seed
        )
    return _classifier(model_name, seed)


def _tfidf(max_features: int):
    extractor = TFIDFExtractor({"max_features": max_features, "ngram_range": (1, 2)})
    return extractor.vectorizer


def _metrics(y_true, y_pred, multilabel: bool, labels: list[str]) -> dict[str, Any]:
    if multilabel:
        per_label = classification_report(
            y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
        )
        return {
            "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "exact_match_accuracy": float(accuracy_score(y_true, y_pred)),
            "hamming_loss": float(hamming_loss(y_true, y_pred)),
            "per_label": per_label,
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_label": classification_report(
            y_true, y_pred, labels=list(range(len(labels))), target_names=labels,
            output_dict=True, zero_division=0,
        ),
    }


def train_news_model(
    dataset: str,
    model_name: str = "xgboost",
    data_dir: Path = Path("data/processed"),
    models_dir: Path = Path("results/models"),
    metrics_dir: Path = Path("results/metrics"),
    max_features: int = 20_000,
    max_train_samples: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Fit one corpus-specific classifier and save a reloadable model bundle."""
    if dataset not in {"ag_news", "reuters21578", "20_newsgroups"}:
        raise ValueError(f"Unsupported dataset: {dataset}")

    records = _read_jsonl(data_dir / f"{dataset}.jsonl")
    multilabel = dataset == "reuters21578"
    eligible = [
        row for row in records
        if row.get("split") in {"train", "test"}
        and row.get("text", "").strip()
        and (row.get("labels") if multilabel else bool(row.get("labels")))
    ]
    train_rows = [row for row in eligible if row["split"] == "train"]
    test_rows = [row for row in eligible if row["split"] == "test"]
    official_validation_rows = [
        row for row in records
        if dataset == "20_newsgroups"
        and row.get("split") == "validation"
        and row.get("text", "").strip()
        and row.get("labels")
    ]
    if not train_rows or not test_rows:
        raise ValueError(f"{dataset} needs non-empty train and test splits")
    official_train_count = len(train_rows)
    official_train_rows = list(train_rows)
    if max_train_samples is not None and max_train_samples > 0 and max_train_samples < len(train_rows):
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(len(train_rows), size=max_train_samples, replace=False))
        train_rows = [train_rows[int(index)] for index in chosen]

    if multilabel:
        label_support = Counter(label for row in train_rows for label in row["labels"])
        class_names = sorted(label for label, support in label_support.items() if support >= 2)
        official_topic_names = sorted({
            label for row in official_train_rows for label in row["labels"]
        })
        unmodeled_labels = sorted(set(official_topic_names) - set(class_names))
        if not class_names:
            raise ValueError("Reuters training split has no topics with at least two examples")
        class_set = set(class_names)
        original_test_count = len(test_rows)
        unsupported_only_train_rows = sum(
            not any(label in class_set for label in row["labels"]) for row in train_rows
        )
        train_rows = [row for row in train_rows
                      if any(label in class_set for label in row["labels"])]
        test_rows = [row for row in test_rows if any(label in class_set for label in row["labels"])]
        unknown_test_label_occurrences = sum(
            label not in class_set for row in test_rows for label in row["labels"]
        )
        train_targets = [[label for label in row["labels"] if label in class_set]
                         for row in train_rows]
        test_targets = [[label for label in row["labels"] if label in class_set]
                        for row in test_rows]
        encoder = MultiLabelBinarizer(classes=class_names)
        y_train = encoder.fit_transform(train_targets)
        y_test = encoder.transform(test_targets)
        model_type = "multilabel"
        validation_metrics = None
        validation_idx = np.array([], dtype=int)
    else:
        encoder = LabelEncoder()
        y_train = encoder.fit_transform([row["labels"][0] for row in train_rows])
        y_test = encoder.transform([row["labels"][0] for row in test_rows])
        class_names = encoder.classes_.tolist()
        model_type = "multiclass"
        unmodeled_labels = []
        unknown_test_label_occurrences = 0
        original_test_count = len(test_rows)
        unsupported_only_train_rows = 0

    # Reuters is multi-label with many rare topics. Keep the official training
    # split intact to avoid random validation subsampling dropping rare labels.
    # AG News has a deterministic stratified validation slice.
    if dataset == "20_newsgroups":
        y_validation = encoder.transform([row["labels"][0] for row in official_validation_rows])
        val_vectorizer = _tfidf(max_features)
        X_fit = val_vectorizer.fit_transform([row["text"] for row in train_rows])
        X_val = val_vectorizer.transform([row["text"] for row in official_validation_rows])
        val_model = _make_model(model_name, False, seed)
        val_model.fit(X_fit, y_train)
        val_predictions = val_model.predict(X_val)
        validation_metrics = _metrics(
            y_validation, val_predictions, False, class_names
        )
        validation_idx = np.arange(len(official_validation_rows))
    elif not multilabel:
        indices = np.arange(len(train_rows))
        fit_idx, validation_idx = train_test_split(
            indices, test_size=0.1, random_state=seed, stratify=y_train
        )
        val_vectorizer = _tfidf(max_features)
        X_fit = val_vectorizer.fit_transform([train_rows[i]["text"] for i in fit_idx])
        X_val = val_vectorizer.transform([train_rows[i]["text"] for i in validation_idx])
        val_model = _make_model(model_name, multilabel, seed)
        val_model.fit(X_fit, y_train[fit_idx])
        val_predictions = val_model.predict(X_val)
        validation_metrics = _metrics(
            y_train[validation_idx], val_predictions, multilabel, class_names
        )

    vectorizer = _tfidf(max_features)
    X_train = vectorizer.fit_transform([row["text"] for row in train_rows])
    X_test = vectorizer.transform([row["text"] for row in test_rows])
    model = _make_model(model_name, multilabel, seed)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    test_metrics = _metrics(y_test, predictions, multilabel, class_names)

    run_name = f"{dataset}_tfidf_{model_name}"
    model_path = models_dir / f"{run_name}.joblib"
    metric_path = metrics_dir / f"{run_name}.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "dataset": dataset,
        "model_name": model_name,
        "model_type": model_type,
        "vectorizer": vectorizer,
        "classifier": model,
        "label_encoder": encoder,
        "label_names": class_names,
        "seed": seed,
        "max_features": max_features,
    }
    joblib.dump(bundle, model_path)
    report = {
        "dataset": dataset,
        "model": model_name,
        "features": "TF-IDF unigrams and bigrams",
        "seed": seed,
        "label_type": model_type,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "class_names": class_names,
        "official_topic_count": len(official_topic_names) if multilabel else len(class_names),
        "modeled_topic_count": len(class_names) if multilabel else len(class_names),
        "eligible_train_records": len(train_rows),
        "official_train_records": official_train_count,
        "training_sample_limit": max_train_samples,
        "validation_records": len(validation_idx),
        "official_test_records": original_test_count,
        "test_records": len(test_rows),
        "excluded_records": len(records) - len(eligible),
        "train_rows_only_with_unmodeled_topics": unsupported_only_train_rows,
        "unmodeled_labels": unmodeled_labels,
        "unknown_test_label_occurrences": unknown_test_label_occurrences,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "model_path": str(model_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metric_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ag_news", "reuters21578", "20_newsgroups"), required=True)
    parser.add_argument("--model", choices=("xgboost", "random_forest"), default="xgboost")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--models-dir", type=Path, default=Path("results/models"))
    parser.add_argument("--metrics-dir", type=Path, default=Path("results/metrics"))
    parser.add_argument("--max-features", type=int, default=20_000)
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="Optional deterministic training subset for a quick pilot")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = train_news_model(
        dataset=args.dataset,
        model_name=args.model,
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        metrics_dir=args.metrics_dir,
        max_features=args.max_features,
        max_train_samples=args.max_train_samples,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
