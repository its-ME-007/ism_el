"""Fit a TF-IDF MiniBatchKMeans topic-clustering baseline on AG News."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.tiering.models.news_training import _read_jsonl


def run_clustering(
    data_dir=Path("data/processed"),
    model_path=Path("results/models/ag_news_tfidf_minibatch_kmeans.joblib"),
    metrics_path=Path("results/metrics/ag_news_tfidf_minibatch_kmeans.json"),
    sample_size=5000,
    seed=42,
):
    records = _read_jsonl(Path(data_dir) / "ag_news.jsonl")
    train = [row for row in records if row["split"] == "train"]
    test = [row for row in records if row["split"] == "test"]
    train_labels = sorted({row["labels"][0] for row in train})
    label_to_id = {label: index for index, label in enumerate(train_labels)}
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(train), size=min(sample_size, len(train)), replace=False))
    fit_rows = [train[int(index)] for index in indices]

    # Use the matching trained AG News vectorizer, which was fit on the same
    # deterministic 5k training sample, and never fit clusters on test data.
    bundle = joblib.load("results/models/ag_news_tfidf_xgboost.joblib")
    vectorizer = bundle["vectorizer"]
    X_fit = vectorizer.transform([row["text"] for row in fit_rows])
    X_test = vectorizer.transform([row["text"] for row in test])
    model = MiniBatchKMeans(
        n_clusters=len(train_labels), random_state=seed,
        batch_size=512, n_init=5, max_iter=100,
    )
    train_clusters = model.fit_predict(X_fit)
    test_clusters = model.predict(X_test)
    y_test = np.asarray([label_to_id[row["labels"][0]] for row in test])

    result = {
        "dataset": "AG News",
        "method": "MiniBatchKMeans on TF-IDF",
        "seed": seed,
        "fit_records": len(fit_rows),
        "test_records": len(test),
        "clusters": len(train_labels),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "train_cluster_sizes": np.bincount(train_clusters, minlength=len(train_labels)).tolist(),
        "test_cluster_sizes": np.bincount(test_clusters, minlength=len(train_labels)).tolist(),
        "test_adjusted_rand_index": float(adjusted_rand_score(y_test, test_clusters)),
        "test_normalized_mutual_information": float(
            normalized_mutual_info_score(y_test, test_clusters)
        ),
        "note": "Clusters were fit on train only; test labels are used only for ARI/NMI evaluation.",
    }
    model_path = Path(model_path); metrics_path = Path(metrics_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"vectorizer": vectorizer, "clusterer": model, "labels": train_labels}, model_path)
    metrics_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["model_path"] = str(model_path)
    return result


if __name__ == "__main__":
    print(json.dumps(run_clustering(), indent=2))
