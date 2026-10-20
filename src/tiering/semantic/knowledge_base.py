"""Similarity-search knowledge base with optional FAISS and NumPy fallback."""

from pathlib import Path
import json

import numpy as np


class KnowledgeBase:
    def __init__(self, config=None, dimension=None, **kwargs):
        options = dict(config or {}) if isinstance(config, dict) else {}
        options.update(kwargs)
        self.dimension = int(dimension or options.get("dimension", 384))
        self.index_type = str(options.get("index_type", "Flat")).upper()
        self.metric = str(options.get("metric", "cosine")).lower()
        self.file_ids = []
        self._vectors = np.empty((0, self.dimension), dtype=np.float32)
        self._faiss = None
        self.n_clusters = max(1, int(options.get("n_clusters", 64)))
        try:
            import faiss
            self._faiss = faiss
            self._index = None if self.index_type == "IVF" else self._make_index(faiss)
        except ImportError:
            self._index = None

    def _make_index(self, faiss, n_clusters=None):
        metric_type = (faiss.METRIC_INNER_PRODUCT
                       if self.metric in ("cosine", "ip", "inner_product")
                       else faiss.METRIC_L2)
        if self.index_type == "IVF":
            cluster_count = max(1, int(n_clusters or self.n_clusters))
            quantizer = (faiss.IndexFlatIP(self.dimension)
                         if metric_type == faiss.METRIC_INNER_PRODUCT
                         else faiss.IndexFlatL2(self.dimension))
            return faiss.IndexIVFFlat(quantizer, self.dimension, cluster_count, metric_type)
        if self.metric in ("cosine", "ip", "inner_product"):
            if self.index_type == "HNSW":
                return faiss.IndexHNSWFlat(self.dimension, 32, faiss.METRIC_INNER_PRODUCT)
            return faiss.IndexFlatIP(self.dimension)
        if self.index_type == "HNSW":
            return faiss.IndexHNSWFlat(self.dimension, 32)
        return faiss.IndexFlatL2(self.dimension)

    def add(self, embeddings: np.ndarray, file_ids):
        vectors = np.asarray(embeddings, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"Expected embeddings with shape (n, {self.dimension})")
        if len(file_ids) != len(vectors):
            raise ValueError("file_ids and embeddings must have equal length")
        if self.metric in ("cosine", "ip", "inner_product"):
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            vectors = vectors / np.maximum(norms, 1e-12)
        self.file_ids.extend(str(file_id) for file_id in file_ids)
        self._vectors = np.vstack((self._vectors, vectors))
        if self._faiss is not None and self.index_type == "IVF" and self._index is None:
            cluster_count = min(self.n_clusters, max(1, len(self._vectors)))
            self._index = self._make_index(self._faiss, cluster_count)
            self._index.train(self._vectors)
            self._index.add(self._vectors)
        elif self._index is not None:
            self._index.add(vectors)

    def search(self, query: np.ndarray, k: int):
        queries = np.asarray(query, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        if queries.shape[1] != self.dimension:
            raise ValueError(f"Expected query dimension {self.dimension}")
        k = max(0, min(int(k), len(self.file_ids)))
        if k == 0:
            return np.empty((len(queries), 0), dtype=np.float32), [[] for _ in queries]
        if self.metric in ("cosine", "ip", "inner_product"):
            norms = np.linalg.norm(queries, axis=1, keepdims=True)
            queries = queries / np.maximum(norms, 1e-12)
        if self._index is not None:
            scores, indices = self._index.search(queries, k)
        else:
            similarities = queries @ self._vectors.T
            if self.metric in ("cosine", "ip", "inner_product"):
                indices = np.argsort(-similarities, axis=1)[:, :k]
                scores = np.take_along_axis(similarities, indices, axis=1)
            else:
                distances = np.sum((queries[:, None, :] - self._vectors[None, :, :]) ** 2, axis=2)
                indices = np.argsort(distances, axis=1)[:, :k]
                scores = np.take_along_axis(distances, indices, axis=1)
        ids = [[self.file_ids[int(index)] for index in row if int(index) >= 0] for row in indices]
        return scores.astype(np.float32), ids

    def save(self, path: Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self._vectors)
        (path / "metadata.json").write_text(json.dumps({
            "dimension": self.dimension, "index_type": self.index_type,
            "metric": self.metric, "file_ids": self.file_ids,
        }), encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        path = Path(path)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        obj = cls(metadata)
        vectors = np.load(path / "vectors.npy").astype(np.float32)
        obj.add(vectors, metadata["file_ids"])
        return obj
