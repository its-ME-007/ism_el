"""Sentence Transformer embedding generation (loaded lazily as an optional extra)."""

from datetime import datetime
from typing import List, Optional

import numpy as np

from src.tiering.ingestion.schemas import EmbeddingRecord, FileDocument


class EmbeddingGenerator:
    def __init__(self, config=None, **kwargs):
        options = dict(config or {}) if isinstance(config, dict) else {}
        options.update(kwargs)
        self.model_name = options.get("model_name", "all-MiniLM-L6-v2")
        self.batch_size = int(options.get("batch_size", 32))
        requested_device = options.get("device", "cpu")
        self.device = requested_device
        self.model = options.get("model")
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "Embedding generation requires the optional semantic dependencies; "
                    "install with `pip install -e '.[semantic]'`."
                ) from exc
            try:
                self.model = SentenceTransformer(self.model_name, device=requested_device)
            except (RuntimeError, ValueError):
                if requested_device == "cpu":
                    raise
                self.device = "cpu"
                self.model = SentenceTransformer(self.model_name, device="cpu")
        self.model_version = self.model_name

    @staticmethod
    def _text(value) -> str:
        if isinstance(value, str):
            return value
        metadata = getattr(value, "metadata", {}) or {}
        return " ".join((str(getattr(value, "filename", "")),
                         str(getattr(value, "category", "")),
                         str(getattr(value, "content", "")), str(metadata)))

    def encode(self, texts) -> np.ndarray:
        values = [self._text(text) for text in texts]
        if not values:
            dimension = int(self.model.get_sentence_embedding_dimension())
            return np.empty((0, dimension), dtype=np.float32)
        vectors = self.model.encode(
            values, batch_size=self.batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_with_metadata(self, files: List[FileDocument]) -> List[EmbeddingRecord]:
        vectors = self.encode(files)
        created_at = datetime.utcnow()
        return [EmbeddingRecord(file_id=file.file_id, embedding=vector.tolist(),
                                model_version=self.model_version, created_at=created_at)
                for file, vector in zip(files, vectors)]
