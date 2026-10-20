"""Retrieval-assisted context features with a deterministic local fallback."""

import logging
from typing import Dict, List, Optional

import numpy as np

from src.tiering.ingestion.schemas import FileDocument, RAGContext
from .embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class RAGContextExtractor:
    def __init__(self, knowledge_base, config=None, embedding_generator=None):
        self.knowledge_base = knowledge_base
        self.config = dict(config or {})
        self.top_k = max(1, int(self.config.get("top_k", self.config.get("top_k_retrieval", 5))))
        self.embedding_generator = embedding_generator
        self.documents: Dict[str, FileDocument] = {}

    def add_documents(self, files: List[FileDocument], embeddings: Optional[np.ndarray] = None):
        if embeddings is None:
            embeddings = self._get_embedder().encode(files)
        self.knowledge_base.add(embeddings, [file.file_id for file in files])
        self.documents.update({file.file_id: file for file in files})

    def _get_embedder(self):
        if self.embedding_generator is None:
            self.embedding_generator = EmbeddingGenerator(self.config.get("embeddings", {}))
        return self.embedding_generator

    def extract(self, file: FileDocument, access_history=None) -> RAGContext:
        vector = self._get_embedder().encode([file])[0]
        scores, related = self.knowledge_base.search(vector, self.top_k + 1)
        ids = related[0]
        values = scores[0]
        ranked = [(file_id, float(score)) for file_id, score in zip(ids, values)
                  if file_id != file.file_id]
        context = self._fallback_extraction(file, ranked)
        related_ids = set(context.related_files)
        access_history = access_history or []
        related_accesses = sum(event.file_id in related_ids for event in access_history)
        access_ratio = related_accesses / max(len(access_history), 1)
        context.feature_vector.append(float(access_ratio))
        if related_accesses:
            context.context_summary += f" {related_accesses} historical access(es) referenced related files."
        logger.info("RAG fallback file=%s related=%s category_confidence=%.4f",
                    file.file_id, context.related_files, context.category_confidence)
        return context

    def _fallback_extraction(self, file: FileDocument, retrieved) -> RAGContext:
        retrieved = list(retrieved or [])
        related_ids = [item[0] if isinstance(item, (tuple, list)) else str(item)
                       for item in retrieved if (item[0] if isinstance(item, (tuple, list)) else item) != file.file_id]
        scores = [max(0.0, float(item[1])) for item in retrieved
                  if isinstance(item, (tuple, list)) and item[0] != file.file_id]
        related_docs = [self.documents[file_id] for file_id in related_ids if file_id in self.documents]
        category = str(getattr(file.category, "value", file.category)).lower()
        category_matches = sum(
            str(getattr(doc.category, "value", doc.category)).lower() == category
            for doc in related_docs
        )
        if related_docs:
            category_confidence = category_matches / len(related_docs)
        else:
            category_confidence = 0.5
        summary = (f"Retrieved {len(related_ids)} related file(s); "
                   f"{category_matches} share category {category}.")
        feature_vector = [float(category_confidence), float(len(related_ids))]
        feature_vector.extend(scores[:max(0, self.top_k - len(feature_vector))])
        return RAGContext(file_id=file.file_id, category_confidence=category_confidence,
                          related_files=related_ids[:self.top_k], context_summary=summary,
                          feature_vector=feature_vector)
