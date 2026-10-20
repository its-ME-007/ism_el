"""Semantic analysis module for TF-IDF, embeddings, knowledge base, and RAG."""

__all__ = []
"""Semantic feature extraction and local retrieval components."""

from .tfidf_extractor import TFIDFExtractor
from .knowledge_base import KnowledgeBase
from .rag import RAGContextExtractor

__all__ = ["TFIDFExtractor", "KnowledgeBase", "RAGContextExtractor"]
