"""Sparse TF-IDF document features with deterministic preprocessing and caching."""

from pathlib import Path
from typing import Iterable, List, Optional, Sequence
import re

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from nltk.stem import PorterStemmer
    _STEMMER = PorterStemmer()
except ImportError:
    _STEMMER = None


class TFIDFExtractor:
    def __init__(self, config=None, **kwargs):
        options = dict(config or {}) if isinstance(config, dict) else {}
        options.update(kwargs)
        self.max_features = int(options.get("max_features", 1000))
        self.ngram_range = tuple(options.get("ngram_range", (1, 2)))
        self.remove_stopwords = options.get("remove_stopwords", True)
        self.cache_path = options.get("cache_path")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words="english" if self.remove_stopwords else None,
            preprocessor=self._preprocess,
            dtype=np.float32,
        )
        self._fitted = False

    @staticmethod
    def _preprocess(text: str) -> str:
        tokens = re.sub(r"[^a-z0-9]+", " ", str(text).lower()).split()
        if _STEMMER is not None:
            tokens = [_STEMMER.stem(token) for token in tokens]
        return " ".join(tokens)

    @staticmethod
    def _texts(documents: Sequence) -> List[str]:
        texts = []
        for doc in documents:
            if isinstance(doc, str):
                texts.append(doc)
            else:
                content = getattr(doc, "content", "") or ""
                filename = getattr(doc, "filename", "") or ""
                category = getattr(doc, "category", "") or ""
                metadata = getattr(doc, "metadata", {}) or {}
                texts.append(" ".join([str(filename), str(category), str(content), str(metadata)]))
        return texts

    def fit(self, documents: Sequence):
        texts = self._texts(documents)
        # sklearn rejects an empty vocabulary; keep a valid one-column
        # representation for empty or punctuation-only corpora.
        if not any(self._preprocess(text) for text in texts):
            texts = ["emptydocument"]
        self.vectorizer.fit(texts)
        self._fitted = True
        return self

    def transform(self, documents: Sequence) -> np.ndarray:
        texts = self._texts(documents)
        if not self._fitted:
            self.fit(texts or [""])
        if not self.vectorizer.vocabulary_:
            return np.zeros((len(texts), 1), dtype=np.float32)
        return self.vectorizer.transform(texts).toarray().astype(np.float32, copy=False)

    def fit_transform(self, documents: Sequence) -> np.ndarray:
        self.fit(documents)
        return self.transform(documents)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Path) -> "TFIDFExtractor":
        obj = joblib.load(Path(path))
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__} at {path}")
        return obj
