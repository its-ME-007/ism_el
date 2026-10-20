"""Feature assembly for workload, semantic, and future access signals."""

from typing import List, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class FeatureFusion:
    FEATURE_TYPES = ("workload", "semantic", "future")

    def __init__(self, config=None):
        self.config = dict(config or {})
        self.imputation_strategy = self.config.get("imputation_strategy", "zeros")
        self.scaling = self.config.get("scaling", "none")
        self.disabled = set(self.config.get("disable_features", []))
        self._dimensions = {}
        self._means = {}
        self._scaler = None

    @staticmethod
    def _matrix(values):
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError("Features must be a 1D or 2D array")
        return array

    def set_ablation(self, disable_features: List[str]):
        unknown = set(disable_features) - set(self.FEATURE_TYPES)
        if unknown:
            raise ValueError(f"Unknown feature types: {sorted(unknown)}")
        self.disabled = set(disable_features)

    def fit(self, workload=None, semantic=None, future=None):
        matrices = {name: self._matrix(value) for name, value in
                    zip(self.FEATURE_TYPES, (workload, semantic, future))}
        for name, matrix in matrices.items():
            if matrix is not None:
                self._dimensions[name] = matrix.shape[1]
                self._means[name] = np.nanmean(matrix, axis=0)
        fused = self._combine(matrices)
        if self.scaling == "standard":
            self._scaler = StandardScaler().fit(fused)
        elif self.scaling == "minmax":
            self._scaler = MinMaxScaler().fit(fused)
        elif self.scaling != "none":
            raise ValueError(f"Unsupported scaling strategy: {self.scaling}")
        return self

    def transform(self, workload=None, semantic=None, future=None):
        matrices = {name: self._matrix(value) for name, value in
                    zip(self.FEATURE_TYPES, (workload, semantic, future))}
        fused = self._combine(matrices)
        return self._scaler.transform(fused).astype(np.float32) if self._scaler else fused

    def fit_transform(self, workload=None, semantic=None, future=None):
        self.fit(workload, semantic, future)
        return self.transform(workload, semantic, future)

    def _combine(self, matrices):
        present = [m for name, m in matrices.items() if name not in self.disabled and m is not None]
        n_rows = max((m.shape[0] for m in present), default=1)
        parts = []
        for name in self.FEATURE_TYPES:
            if name in self.disabled:
                continue
            matrix = matrices[name]
            width = self._dimensions.get(name, matrix.shape[1] if matrix is not None else 0)
            if width == 0:
                continue
            if matrix is None:
                if self.imputation_strategy == "mean":
                    fill = self._means.get(name, np.zeros(width, dtype=np.float32))
                    matrix = np.repeat(np.asarray(fill, dtype=np.float32).reshape(1, -1), n_rows, axis=0)
                else:
                    fill = 0.0 if self.imputation_strategy == "zeros" else np.nan
                    matrix = np.full((n_rows, width), fill, dtype=np.float32)
            elif matrix.shape[0] == 1 and n_rows > 1:
                matrix = np.repeat(matrix, n_rows, axis=0)
            elif matrix.shape[0] != n_rows:
                raise ValueError("Feature matrices must have the same number of rows")
            if self.imputation_strategy == "mean":
                mean = (self._means[name] if name in self._means
                        else np.nanmean(matrix, axis=0))
                matrix = np.where(np.isnan(matrix), mean, matrix)
            elif self.imputation_strategy == "zeros":
                matrix = np.nan_to_num(matrix, nan=0.0)
            elif self.imputation_strategy != "none":
                raise ValueError(f"Unsupported imputation strategy: {self.imputation_strategy}")
            parts.append(matrix)
        return np.concatenate(parts, axis=1).astype(np.float32) if parts else np.empty((n_rows, 0), dtype=np.float32)

    def fuse(self, workload=None, semantic=None, future=None):
        return self.transform(workload, semantic, future)

    def get_feature_names(self):
        names = []
        for feature_type in self.FEATURE_TYPES:
            if feature_type not in self.disabled:
                names.extend(f"{feature_type}_{i}" for i in range(self._dimensions.get(feature_type, 0)))
        return names
