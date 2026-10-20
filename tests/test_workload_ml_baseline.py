"""Tests for Workload-ML baseline."""

import pytest
import numpy as np
from src.tiering.baselines.workload_ml import WorkloadMLBaseline
from src.tiering.models.tier_classifier import TierClassifier
from src.tiering.ingestion.schemas import Tier


class TestWorkloadMLBaselineInitialization:
    """Test WorkloadMLBaseline initialization."""
    
    def test_initialization(self):
        """Test that WorkloadMLBaseline initializes correctly."""
        classifier = TierClassifier()
        workload_indices = [0, 1, 2, 3, 4]
        
        baseline = WorkloadMLBaseline(classifier, workload_indices)
        
        assert baseline.classifier is classifier
        assert baseline.workload_indices == workload_indices


class TestPredictTier:
    """Test tier prediction using only workload features."""
    
    @pytest.fixture
    def trained_baseline(self):
        """Create a trained Workload-ML baseline."""
        np.random.seed(42)
        # Total features: 10 (5 workload + 5 semantic)
        n_samples = 51
        n_total_features = 10
        workload_indices = [0, 1, 2, 3, 4]
        
        X = np.random.randn(n_samples, n_total_features)
        y = np.tile([0, 1, 2], n_samples // 3)
        assert len(y) == n_samples
        
        baseline = WorkloadMLBaseline.train(X, y, workload_indices)
        return baseline, X, y
    
    def test_predict_tier_uses_only_workload_features(self, trained_baseline):
        """Test that prediction uses only workload features."""
        baseline, X, y = trained_baseline
        
        # Modify semantic features (indices 5-9)
        X_modified = X.copy()
        X_modified[:, 5:] = 999.0  # Change semantic features dramatically
        
        # Predictions should be the same because only workload features are used
        preds_original = baseline.predict_tier(X)
        preds_modified = baseline.predict_tier(X_modified)
        
        np.testing.assert_array_equal(preds_original, preds_modified)
    
    def test_predict_tier_returns_tier_objects(self, trained_baseline):
        """Test that predictions return Tier objects."""
        baseline, X, y = trained_baseline
        
        preds = baseline.predict_tier(X[:10])
        
        assert len(preds) == 10
        for pred in preds:
            assert pred in [Tier.HOT, Tier.WARM, Tier.COLD]
    
    def test_predict_tier_shape(self, trained_baseline):
        """Test that prediction output shape is correct."""
        baseline, X, y = trained_baseline
        
        preds = baseline.predict_tier(X)
        
        assert preds.shape == (X.shape[0],)


class TestPredictProba:
    """Test probability prediction."""
    
    @pytest.fixture
    def trained_baseline(self):
        """Create a trained Workload-ML baseline."""
        np.random.seed(42)
        n_samples = 51
        n_total_features = 10
        workload_indices = [0, 1, 2, 3, 4]
        
        X = np.random.randn(n_samples, n_total_features)
        y = np.tile([0, 1, 2], n_samples // 3)
        assert len(y) == n_samples
        
        baseline = WorkloadMLBaseline.train(X, y, workload_indices)
        return baseline, X
    
    def test_predict_proba_shape(self, trained_baseline):
        """Test probability matrix shape."""
        baseline, X = trained_baseline
        
        proba = baseline.predict_proba(X[:10])
        
        assert proba.shape == (10, 3)
    
    def test_predict_proba_sums_to_one(self, trained_baseline):
        """Test that probabilities sum to 1 for each sample."""
        baseline, X = trained_baseline
        
        proba = baseline.predict_proba(X)
        
        row_sums = np.sum(proba, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-6)
    
    def test_predict_proba_valid_range(self, trained_baseline):
        """Test that all probabilities are in [0, 1]."""
        baseline, X = trained_baseline
        
        proba = baseline.predict_proba(X)
        
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
    
    def test_predict_proba_uses_only_workload_features(self, trained_baseline):
        """Test that probability prediction uses only workload features."""
        baseline, X = trained_baseline
        
        # Modify semantic features
        X_modified = X.copy()
        X_modified[:, 5:] = 999.0
        
        proba_original = baseline.predict_proba(X)
        proba_modified = baseline.predict_proba(X_modified)
        
        np.testing.assert_array_almost_equal(proba_original, proba_modified, decimal=6)


class TestFeatureImportance:
    """Test feature importance extraction."""
    
    @pytest.fixture
    def trained_baseline(self):
        """Create a trained Workload-ML baseline."""
        np.random.seed(42)
        n_samples = 51
        n_total_features = 10
        workload_indices = [0, 1, 2, 3, 4]
        
        X = np.random.randn(n_samples, n_total_features)
        y = np.tile([0, 1, 2], n_samples // 3)
        assert len(y) == n_samples
        
        baseline = WorkloadMLBaseline.train(X, y, workload_indices)
        return baseline
    
    def test_get_feature_importance(self, trained_baseline):
        """Test feature importance extraction."""
        importance = trained_baseline.get_feature_importance()
        
        assert isinstance(importance, dict)
    
    def test_feature_importance_values_normalized(self, trained_baseline):
        """Test that feature importance values are normalized."""
        importance = trained_baseline.get_feature_importance()
        
        if importance:
            values = list(importance.values())
            assert all(0 <= v <= 1 for v in values)


class TestTraining:
    """Test training functionality."""
    
    def test_train_basic(self):
        """Test basic training."""
        np.random.seed(42)
        n_samples = 51
        n_total_features = 10
        workload_indices = [0, 1, 2, 3, 4]
        
        X = np.random.randn(n_samples, n_total_features)
        y = np.tile([0, 1, 2], n_samples // 3)
        assert len(y) == n_samples
        
        baseline = WorkloadMLBaseline.train(X, y, workload_indices)
        
        assert baseline is not None
        assert baseline.classifier._is_fitted
        assert baseline.workload_indices == workload_indices
    
    def test_train_with_validation_set(self):
        """Test training with validation set."""
        np.random.seed(42)
        n_samples = 51
        n_total_features = 10
        workload_indices = [0, 1, 2, 3, 4]
        
        X = np.random.randn(n_samples, n_total_features)
        y = np.tile([0, 1, 2], n_samples // 3)
        assert len(y) == n_samples
        
        X_val = np.random.randn(30, n_total_features)
        y_val = np.tile([0, 1, 2], 30 // 3)
        
        baseline = WorkloadMLBaseline.train(X, y, workload_indices, X_val, y_val)
        
        assert baseline.classifier._is_fitted
    
    def test_train_creates_correct_feature_set(self):
        """Test that training uses only specified workload features."""
        np.random.seed(42)
        n_samples = 51
        n_total_features = 15
        workload_indices = [0, 1, 2, 3]  # Only 4 features
        
        X = np.random.randn(n_samples, n_total_features)
        y = np.tile([0, 1, 2], n_samples // 3)
        assert len(y) == n_samples
        
        baseline = WorkloadMLBaseline.train(X, y, workload_indices)
        
        # The classifier should have 4 feature names
        assert len(baseline.classifier.feature_names) == len(workload_indices)


class TestPredictionConsistency:
    """Test consistency between different prediction methods."""
    
    @pytest.fixture
    def trained_baseline(self):
        """Create a trained Workload-ML baseline."""
        np.random.seed(42)
        n_samples = 51
        n_total_features = 10
        workload_indices = [0, 1, 2, 3, 4]
        
        X = np.random.randn(n_samples, n_total_features)
        y = np.tile([0, 1, 2], n_samples // 3)
        assert len(y) == n_samples
        
        baseline = WorkloadMLBaseline.train(X, y, workload_indices)
        return baseline, X
    
    def test_tier_prediction_matches_proba_argmax(self, trained_baseline):
        """Test that tier predictions match argmax of probabilities."""
        baseline, X = trained_baseline
        
        tier_map = {0: 'HOT', 1: 'WARM', 2: 'COLD'}
        
        preds = baseline.predict_tier(X)
        proba = baseline.predict_proba(X)
        argmax_preds = np.argmax(proba, axis=1)
        
        # Convert to tier names for comparison
        argmax_tiers = np.array([tier_map[idx] for idx in argmax_preds])
        pred_tiers = np.array([t.value for t in preds])
        
        assert np.array_equal(argmax_tiers, pred_tiers)


class TestDifferentFeatureIndices:
    """Test behavior with different feature index configurations."""
    
    def test_single_feature(self):
        """Test with single workload feature."""
        np.random.seed(42)
        n_samples = 51
        n_total_features = 10
        workload_indices = [0]  # Only first feature
        
        X = np.random.randn(n_samples, n_total_features)
        y = np.tile([0, 1, 2], n_samples // 3)
        assert len(y) == n_samples
        
        baseline = WorkloadMLBaseline.train(X, y, workload_indices)
        preds = baseline.predict_tier(X)
        
        assert len(preds) == n_samples
    
    def test_all_features(self):
        """Test when all features are workload features."""
        np.random.seed(42)
        n_samples = 51
        n_total_features = 5
        workload_indices = [0, 1, 2, 3, 4]
        
        X = np.random.randn(n_samples, n_total_features)
        y = np.tile([0, 1, 2], n_samples // 3)
        assert len(y) == n_samples
        
        baseline = WorkloadMLBaseline.train(X, y, workload_indices)
        preds = baseline.predict_tier(X)
        
        assert len(preds) == n_samples


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
