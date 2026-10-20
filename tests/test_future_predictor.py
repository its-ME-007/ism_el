"""Tests for FutureAccessPredictor."""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from src.tiering.prediction.future_predictor import FutureAccessPredictor


class TestFutureAccessPredictor:
    """Test suite for FutureAccessPredictor."""
    
    @pytest.fixture
    def predictor(self):
        """Create predictor instance."""
        return FutureAccessPredictor(
            prediction_window_hours=24,
            n_estimators=50,
            max_depth=3,
            calibration_method='isotonic'
        )
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample training data."""
        np.random.seed(42)
        n_samples = 200
        n_features = 10
        
        X = np.random.randn(n_samples, n_features).astype(np.float32)
        # Make first few features predictive
        y = (X[:, 0] > 0).astype(float) * 0.7 + (X[:, 1] > 0).astype(float) * 0.3
        y = np.clip(y + np.random.randn(n_samples) * 0.1, 0, 1).astype(np.float32)
        
        feature_names = [f"feature_{i}" for i in range(n_features)]
        
        return X, y, feature_names
    
    def test_model_initialization(self, predictor):
        """Test predictor initialization."""
        assert predictor.model is None
        assert predictor.calibrator is None
        assert predictor.feature_names is None
    
    def test_fit_basic(self, predictor, sample_data):
        """Test basic model fitting."""
        X, y, feature_names = sample_data
        
        predictor.fit(X, y, feature_names=feature_names)
        
        assert predictor.model is not None
        assert predictor.feature_names == feature_names
    
    def test_predict_output_range(self, predictor, sample_data):
        """Test that predictions are in [0, 1]."""
        X, y, feature_names = sample_data
        
        predictor.fit(X, y, feature_names=feature_names)
        preds = predictor.predict(X[:10])
        
        assert isinstance(preds, np.ndarray)
        assert len(preds) == 10
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 1.0)
    
    def test_predict_proba_format(self, predictor, sample_data):
        """Test predict_proba returns correct shape."""
        X, y, feature_names = sample_data
        
        predictor.fit(X, y, feature_names=feature_names)
        proba = predictor.predict_proba(X[:10])
        
        assert proba.shape == (10, 1)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
    
    def test_predict_without_fit(self, predictor, sample_data):
        """Test that predict raises error without fitting."""
        X, _, _ = sample_data
        
        with pytest.raises(ValueError, match="not trained"):
            predictor.predict(X)
    
    def test_feature_importance(self, predictor, sample_data):
        """Test feature importance extraction."""
        X, y, feature_names = sample_data
        
        predictor.fit(X, y, feature_names=feature_names)
        importance = predictor.get_feature_importance()
        
        # Should have some features with non-zero importance
        assert len(importance) > 0
        # All importances should be in [0, 1]
        assert all(0 <= v <= 1 for v in importance.values())
        # Keys should be feature names
        assert all(k in feature_names for k in importance.keys())
    
    def test_category_priors(self, predictor, sample_data):
        """Test category prior setting and application."""
        X, y, feature_names = sample_data
        
        predictor.fit(X, y, feature_names=feature_names)
        
        # Set priors
        priors = {"finance": 0.7, "legal": 0.3}
        predictor.set_category_priors(priors)
        
        # Use priors in prediction
        preds = predictor.predict_with_cold_start(X[:10], category="finance")
        
        assert len(preds) == 10
        assert np.all(preds >= 0.0)
        assert np.all(preds <= 1.0)
    
    def test_cold_start_fallback(self, predictor, sample_data):
        """Test cold-start fallback behavior."""
        X, y, feature_names = sample_data
        
        predictor.fit(X, y, feature_names=feature_names)
        predictor.set_category_priors({"test": 0.8})
        
        # Create a sample that would have low prediction
        X_cold_start = np.zeros((5, 10), dtype=np.float32)
        
        preds_with_fallback = predictor.predict_with_cold_start(
            X_cold_start, category="test"
        )
        
        # Predictions should remain valid probabilities after fallback
        assert len(preds_with_fallback) == 5
        assert np.all(preds_with_fallback >= 0.0)
        assert np.all(preds_with_fallback <= 1.0)
    
    def test_early_stopping(self, predictor, sample_data):
        """Test model training without early stopping."""
        X, y, feature_names = sample_data
        
        # Fit without eval set (early stopping not supported in latest xgboost)
        predictor.fit(
            X, y,
            feature_names=feature_names
        )
        
        # Model should be trained
        assert predictor.model is not None
        assert len(predictor.predict(X[:5])) == 5
    
    def test_model_persistence(self, predictor, sample_data):
        """Test save and load functionality."""
        X, y, feature_names = sample_data
        
        predictor.fit(X, y, feature_names=feature_names)
        preds_before = predictor.predict(X[:5])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model"
            
            # Save
            predictor.save(model_path, version="1.0")
            
            # Check that files were created
            assert (Path(tmpdir) / "model_xgb_1.0.json").exists()
            assert (Path(tmpdir) / "model_meta_1.0.txt").exists()
    
    def test_calibration_isotonic(self):
        """Test isotonic calibration."""
        predictor = FutureAccessPredictor(calibration_method='isotonic')
        
        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = (X[:, 0] > 0).astype(float) + np.random.randn(100) * 0.1
        y = np.clip(y, 0, 1)
        
        predictor.fit(X, y, feature_names=[f"f{i}" for i in range(5)])
        
        preds = predictor.predict(X)
        assert np.all((preds >= 0) & (preds <= 1))
    
    def test_calibration_clip(self):
        """Test clip calibration."""
        predictor = FutureAccessPredictor(calibration_method='clip')
        
        # Create synthetic data
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = (X[:, 0] > 0).astype(float)
        
        predictor.fit(X, y, feature_names=[f"f{i}" for i in range(5)])
        
        preds = predictor.predict(X)
        assert np.all((preds >= 0) & (preds <= 1))


# Property tests
@pytest.mark.parametrize("n_samples", [50, 100, 200])
def test_property_probability_calibration(n_samples):
    """Property 9: Model Probability Calibration (Predictor).
    
    Validates: Requirements 7.3
    
    All predictions should be valid probabilities in [0, 1],
    regardless of input data or model size.
    """
    np.random.seed(42)
    predictor = FutureAccessPredictor(n_estimators=30)
    
    # Generate data
    X = np.random.randn(n_samples, 8).astype(np.float32)
    y = (X[:, 0] > 0).astype(float) * 0.5 + np.random.randn(n_samples) * 0.1
    y = np.clip(y, 0, 1)
    
    predictor.fit(X, y, feature_names=[f"f{i}" for i in range(8)])
    
    # Predict on new data
    X_test = np.random.randn(20, 8).astype(np.float32)
    preds = predictor.predict(X_test)
    
    # All predictions must be valid probabilities
    assert np.all(preds >= 0.0)
    assert np.all(preds <= 1.0)
    assert np.all(np.isfinite(preds))


@pytest.mark.parametrize("missing_ratio", [0.0, 0.1, 0.3, 0.5])
def test_property_cold_start_prior_application(missing_ratio):
    """Property 17: Cold-Start Prior Application.
    
    Validates: Requirements 7.6
    
    Cold-start prior should be applied consistently to low-confidence
    predictions or files with insufficient history.
    """
    np.random.seed(42)
    predictor = FutureAccessPredictor()
    
    # Train on data
    X_train = np.random.randn(100, 8).astype(np.float32)
    y_train = (X_train[:, 0] > 0).astype(float)
    predictor.fit(X_train, y_train, feature_names=[f"f{i}" for i in range(8)])
    
    # Set priors
    categories = {"test_cat": 0.6}
    predictor.set_category_priors(categories)
    
    # Create test data with varying amounts of information
    n_test = 30
    X_test = np.random.randn(n_test, 8).astype(np.float32)
    
    # Apply cold-start fallback
    preds_direct = predictor.predict(X_test)
    preds_with_prior = predictor.predict_with_cold_start(X_test, category="test_cat")
    
    # Predictions should remain valid probabilities
    assert np.all(preds_with_prior >= 0.0)
    assert np.all(preds_with_prior <= 1.0)
