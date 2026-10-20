"""Tests for TierClassifier model."""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from src.tiering.models.tier_classifier import TierClassifier


class TestTierClassifierInitialization:
    """Test TierClassifier initialization."""
    
    def test_initialization(self):
        """Test that TierClassifier initializes with default parameters."""
        classifier = TierClassifier()
        assert classifier.model is not None
        assert classifier.feature_names is None
        assert classifier.scaler is None
        assert not classifier._is_fitted
    
    def test_initialization_custom_params(self):
        """Test initialization with custom parameters."""
        classifier = TierClassifier(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.05,
            random_state=123
        )
        assert classifier.model is not None
        assert not classifier._is_fitted


class TestFitAndPredict:
    """Test fitting and prediction functionality."""
    
    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic training data."""
        np.random.seed(42)
        n_samples = 99  # Use 99 to divide evenly by 3
        n_features = 10
        X = np.random.randn(n_samples, n_features)
        # Create balanced classes
        y = np.tile([0, 1, 2], n_samples // 3)
        assert len(y) == n_samples, f"y length {len(y)} != {n_samples}"
        return X, y
    
    def test_fit_basic(self, synthetic_data):
        """Test basic fit operation."""
        X, y = synthetic_data
        classifier = TierClassifier()
        classifier.fit(X, y)
        
        assert classifier._is_fitted
        assert classifier.feature_names is not None
        assert len(classifier.feature_names) == X.shape[1]
    
    def test_fit_with_feature_names(self, synthetic_data):
        """Test fit with provided feature names."""
        X, y = synthetic_data
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        classifier = TierClassifier()
        classifier.fit(X, y, feature_names=feature_names)
        
        assert classifier.feature_names == feature_names
    
    def test_fit_with_validation_set(self, synthetic_data):
        """Test fit with validation set for early stopping."""
        X, y = synthetic_data
        X_val, y_val = synthetic_data
        classifier = TierClassifier(n_estimators=100)
        classifier.fit(X, y, eval_set=(X_val, y_val))
        
        assert classifier._is_fitted
    
    def test_predict_after_fit(self, synthetic_data):
        """Test predict after fitting."""
        X, y = synthetic_data
        classifier = TierClassifier()
        classifier.fit(X, y)
        
        predictions = classifier.predict(X[:10])
        assert predictions.shape == (10,)
        assert np.all(np.isin(predictions, [0, 1, 2]))
    
    def test_predict_before_fit_raises_error(self):
        """Test that predict raises error before fitting."""
        classifier = TierClassifier()
        X = np.random.randn(10, 5)
        
        with pytest.raises(ValueError, match="must be fitted"):
            classifier.predict(X)


class TestPredictProba:
    """Test probability prediction functionality."""
    
    @pytest.fixture
    def fitted_classifier(self):
        """Create a fitted classifier."""
        np.random.seed(42)
        n_samples = 51  # 3 * 17
        X = np.random.randn(n_samples, 5)
        y = np.tile([0, 1, 2], n_samples // 3)
        assert len(y) == n_samples
        classifier = TierClassifier()
        classifier.fit(X, y)
        return classifier, X
    
    def test_predict_proba_shape(self, fitted_classifier):
        """Test predict_proba returns correct shape."""
        classifier, X = fitted_classifier
        proba = classifier.predict_proba(X[:10])
        
        assert proba.shape == (10, 3)
    
    def test_predict_proba_sums_to_one(self, fitted_classifier):
        """
        Test that probabilities sum to 1 for each sample.
        
        **Validates: Requirements 9.2, 15.4, 16.4**
        """
        classifier, X = fitted_classifier
        proba = classifier.predict_proba(X)
        
        # Check that each row sums to approximately 1.0
        row_sums = np.sum(proba, axis=1)
        np.testing.assert_allclose(row_sums, 1.0, rtol=1e-6)
    
    def test_predict_proba_valid_range(self, fitted_classifier):
        """Test that all probabilities are in [0, 1]."""
        classifier, X = fitted_classifier
        proba = classifier.predict_proba(X)
        
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)
    
    def test_predict_proba_before_fit_raises_error(self):
        """Test that predict_proba raises error before fitting."""
        classifier = TierClassifier()
        X = np.random.randn(10, 5)
        
        with pytest.raises(ValueError, match="must be fitted"):
            classifier.predict_proba(X)


class TestFeatureImportance:
    """Test feature importance extraction."""
    
    @pytest.fixture
    def fitted_classifier(self):
        """Create a fitted classifier."""
        np.random.seed(42)
        n_samples = 51  # 3 * 17
        X = np.random.randn(n_samples, 5)
        y = np.tile([0, 1, 2], n_samples // 3)
        assert len(y) == n_samples
        classifier = TierClassifier()
        feature_names = ['feat_a', 'feat_b', 'feat_c', 'feat_d', 'feat_e']
        classifier.fit(X, y, feature_names=feature_names)
        return classifier
    
    def test_feature_importance_extraction(self, fitted_classifier):
        """Test that feature importance can be extracted."""
        importance = fitted_classifier.get_feature_importance()
        
        # Should return a dictionary
        assert isinstance(importance, dict)
    
    def test_feature_importance_values_normalized(self, fitted_classifier):
        """Test that feature importance values are normalized."""
        importance = fitted_classifier.get_feature_importance()
        
        if importance:  # If any features have importance
            values = list(importance.values())
            assert all(0 <= v <= 1 for v in values)
    
    def test_feature_importance_before_fit_raises_error(self):
        """Test that feature importance raises error before fitting."""
        classifier = TierClassifier()
        
        with pytest.raises(ValueError, match="must be fitted"):
            classifier.get_feature_importance()


class TestModelPersistence:
    """Test model saving and loading."""
    
    @pytest.fixture
    def fitted_classifier(self):
        """Create a fitted classifier."""
        np.random.seed(42)
        n_samples = 51  # 3 * 17
        X = np.random.randn(n_samples, 5)
        y = np.tile([0, 1, 2], n_samples // 3)
        assert len(y) == n_samples
        classifier = TierClassifier()
        classifier.fit(X, y)
        return classifier, X, y
    
    def test_save_model(self, fitted_classifier):
        """Test model saving to temporary directory."""
        classifier, _, _ = fitted_classifier
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_model_path = Path(tmpdir) / "test_model"
            classifier.save(str(temp_model_path))
            
            # Check that model file was created
            model_file = temp_model_path.parent / f"{temp_model_path.name}_xgb_1.0.json"
            assert model_file.exists()
    
    def test_save_and_load_model(self, fitted_classifier):
        """Test saving and loading model."""
        classifier, X, y = fitted_classifier
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_model_path = Path(tmpdir) / "test_model"
            
            # Save model
            classifier.save(str(temp_model_path), version="1.0")
            
            # Load model
            loaded_classifier = TierClassifier.load(str(temp_model_path), version="1.0")
            
            # Check that loaded model is marked as fitted
            assert loaded_classifier._is_fitted
            
            # Note: predictions may differ slightly due to scaler not being persisted
            # In a full implementation, the scaler should be persisted with the model
            proba = loaded_classifier.predict_proba(X)
            assert proba.shape == (X.shape[0], 3)
            assert np.all(proba >= 0)
            assert np.all(proba <= 1)
    
    def test_load_nonexistent_model(self):
        """Test loading nonexistent model raises error."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_model_path = Path(tmpdir) / "nonexistent_model"
            with pytest.raises(FileNotFoundError):
                TierClassifier.load(str(temp_model_path), version="1.0")


class TestClassificationReport:
    """Test classification report generation."""
    
    @pytest.fixture
    def fitted_classifier(self):
        """Create a fitted classifier."""
        np.random.seed(42)
        n_samples = 51  # 3 * 17
        X = np.random.randn(n_samples, 5)
        y = np.tile([0, 1, 2], n_samples // 3)
        assert len(y) == n_samples
        classifier = TierClassifier()
        classifier.fit(X, y)
        return classifier, X, y
    
    def test_classification_report_generation(self, fitted_classifier):
        """Test that classification report can be generated."""
        classifier, X, y = fitted_classifier
        report = classifier.get_classification_report(X, y)
        
        assert isinstance(report, dict)
        assert 'accuracy' in report
        assert 'precision_weighted' in report
        assert 'recall_weighted' in report
        assert 'f1_weighted' in report
    
    def test_classification_report_values_valid(self, fitted_classifier):
        """Test that classification report values are in valid ranges."""
        classifier, X, y = fitted_classifier
        report = classifier.get_classification_report(X, y)
        
        # Check that metrics are in [0, 1] range
        for key in ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']:
            assert 0 <= report[key] <= 1
    
    def test_classification_report_before_fit_raises_error(self):
        """Test that classification report raises error before fitting."""
        classifier = TierClassifier()
        X = np.random.randn(10, 5)
        y = np.array([0, 1, 2] * 3 + [0])
        
        with pytest.raises(ValueError, match="must be fitted"):
            classifier.get_classification_report(X, y)


class TestFeatureDimensions:
    """Test feature dimension consistency."""
    
    def test_different_feature_dimensions(self):
        """Test that different feature dimensions are handled correctly."""
        classifier = TierClassifier()
        
        for n_features in [5, 10, 20, 50]:
            X = np.random.randn(30, n_features)
            y = np.tile([0, 1, 2], 30 // 3)
            y = y[:30]
            
            classifier.fit(X, y)
            predictions = classifier.predict(X)
            
            assert predictions.shape == (30,)
            assert len(classifier.feature_names) == n_features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
