"""Tests for training pipeline."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from src.tiering.models.training import (
    create_temporal_train_test_split,
    generate_tier_labels,
    train_future_access_predictor,
    train_tier_classifier,
    time_series_train_test_split,
    validate_no_data_leakage,
    TrainingPipeline
)
from src.tiering.ingestion.schemas import AccessEvent, OperationType


class TestTemporalTrainTestSplit:
    """Test temporal train/test splitting."""
    
    @pytest.fixture
    def sample_traces(self):
        """Create sample access traces."""
        base_time = datetime(2024, 1, 1)
        traces = []
        
        for i in range(100):
            traces.append(AccessEvent(
                timestamp=base_time + timedelta(hours=i),
                file_id=f"file_{i % 10}",
                operation=OperationType.READ,
                size_bytes=1024
            ))
        
        return traces
    
    def test_temporal_split_basic(self, sample_traces):
        """Test basic temporal split."""
        split_date = datetime(2024, 1, 3)
        
        train_events, test_events, test_start = create_temporal_train_test_split(
            [], sample_traces, split_date, 24
        )
        
        assert len(train_events) > 0
        assert len(test_events) > 0
        assert test_start == split_date
    
    def test_all_train_events_before_split(self, sample_traces):
        """Test that all train events are before split date."""
        split_date = datetime(2024, 1, 3)
        
        train_events, _, _ = create_temporal_train_test_split(
            [], sample_traces, split_date, 24
        )
        
        for event in train_events:
            assert event.timestamp < split_date
    
    def test_all_test_events_after_split(self, sample_traces):
        """Test that all test events are after split date."""
        split_date = datetime(2024, 1, 3)
        
        _, test_events, _ = create_temporal_train_test_split(
            [], sample_traces, split_date, 24
        )
        
        for event in test_events:
            assert event.timestamp >= split_date


class TestGenerateTierLabels:
    """Test tier label generation."""
    
    @pytest.fixture
    def sample_traces_and_files(self):
        """Create sample traces and file list."""
        base_time = datetime(2024, 1, 1)
        traces = []
        
        # Create traces with known access patterns
        # file_0: 80 accesses (HOT)
        # file_1: 50 accesses (WARM)
        # file_2: 20 accesses (COLD)
        
        for i in range(80):
            traces.append(AccessEvent(
                timestamp=base_time + timedelta(hours=i),
                file_id="file_0",
                operation=OperationType.READ
            ))
        
        for i in range(50):
            traces.append(AccessEvent(
                timestamp=base_time + timedelta(hours=i + 100),
                file_id="file_1",
                operation=OperationType.READ
            ))
        
        for i in range(20):
            traces.append(AccessEvent(
                timestamp=base_time + timedelta(hours=i + 200),
                file_id="file_2",
                operation=OperationType.READ
            ))
        
        # Add a file with no accesses
        traces.append(AccessEvent(
            timestamp=base_time + timedelta(hours=1),
            file_id="file_3",
            operation=OperationType.READ
        ))
        
        file_ids = ["file_0", "file_1", "file_2", "file_3"]
        
        return traces, file_ids, base_time
    
    def test_labels_generated(self, sample_traces_and_files):
        """Test that labels are generated for all files."""
        traces, file_ids, base_time = sample_traces_and_files
        
        labels = generate_tier_labels(file_ids, traces, base_time, 24)
        
        assert len(labels) == len(file_ids)
        assert all(fid in labels for fid in file_ids)
    
    def test_label_values_valid(self, sample_traces_and_files):
        """Test that label values are valid tier classes."""
        traces, file_ids, base_time = sample_traces_and_files
        
        labels = generate_tier_labels(file_ids, traces, base_time, 24)
        
        for label in labels.values():
            assert label in [0, 1, 2]  # HOT, WARM, COLD
    
    def test_no_accesses_generates_cold(self):
        """Test that files with no accesses are labeled COLD."""
        base_time = datetime(2024, 1, 1)
        traces = []  # No traces
        file_ids = ["file_0", "file_1"]
        
        labels = generate_tier_labels(file_ids, traces, base_time, 24)
        
        # All should be COLD
        for label in labels.values():
            assert label == 2


class TestTrainFutureAccessPredictor:
    """Test future access predictor training."""
    
    @pytest.fixture
    def synthetic_training_data(self):
        """Create synthetic training data."""
        np.random.seed(42)
        n_samples = 50
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)  # Binary: access or not
        
        feature_names = [f"feat_{i}" for i in range(n_features)]
        
        return X, y, feature_names
    
    def test_train_basic(self, synthetic_training_data):
        """Test basic training."""
        X, y, feature_names = synthetic_training_data
        
        predictor = train_future_access_predictor(X, y, feature_names, prediction_window_hours=24)
        
        assert predictor is not None
        assert predictor.model is not None
    
    def test_predict_after_train(self, synthetic_training_data):
        """Test prediction after training."""
        X, y, feature_names = synthetic_training_data
        
        predictor = train_future_access_predictor(X, y, feature_names)
        preds = predictor.predict(X)
        
        assert preds.shape == (X.shape[0],)
        assert np.all(preds >= 0)
        assert np.all(preds <= 1)
    
    def test_train_with_validation_set(self, synthetic_training_data):
        """Test training with validation set."""
        X, y, feature_names = synthetic_training_data
        X_val = np.random.randn(20, X.shape[1])
        y_val = np.random.randint(0, 2, 20)
        
        predictor = train_future_access_predictor(
            X, y, feature_names,
            X_test=X_val,
            y_test=y_val
        )
        
        assert predictor is not None


class TestTrainTierClassifier:
    """Test tier classifier training."""
    
    @pytest.fixture
    def synthetic_training_data(self):
        """Create synthetic training data."""
        np.random.seed(42)
        n_samples = 51
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y = np.tile([0, 1, 2], n_samples // 3)
        assert len(y) == n_samples
        
        feature_names = [f"feat_{i}" for i in range(n_features)]
        
        return X, y, feature_names
    
    def test_train_basic(self, synthetic_training_data):
        """Test basic training."""
        X, y, feature_names = synthetic_training_data
        
        classifier = train_tier_classifier(X, y, feature_names)
        
        assert classifier is not None
        assert classifier._is_fitted
    
    def test_predict_after_train(self, synthetic_training_data):
        """Test prediction after training."""
        X, y, feature_names = synthetic_training_data
        
        classifier = train_tier_classifier(X, y, feature_names)
        preds = classifier.predict(X)
        
        assert preds.shape == (X.shape[0],)
        assert np.all(np.isin(preds, [0, 1, 2]))
    
    def test_train_with_validation_set(self, synthetic_training_data):
        """Test training with validation set."""
        X, y, feature_names = synthetic_training_data
        X_val = np.random.randn(30, X.shape[1])
        y_val = np.tile([0, 1, 2], 30 // 3)
        
        classifier = train_tier_classifier(
            X, y, feature_names,
            X_test=X_val,
            y_test=y_val
        )
        
        assert classifier._is_fitted


class TestTimeSeriesSplit:
    """Test time-series aware train/test split."""
    
    def test_basic_split(self):
        """Test basic time-series split."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)
        
        X_train, X_test, y_train, y_test = time_series_train_test_split(X, y, test_size=0.2)
        
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
    
    def test_gap_application(self):
        """Test that gap is applied correctly."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 3, 100)
        
        X_train1, X_test1, _, _ = time_series_train_test_split(X, y, test_size=0.2, gap=0)
        X_train2, X_test2, _, _ = time_series_train_test_split(X, y, test_size=0.2, gap=5)
        
        # With gap, test should start later
        assert len(X_test2) <= len(X_test1)


class TestValidateNoDataLeakage:
    """Test data leakage validation."""
    
    def test_no_leakage_with_separate_indices(self):
        """Test validation passes with separate indices."""
        traces = [
            AccessEvent(timestamp=datetime(2024, 1, 1), file_id="f1", operation=OperationType.READ),
            AccessEvent(timestamp=datetime(2024, 1, 2), file_id="f2", operation=OperationType.READ),
        ]
        
        train_idx = np.array([0])
        test_idx = np.array([1])
        
        result = validate_no_data_leakage(train_idx, test_idx, traces, 24)
        
        assert result is True


class TestTrainingPipeline:
    """Test complete training pipeline."""
    
    @pytest.fixture
    def training_data(self):
        """Create training data."""
        np.random.seed(42)
        n_samples = 51
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        y_tier = np.tile([0, 1, 2], n_samples // 3)
        y_access = np.random.randint(0, 2, n_samples)
        
        assert len(y_tier) == n_samples
        
        feature_names = [f"feat_{i}" for i in range(n_features)]
        
        return X, y_tier, y_access, feature_names
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        pipeline = TrainingPipeline(random_state=42)
        
        assert pipeline.random_state == 42
        assert pipeline.tier_classifier is None
        assert pipeline.future_predictor is None
    
    def test_train_models(self, training_data):
        """Test complete model training."""
        X, y_tier, y_access, feature_names = training_data
        
        pipeline = TrainingPipeline()
        results = pipeline.train_models(
            X, y_tier, y_access,
            feature_names=feature_names
        )
        
        assert results['tier_classifier_trained']
        assert results['future_predictor_trained']
        assert results['n_training_samples'] == len(X)
        assert results['feature_count'] == X.shape[1]
    
    def test_models_are_usable_after_training(self, training_data):
        """Test that trained models can make predictions."""
        X, y_tier, y_access, feature_names = training_data
        
        pipeline = TrainingPipeline()
        pipeline.train_models(X, y_tier, y_access, feature_names=feature_names)
        
        # Make predictions
        tier_preds = pipeline.tier_classifier.predict(X)
        access_preds = pipeline.future_predictor.predict(X)
        
        assert tier_preds.shape == (len(X),)
        assert access_preds.shape == (len(X),)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
