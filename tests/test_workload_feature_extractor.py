"""Tests for WorkloadFeatureExtractor."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from src.tiering.ingestion.schemas import AccessEvent, OperationType, FileDocument, Category
from src.tiering.workload.feature_extractor import WorkloadFeatureExtractor, WorkloadFeatures


class TestWorkloadFeatureExtractor:
    """Test suite for WorkloadFeatureExtractor."""
    
    @pytest.fixture
    def extractor(self):
        """Create feature extractor instance."""
        return WorkloadFeatureExtractor(normalize=True)
    
    @pytest.fixture
    def base_time(self):
        """Base reference time for tests."""
        return datetime(2024, 1, 15, 12, 0, 0)
    
    @pytest.fixture
    def created_at(self, base_time):
        """File creation time."""
        return base_time - timedelta(days=30)
    
    def test_empty_events_returns_cold_start_features(self, extractor, base_time, created_at):
        """Test that empty events list returns cold-start features."""
        # Cold-start: no events before reference time
        dummy_event = AccessEvent(
            timestamp=base_time + timedelta(hours=1),
            file_id="test_file",
            operation=OperationType.READ
        )
        
        features = extractor.extract(
            [dummy_event],
            base_time,
            created_at
        )
        
        assert features.access_count_1h == 0
        assert features.access_count_24h == 0
        assert features.access_count_7d == 0
        assert features.access_count_30d == 0
        assert features.mean_inter_arrival_hours == 0.0
        assert features.std_inter_arrival_hours == 0.0
    
    def test_single_recent_access(self, extractor, base_time, created_at):
        """Test with single access event."""
        event = AccessEvent(
            timestamp=base_time - timedelta(hours=2),
            file_id="test_file",
            operation=OperationType.READ
        )
        
        features = extractor.extract([event], base_time, created_at)
        
        assert features.access_count_1h == 0
        assert features.access_count_24h == 1
        assert features.access_count_7d == 1
        assert features.access_count_30d == 1
        assert features.last_access_hours == 2.0
        assert features.read_write_ratio > 0
    
    def test_access_count_windows(self, extractor, base_time, created_at):
        """Test access counting in different time windows."""
        events = [
            AccessEvent(
                timestamp=base_time - timedelta(minutes=30),
                file_id="test_file",
                operation=OperationType.READ
            ),
            AccessEvent(
                timestamp=base_time - timedelta(hours=2),
                file_id="test_file",
                operation=OperationType.READ
            ),
            AccessEvent(
                timestamp=base_time - timedelta(hours=12),
                file_id="test_file",
                operation=OperationType.READ
            ),
            AccessEvent(
                timestamp=base_time - timedelta(days=2),
                file_id="test_file",
                operation=OperationType.READ
            ),
        ]
        
        features = extractor.extract(events, base_time, created_at)
        
        assert features.access_count_1h == 1  # Only the 30-min-old event
        assert features.access_count_24h == 3  # All except the 2-day-old
        assert features.access_count_7d == 4
        assert features.access_count_30d == 4
    
    def test_data_leakage_prevention(self, extractor, base_time, created_at):
        """Test that future events are not included (critical for preventing data leakage)."""
        past_event = AccessEvent(
            timestamp=base_time - timedelta(hours=1),
            file_id="test_file",
            operation=OperationType.READ
        )
        future_event = AccessEvent(
            timestamp=base_time + timedelta(hours=1),  # In the future
            file_id="test_file",
            operation=OperationType.READ
        )
        
        features = extractor.extract(
            [past_event, future_event],
            base_time,
            created_at
        )
        
        # Future event should NOT be counted
        assert features.access_count_1h == 1
        assert features.access_count_24h == 1
    
    def test_read_write_ratio(self, extractor, base_time, created_at):
        """Test read/write ratio computation."""
        events = [
            AccessEvent(
                timestamp=base_time - timedelta(hours=i),
                file_id="test_file",
                operation=OperationType.READ if i % 2 == 0 else OperationType.WRITE
            )
            for i in range(1, 5)
        ]
        
        features = extractor.extract(events, base_time, created_at)
        
        # 2 reads, 2 writes → ratio = 2/2 = 1.0
        assert features.read_write_ratio == 1.0
    
    def test_read_write_ratio_all_reads(self, extractor, base_time, created_at):
        """Test read/write ratio when all operations are reads."""
        events = [
            AccessEvent(
                timestamp=base_time - timedelta(hours=i),
                file_id="test_file",
                operation=OperationType.READ
            )
            for i in range(1, 4)
        ]
        
        features = extractor.extract(events, base_time, created_at)
        
        # 3 reads, 0 writes → ratio = 3/0 = infinity (handled as count)
        assert features.read_write_ratio == 3.0
    
    def test_inter_arrival_statistics(self, extractor, base_time, created_at):
        """Test inter-arrival time computation."""
        # Events with regular intervals (every 1 hour)
        events = [
            AccessEvent(
                timestamp=base_time - timedelta(hours=i),
                file_id="test_file",
                operation=OperationType.READ
            )
            for i in range(5, 0, -1)
        ]
        
        features = extractor.extract(events, base_time, created_at)
        
        # With regular 1-hour intervals, mean and std should be close to 1 and 0
        assert features.mean_inter_arrival_hours == pytest.approx(1.0, abs=0.01)
        assert features.std_inter_arrival_hours == pytest.approx(0.0, abs=0.01)
    
    def test_inter_arrival_statistics_irregular(self, extractor, base_time, created_at):
        """Test inter-arrival stats with irregular intervals."""
        events = [
            AccessEvent(
                timestamp=base_time - timedelta(hours=10),
                file_id="test_file",
                operation=OperationType.READ
            ),
            AccessEvent(
                timestamp=base_time - timedelta(hours=5),
                file_id="test_file",
                operation=OperationType.READ
            ),
            AccessEvent(
                timestamp=base_time - timedelta(hours=1),
                file_id="test_file",
                operation=OperationType.READ
            ),
        ]
        
        features = extractor.extract(events, base_time, created_at)
        
        # Inter-arrivals: 5 hours, 4 hours
        # Mean = 4.5, std > 0
        assert features.mean_inter_arrival_hours == pytest.approx(4.5, abs=0.01)
        assert features.std_inter_arrival_hours > 0
    
    def test_periodicity_score_regular(self, extractor, base_time, created_at):
        """Test periodicity score for regular access patterns."""
        # Perfectly regular: access every hour
        events = [
            AccessEvent(
                timestamp=base_time - timedelta(hours=i),
                file_id="test_file",
                operation=OperationType.READ
            )
            for i in range(1, 10)
        ]
        
        features = extractor.extract(events, base_time, created_at)
        
        # Regular pattern should have high periodicity score
        assert features.periodicity_score > 0.5
    
    def test_periodicity_score_irregular(self, extractor, base_time, created_at):
        """Test periodicity score for irregular access patterns."""
        # Highly irregular intervals
        intervals = [1, 10, 2, 50, 1, 15, 2]
        events = []
        current_time = base_time
        
        for interval in intervals:
            current_time = current_time - timedelta(hours=interval)
            events.append(
                AccessEvent(
                    timestamp=current_time,
                    file_id="test_file",
                    operation=OperationType.READ
                )
            )
        
        # Reverse to chronological order
        events = list(reversed(events))
        
        features = extractor.extract(events, base_time, created_at)
        
        # Irregular pattern should have lower periodicity score
        assert features.periodicity_score < 0.5
    
    def test_time_since_creation(self, extractor, base_time, created_at):
        """Test time since creation calculation."""
        event = AccessEvent(
            timestamp=base_time - timedelta(hours=1),
            file_id="test_file",
            operation=OperationType.READ
        )
        
        features = extractor.extract([event], base_time, created_at)
        
        expected_hours = (base_time - created_at).total_seconds() / 3600.0
        assert features.time_since_creation_hours == pytest.approx(expected_hours, rel=0.01)
    
    def test_features_to_vector(self, extractor, base_time, created_at):
        """Test conversion of features to numpy vector."""
        events = [
            AccessEvent(
                timestamp=base_time - timedelta(hours=i),
                file_id="test_file",
                operation=OperationType.READ
            )
            for i in range(1, 5)
        ]
        
        features = extractor.extract(events, base_time, created_at)
        vector = features.to_vector()
        
        assert isinstance(vector, np.ndarray)
        assert vector.dtype == np.float32
        assert len(vector) == 10  # 10 features
        assert np.all(np.isfinite(vector))  # No NaN or inf
    
    def test_extract_batch(self, extractor, base_time, created_at):
        """Test batch extraction for multiple files."""
        events_by_file = {
            "file_1": [
                AccessEvent(
                    timestamp=base_time - timedelta(hours=i),
                    file_id="file_1",
                    operation=OperationType.READ
                )
                for i in range(1, 4)
            ],
            "file_2": [
                AccessEvent(
                    timestamp=base_time - timedelta(hours=i),
                    file_id="file_2",
                    operation=OperationType.READ
                )
                for i in range(1, 3)
            ],
        }
        
        file_metadata = {
            "file_1": created_at,
            "file_2": created_at,
        }
        
        features_dict = extractor.extract_batch(
            events_by_file, base_time, file_metadata
        )
        
        assert "file_1" in features_dict
        assert "file_2" in features_dict
        assert features_dict["file_1"].access_count_24h == 3
        assert features_dict["file_2"].access_count_24h == 2


# Property tests
@pytest.mark.parametrize("num_events", [0, 1, 5, 20, 100])
def test_property_feature_dimensions_consistent(num_events):
    """Property 4: Feature Vector Dimension Consistency.
    
    Validates: Requirements 6.4, 3.3, 3.5, 4.1, 4.2
    
    For any set of access events, the extracted features should
    produce a vector of consistent dimension.
    """
    extractor = WorkloadFeatureExtractor()
    base_time = datetime(2024, 1, 15, 12, 0, 0)
    created_at = base_time - timedelta(days=30)
    
    events = [
        AccessEvent(
            timestamp=base_time - timedelta(hours=i),
            file_id="test_file",
            operation=OperationType.READ if i % 2 == 0 else OperationType.WRITE
        )
        for i in range(1, num_events + 1)
    ]
    
    features = extractor.extract(events, base_time, created_at)
    vector = features.to_vector()
    
    # Dimension must always be 10
    assert len(vector) == 10
    # All values must be finite
    assert np.all(np.isfinite(vector))
    # All values must be in reasonable range
    assert np.all(vector >= 0)


@pytest.mark.parametrize("reference_offset_hours", [0, 1, 24, 168, 720])
def test_property_temporal_integrity(reference_offset_hours):
    """Property 7: Temporal Integrity in Feature Computation.
    
    Validates: Requirements 6.5, 7.4, 21.1, 21.2, 21.3
    
    Features computed at different reference times should only
    include events strictly before that reference time (no data leakage).
    """
    extractor = WorkloadFeatureExtractor()
    base_time = datetime(2024, 1, 15, 12, 0, 0)
    created_at = base_time - timedelta(days=30)
    reference_time = base_time + timedelta(hours=reference_offset_hours)
    
    # Create events spanning from past to future of any reference time
    events = [
        AccessEvent(
            timestamp=base_time - timedelta(hours=i),
            file_id="test_file",
            operation=OperationType.READ
        )
        for i in range(0, 100, 10)
    ] + [
        AccessEvent(
            timestamp=base_time + timedelta(hours=i),
            file_id="test_file",
            operation=OperationType.READ
        )
        for i in range(10, 100, 10)
    ]
    
    features = extractor.extract(events, reference_time, created_at)
    
    # Only events before reference_time should be counted
    historical_events = [e for e in events if e.timestamp < reference_time]
    expected_count_24h = sum(
        1 for e in historical_events
        if reference_time - timedelta(hours=24) <= e.timestamp < reference_time
    )
    
    assert features.access_count_24h == expected_count_24h
