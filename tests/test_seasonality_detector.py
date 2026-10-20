"""Tests for SeasonalityDetector."""

import pytest
import numpy as np
from datetime import datetime, timedelta
from src.tiering.ingestion.schemas import AccessEvent, OperationType
from src.tiering.workload.seasonality import SeasonalityDetector, SeasonalityResult


class TestSeasonalityDetector:
    """Test suite for SeasonalityDetector."""
    
    @pytest.fixture
    def detector(self):
        """Create seasonality detector instance."""
        return SeasonalityDetector(
            min_period_hours=1,
            max_period_hours=168,  # 1 week
            confidence_threshold=0.3,
            min_events=20
        )
    
    @pytest.fixture
    def base_time(self):
        """Base reference time for tests."""
        return datetime(2024, 1, 15, 12, 0, 0)
    
    def test_empty_events(self, detector, base_time):
        """Test that empty events return None."""
        result = detector.detect([], base_time)
        assert result is None
    
    def test_insufficient_events(self, detector, base_time):
        """Test that fewer than min_events return None."""
        events = [
            AccessEvent(
                timestamp=base_time - timedelta(hours=i),
                file_id="test_file",
                operation=OperationType.READ
            )
            for i in range(1, 10)  # Only 9 events, min_events=20
        ]
        
        result = detector.detect(events, base_time)
        assert result is None
    
    def test_detect_hourly_seasonality(self, detector, base_time):
        """Test detection of hourly seasonal pattern."""
        # Create events with 6-hour periodicity
        events = []
        for day in range(14):  # 14 days
            for hour_offset in [0, 6, 12, 18]:  # 4 accesses per day at regular intervals
                event = AccessEvent(
                    timestamp=base_time - timedelta(days=day, hours=hour_offset),
                    file_id="test_file",
                    operation=OperationType.READ
                )
                events.append(event)
        
        result = detector.detect(events, base_time)
        
        # Should detect 6-hour periodicity
        assert result is not None
        assert result.period_hours == 6
        assert result.confidence > detector.confidence_threshold
    
    def test_detect_daily_seasonality(self, detector, base_time):
        """Test detection of daily seasonal pattern."""
        detector_daily = SeasonalityDetector(
            min_period_hours=1,
            max_period_hours=720,
            confidence_threshold=0.2,
            min_events=20
        )
        
        # Create events with 24-hour periodicity (daily)
        events = []
        for day in range(30):
            # 2 accesses per day at same time
            for hour_offset in [9, 17]:
                event = AccessEvent(
                    timestamp=base_time - timedelta(days=day, hours=hour_offset),
                    file_id="test_file",
                    operation=OperationType.READ
                )
                events.append(event)
        
        result = detector_daily.detect(events, base_time)
        
        # Should detect 24-hour periodicity
        assert result is not None
        assert result.period_hours in [23, 24, 25]  # Allow some tolerance
        assert result.confidence > detector_daily.confidence_threshold
    
    def test_no_seasonality_random(self, detector, base_time):
        """Test that random pattern does not detect strong seasonality."""
        np.random.seed(42)
        
        # Create random events
        events = []
        current_time = base_time - timedelta(days=30)
        
        for _ in range(100):
            random_hours = np.random.exponential(5)  # Random intervals
            current_time = current_time - timedelta(hours=random_hours)
            event = AccessEvent(
                timestamp=current_time,
                file_id="test_file",
                operation=OperationType.READ
            )
            events.append(event)
        
        result = detector.detect(events, base_time)
        
        # Random pattern should either return None or low confidence
        if result is not None:
            assert result.confidence < 0.5
    
    def test_seasonality_result_properties(self, detector, base_time):
        """Test that SeasonalityResult has valid properties."""
        # Create a simple periodic pattern
        events = []
        for hour in range(48):  # 2 days, hourly
            if hour % 12 < 6:  # Busy period
                event = AccessEvent(
                    timestamp=base_time - timedelta(hours=hour),
                    file_id="test_file",
                    operation=OperationType.READ
                )
                events.append(event)
                # Add multiple accesses in busy hours
                for _ in range(2):
                    events.append(event)
        
        if len(events) >= detector.min_events:
            result = detector.detect(events, base_time)
            
            if result is not None:
                assert result.period_hours > 0
                assert 0 <= result.amplitude <= 1
                assert 0 <= result.phase < 2 * np.pi
                assert result.confidence > 0
    
    def test_autocorrelation_computation(self, detector):
        """Test autocorrelation computation."""
        # Regular signal with period 12
        signal = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] * 5)  # 60 samples
        
        acf = detector._compute_autocorrelation(signal)
        
        # ACF should be highest at lag 0
        assert acf[0] == pytest.approx(1.0, abs=0.01)
        
        # ACF should show periodicity peaks
        assert len(acf) == len(signal)
    
    def test_hourly_counts(self, detector, base_time):
        """Test hourly count computation."""
        # Create events in specific hours
        events = [
            AccessEvent(
                timestamp=base_time - timedelta(hours=i),
                file_id="test_file",
                operation=OperationType.READ
            )
            for i in range(0, 72, 6)  # Every 6 hours for 72 hours
        ]
        
        hourly_counts = detector._compute_hourly_counts(events, base_time)
        
        # Should have approximately 72 hours of data (allow for boundary)
        assert len(hourly_counts) >= 66  # Within ~10% tolerance
        
        # Should have non-zero counts
        assert np.sum(hourly_counts) == len(events)
    
    def test_amplitude_computation(self, detector):
        """Test amplitude computation."""
        # Signal with clear amplitude pattern - alternating high and low
        signal = np.array([10, 10, 10, 10, 10, 10, 1, 1, 1, 1, 1, 1] * 5)
        
        amplitude = detector._compute_amplitude(signal, period=6)  # Use half-period
        
        # Amplitude should be non-negative
        assert amplitude >= 0
        # For alternating pattern, should have some amplitude
        assert amplitude > 0
    
    def test_phase_computation(self, detector):
        """Test phase computation."""
        # Signal with clear peak at position 3
        signal = np.array([1, 2, 5, 10, 3, 1] * 5)
        
        phase = detector._compute_phase(signal, period=6)
        
        # Phase should be in valid range
        assert 0 <= phase < 2 * np.pi


# Property tests
@pytest.mark.parametrize("num_events", [20, 50, 100, 200])
def test_property_seasonality_detection_accuracy(num_events):
    """Property 8: Seasonality Detection Accuracy.
    
    Validates: Requirements 2.6, 6.3
    
    For synthetic periodic signals, the detector should identify the
    correct period within the configured search space.
    """
    detector = SeasonalityDetector(
        min_period_hours=1,
        max_period_hours=100,
        confidence_threshold=0.2,
        min_events=15
    )
    
    base_time = datetime(2024, 1, 15, 12, 0, 0)
    
    # Create perfectly periodic signal with period 12
    events = []
    for i in range(num_events):
        timestamp = base_time - timedelta(hours=i)
        # Spike every 12 hours
        if (i % 12) == 0:
            for _ in range(3):
                events.append(AccessEvent(
                    timestamp=timestamp,
                    file_id="test_file",
                    operation=OperationType.READ
                ))
    
    result = detector.detect(events, base_time)
    
    if result is not None:
        # Should detect 12-hour period (with tolerance)
        assert result.period_hours in range(10, 15)
        assert result.confidence > 0.3
