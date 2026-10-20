"""Seasonality detection in access traces using autocorrelation."""

from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from pydantic import BaseModel, Field

from src.tiering.ingestion.schemas import AccessEvent


@dataclass
class SeasonalityResult:
    """Result of seasonality detection."""
    period_hours: int
    amplitude: float
    phase: float
    confidence: float
    
    def __repr__(self) -> str:
        return f"SeasonalityResult(period={self.period_hours}h, amplitude={self.amplitude:.3f}, confidence={self.confidence:.3f})"


class SeasonalityDetector:
    """
    Detect periodic patterns in access traces using autocorrelation analysis.
    
    Requirements:
    - Requirement 2.6: Detect seasonal patterns using autocorrelation
    - Requirement 6.3: Detect seasonal patterns using autocorrelation
    """
    
    def __init__(
        self,
        min_period_hours: int = 1,
        max_period_hours: int = 720,  # 30 days
        confidence_threshold: float = 0.3,
        min_events: int = 20
    ):
        """
        Initialize seasonality detector.
        
        Args:
            min_period_hours: Minimum period to consider (hours)
            max_period_hours: Maximum period to consider (hours)
            confidence_threshold: Minimum autocorrelation peak height to report seasonality
            min_events: Minimum number of events required for detection
        """
        self.min_period_hours = min_period_hours
        self.max_period_hours = max_period_hours
        self.confidence_threshold = confidence_threshold
        self.min_events = min_events
    
    def detect(
        self,
        events: List[AccessEvent],
        reference_time: Optional[datetime] = None
    ) -> Optional[SeasonalityResult]:
        """
        Detect seasonal patterns using autocorrelation analysis.
        
        Args:
            events: List of access events (should be chronologically sorted)
            reference_time: Reference time for computing access counts
                           (if None, uses time of last event)
        
        Returns:
            SeasonalityResult with period, amplitude, phase if detected; None otherwise
        """
        if len(events) < self.min_events:
            return None
        
        if not events:
            return None
        
        # Set reference time if not provided
        if reference_time is None:
            reference_time = events[-1].timestamp
        
        # Compute access counts per hour
        hourly_counts = self._compute_hourly_counts(events, reference_time)
        
        if len(hourly_counts) < 2 * self.min_period_hours:
            return None
        
        # Compute autocorrelation
        autocorr = self._compute_autocorrelation(hourly_counts)
        
        # Find dominant period
        period_hours, acf_value = self._find_dominant_period(
            autocorr,
            self.min_period_hours,
            self.max_period_hours
        )
        
        if period_hours is None or acf_value < self.confidence_threshold:
            return None
        
        # Compute amplitude and phase
        amplitude = self._compute_amplitude(hourly_counts, period_hours)
        phase = self._compute_phase(hourly_counts, period_hours)
        
        return SeasonalityResult(
            period_hours=period_hours,
            amplitude=amplitude,
            phase=phase,
            confidence=acf_value
        )
    
    @staticmethod
    def _compute_hourly_counts(
        events: List[AccessEvent],
        reference_time: datetime
    ) -> np.ndarray:
        """
        Compute access counts binned by hour.
        
        Returns:
            Array of hourly access counts
        """
        if not events:
            return np.array([])
        
        # Find earliest and latest events
        earliest = min(e.timestamp for e in events)
        latest = max(e.timestamp for e in events)
        
        # Calculate number of hours from earliest to reference time
        num_hours = int((reference_time - earliest).total_seconds() / 3600) + 1
        
        if num_hours <= 0:
            return np.array([0])
        
        # Initialize hourly counts
        hourly_counts = np.zeros(num_hours, dtype=int)
        
        # Bin events into hours
        for event in events:
            hour_offset = int((event.timestamp - earliest).total_seconds() / 3600)
            if 0 <= hour_offset < num_hours:
                hourly_counts[hour_offset] += 1
        
        return hourly_counts
    
    @staticmethod
    def _compute_autocorrelation(signal: np.ndarray, max_lag: Optional[int] = None) -> np.ndarray:
        """
        Compute autocorrelation using numpy.correlate.
        
        Args:
            signal: Input signal (hourly counts)
            max_lag: Maximum lag to compute (if None, uses length of signal)
        
        Returns:
            Autocorrelation values normalized to [0, 1]
        """
        if len(signal) < 2:
            return np.array([1.0])
        
        # Normalize signal to zero mean
        signal_normalized = signal - np.mean(signal)
        
        # Compute autocorrelation using correlate
        # 'full' mode returns correlations for all lags from -len(signal)+1 to len(signal)-1
        c = np.correlate(signal_normalized, signal_normalized, mode='full')
        
        # Keep only positive lags (including lag 0)
        c = c[len(c)//2:]
        
        # Normalize by variance (c[0] = variance * len(signal))
        if c[0] > 0:
            c = c / c[0]
        
        # Limit to max_lag
        if max_lag is not None:
            c = c[:max_lag+1]
        
        return c
    
    def _find_dominant_period(
        self,
        acf: np.ndarray,
        min_period: int,
        max_period: int
    ) -> Tuple[Optional[int], float]:
        """
        Find the dominant period from autocorrelation.
        
        Returns:
            Tuple of (period_hours, acf_value) or (None, 0.0) if no significant peak found
        """
        # Look for peaks in ACF between min and max period
        if max_period >= len(acf):
            max_period = len(acf) - 1
        
        # Find peaks: local maxima with sufficient height
        peaks = []
        for lag in range(min_period, min(max_period + 1, len(acf))):
            # Check if this is a local maximum
            is_peak = True
            
            # Check neighbors
            if lag > 0 and acf[lag] <= acf[lag - 1]:
                is_peak = False
            if lag < len(acf) - 1 and acf[lag] <= acf[lag + 1]:
                is_peak = False
            
            if is_peak and acf[lag] > 0.1:  # Minimum height threshold
                peaks.append((lag, acf[lag]))
        
        if not peaks:
            return None, 0.0
        
        # Return strongest peak
        period_hours, acf_value = max(peaks, key=lambda x: x[1])
        
        return period_hours, acf_value
    
    @staticmethod
    def _compute_amplitude(signal: np.ndarray, period: int) -> float:
        """
        Compute amplitude of periodic component.
        
        Uses the periodic component strength as a proxy for amplitude.
        
        Returns:
            Amplitude as a fraction of signal range
        """
        if len(signal) < period * 2:
            return 0.0
        
        signal_min = np.min(signal)
        signal_max = np.max(signal)
        
        if signal_max == signal_min:
            return 0.0
        
        # Normalize signal to [0, 1]
        signal_norm = (signal - signal_min) / (signal_max - signal_min)
        
        # Compute power at the detected frequency
        num_periods = len(signal) // period
        if num_periods < 2:
            return 0.0
        
        periodic_values = []
        for i in range(num_periods):
            segment = signal_norm[i * period : (i + 1) * period]
            periodic_values.append(np.mean(segment))
        
        # Amplitude as range of periodic averages
        if len(periodic_values) < 2:
            return 0.0
        
        amplitude = (np.max(periodic_values) - np.min(periodic_values))
        
        return float(np.clip(amplitude, 0.0, 1.0))
    
    @staticmethod
    def _compute_phase(signal: np.ndarray, period: int) -> float:
        """
        Compute phase of periodic component.
        
        Returns:
            Phase in radians [0, 2π)
        """
        if len(signal) < period:
            return 0.0
        
        # Take first period as reference
        first_period = signal[:period]
        
        # Find maximum within the period
        max_idx = np.argmax(first_period)
        
        # Phase as fraction of period converted to radians
        phase = (max_idx / period) * 2 * np.pi
        
        return float(phase)
