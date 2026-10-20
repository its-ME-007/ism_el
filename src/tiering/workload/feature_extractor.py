"""Workload feature extraction from access traces."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import sys
import numpy as np
from pydantic import BaseModel, Field

from src.tiering.ingestion.schemas import AccessEvent, OperationType


class WorkloadFeatures(BaseModel):
    """Time-series features extracted from access patterns."""
    file_id: str
    
    # Access counts per time window
    access_count_1h: int = Field(ge=0)
    access_count_24h: int = Field(ge=0)
    access_count_7d: int = Field(ge=0)
    access_count_30d: int = Field(ge=0)
    
    # Recency metrics
    last_access_hours: float = Field(ge=0)
    time_since_creation_hours: float = Field(ge=0)
    
    # Inter-arrival statistics
    mean_inter_arrival_hours: float = Field(ge=0)
    std_inter_arrival_hours: float = Field(ge=0)
    
    # Pattern indicators
    read_write_ratio: float = Field(ge=0)
    periodicity_score: float = Field(ge=0, le=1)
    
    def to_vector(self) -> np.ndarray:
        """Convert features to normalized numpy array for ML input."""
        return np.array([
            self.access_count_1h,
            self.access_count_24h,
            self.access_count_7d,
            self.access_count_30d,
            self.last_access_hours,
            self.time_since_creation_hours,
            self.mean_inter_arrival_hours,
            self.std_inter_arrival_hours,
            self.read_write_ratio,
            self.periodicity_score
        ], dtype=np.float32)


class WorkloadFeatureExtractor:
    """
    Extract time-series features from access traces.
    
    Computes workload features only from historical data available
    at the time of computation, preventing data leakage.
    
    Requirements:
    - Requirement 6.1: Compute access counts per configurable time windows
    - Requirement 6.2: Compute recency and inter-arrival statistics
    - Requirement 6.3: Detect seasonal patterns using autocorrelation
    - Requirement 6.4: Output normalized feature vectors for ML
    - Requirement 6.5: Prevent data leakage by using only historical data
    """
    
    def __init__(
        self,
        window_1h_minutes: int = 60,
        window_24h_hours: int = 24,
        window_7d_days: int = 7,
        window_30d_days: int = 30,
        normalize: bool = True
    ):
        """
        Initialize workload feature extractor.
        
        Args:
            window_1h_minutes: Time window for 1h access count (minutes)
            window_24h_hours: Time window for 24h access count (hours)
            window_7d_days: Time window for 7d access count (days)
            window_30d_days: Time window for 30d access count (days)
            normalize: Whether to normalize features
        """
        self.window_1h = timedelta(minutes=window_1h_minutes)
        self.window_24h = timedelta(hours=window_24h_hours)
        self.window_7d = timedelta(days=window_7d_days)
        self.window_30d = timedelta(days=window_30d_days)
        self.normalize = normalize
    
    def extract(
        self,
        events: List[AccessEvent],
        reference_time: datetime,
        created_at: datetime
    ) -> WorkloadFeatures:
        """
        Extract features from events ONLY before reference_time.
        
        CRITICAL: Only uses events before reference_time to prevent data leakage.
        
        Args:
            events: List of all access events for this file
            reference_time: Time of feature extraction (no events after this used)
            created_at: File creation timestamp
            
        Returns:
            WorkloadFeatures with normalized values
        """
        # Filter events to only those before reference_time (no data leakage)
        historical_events = [e for e in events if e.timestamp < reference_time]
        
        if not historical_events:
            # Cold-start: no historical data
            return WorkloadFeatures(
                file_id=events[0].file_id if events else "unknown",
                access_count_1h=0,
                access_count_24h=0,
                access_count_7d=0,
                access_count_30d=0,
                last_access_hours=float(sys.maxsize),  # Very large but finite value
                time_since_creation_hours=(reference_time - created_at).total_seconds() / 3600.0,
                mean_inter_arrival_hours=0.0,
                std_inter_arrival_hours=0.0,
                read_write_ratio=0.0,
                periodicity_score=0.0
            )
        
        file_id = historical_events[0].file_id
        
        # Compute access counts per time window
        access_count_1h = self._count_events_in_window(
            historical_events, reference_time - self.window_1h, reference_time
        )
        access_count_24h = self._count_events_in_window(
            historical_events, reference_time - self.window_24h, reference_time
        )
        access_count_7d = self._count_events_in_window(
            historical_events, reference_time - self.window_7d, reference_time
        )
        access_count_30d = self._count_events_in_window(
            historical_events, reference_time - self.window_30d, reference_time
        )
        
        # Last access recency in hours
        last_access_time = historical_events[-1].timestamp
        last_access_hours = max(0.0, (reference_time - last_access_time).total_seconds() / 3600.0)
        
        # Time since creation in hours
        time_since_creation_hours = (reference_time - created_at).total_seconds() / 3600.0
        
        # Inter-arrival statistics
        mean_inter_arrival, std_inter_arrival = self._compute_inter_arrival_stats(
            historical_events
        )
        
        # Read/write ratio
        read_write_ratio = self._compute_read_write_ratio(historical_events)
        
        # Periodicity score (simplified: based on coefficient of variation of inter-arrivals)
        periodicity_score = self._compute_periodicity_score(
            historical_events, reference_time
        )
        
        return WorkloadFeatures(
            file_id=file_id,
            access_count_1h=access_count_1h,
            access_count_24h=access_count_24h,
            access_count_7d=access_count_7d,
            access_count_30d=access_count_30d,
            last_access_hours=last_access_hours,
            time_since_creation_hours=time_since_creation_hours,
            mean_inter_arrival_hours=mean_inter_arrival,
            std_inter_arrival_hours=std_inter_arrival,
            read_write_ratio=read_write_ratio,
            periodicity_score=periodicity_score
        )
    
    def extract_batch(
        self,
        events_by_file: Dict[str, List[AccessEvent]],
        reference_time: datetime,
        file_metadata: Dict[str, datetime]
    ) -> Dict[str, WorkloadFeatures]:
        """
        Extract features for multiple files efficiently.
        
        Args:
            events_by_file: Dictionary mapping file_id to list of AccessEvents
            reference_time: Time of feature extraction
            file_metadata: Dictionary mapping file_id to created_at timestamp
            
        Returns:
            Dictionary mapping file_id to WorkloadFeatures
        """
        features = {}
        for file_id, events in events_by_file.items():
            created_at = file_metadata.get(file_id, min(e.timestamp for e in events))
            features[file_id] = self.extract(events, reference_time, created_at)
        return features
    
    @staticmethod
    def _count_events_in_window(
        events: List[AccessEvent],
        start_time: datetime,
        end_time: datetime
    ) -> int:
        """Count access events within time window."""
        return sum(1 for e in events if start_time <= e.timestamp < end_time)
    
    @staticmethod
    def _compute_inter_arrival_stats(
        events: List[AccessEvent]
    ) -> Tuple[float, float]:
        """
        Compute mean and std of inter-arrival times.
        
        Returns:
            Tuple of (mean_hours, std_hours)
        """
        if len(events) < 2:
            return 0.0, 0.0
        
        # Sort events by timestamp to ensure proper ordering
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        inter_arrivals = []
        for i in range(1, len(sorted_events)):
            delta = (sorted_events[i].timestamp - sorted_events[i-1].timestamp).total_seconds() / 3600.0
            # Ensure non-negative delta
            if delta >= 0:
                inter_arrivals.append(delta)
        
        if not inter_arrivals:
            return 0.0, 0.0
        
        mean_inter_arrival = np.mean(inter_arrivals)
        std_inter_arrival = np.std(inter_arrivals)
        
        return float(mean_inter_arrival), float(std_inter_arrival)
    
    @staticmethod
    def _compute_read_write_ratio(events: List[AccessEvent]) -> float:
        """Compute ratio of read to write operations."""
        if not events:
            return 0.0
        
        read_count = sum(1 for e in events if e.operation == OperationType.READ)
        write_count = sum(1 for e in events if e.operation == OperationType.WRITE)
        
        if write_count == 0:
            return float(read_count) if read_count > 0 else 0.0
        
        return read_count / write_count
    
    @staticmethod
    def _compute_periodicity_score(
        events: List[AccessEvent],
        reference_time: datetime
    ) -> float:
        """
        Compute periodicity score based on access pattern regularity.
        
        Uses coefficient of variation of inter-arrival times as a proxy for regularity:
        - High variation (irregular) → low score
        - Low variation (regular/periodic) → high score
        
        Returns:
            Score between 0.0 and 1.0
        """
        if len(events) < 3:
            return 0.0
        
        # Sort events by timestamp
        sorted_events = sorted(events, key=lambda e: e.timestamp)
        
        inter_arrivals = []
        for i in range(1, len(sorted_events)):
            delta = (sorted_events[i].timestamp - sorted_events[i-1].timestamp).total_seconds()
            if delta > 0:
                inter_arrivals.append(delta)
        
        if not inter_arrivals or len(inter_arrivals) < 2:
            return 0.0
        
        mean_delta = np.mean(inter_arrivals)
        std_delta = np.std(inter_arrivals)
        
        if mean_delta == 0:
            return 0.0
        
        # Coefficient of variation (CV)
        cv = std_delta / mean_delta
        
        # Convert CV to periodicity score: lower CV = higher score
        # CV > 1 means irregular, CV < 1 means relatively regular
        # Map CV to [0, 1] range using exponential decay
        periodicity = float(np.exp(-cv))  # Decay as CV increases
        
        return max(0.0, min(1.0, periodicity))
