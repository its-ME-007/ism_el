"""Workload trace generator for access patterns"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import json

from .schemas import (
    FileDocument, AccessEvent, OperationType, WorkloadPatternType,
    WorkloadPatternSpec
)


class WorkloadGenerator:
    """Generates realistic access traces with configurable patterns."""

    def __init__(self, config: Dict, seed: int):
        """
        Initialize generator with pattern distribution and seed.

        Args:
            config: Dictionary with pattern ratios, time parameters, periodicity settings
                   Expected keys:
                   - hot_ratio: float (0-1) - fraction of files with hot pattern
                   - warm_ratio: float (0-1) - fraction of files with warm pattern
                   - cold_ratio: float (0-1) - fraction of files with cold pattern
                   - seasonal_ratio: float (0-1) - fraction of files with seasonal pattern
                   - hot_frequency: float - accesses per hour for hot files
                   - warm_frequency: float - accesses per hour for warm files
                   - cold_frequency: float - accesses per hour for cold files
                   - seasonal_period_hours: float - period for seasonal pattern
                   - seasonal_amplitude: float - amplitude multiplier for seasonal
                   - read_write_ratio: float - ratio of reads to writes
            seed: Random seed for reproducibility
        """
        self.config = config
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Normalize pattern ratios to sum to 1
        total_ratio = (config.get('hot_ratio', 0.3) + 
                      config.get('warm_ratio', 0.4) +
                      config.get('cold_ratio', 0.2) +
                      config.get('seasonal_ratio', 0.1))
        
        self.hot_ratio = config.get('hot_ratio', 0.3) / total_ratio
        self.warm_ratio = config.get('warm_ratio', 0.4) / total_ratio
        self.cold_ratio = config.get('cold_ratio', 0.2) / total_ratio
        self.seasonal_ratio = config.get('seasonal_ratio', 0.1) / total_ratio

        # Access frequencies (accesses per hour)
        self.hot_frequency = config.get('hot_frequency', 10.0)
        self.warm_frequency = config.get('warm_frequency', 1.0)
        self.cold_frequency = config.get('cold_frequency', 0.1)

        # Seasonal pattern parameters
        self.seasonal_period_hours = config.get('seasonal_period_hours', 24.0)
        self.seasonal_amplitude = config.get('seasonal_amplitude', 5.0)

        # Read/write ratio (read_ratio: write_ratio)
        read_ratio = config.get('read_write_ratio', 0.8)
        write_ratio = 1.0 - read_ratio
        self.read_probability = read_ratio / (read_ratio + write_ratio)

    def assign_patterns(self, files: List[FileDocument]) -> Dict[str, WorkloadPatternSpec]:
        """
        Assign workload patterns (hot/warm/cold/seasonal) to files.

        Returns:
            Mapping from file_id to WorkloadPatternSpec
        """
        patterns: Dict[str, WorkloadPatternSpec] = {}
        num_files = len(files)

        # Generate pattern assignments based on ratios
        pattern_counts = {
            WorkloadPatternType.HOT: int(np.ceil(num_files * self.hot_ratio)),
            WorkloadPatternType.WARM: int(np.ceil(num_files * self.warm_ratio)),
            WorkloadPatternType.COLD: int(np.ceil(num_files * self.cold_ratio)),
            WorkloadPatternType.SEASONAL: int(np.floor(num_files * self.seasonal_ratio)),
        }

        # Adjust for any rounding errors
        total_assigned = sum(pattern_counts.values())
        if total_assigned > num_files:
            pattern_counts[WorkloadPatternType.COLD] -= (total_assigned - num_files)

        # Create shuffled list of patterns
        pattern_list = []
        for pattern, count in pattern_counts.items():
            pattern_list.extend([pattern] * count)
        
        # Shuffle and truncate to exact number of files
        self.rng.shuffle(pattern_list)
        pattern_list = pattern_list[:num_files]

        # Assign patterns to files
        for file, pattern in zip(files, pattern_list):
            if pattern == WorkloadPatternType.HOT:
                spec = WorkloadPatternSpec(
                    file_id=file.file_id,
                    pattern_type=pattern,
                    frequency=self.hot_frequency
                )
            elif pattern == WorkloadPatternType.WARM:
                spec = WorkloadPatternSpec(
                    file_id=file.file_id,
                    pattern_type=pattern,
                    frequency=self.warm_frequency
                )
            elif pattern == WorkloadPatternType.COLD:
                spec = WorkloadPatternSpec(
                    file_id=file.file_id,
                    pattern_type=pattern,
                    frequency=self.cold_frequency
                )
            else:  # SEASONAL
                spec = WorkloadPatternSpec(
                    file_id=file.file_id,
                    pattern_type=pattern,
                    frequency=self.warm_frequency,  # Base frequency
                    seasonal_period_hours=self.seasonal_period_hours,
                    seasonal_amplitude=self.seasonal_amplitude
                )
            patterns[file.file_id] = spec

        return patterns

    def generate_traces(
        self,
        files: List[FileDocument],
        duration_hours: float,
        start_time: datetime
    ) -> List[AccessEvent]:
        """
        Generate access traces for files over simulation duration.

        Args:
            files: List of FileDocument instances
            duration_hours: Simulation duration in hours
            start_time: Start time for the trace

        Returns:
            Chronologically sorted list of AccessEvent instances
        """
        # Assign patterns to files
        patterns = self.assign_patterns(files)

        # Generate events
        events: List[AccessEvent] = []
        end_time = start_time + timedelta(hours=duration_hours)

        for file in files:
            pattern_spec = patterns[file.file_id]
            file_events = self._generate_file_events(
                file, pattern_spec, start_time, end_time
            )
            events.extend(file_events)

        # Sort events chronologically
        events.sort(key=lambda e: e.timestamp)

        return events

    def _generate_file_events(
        self,
        file: FileDocument,
        pattern_spec: WorkloadPatternSpec,
        start_time: datetime,
        end_time: datetime
    ) -> List[AccessEvent]:
        """
        Generate events for a single file with its assigned pattern.

        Args:
            file: The file to generate events for
            pattern_spec: The workload pattern specification
            start_time: Start time for event generation
            end_time: End time for event generation

        Returns:
            List of AccessEvent instances
        """
        events: List[AccessEvent] = []
        duration_hours = (end_time - start_time).total_seconds() / 3600.0

        if pattern_spec.pattern_type == WorkloadPatternType.HOT:
            events = self._generate_hot_events(
                file, pattern_spec, start_time, end_time
            )
        elif pattern_spec.pattern_type == WorkloadPatternType.WARM:
            events = self._generate_warm_events(
                file, pattern_spec, start_time, end_time
            )
        elif pattern_spec.pattern_type == WorkloadPatternType.COLD:
            events = self._generate_cold_events(
                file, pattern_spec, start_time, end_time
            )
        elif pattern_spec.pattern_type == WorkloadPatternType.SEASONAL:
            events = self._generate_seasonal_events(
                file, pattern_spec, start_time, end_time
            )

        return events

    def _generate_hot_events(
        self,
        file: FileDocument,
        pattern_spec: WorkloadPatternSpec,
        start_time: datetime,
        end_time: datetime
    ) -> List[AccessEvent]:
        """Generate events for hot pattern (frequent recent access)"""
        events: List[AccessEvent] = []
        duration_seconds = (end_time - start_time).total_seconds()

        # Expected number of events
        frequency_per_second = self.hot_frequency / 3600.0
        expected_events = int(np.ceil(frequency_per_second * duration_seconds))

        # Generate inter-arrival times using exponential distribution
        inter_arrival_seconds = self.rng.exponential(
            1.0 / frequency_per_second, 
            expected_events + 10  # Add buffer
        )

        current_time = start_time
        for inter_arrival in inter_arrival_seconds:
            current_time = current_time + timedelta(seconds=inter_arrival)
            if current_time >= end_time:
                break

            # Add nanosecond precision via adjustment
            nanosecond_offset = self.rng.randint(0, 1000000)
            timestamp = current_time.replace(microsecond=nanosecond_offset // 1000)

            operation = (OperationType.READ if self.rng.random() < self.read_probability 
                        else OperationType.WRITE)

            event = AccessEvent(
                timestamp=timestamp,
                file_id=file.file_id,
                operation=operation,
                size_bytes=file.size_bytes if operation == OperationType.READ else None
            )
            events.append(event)

        return events

    def _generate_warm_events(
        self,
        file: FileDocument,
        pattern_spec: WorkloadPatternSpec,
        start_time: datetime,
        end_time: datetime
    ) -> List[AccessEvent]:
        """Generate events for warm pattern (moderate access)"""
        events: List[AccessEvent] = []
        duration_seconds = (end_time - start_time).total_seconds()

        # Expected number of events
        frequency_per_second = self.warm_frequency / 3600.0
        expected_events = int(np.ceil(frequency_per_second * duration_seconds))

        # Generate more regular inter-arrival times than hot (less variance)
        inter_arrival_seconds = self.rng.exponential(
            1.0 / frequency_per_second,
            expected_events + 10
        )

        current_time = start_time
        for inter_arrival in inter_arrival_seconds:
            current_time = current_time + timedelta(seconds=inter_arrival)
            if current_time >= end_time:
                break

            # Add nanosecond precision
            nanosecond_offset = self.rng.randint(0, 1000000)
            timestamp = current_time.replace(microsecond=nanosecond_offset // 1000)

            operation = (OperationType.READ if self.rng.random() < self.read_probability
                        else OperationType.WRITE)

            event = AccessEvent(
                timestamp=timestamp,
                file_id=file.file_id,
                operation=operation,
                size_bytes=file.size_bytes if operation == OperationType.READ else None
            )
            events.append(event)

        return events

    def _generate_cold_events(
        self,
        file: FileDocument,
        pattern_spec: WorkloadPatternSpec,
        start_time: datetime,
        end_time: datetime
    ) -> List[AccessEvent]:
        """Generate events for cold pattern (rare access)"""
        events: List[AccessEvent] = []
        duration_seconds = (end_time - start_time).total_seconds()

        # Expected number of events
        frequency_per_second = self.cold_frequency / 3600.0
        expected_events = int(np.ceil(frequency_per_second * duration_seconds))

        # For cold files, generate uniformly random events
        if expected_events > 0:
            event_times = self.rng.uniform(0, duration_seconds, expected_events)
            event_times = np.sort(event_times)

            for time_offset in event_times:
                timestamp = start_time + timedelta(seconds=time_offset)
                # Add nanosecond precision
                nanosecond_offset = self.rng.randint(0, 1000000)
                timestamp = timestamp.replace(microsecond=nanosecond_offset // 1000)

                operation = (OperationType.READ if self.rng.random() < self.read_probability
                            else OperationType.WRITE)

                event = AccessEvent(
                    timestamp=timestamp,
                    file_id=file.file_id,
                    operation=operation,
                    size_bytes=file.size_bytes if operation == OperationType.READ else None
                )
                events.append(event)

        return events

    def _generate_seasonal_events(
        self,
        file: FileDocument,
        pattern_spec: WorkloadPatternSpec,
        start_time: datetime,
        end_time: datetime
    ) -> List[AccessEvent]:
        """Generate events for seasonal pattern (periodic access)"""
        events: List[AccessEvent] = []
        period_seconds = pattern_spec.seasonal_period_hours * 3600.0
        amplitude = pattern_spec.seasonal_amplitude
        base_frequency = self.warm_frequency / 3600.0  # Convert to per second
        peak_multiplier = max(1.0, amplitude)
        peak_frequency = base_frequency * peak_multiplier

        # Use thinning to sample a non-homogeneous Poisson process. Candidate
        # events arrive at the peak rate and are accepted according to the
        # seasonal rate at their phase. This keeps the rate non-negative and
        # avoids long stalls when the sinusoid is near its trough.
        current_time = start_time
        while current_time < end_time:
            inter_arrival = self.rng.exponential(1.0 / max(peak_frequency, 1e-6))
            current_time = current_time + timedelta(seconds=inter_arrival)
            if current_time >= end_time:
                break

            # Calculate phase within current period
            seconds_since_start = (current_time - start_time).total_seconds()
            phase = (seconds_since_start % period_seconds) / period_seconds

            # Modulate frequency using sinusoidal pattern
            # Peak at phase=0.25 (quarter cycle), trough at phase=0.75
            frequency_multiplier = max(
                0.0,
                1.0 + (amplitude - 1.0) * np.sin(2 * np.pi * phase)
            )
            if self.rng.random_sample() > frequency_multiplier / peak_multiplier:
                continue

            # Add nanosecond precision
            nanosecond_offset = self.rng.randint(0, 1000000)
            timestamp = current_time.replace(microsecond=nanosecond_offset // 1000)

            operation = (OperationType.READ if self.rng.random() < self.read_probability
                        else OperationType.WRITE)

            event = AccessEvent(
                timestamp=timestamp,
                file_id=file.file_id,
                operation=operation,
                size_bytes=file.size_bytes if operation == OperationType.READ else None
            )
            events.append(event)

        return events

    def to_jsonl(self, events: List[AccessEvent], path: Path) -> None:
        """
        Write events to JSON Lines format.

        Args:
            events: List of AccessEvent instances
            path: Output file path
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            for event in events:
                f.write(event.model_dump_json() + '\n')
