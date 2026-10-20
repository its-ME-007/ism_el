"""Unit tests for workload trace generation"""

import pytest
from datetime import datetime, timedelta
from pathlib import Path
import json
import tempfile

from src.tiering.ingestion.workload_generator import WorkloadGenerator
from src.tiering.ingestion.schemas import (
    FileDocument, Category, AccessEvent, OperationType, WorkloadPatternType,
    WorkloadPatternSpec
)


@pytest.fixture
def sample_files():
    """Create sample files for testing"""
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    files = [
        FileDocument(
            file_id=f"file_{i}",
            filename=f"file_{i}.txt",
            category=Category.FINANCE,
            content=f"Test content {i}",
            size_bytes=1000 * (i + 1),
            created_at=base_time,
            modified_at=base_time
        )
        for i in range(10)
    ]
    return files


@pytest.fixture
def default_config():
    """Create default workload configuration"""
    return {
        'hot_ratio': 0.3,
        'warm_ratio': 0.4,
        'cold_ratio': 0.2,
        'seasonal_ratio': 0.1,
        'hot_frequency': 10.0,
        'warm_frequency': 1.0,
        'cold_frequency': 0.1,
        'seasonal_period_hours': 24.0,
        'seasonal_amplitude': 5.0,
        'read_write_ratio': 0.8
    }


class TestWorkloadGeneratorInit:
    """Test WorkloadGenerator initialization"""

    def test_init_with_default_config(self, default_config):
        """Test initialization with valid configuration"""
        gen = WorkloadGenerator(default_config, seed=42)
        assert gen.seed == 42
        assert gen.hot_frequency == 10.0
        assert gen.warm_frequency == 1.0
        assert gen.cold_frequency == 0.1

    def test_init_normalizes_pattern_ratios(self, default_config):
        """Test that pattern ratios are normalized to sum to 1"""
        gen = WorkloadGenerator(default_config, seed=42)
        total = gen.hot_ratio + gen.warm_ratio + gen.cold_ratio + gen.seasonal_ratio
        assert abs(total - 1.0) < 1e-6

    def test_init_read_write_ratio(self, default_config):
        """Test read/write probability calculation"""
        gen = WorkloadGenerator(default_config, seed=42)
        assert gen.read_probability > 0.7  # Should be close to 0.8
        assert gen.read_probability < 0.9


class TestPatternAssignment:
    """Test workload pattern assignment"""

    def test_assign_patterns_returns_all_files(self, sample_files, default_config):
        """Test that all files get assigned a pattern"""
        gen = WorkloadGenerator(default_config, seed=42)
        patterns = gen.assign_patterns(sample_files)
        
        assert len(patterns) == len(sample_files)
        assert all(f.file_id in patterns for f in sample_files)

    def test_assign_patterns_distribution(self, default_config):
        """Test that pattern distribution matches configuration"""
        # Create larger file set for statistical test
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        files = [
            FileDocument(
                file_id=f"file_{i}",
                filename=f"file_{i}.txt",
                category=Category.FINANCE,
                content=f"Test {i}",
                size_bytes=1000,
                created_at=base_time,
                modified_at=base_time
            )
            for i in range(100)
        ]

        gen = WorkloadGenerator(default_config, seed=42)
        patterns = gen.assign_patterns(files)

        # Count patterns
        pattern_counts = {p: 0 for p in WorkloadPatternType}
        for spec in patterns.values():
            pattern_counts[spec.pattern_type] += 1

        # Check rough alignment with expected ratios (with tolerance)
        total = len(files)
        assert pattern_counts[WorkloadPatternType.HOT] > total * 0.25  # ~30% expected
        assert pattern_counts[WorkloadPatternType.WARM] > total * 0.35  # ~40% expected
        assert pattern_counts[WorkloadPatternType.COLD] > total * 0.15  # ~20% expected

    def test_assign_patterns_hot_has_high_frequency(self, sample_files, default_config):
        """Test that hot pattern gets high frequency"""
        gen = WorkloadGenerator(default_config, seed=42)
        patterns = gen.assign_patterns(sample_files)

        hot_patterns = [p for p in patterns.values() if p.pattern_type == WorkloadPatternType.HOT]
        if hot_patterns:
            for pattern in hot_patterns:
                assert pattern.frequency == gen.hot_frequency

    def test_assign_patterns_seasonal_has_periodicity(self, sample_files, default_config):
        """Test that seasonal pattern has period and amplitude"""
        gen = WorkloadGenerator(default_config, seed=42)
        patterns = gen.assign_patterns(sample_files)

        seasonal_patterns = [p for p in patterns.values() 
                            if p.pattern_type == WorkloadPatternType.SEASONAL]
        if seasonal_patterns:
            for pattern in seasonal_patterns:
                assert pattern.seasonal_period_hours is not None
                assert pattern.seasonal_amplitude is not None
                assert pattern.seasonal_period_hours == gen.seasonal_period_hours
                assert pattern.seasonal_amplitude == gen.seasonal_amplitude


class TestTraceGeneration:
    """Test access trace generation"""

    def test_generate_traces_returns_list(self, sample_files, default_config):
        """Test that generate_traces returns a list of events"""
        gen = WorkloadGenerator(default_config, seed=42)
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        events = gen.generate_traces(sample_files, duration_hours=1.0, start_time=start_time)

        assert isinstance(events, list)
        assert len(events) > 0
        assert all(isinstance(e, AccessEvent) for e in events)

    def test_generate_traces_chronological_ordering(self, sample_files, default_config):
        """Test that events are in chronological order"""
        gen = WorkloadGenerator(default_config, seed=42)
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        events = gen.generate_traces(sample_files, duration_hours=1.0, start_time=start_time)

        # Check strictly ascending order
        for i in range(len(events) - 1):
            assert events[i].timestamp <= events[i + 1].timestamp

    def test_generate_traces_within_time_bounds(self, sample_files, default_config):
        """Test that all events are within specified time range"""
        gen = WorkloadGenerator(default_config, seed=42)
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        duration_hours = 2.0
        end_time = start_time + timedelta(hours=duration_hours)

        events = gen.generate_traces(sample_files, duration_hours=duration_hours, 
                                    start_time=start_time)

        for event in events:
            assert event.timestamp >= start_time
            assert event.timestamp <= end_time

    def test_generate_traces_uses_file_ids(self, sample_files, default_config):
        """Test that all events reference valid file IDs"""
        gen = WorkloadGenerator(default_config, seed=42)
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        events = gen.generate_traces(sample_files, duration_hours=1.0, start_time=start_time)

        file_ids = {f.file_id for f in sample_files}
        for event in events:
            assert event.file_id in file_ids

    def test_generate_traces_has_operations(self, sample_files, default_config):
        """Test that events have read/write operations"""
        gen = WorkloadGenerator(default_config, seed=42)
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        events = gen.generate_traces(sample_files, duration_hours=1.0, start_time=start_time)

        for event in events:
            assert event.operation in [OperationType.READ, OperationType.WRITE]

    def test_generate_traces_read_write_ratio(self, default_config):
        """Test that read/write ratio is approximately correct"""
        # Use many files and long duration for statistical test
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        files = [
            FileDocument(
                file_id=f"file_{i}",
                filename=f"file_{i}.txt",
                category=Category.FINANCE,
                content=f"Test {i}",
                size_bytes=1000,
                created_at=base_time,
                modified_at=base_time
            )
            for i in range(5)
        ]

        gen = WorkloadGenerator(default_config, seed=42)
        events = gen.generate_traces(files, duration_hours=10.0, start_time=base_time)

        reads = sum(1 for e in events if e.operation == OperationType.READ)
        total = len(events)
        
        if total > 0:
            read_ratio = reads / total
            # Allow 10% tolerance on the 0.8 expected ratio
            assert 0.70 < read_ratio < 0.90

    def test_generate_traces_nanosecond_precision(self, sample_files, default_config):
        """Test that timestamps have microsecond precision (nanosecond via microsecond)"""
        gen = WorkloadGenerator(default_config, seed=42)
        start_time = datetime(2024, 1, 1, 0, 0, 0)
        events = gen.generate_traces(sample_files, duration_hours=1.0, start_time=start_time)

        # Check that at least some events have non-zero microseconds
        # (indicating use of sub-second precision)
        microseconds = [e.timestamp.microsecond for e in events]
        assert len(set(microseconds)) > 1  # Not all the same

    def test_generate_traces_reproducible_with_seed(self, sample_files, default_config):
        """Test that same seed produces same traces"""
        gen1 = WorkloadGenerator(default_config, seed=42)
        events1 = gen1.generate_traces(sample_files, duration_hours=1.0, 
                                      start_time=datetime(2024, 1, 1, 0, 0, 0))

        gen2 = WorkloadGenerator(default_config, seed=42)
        events2 = gen2.generate_traces(sample_files, duration_hours=1.0,
                                      start_time=datetime(2024, 1, 1, 0, 0, 0))

        assert len(events1) == len(events2)
        for e1, e2 in zip(events1, events2):
            assert e1.timestamp == e2.timestamp
            assert e1.file_id == e2.file_id
            assert e1.operation == e2.operation

    def test_generate_traces_different_seeds_produce_different_traces(self, sample_files, default_config):
        """Test that different seeds produce different traces"""
        gen1 = WorkloadGenerator(default_config, seed=42)
        events1 = gen1.generate_traces(sample_files, duration_hours=1.0,
                                      start_time=datetime(2024, 1, 1, 0, 0, 0))

        gen2 = WorkloadGenerator(default_config, seed=43)
        events2 = gen2.generate_traces(sample_files, duration_hours=1.0,
                                      start_time=datetime(2024, 1, 1, 0, 0, 0))

        # Should have different number of events or different patterns
        if len(events1) == len(events2):
            # At least some events should be different
            differences = sum(1 for e1, e2 in zip(events1, events2)
                            if e1.timestamp != e2.timestamp or e1.operation != e2.operation)
            assert differences > 0

    def test_generate_traces_hot_produces_more_events(self, default_config):
        """Test that hot pattern produces more events than cold"""
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        
        # Create one hot file and one cold file
        hot_file = FileDocument(
            file_id="hot_file",
            filename="hot.txt",
            category=Category.FINANCE,
            content="Hot file",
            size_bytes=1000,
            created_at=base_time,
            modified_at=base_time
        )
        
        cold_file = FileDocument(
            file_id="cold_file",
            filename="cold.txt",
            category=Category.FINANCE,
            content="Cold file",
            size_bytes=1000,
            created_at=base_time,
            modified_at=base_time
        )

        gen = WorkloadGenerator(default_config, seed=42)
        
        # Generate many traces to get consistent pattern
        total_hot_events = 0
        total_cold_events = 0
        
        for seed in range(5):
            gen = WorkloadGenerator(default_config, seed=seed)
            patterns = gen.assign_patterns([hot_file, cold_file])
            
            # Set explicit patterns
            patterns[hot_file.file_id].pattern_type = WorkloadPatternType.HOT
            patterns[cold_file.file_id].pattern_type = WorkloadPatternType.COLD
            
            hot_events = gen._generate_file_events(
                hot_file, patterns[hot_file.file_id],
                base_time, base_time + timedelta(hours=1)
            )
            cold_events = gen._generate_file_events(
                cold_file, patterns[cold_file.file_id],
                base_time, base_time + timedelta(hours=1)
            )
            
            total_hot_events += len(hot_events)
            total_cold_events += len(cold_events)
        
        # Hot should produce significantly more events
        assert total_hot_events > total_cold_events * 2


class TestEventGeneration:
    """Test individual event generation methods"""

    def test_generate_hot_events_uses_high_frequency(self, sample_files, default_config):
        """Test hot events are generated frequently"""
        gen = WorkloadGenerator(default_config, seed=42)
        file = sample_files[0]

        start_time = datetime(2024, 1, 1, 0, 0, 0)
        end_time = start_time + timedelta(hours=1)

        from src.tiering.ingestion.schemas import WorkloadPatternSpec
        pattern = WorkloadPatternSpec(
            file_id=file.file_id,
            pattern_type=WorkloadPatternType.HOT,
            frequency=gen.hot_frequency
        )

        events = gen._generate_hot_events(file, pattern, start_time, end_time)

        # Should generate roughly hot_frequency * duration_hours events
        expected = int(gen.hot_frequency * 1.0)  # 1 hour duration
        assert len(events) > expected * 0.5  # Allow some variation
        assert len(events) < expected * 1.5

    def test_generate_seasonal_events_has_period(self, sample_files, default_config):
        """Test seasonal events follow periodic pattern"""
        gen = WorkloadGenerator(default_config, seed=42)
        file = sample_files[0]

        start_time = datetime(2024, 1, 1, 0, 0, 0)
        end_time = start_time + timedelta(hours=48)  # 2 days = 2 full periods

        from src.tiering.ingestion.schemas import WorkloadPatternSpec
        pattern = WorkloadPatternSpec(
            file_id=file.file_id,
            pattern_type=WorkloadPatternType.SEASONAL,
            frequency=1.0,
            seasonal_period_hours=gen.seasonal_period_hours,
            seasonal_amplitude=gen.seasonal_amplitude
        )

        events = gen._generate_seasonal_events(file, pattern, start_time, end_time)

        # Should have events across the period
        assert len(events) > 0
        
        # Check that events span the full time range reasonably
        first_event_hour = (events[0].timestamp - start_time).total_seconds() / 3600
        last_event_hour = (events[-1].timestamp - start_time).total_seconds() / 3600
        
        assert last_event_hour > 24  # Should span beyond first period


class TestJSONLOutput:
    """Test JSON Lines output functionality"""

    def test_to_jsonl_creates_file(self, sample_files, default_config):
        """Test that to_jsonl creates output file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "traces.jsonl"
            
            gen = WorkloadGenerator(default_config, seed=42)
            start_time = datetime(2024, 1, 1, 0, 0, 0)
            events = gen.generate_traces(sample_files, duration_hours=1.0, start_time=start_time)
            
            gen.to_jsonl(events, output_path)

            assert output_path.exists()

    def test_to_jsonl_valid_json_lines(self, sample_files, default_config):
        """Test that output file contains valid JSON lines"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "traces.jsonl"
            
            gen = WorkloadGenerator(default_config, seed=42)
            start_time = datetime(2024, 1, 1, 0, 0, 0)
            events = gen.generate_traces(sample_files, duration_hours=1.0, start_time=start_time)
            
            gen.to_jsonl(events, output_path)

            # Parse JSON lines
            with open(output_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) == len(events)
                
                for line in lines:
                    parsed = json.loads(line)
                    assert 'timestamp' in parsed
                    assert 'file_id' in parsed
                    assert 'operation' in parsed

    def test_to_jsonl_creates_directories(self, sample_files, default_config):
        """Test that to_jsonl creates parent directories"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "deep" / "nested" / "traces.jsonl"
            
            gen = WorkloadGenerator(default_config, seed=42)
            start_time = datetime(2024, 1, 1, 0, 0, 0)
            events = gen.generate_traces(sample_files, duration_hours=1.0, start_time=start_time)
            
            gen.to_jsonl(events, output_path)

            assert output_path.exists()
            assert output_path.parent.exists()
