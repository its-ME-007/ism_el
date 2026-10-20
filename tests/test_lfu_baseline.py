"""Unit tests for LFU baseline tiering policy."""

import pytest
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path to import the module directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tiering.baselines.lfu import (
    LFUBaseline,
    Tier,
    MigrationAction,
    MigrationDecision
)


class TestLFUBaselineInitialization:
    """Tests for LFUBaseline initialization."""
    
    def test_initialization_with_tier_capacities(self):
        """Test LFU baseline can be initialized with tier capacities."""
        capacities = {
            Tier.HOT: 10.0,
            Tier.WARM: 50.0,
            Tier.COLD: 500.0
        }
        baseline = LFUBaseline(tier_capacities=capacities)
        
        assert baseline.tier_capacities == capacities
        assert baseline.hot_threshold == 10
        assert baseline.cold_threshold == 2
        assert baseline.frequency_window_hours == 24.0
    
    def test_initialization_with_custom_thresholds(self):
        """Test LFU baseline with custom frequency thresholds."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LFUBaseline(
            tier_capacities=capacities,
            hot_threshold=20,
            cold_threshold=5,
            frequency_window_hours=48.0
        )
        
        assert baseline.hot_threshold == 20
        assert baseline.cold_threshold == 5
        assert baseline.frequency_window_hours == 48.0
    
    def test_initial_tier_occupancy_is_zero(self):
        """Test tier occupancy starts at zero."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LFUBaseline(tier_capacities=capacities)
        
        occupancy = baseline.get_tier_occupancy()
        assert occupancy[Tier.HOT] == 0.0
        assert occupancy[Tier.WARM] == 0.0
        assert occupancy[Tier.COLD] == 0.0


class TestLFUAccessFrequencyTracking:
    """Tests for access frequency counting."""
    
    def test_new_file_placement_in_cold_tier(self):
        """Test that new files are placed in COLD tier on first access."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LFUBaseline(tier_capacities=capacities)
        timestamp = datetime(2024, 1, 1, 12, 0)
        
        # First access to new file
        baseline.on_access("file1", timestamp, file_size=1.0)
        
        assert baseline.get_tier("file1") == Tier.COLD
    
    def test_frequency_counting_single_window(self):
        """Test that access frequency is counted within frequency window."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LFUBaseline(
            tier_capacities=capacities,
            frequency_window_hours=24.0,
            hot_threshold=5,
            cold_threshold=2
        )
        base_time = datetime(2024, 1, 1, 12, 0)
        
        # First access
        baseline.on_access("file1", base_time, file_size=1.0)
        baseline.on_access("file1", base_time + timedelta(hours=1), file_size=1.0)
        baseline.on_access("file1", base_time + timedelta(hours=2), file_size=1.0)
        baseline.on_access("file1", base_time + timedelta(hours=3), file_size=1.0)
        baseline.on_access("file1", base_time + timedelta(hours=4), file_size=1.0)
        
        # After 5 accesses, should be promoted to HOT (>= threshold of 5)
        assert baseline.get_tier("file1") == Tier.HOT
    
    def test_promotion_to_warm_tier(self):
        """Test promotion from COLD to WARM tier at threshold."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LFUBaseline(
            tier_capacities=capacities,
            frequency_window_hours=24.0,
            hot_threshold=10,
            cold_threshold=3
        )
        base_time = datetime(2024, 1, 1, 12, 0)
        
        # Access 3 times to hit WARM threshold
        for i in range(3):
            baseline.on_access(
                "file1",
                base_time + timedelta(hours=i),
                file_size=1.0
            )
        
        assert baseline.get_tier("file1") == Tier.WARM
    
    def test_promotion_to_hot_tier(self):
        """Test promotion from WARM to HOT tier at threshold."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LFUBaseline(
            tier_capacities=capacities,
            frequency_window_hours=24.0,
            hot_threshold=5,
            cold_threshold=2
        )
        base_time = datetime(2024, 1, 1, 12, 0)
        
        # Access 5 times to hit HOT threshold
        for i in range(5):
            baseline.on_access(
                "file1",
                base_time + timedelta(hours=i),
                file_size=1.0
            )
        
        assert baseline.get_tier("file1") == Tier.HOT
    
    def test_multiple_accesses_same_timestamp(self):
        """Test that multiple accesses at same timestamp are counted."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LFUBaseline(
            tier_capacities=capacities,
            frequency_window_hours=24.0,
            hot_threshold=3,
            cold_threshold=2
        )
        timestamp = datetime(2024, 1, 1, 12, 0)
        
        # Multiple accesses at same timestamp
        baseline.on_access("file1", timestamp, file_size=1.0)
        baseline.on_access("file1", timestamp, file_size=1.0)
        baseline.on_access("file1", timestamp, file_size=1.0)
        
        assert baseline.get_tier("file1") == Tier.HOT


class TestFrequencyDecay:
    """Tests for frequency decay and sliding window behavior."""
    
    def test_frequency_decay_outside_window(self):
        """Test that accesses outside window have reduced weight."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LFUBaseline(
            tier_capacities=capacities,
            frequency_window_hours=24.0,
            hot_threshold=5,
            cold_threshold=2,
            window_decay_enabled=True
        )
        base_time = datetime(2024, 1, 1, 12, 0)
        
        # Access 5 times at time T
        for i in range(5):
            baseline.on_access(
                "file1",
                base_time + timedelta(hours=i),
                file_size=1.0
            )
        
        # Should be in HOT tier (5 accesses >= threshold)
        assert baseline.get_tier("file1") == Tier.HOT
        
        # Now check frequency much later (outside window)
        much_later = base_time + timedelta(days=3)
        frequency = baseline._get_current_frequency("file1", much_later)
        
        # Decay should reduce frequency below threshold
        assert frequency < 5
    
    def test_no_frequency_decay_when_disabled(self):
        """Test frequency decay can be disabled."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LFUBaseline(
            tier_capacities=capacities,
            frequency_window_hours=24.0,
            hot_threshold=5,
            cold_threshold=2,
            window_decay_enabled=False
        )
        base_time = datetime(2024, 1, 1, 12, 0)
        
        # Access 5 times
        for i in range(5):
            baseline.on_access(
                "file1",
                base_time + timedelta(hours=i),
                file_size=1.0
            )
        
        # Check frequency much later - should still count all accesses
        much_later = base_time + timedelta(days=3)
        frequency = baseline._get_current_frequency("file1", much_later)
        
        # Without decay, should keep all 5 accesses
        assert frequency == 5


class TestCapacityManagement:
    """Tests for tier capacity enforcement."""
    
    def test_occupancy_tracking_on_access(self):
        """Test that occupancy is updated on file access."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LFUBaseline(tier_capacities=capacities)
        timestamp = datetime(2024, 1, 1, 12, 0)
        
        baseline.on_access("file1", timestamp, file_size=2.5)
        
        occupancy = baseline.get_tier_occupancy()
        assert occupancy[Tier.COLD] == 2.5
    
    def test_occupancy_update_on_promotion(self):
        """Test that occupancy is updated when file is promoted."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LFUBaseline(
            tier_capacities=capacities,
            frequency_window_hours=24.0,
            hot_threshold=3,
            cold_threshold=2
        )
        base_time = datetime(2024, 1, 1, 12, 0)
        
        # First access - file stays in COLD (frequency=1 < cold_threshold=2)
        baseline.on_access("file1", base_time, file_size=1.0)
        assert baseline.get_tier("file1") == Tier.COLD
        occupancy_cold_1 = baseline.get_tier_occupancy()[Tier.COLD]
        
        # Second access - promotes to WARM (frequency=2 >= cold_threshold=2)
        baseline.on_access("file1", base_time + timedelta(hours=1), file_size=1.0)
        assert baseline.get_tier("file1") == Tier.WARM
        occupancy_cold_2 = baseline.get_tier_occupancy()[Tier.COLD]
        occupancy_warm = baseline.get_tier_occupancy()[Tier.WARM]
        
        assert occupancy_cold_1 == 1.0
        assert occupancy_cold_2 == 0.0
        assert occupancy_warm == 1.0
    
    def test_capacity_overflow_cascading_demotion(self):
        """Test that files are demoted when tier capacity exceeded."""
        capacities = {Tier.HOT: 5.0, Tier.WARM: 10.0, Tier.COLD: 500.0}
        baseline = LFUBaseline(
            tier_capacities=capacities,
            frequency_window_hours=24.0,
            hot_threshold=3,
            cold_threshold=2
        )
        base_time = datetime(2024, 1, 1, 12, 0)
        
        # Create file that would be promoted to HOT
        baseline.on_access("file1", base_time, file_size=3.0)
        baseline.on_access("file1", base_time + timedelta(hours=1), file_size=3.0)
        baseline.on_access("file1", base_time + timedelta(hours=2), file_size=3.0)
        
        # File should be in HOT (3.0 < 5.0 capacity)
        assert baseline.get_tier("file1") == Tier.HOT
        
        # Create second file that would overflow HOT
        baseline.on_access("file2", base_time, file_size=3.0)
        baseline.on_access("file2", base_time + timedelta(hours=1), file_size=3.0)
        baseline.on_access("file2", base_time + timedelta(hours=2), file_size=3.0)
        
        # File2 should be demoted to WARM (HOT is full)
        occupancy = baseline.get_tier_occupancy()
        assert occupancy[Tier.HOT] == 3.0  # Only file1
        assert occupancy[Tier.WARM] == 3.0  # File2 demoted
    
    def test_tier_utilization_percentage(self):
        """Test tier utilization percentage calculation."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LFUBaseline(tier_capacities=capacities)
        timestamp = datetime(2024, 1, 1, 12, 0)
        
        baseline.on_access("file1", timestamp, file_size=2.5)
        
        utilization = baseline.get_tier_utilization()
        assert utilization[Tier.COLD] == 0.5  # 2.5 / 500 = 0.5%
        assert utilization[Tier.HOT] == 0.0


class TestDecisionInterval:
    """Tests for on_decision_interval behavior."""
    
    def test_decision_interval_evaluates_all_files(self):
        """Test that decision interval checks all files."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LFUBaseline(
            tier_capacities=capacities,
            frequency_window_hours=24.0,
            hot_threshold=5,
            cold_threshold=2
        )
        base_time = datetime(2024, 1, 1, 12, 0)
        
        # Create files with different frequencies
        for i in range(5):
            baseline.on_access(
                "file1",
                base_time + timedelta(hours=i),
                file_size=1.0
            )
        
        for i in range(2):
            baseline.on_access(
                "file2",
                base_time + timedelta(hours=i),
                file_size=1.0
            )
        
        baseline.on_access("file3", base_time, file_size=1.0)
        
        # file1 in HOT, file2 in WARM, file3 in COLD
        assert baseline.get_tier("file1") == Tier.HOT
        assert baseline.get_tier("file2") == Tier.WARM
        assert baseline.get_tier("file3") == Tier.COLD
        
        # Decision interval far in future - should trigger frequency decay
        future_time = base_time + timedelta(days=2)
        decisions = baseline.on_decision_interval(future_time)
        
        # Files should be demoted due to decayed frequency
        assert baseline.get_tier("file1") == Tier.WARM or baseline.get_tier("file1") == Tier.COLD
    
    def test_decision_interval_specific_files(self):
        """Test decision interval can evaluate specific files."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LFUBaseline(
            tier_capacities=capacities,
            frequency_window_hours=24.0,
            hot_threshold=5,
            cold_threshold=2
        )
        base_time = datetime(2024, 1, 1, 12, 0)
        
        # Create multiple files
        for i in range(5):
            baseline.on_access("file1", base_time + timedelta(hours=i), file_size=1.0)
        
        baseline.on_access("file2", base_time, file_size=1.0)
        
        # Call decision interval for specific file
        future_time = base_time + timedelta(hours=25)
        decisions = baseline.on_decision_interval(future_time, file_ids=["file1"])
        
        # Only file1 should be evaluated
        assert all(d.file_id == "file1" for d in decisions)


class TestFileStateManagement:
    """Tests for file state tracking and reset."""
    
    def test_reset_file_state_clears_frequency(self):
        """Test that reset_file_state clears frequency counter."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LFUBaseline(
            tier_capacities=capacities,
            frequency_window_hours=24.0,
            hot_threshold=5,
            cold_threshold=2
        )
        base_time = datetime(2024, 1, 1, 12, 0)
        
        # Create file with high frequency
        for i in range(5):
            baseline.on_access("file1", base_time + timedelta(hours=i), file_size=1.0)
        
        assert baseline.get_tier("file1") == Tier.HOT
        
        # Reset state
        baseline.reset_file_state("file1")
        
        # Frequency should be 0 now
        assert baseline._get_current_frequency("file1", base_time) == 0
    
    def test_get_tier_returns_none_for_unknown_file(self):
        """Test that get_tier returns None for files not tracked."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LFUBaseline(tier_capacities=capacities)
        
        assert baseline.get_tier("unknown_file") is None


class TestNoSemanticInformation:
    """Verify LFU does not use semantic or future-access information."""
    
    def test_lfu_only_uses_access_frequency(self):
        """Test that LFU baseline only considers access frequency."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LFUBaseline(tier_capacities=capacities)
        
        # LFU should only have frequency tracking attributes
        # No semantic features or future predictions
        assert not hasattr(baseline, "semantic_features")
        assert not hasattr(baseline, "future_predictions")
        assert not hasattr(baseline, "rag_context")
        
        # Only frequency-related tracking
        assert hasattr(baseline, "_access_frequencies")
        assert hasattr(baseline, "_file_tiers")
        assert hasattr(baseline, "_last_access_time")


class TestMigrationDecisions:
    """Tests for migration decision generation."""
    
    def test_migration_decision_format(self):
        """Test that migration decisions have proper format."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LFUBaseline(
            tier_capacities=capacities,
            frequency_window_hours=24.0,
            hot_threshold=3,
            cold_threshold=2
        )
        base_time = datetime(2024, 1, 1, 12, 0)
        
        # Create file with access history
        for i in range(3):
            baseline.on_access("file1", base_time + timedelta(hours=i), file_size=1.0)
        
        # Decision interval should generate decisions
        future_time = base_time + timedelta(days=2)
        decisions = baseline.on_decision_interval(future_time)
        
        # Check decision format
        if decisions:
            decision = decisions[0]
            assert isinstance(decision, MigrationDecision)
            assert decision.file_id == "file1"
            assert isinstance(decision.action, MigrationAction)
            assert isinstance(decision.rationale, str)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_baseline_state(self):
        """Test baseline with no files."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LFUBaseline(tier_capacities=capacities)
        
        occupancy = baseline.get_tier_occupancy()
        assert all(v == 0.0 for v in occupancy.values())
        
        decisions = baseline.on_decision_interval(datetime.now())
        assert len(decisions) == 0
    
    def test_single_file_in_system(self):
        """Test with single file."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LFUBaseline(tier_capacities=capacities)
        timestamp = datetime(2024, 1, 1, 12, 0)
        
        baseline.on_access("file1", timestamp, file_size=5.0)
        
        assert baseline.get_tier("file1") == Tier.COLD
        occupancy = baseline.get_tier_occupancy()
        assert occupancy[Tier.COLD] == 5.0
    
    def test_zero_frequency_threshold(self):
        """Test with zero frequency threshold for edge case."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LFUBaseline(
            tier_capacities=capacities,
            hot_threshold=0,
            cold_threshold=0
        )
        timestamp = datetime(2024, 1, 1, 12, 0)
        
        # Even first access should trigger HOT tier
        baseline.on_access("file1", timestamp, file_size=1.0)
        assert baseline.get_tier("file1") == Tier.HOT
    
    def test_file_size_validation_on_first_access(self):
        """Test that file size is required for new files."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LFUBaseline(tier_capacities=capacities)
        timestamp = datetime(2024, 1, 1, 12, 0)
        
        # Should raise error if file_size not provided for new file
        with pytest.raises(ValueError):
            baseline.on_access("file1", timestamp, file_size=None)
    
    def test_subsequent_access_without_file_size(self):
        """Test that file_size can be omitted for known files."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LFUBaseline(tier_capacities=capacities)
        base_time = datetime(2024, 1, 1, 12, 0)
        
        # First access with size
        baseline.on_access("file1", base_time, file_size=1.0)
        
        # Subsequent accesses without size should work
        baseline.on_access("file1", base_time + timedelta(hours=1))
        baseline.on_access("file1", base_time + timedelta(hours=2))
        
        assert baseline.get_tier("file1") is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
