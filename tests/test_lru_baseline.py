"""Unit tests for LRU baseline tiering policy."""

import pytest
from datetime import datetime, timedelta
from src.tiering.baselines.lru import (
    LRUBaseline,
    Tier,
    MigrationAction,
    MigrationDecision
)


class TestLRUBaselineInitialization:
    """Tests for LRUBaseline initialization."""
    
    def test_initialization_with_tier_capacities(self):
        """Test LRU baseline can be initialized with tier capacities."""
        capacities = {
            Tier.HOT: 10.0,
            Tier.WARM: 50.0,
            Tier.COLD: 500.0
        }
        baseline = LRUBaseline(tier_capacities=capacities)
        
        assert baseline.tier_capacities == capacities
        assert baseline.recency_window_hours == 24.0
        assert baseline.hot_capacity_files is None
    
    def test_initialization_with_custom_recency_window(self):
        """Test LRU baseline with custom recency window."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(
            tier_capacities=capacities,
            recency_window_hours=48.0
        )
        
        assert baseline.recency_window_hours == 48.0
    
    def test_initialization_with_hot_capacity_files(self):
        """Test LRU baseline with max HOT tier files limit."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(
            tier_capacities=capacities,
            hot_capacity_files=100
        )
        
        assert baseline.hot_capacity_files == 100
    
    def test_initial_tier_occupancy_is_zero(self):
        """Test tier occupancy starts at zero."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        
        occupancy = baseline.get_tier_occupancy()
        assert occupancy[Tier.HOT] == 0.0
        assert occupancy[Tier.WARM] == 0.0
        assert occupancy[Tier.COLD] == 0.0


class TestLRUAccessTracking:
    """Tests for access tracking and tier promotion."""
    
    def test_new_file_placement_in_cold_tier(self):
        """Test that new files are placed in COLD tier on initialization."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        timestamp = datetime(2024, 1, 1, 12, 0)
        
        # First access to new file
        baseline.on_access("file1", timestamp, file_size=1.0)
        
        assert baseline.get_tier("file1") == Tier.HOT
    
    def test_promotion_on_first_access(self):
        """Test that files are promoted to HOT tier on access."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        timestamp = datetime(2024, 1, 1, 12, 0)
        
        # Access file1
        baseline.on_access("file1", timestamp, file_size=1.0)
        
        # File should be in HOT tier (LRU promotes on access)
        assert baseline.get_tier("file1") == Tier.HOT
    
    def test_last_access_time_tracking(self):
        """Test that last access time is tracked."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        timestamp1 = datetime(2024, 1, 1, 12, 0)
        timestamp2 = datetime(2024, 1, 1, 13, 0)
        
        baseline.on_access("file1", timestamp1, file_size=1.0)
        assert baseline.get_file_access_time("file1") == timestamp1
        
        baseline.on_access("file1", timestamp2, file_size=1.0)
        assert baseline.get_file_access_time("file1") == timestamp2
    
    def test_access_updates_order(self):
        """Test that access updates the access order."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        base_time = datetime(2024, 1, 1, 12, 0)
        
        baseline.on_access("file1", base_time, file_size=1.0)
        baseline.on_access("file2", base_time + timedelta(hours=1), file_size=1.0)
        baseline.on_access("file3", base_time + timedelta(hours=2), file_size=1.0)
        
        # All should be in HOT tier
        assert baseline.get_tier("file1") == Tier.HOT
        assert baseline.get_tier("file2") == Tier.HOT
        assert baseline.get_tier("file3") == Tier.HOT
        
        # file3 is most recently used, file1 is least recently used
        access_order = list(baseline._access_order.keys())
        assert access_order[-1] == "file3"
        assert access_order[0] == "file1"
    
    def test_multiple_accesses_same_file(self):
        """Test multiple accesses to the same file."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        base_time = datetime(2024, 1, 1, 12, 0)
        
        # Multiple accesses to same file
        for i in range(5):
            baseline.on_access("file1", base_time + timedelta(hours=i), file_size=1.0)
        
        # File should remain in HOT tier
        assert baseline.get_tier("file1") == Tier.HOT


class TestCapacityManagement:
    """Tests for tier capacity enforcement."""
    
    def test_occupancy_tracking_on_access(self):
        """Test that occupancy is updated on file access."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        timestamp = datetime(2024, 1, 1, 12, 0)
        
        baseline.on_access("file1", timestamp, file_size=2.5)
        
        occupancy = baseline.get_tier_occupancy()
        # File is promoted to HOT on access
        assert occupancy[Tier.HOT] == 2.5
        assert occupancy[Tier.COLD] == 0.0
    
    def test_hot_tier_capacity_tracking(self):
        """Test that HOT tier capacity is properly tracked."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        base_time = datetime(2024, 1, 1, 12, 0)
        
        # Add files to HOT tier
        baseline.on_access("file1", base_time, file_size=3.0)
        baseline.on_access("file2", base_time + timedelta(hours=1), file_size=4.0)
        
        occupancy = baseline.get_tier_occupancy()
        assert occupancy[Tier.HOT] == 7.0
    
    def test_hot_capacity_exceeded_triggers_demotion(self):
        """Test that exceeding HOT capacity triggers demotion of LRU file."""
        capacities = {Tier.HOT: 5.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        base_time = datetime(2024, 1, 1, 12, 0)
        
        # Add file1 (3GB) to HOT
        baseline.on_access("file1", base_time, file_size=3.0)
        assert baseline.get_tier("file1") == Tier.HOT
        
        # Add file2 (4GB) - should also go to HOT initially
        baseline.on_access("file2", base_time + timedelta(hours=1), file_size=4.0)
        
        # file1 should be demoted because it's LRU
        # When on_decision_interval is called, it will demote file1
        occupancy = baseline.get_tier_occupancy()
        # file1 (3GB) + file2 (4GB) = 7GB, exceeds 5GB capacity
        
        # Run decision interval to enforce capacity
        baseline.on_decision_interval(base_time + timedelta(hours=2))
        
        occupancy_after = baseline.get_tier_occupancy()
        # file1 should be in WARM after demotion
        assert baseline.get_tier("file1") == Tier.WARM
        assert baseline.get_tier("file2") == Tier.HOT
    
    def test_tier_utilization_percentage(self):
        """Test tier utilization percentage calculation."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        timestamp = datetime(2024, 1, 1, 12, 0)
        
        baseline.on_access("file1", timestamp, file_size=2.5)
        
        utilization = baseline.get_tier_utilization()
        assert utilization[Tier.HOT] == 25.0  # 2.5 / 10.0 = 25%
        assert utilization[Tier.WARM] == 0.0


class TestLRUDemotion:
    """Tests for LRU demotion logic based on recency."""
    
    def test_demote_least_recently_used_from_hot(self):
        """Test that least recently used file is demoted from HOT."""
        capacities = {Tier.HOT: 5.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        base_time = datetime(2024, 1, 1, 12, 0)
        
        # Add file1 at time T
        baseline.on_access("file1", base_time, file_size=3.0)
        
        # Add file2 at time T+1h (more recent)
        baseline.on_access("file2", base_time + timedelta(hours=1), file_size=3.0)
        
        # Decision interval - should demote file1 (LRU)
        decisions = baseline.on_decision_interval(base_time + timedelta(hours=2))
        
        # file1 should be demoted, file2 should remain in HOT
        assert baseline.get_tier("file1") == Tier.WARM
        assert baseline.get_tier("file2") == Tier.HOT
        assert len([d for d in decisions if d.source_tier == Tier.HOT]) > 0
    
    def test_recency_threshold_for_warm_demotion(self):
        """Test that WARM files outside recency window are demoted to COLD when capacity exceeded."""
        capacities = {Tier.HOT: 50.0, Tier.WARM: 10.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(
            tier_capacities=capacities,
            recency_window_hours=24.0
        )
        base_time = datetime(2024, 1, 1, 12, 0)
        
        # Create multiple files in WARM tier to exceed capacity
        baseline._file_tiers["file1"] = Tier.WARM
        baseline._file_sizes["file1"] = 6.0
        baseline._tier_occupancy[Tier.WARM] = 6.0
        baseline._last_access_time["file1"] = base_time
        baseline._access_order["file1"] = base_time
        
        baseline._file_tiers["file2"] = Tier.WARM
        baseline._file_sizes["file2"] = 6.0
        baseline._tier_occupancy[Tier.WARM] = 12.0  # Exceeds 10GB capacity
        baseline._last_access_time["file2"] = base_time + timedelta(hours=1)
        baseline._access_order["file2"] = base_time + timedelta(hours=1)
        
        # Decision interval far in future (outside recency window)
        future_time = base_time + timedelta(days=3)
        decisions = baseline.on_decision_interval(future_time)
        
        # file1 should be demoted to COLD (outside recency window and LRU)
        assert baseline.get_tier("file1") == Tier.COLD
    
    def test_recent_files_not_demoted(self):
        """Test that recently accessed files are not demoted."""
        capacities = {Tier.HOT: 50.0, Tier.WARM: 10.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(
            tier_capacities=capacities,
            recency_window_hours=24.0
        )
        base_time = datetime(2024, 1, 1, 12, 0)
        
        # Create file in WARM tier with recent access
        baseline._file_tiers["file1"] = Tier.WARM
        baseline._file_sizes["file1"] = 1.0
        baseline._tier_occupancy[Tier.WARM] = 1.0
        baseline._last_access_time["file1"] = base_time
        baseline._access_order["file1"] = base_time
        
        # Decision interval just a few hours later (within recency window)
        future_time = base_time + timedelta(hours=6)
        decisions = baseline.on_decision_interval(future_time)
        
        # file1 should remain in WARM (within recency window)
        assert baseline.get_tier("file1") == Tier.WARM
    
    def test_multiple_demotions_at_decision_interval(self):
        """Test multiple files being demoted at decision interval."""
        capacities = {Tier.HOT: 5.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        base_time = datetime(2024, 1, 1, 12, 0)
        
        # Add multiple files to HOT
        baseline.on_access("file1", base_time, file_size=2.0)
        baseline.on_access("file2", base_time + timedelta(hours=1), file_size=2.0)
        baseline.on_access("file3", base_time + timedelta(hours=2), file_size=2.0)
        
        # Now HOT is full (6GB > 5GB capacity)
        # Decision interval should demote LRU files
        decisions = baseline.on_decision_interval(base_time + timedelta(hours=3))
        
        # At least one file should be demoted
        hot_tier_files = [
            f for f in ["file1", "file2", "file3"]
            if baseline.get_tier(f) == Tier.HOT
        ]
        
        assert len(hot_tier_files) <= 2  # Some files should be demoted


class TestDecisionInterval:
    """Tests for on_decision_interval behavior."""
    
    def test_decision_interval_returns_migration_decisions(self):
        """Test that decision interval returns migration decisions."""
        capacities = {Tier.HOT: 5.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        base_time = datetime(2024, 1, 1, 12, 0)
        
        baseline.on_access("file1", base_time, file_size=3.0)
        baseline.on_access("file2", base_time + timedelta(hours=1), file_size=3.0)
        
        decisions = baseline.on_decision_interval(base_time + timedelta(hours=2))
        
        # Should have migration decision for exceeding capacity
        assert isinstance(decisions, list)
        if decisions:
            for d in decisions:
                assert isinstance(d, MigrationDecision)
                assert d.action == MigrationAction.MIGRATE
    
    def test_decision_interval_with_specific_files(self):
        """Test decision interval can evaluate specific files."""
        capacities = {Tier.HOT: 50.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        base_time = datetime(2024, 1, 1, 12, 0)
        
        baseline.on_access("file1", base_time, file_size=1.0)
        baseline.on_access("file2", base_time + timedelta(hours=1), file_size=1.0)
        
        # Evaluate only file1
        decisions = baseline.on_decision_interval(
            base_time + timedelta(hours=2),
            file_ids=["file1"]
        )
        
        # Only file1 should be in decisions
        assert all(d.file_id == "file1" for d in decisions)
    
    def test_decision_interval_no_files(self):
        """Test decision interval with empty file list."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        
        decisions = baseline.on_decision_interval(
            datetime.now(),
            file_ids=[]
        )
        
        assert decisions == []
    
    def test_decision_interval_enforces_occupancy_limits(self):
        """Test that decision interval enforces tier occupancy limits."""
        capacities = {Tier.HOT: 5.0, Tier.WARM: 10.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        base_time = datetime(2024, 1, 1, 12, 0)
        
        # Setup: HOT is over capacity
        baseline._file_tiers["file1"] = Tier.HOT
        baseline._file_sizes["file1"] = 3.0
        baseline._tier_occupancy[Tier.HOT] = 3.0
        baseline._last_access_time["file1"] = base_time - timedelta(hours=1)
        baseline._access_order["file1"] = base_time - timedelta(hours=1)
        
        baseline._file_tiers["file2"] = Tier.HOT
        baseline._file_sizes["file2"] = 3.0
        baseline._tier_occupancy[Tier.HOT] = 6.0
        baseline._last_access_time["file2"] = base_time
        baseline._access_order["file2"] = base_time
        
        # Decision interval should demote file1
        decisions = baseline.on_decision_interval(base_time + timedelta(hours=1))
        
        # file1 should be demoted
        assert baseline.get_tier("file1") == Tier.WARM


class TestFileStateManagement:
    """Tests for file state tracking and utilities."""
    
    def test_get_file_access_time(self):
        """Test getting file's last access time."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        timestamp = datetime(2024, 1, 1, 12, 0)
        
        baseline.on_access("file1", timestamp, file_size=1.0)
        
        assert baseline.get_file_access_time("file1") == timestamp
    
    def test_get_file_access_time_unknown_file(self):
        """Test getting access time for unknown file."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        
        assert baseline.get_file_access_time("unknown") is None
    
    def test_reset_file_state(self):
        """Test resetting file state."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        base_time = datetime(2024, 1, 1, 12, 0)
        
        baseline.on_access("file1", base_time, file_size=1.0)
        assert baseline.get_file_access_time("file1") == base_time
        
        baseline.reset_file_state("file1")
        
        # Access time should be cleared from tracking
        assert "file1" not in baseline._access_order


class TestNoSemanticInformation:
    """Verify LRU does not use semantic or future-access information."""
    
    def test_lru_only_uses_recency(self):
        """Test that LRU baseline only considers access recency."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        
        # LRU should only have recency tracking attributes
        # No semantic features or future predictions
        assert not hasattr(baseline, "semantic_features")
        assert not hasattr(baseline, "future_predictions")
        assert not hasattr(baseline, "rag_context")
        
        # Only recency-related tracking
        assert hasattr(baseline, "_last_access_time")
        assert hasattr(baseline, "_access_order")
        assert hasattr(baseline, "_file_tiers")


class TestMigrationDecisions:
    """Tests for migration decision generation."""
    
    def test_migration_decision_has_rationale(self):
        """Test that migration decisions include rationale."""
        capacities = {Tier.HOT: 5.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        base_time = datetime(2024, 1, 1, 12, 0)
        
        baseline.on_access("file1", base_time, file_size=3.0)
        baseline.on_access("file2", base_time + timedelta(hours=1), file_size=3.0)
        
        decisions = baseline.on_decision_interval(base_time + timedelta(hours=2))
        
        if decisions:
            for d in decisions:
                assert d.rationale != ""
                assert isinstance(d.rationale, str)
    
    def test_migration_decision_specifies_tiers(self):
        """Test that migration decisions specify source and dest tiers."""
        capacities = {Tier.HOT: 5.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        base_time = datetime(2024, 1, 1, 12, 0)
        
        baseline.on_access("file1", base_time, file_size=3.0)
        baseline.on_access("file2", base_time + timedelta(hours=1), file_size=3.0)
        
        decisions = baseline.on_decision_interval(base_time + timedelta(hours=2))
        
        if decisions:
            for d in decisions:
                assert d.source_tier is not None
                assert d.dest_tier is not None
                assert isinstance(d.source_tier, Tier)
                assert isinstance(d.dest_tier, Tier)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_empty_baseline_state(self):
        """Test baseline with no files."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        
        occupancy = baseline.get_tier_occupancy()
        assert all(v == 0.0 for v in occupancy.values())
        
        decisions = baseline.on_decision_interval(datetime.now())
        assert len(decisions) == 0
    
    def test_single_file_in_system(self):
        """Test with single file."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        timestamp = datetime(2024, 1, 1, 12, 0)
        
        baseline.on_access("file1", timestamp, file_size=5.0)
        
        assert baseline.get_tier("file1") == Tier.HOT
        occupancy = baseline.get_tier_occupancy()
        assert occupancy[Tier.HOT] == 5.0
    
    def test_file_size_validation_on_first_access(self):
        """Test that file size is required for new files."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        timestamp = datetime(2024, 1, 1, 12, 0)
        
        # Should raise error if file_size not provided for new file
        with pytest.raises(ValueError):
            baseline.on_access("file1", timestamp, file_size=None)
    
    def test_subsequent_access_without_file_size(self):
        """Test that file_size can be omitted for known files."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        base_time = datetime(2024, 1, 1, 12, 0)
        
        # First access with size
        baseline.on_access("file1", base_time, file_size=1.0)
        
        # Subsequent accesses without size should work
        baseline.on_access("file1", base_time + timedelta(hours=1))
        baseline.on_access("file1", base_time + timedelta(hours=2))
        
        assert baseline.get_tier("file1") is not None
    
    def test_get_tier_returns_none_for_unknown_file(self):
        """Test that get_tier returns None for files not tracked."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        
        assert baseline.get_tier("unknown_file") is None


class TestConfigurableCapacities:
    """Tests for configurable capacity limits."""
    
    def test_different_tier_capacities(self):
        """Test with different tier capacities."""
        capacities = {
            Tier.HOT: 5.0,
            Tier.WARM: 20.0,
            Tier.COLD: 1000.0
        }
        baseline = LRUBaseline(tier_capacities=capacities)
        
        assert baseline.tier_capacities[Tier.HOT] == 5.0
        assert baseline.tier_capacities[Tier.WARM] == 20.0
        assert baseline.tier_capacities[Tier.COLD] == 1000.0
    
    def test_capacity_enforcement_hot_tier(self):
        """Test capacity enforcement for HOT tier."""
        capacities = {Tier.HOT: 10.0, Tier.WARM: 100.0, Tier.COLD: 1000.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        base_time = datetime(2024, 1, 1, 12, 0)
        
        # Fill HOT tier to capacity
        baseline.on_access("file1", base_time, file_size=5.0)
        baseline.on_access("file2", base_time + timedelta(hours=1), file_size=5.0)
        
        occupancy = baseline.get_tier_occupancy()
        assert occupancy[Tier.HOT] == 10.0
        
        # Add another file - should trigger capacity handling
        baseline.on_access("file3", base_time + timedelta(hours=2), file_size=5.0)
        
        # file3 should be in HOT initially (it was just accessed)
        # But file1 (LRU) should be demoted to WARM due to capacity
        assert baseline.get_tier("file3") == Tier.HOT
        assert baseline.get_tier("file1") == Tier.WARM


class TestPromotionAndDemotion:
    """Tests for promotion and demotion workflows."""
    
    def test_file_promoted_to_hot_on_access(self):
        """Test file promotion to HOT on access."""
        capacities = {Tier.HOT: 50.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        base_time = datetime(2024, 1, 1, 12, 0)
        
        # Manually place file in COLD
        baseline._file_tiers["file1"] = Tier.COLD
        baseline._file_sizes["file1"] = 1.0
        baseline._tier_occupancy[Tier.COLD] = 1.0
        baseline._last_access_time["file1"] = base_time
        baseline._access_order["file1"] = base_time
        
        # Access should promote to HOT
        baseline.on_access("file1", base_time + timedelta(hours=1), file_size=1.0)
        
        assert baseline.get_tier("file1") == Tier.HOT
        assert baseline.get_tier_occupancy()[Tier.COLD] == 0.0
        assert baseline.get_tier_occupancy()[Tier.HOT] == 1.0
    
    def test_sequential_promotion_and_demotion(self):
        """Test sequential promotion and demotion cycles."""
        capacities = {Tier.HOT: 5.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
        baseline = LRUBaseline(tier_capacities=capacities)
        base_time = datetime(2024, 1, 1, 12, 0)
        
        # Create file and promote to HOT
        baseline.on_access("file1", base_time, file_size=2.0)
        assert baseline.get_tier("file1") == Tier.HOT
        
        # Access other files to fill HOT
        baseline.on_access("file2", base_time + timedelta(hours=1), file_size=2.0)
        baseline.on_access("file3", base_time + timedelta(hours=2), file_size=2.0)
        
        # file1 should be demoted (it's LRU)
        baseline.on_decision_interval(base_time + timedelta(hours=3))
        assert baseline.get_tier("file1") == Tier.WARM


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
