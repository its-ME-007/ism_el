"""Unit tests for Age-Based baseline tiering policy."""

import pytest
from datetime import datetime, timedelta
from src.tiering.baselines.age_based import AgeBasedBaseline, Tier, MigrationAction


class TestAgeBasedBaseline:
    """Tests for AgeBasedBaseline class."""
    
    @pytest.fixture
    def baseline(self):
        """Create a baseline instance with default configuration."""
        return AgeBasedBaseline(
            hot_age_threshold_hours=24.0,
            warm_age_threshold_hours=168.0
        )
    
    @pytest.fixture
    def base_time(self):
        """Base timestamp for tests."""
        return datetime(2024, 1, 1, 12, 0, 0)
    
    def test_init_valid_configuration(self):
        """Test initialization with valid configuration."""
        baseline = AgeBasedBaseline(
            hot_age_threshold_hours=24.0,
            warm_age_threshold_hours=168.0
        )
        assert baseline.hot_age_threshold_hours == 24.0
        assert baseline.warm_age_threshold_hours == 168.0
        assert baseline.get_tier_occupancy() == {Tier.HOT: 0.0, Tier.WARM: 0.0, Tier.COLD: 0.0}
    
    def test_init_invalid_thresholds(self):
        """Test initialization with invalid thresholds (warm <= hot)."""
        with pytest.raises(ValueError):
            AgeBasedBaseline(
                hot_age_threshold_hours=168.0,
                warm_age_threshold_hours=24.0
            )
    
    def test_init_equal_thresholds(self):
        """Test initialization with equal thresholds."""
        with pytest.raises(ValueError):
            AgeBasedBaseline(
                hot_age_threshold_hours=24.0,
                warm_age_threshold_hours=24.0
            )
    
    def test_add_file_new_file(self, baseline, base_time):
        """Test adding a new file."""
        created_at = base_time - timedelta(days=1)
        modified_at = base_time
        
        baseline.add_file(
            file_id="file1",
            created_at=created_at,
            modified_at=modified_at,
            file_size=1.0
        )
        
        assert baseline.get_tier("file1") == Tier.COLD
    
    def test_add_file_duplicate(self, baseline, base_time):
        """Test adding a duplicate file raises error."""
        created_at = base_time - timedelta(days=1)
        modified_at = base_time
        
        baseline.add_file(
            file_id="file1",
            created_at=created_at,
            modified_at=modified_at,
            file_size=1.0
        )
        
        with pytest.raises(ValueError, match="already tracked"):
            baseline.add_file(
                file_id="file1",
                created_at=created_at,
                modified_at=modified_at,
                file_size=1.0
            )
    
    def test_add_file_invalid_timestamps(self, baseline, base_time):
        """Test adding file with modified_at before created_at."""
        created_at = base_time
        modified_at = base_time - timedelta(days=1)
        
        with pytest.raises(ValueError, match="modified_at cannot be before created_at"):
            baseline.add_file(
                file_id="file1",
                created_at=created_at,
                modified_at=modified_at,
                file_size=1.0
            )
    
    def test_determine_tier_hot(self, baseline, base_time):
        """Test tier determination for recently modified file (HOT)."""
        created_at = base_time - timedelta(days=10)
        modified_at = base_time - timedelta(hours=12)  # 12 hours ago
        
        baseline.add_file(
            file_id="file1",
            created_at=created_at,
            modified_at=modified_at,
            file_size=1.0
        )
        
        tier = baseline._determine_tier("file1", base_time)
        assert tier == Tier.HOT
    
    def test_determine_tier_warm(self, baseline, base_time):
        """Test tier determination for moderately aged file (WARM)."""
        created_at = base_time - timedelta(days=30)
        modified_at = base_time - timedelta(days=5)  # 5 days ago
        
        baseline.add_file(
            file_id="file1",
            created_at=created_at,
            modified_at=modified_at,
            file_size=1.0
        )
        
        tier = baseline._determine_tier("file1", base_time)
        assert tier == Tier.WARM
    
    def test_determine_tier_cold(self, baseline, base_time):
        """Test tier determination for old file (COLD)."""
        created_at = base_time - timedelta(days=100)
        modified_at = base_time - timedelta(days=20)  # 20 days ago
        
        baseline.add_file(
            file_id="file1",
            created_at=created_at,
            modified_at=modified_at,
            file_size=1.0
        )
        
        tier = baseline._determine_tier("file1", base_time)
        assert tier == Tier.COLD
    
    def test_determine_tier_boundary_hot_warm(self, baseline, base_time):
        """Test boundary between HOT and WARM tiers."""
        created_at = base_time - timedelta(days=10)
        # Exactly at the boundary
        modified_at = base_time - timedelta(hours=24)
        
        baseline.add_file(
            file_id="file1",
            created_at=created_at,
            modified_at=modified_at,
            file_size=1.0
        )
        
        tier = baseline._determine_tier("file1", base_time)
        assert tier == Tier.WARM  # At boundary, should be in WARM
    
    def test_determine_tier_boundary_warm_cold(self, baseline, base_time):
        """Test boundary between WARM and COLD tiers."""
        created_at = base_time - timedelta(days=30)
        # Exactly at the boundary
        modified_at = base_time - timedelta(hours=168)
        
        baseline.add_file(
            file_id="file1",
            created_at=created_at,
            modified_at=modified_at,
            file_size=1.0
        )
        
        tier = baseline._determine_tier("file1", base_time)
        assert tier == Tier.COLD  # At boundary, should be in COLD
    
    def test_on_decision_interval_no_migration_needed(self, baseline, base_time):
        """Test decision interval when no migrations are needed."""
        created_at = base_time - timedelta(days=1)
        modified_at = base_time - timedelta(hours=12)
        
        baseline.add_file(
            file_id="file1",
            created_at=created_at,
            modified_at=modified_at,
            file_size=1.0
        )
        
        # Set initial tier
        baseline._file_tiers["file1"] = Tier.HOT
        baseline._tier_occupancy[Tier.HOT] = 1.0
        
        decisions = baseline.on_decision_interval(base_time)
        assert len(decisions) == 0
    
    def test_on_decision_interval_cold_to_hot_migration(self, baseline, base_time):
        """Test migration from COLD to HOT when file is recently modified."""
        created_at = base_time - timedelta(days=1)
        modified_at = base_time - timedelta(hours=12)  # Recently modified
        
        baseline.add_file(
            file_id="file1",
            created_at=created_at,
            modified_at=modified_at,
            file_size=1.0
        )
        
        # File starts in COLD tier
        baseline._file_tiers["file1"] = Tier.COLD
        baseline._tier_occupancy[Tier.COLD] = 1.0
        
        decisions = baseline.on_decision_interval(base_time)
        
        assert len(decisions) == 1
        assert decisions[0].file_id == "file1"
        assert decisions[0].action == MigrationAction.MIGRATE
        assert decisions[0].source_tier == Tier.COLD
        assert decisions[0].dest_tier == Tier.HOT
        assert baseline.get_tier("file1") == Tier.HOT
    
    def test_on_decision_interval_hot_to_warm_migration(self, baseline, base_time):
        """Test migration from HOT to WARM when file ages."""
        created_at = base_time - timedelta(days=10)
        modified_at = base_time - timedelta(days=1.5)  # 1.5 days ago
        
        baseline.add_file(
            file_id="file1",
            created_at=created_at,
            modified_at=modified_at,
            file_size=1.0
        )
        
        # File starts in HOT tier
        baseline._file_tiers["file1"] = Tier.HOT
        baseline._tier_occupancy[Tier.HOT] = 1.0
        
        decisions = baseline.on_decision_interval(base_time)
        
        assert len(decisions) == 1
        assert decisions[0].file_id == "file1"
        assert decisions[0].source_tier == Tier.HOT
        assert decisions[0].dest_tier == Tier.WARM
        assert baseline.get_tier("file1") == Tier.WARM
    
    def test_on_decision_interval_warm_to_cold_migration(self, baseline, base_time):
        """Test migration from WARM to COLD when file is old."""
        created_at = base_time - timedelta(days=100)
        modified_at = base_time - timedelta(days=20)  # 20 days ago
        
        baseline.add_file(
            file_id="file1",
            created_at=created_at,
            modified_at=modified_at,
            file_size=1.0
        )
        
        # File starts in WARM tier
        baseline._file_tiers["file1"] = Tier.WARM
        baseline._tier_occupancy[Tier.WARM] = 1.0
        
        decisions = baseline.on_decision_interval(base_time)
        
        assert len(decisions) == 1
        assert decisions[0].file_id == "file1"
        assert decisions[0].source_tier == Tier.WARM
        assert decisions[0].dest_tier == Tier.COLD
        assert baseline.get_tier("file1") == Tier.COLD
    
    def test_on_decision_interval_multiple_files(self, baseline, base_time):
        """Test decision interval with multiple files in different tiers."""
        # Recent file
        baseline.add_file(
            file_id="file1",
            created_at=base_time - timedelta(days=1),
            modified_at=base_time - timedelta(hours=6),
            file_size=1.0
        )
        baseline._file_tiers["file1"] = Tier.COLD
        baseline._tier_occupancy[Tier.COLD] += 1.0
        
        # Medium-age file
        baseline.add_file(
            file_id="file2",
            created_at=base_time - timedelta(days=10),
            modified_at=base_time - timedelta(days=3),
            file_size=2.0
        )
        baseline._file_tiers["file2"] = Tier.HOT
        baseline._tier_occupancy[Tier.HOT] += 2.0
        
        # Old file
        baseline.add_file(
            file_id="file3",
            created_at=base_time - timedelta(days=100),
            modified_at=base_time - timedelta(days=25),
            file_size=3.0
        )
        baseline._file_tiers["file3"] = Tier.WARM
        baseline._tier_occupancy[Tier.WARM] += 3.0
        
        decisions = baseline.on_decision_interval(base_time)
        
        # file1: COLD -> HOT (recently modified)
        # file2: HOT -> WARM (aged)
        # file3: WARM -> COLD (very old)
        assert len(decisions) == 3
        
        file1_decision = next(d for d in decisions if d.file_id == "file1")
        assert file1_decision.dest_tier == Tier.HOT
        
        file2_decision = next(d for d in decisions if d.file_id == "file2")
        assert file2_decision.dest_tier == Tier.WARM
        
        file3_decision = next(d for d in decisions if d.file_id == "file3")
        assert file3_decision.dest_tier == Tier.COLD
    
    def test_on_decision_interval_specific_file_ids(self, baseline, base_time):
        """Test decision interval with specific file_ids list."""
        baseline.add_file(
            file_id="file1",
            created_at=base_time - timedelta(days=1),
            modified_at=base_time - timedelta(hours=12),
            file_size=1.0
        )
        baseline._file_tiers["file1"] = Tier.COLD
        baseline._tier_occupancy[Tier.COLD] += 1.0
        
        baseline.add_file(
            file_id="file2",
            created_at=base_time - timedelta(days=10),
            modified_at=base_time - timedelta(days=3),
            file_size=1.0
        )
        baseline._file_tiers["file2"] = Tier.HOT
        baseline._tier_occupancy[Tier.HOT] += 1.0
        
        # Only evaluate file1
        decisions = baseline.on_decision_interval(base_time, file_ids=["file1"])
        
        assert len(decisions) == 1
        assert decisions[0].file_id == "file1"
    
    def test_get_tier_occupancy(self, baseline, base_time):
        """Test getting tier occupancy."""
        baseline.add_file(
            file_id="file1",
            created_at=base_time - timedelta(days=1),
            modified_at=base_time - timedelta(hours=12),
            file_size=1.5
        )
        baseline._file_tiers["file1"] = Tier.HOT
        baseline._tier_occupancy[Tier.HOT] = 1.5
        
        occupancy = baseline.get_tier_occupancy()
        assert occupancy[Tier.HOT] == 1.5
        assert occupancy[Tier.WARM] == 0.0
        assert occupancy[Tier.COLD] == 0.0
    
    def test_get_tier_utilization(self, baseline, base_time):
        """Test getting tier utilization."""
        baseline_with_capacity = AgeBasedBaseline(
            hot_age_threshold_hours=24.0,
            warm_age_threshold_hours=168.0,
            tier_capacities={
                Tier.HOT: 10.0,
                Tier.WARM: 20.0,
                Tier.COLD: 100.0
            }
        )
        
        baseline_with_capacity.add_file(
            file_id="file1",
            created_at=base_time - timedelta(days=1),
            modified_at=base_time - timedelta(hours=12),
            file_size=5.0
        )
        baseline_with_capacity._file_tiers["file1"] = Tier.HOT
        baseline_with_capacity._tier_occupancy[Tier.HOT] = 5.0
        
        utilization = baseline_with_capacity.get_tier_utilization()
        assert utilization[Tier.HOT] == 50.0  # 5.0 / 10.0 * 100
    
    def test_get_file_age_hours(self, baseline, base_time):
        """Test getting file age in hours."""
        created_at = base_time - timedelta(days=10)
        modified_at = base_time - timedelta(hours=24)
        
        baseline.add_file(
            file_id="file1",
            created_at=created_at,
            modified_at=modified_at,
            file_size=1.0
        )
        
        age = baseline.get_file_age_hours("file1", base_time)
        assert age == 24.0
    
    def test_get_file_age_hours_nonexistent_file(self, baseline, base_time):
        """Test getting age for nonexistent file."""
        age = baseline.get_file_age_hours("nonexistent", base_time)
        assert age is None
    
    def test_get_tier_nonexistent_file(self, baseline):
        """Test getting tier for nonexistent file."""
        tier = baseline.get_tier("nonexistent")
        assert tier is None
    
    def test_reset_file_state(self, baseline, base_time):
        """Test resetting file state."""
        baseline.add_file(
            file_id="file1",
            created_at=base_time - timedelta(days=1),
            modified_at=base_time - timedelta(hours=12),
            file_size=1.0
        )
        baseline._file_tiers["file1"] = Tier.HOT
        baseline._tier_occupancy[Tier.HOT] = 1.0
        
        baseline.reset_file_state("file1")
        
        assert baseline.get_tier("file1") is None
        assert baseline.get_file_age_hours("file1", base_time) is None
        assert baseline._tier_occupancy[Tier.HOT] == 0.0
    
    def test_tier_occupancy_updates(self, baseline, base_time):
        """Test that occupancy is correctly updated during migrations."""
        baseline.add_file(
            file_id="file1",
            created_at=base_time - timedelta(days=1),
            modified_at=base_time - timedelta(hours=12),
            file_size=2.5
        )
        baseline._file_tiers["file1"] = Tier.COLD
        baseline._tier_occupancy[Tier.COLD] = 2.5
        
        decisions = baseline.on_decision_interval(base_time)
        
        # After migration to HOT
        occupancy = baseline.get_tier_occupancy()
        assert occupancy[Tier.COLD] == 0.0
        assert occupancy[Tier.HOT] == 2.5
