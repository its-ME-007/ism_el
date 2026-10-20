"""Age-based tiering policy that migrates files based on creation/modification time."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional


class Tier(str, Enum):
    """Storage tier classification."""
    HOT = "HOT"
    WARM = "WARM"
    COLD = "COLD"


class MigrationAction(str, Enum):
    """Action to take for a file."""
    MIGRATE = "migrate"
    HOLD = "hold"
    NONE = "none"


class MigrationDecision:
    """Result of a migration evaluation."""
    
    def __init__(
        self,
        file_id: str,
        action: MigrationAction,
        source_tier: Optional[Tier] = None,
        dest_tier: Optional[Tier] = None,
        rationale: str = "",
        estimated_cost: float = 0.0
    ):
        self.file_id = file_id
        self.action = action
        self.source_tier = source_tier
        self.dest_tier = dest_tier
        self.rationale = rationale
        self.estimated_cost = estimated_cost


class AgeBasedBaseline:
    """
    Age-based tiering policy that migrates files based on file age.
    
    Tracks file creation and modification times and gradually migrates
    files to colder tiers based on configurable age thresholds. Files
    are evaluated at decision intervals to determine if they should be
    demoted based on their age since creation or last modification.
    
    Does not use semantic, workload, or future-access information,
    relying solely on file age for tiering decisions.
    
    **Tier Assignment Logic:**
    - HOT: Files modified within the last `hot_age_threshold_hours`
    - WARM: Files modified within the last `warm_age_threshold_hours`
    - COLD: Files older than `warm_age_threshold_hours`
    """
    
    def __init__(
        self,
        hot_age_threshold_hours: float = 24.0,
        warm_age_threshold_hours: float = 168.0,
        tier_capacities: Optional[Dict[Tier, float]] = None
    ):
        """
        Initialize age-based baseline with configuration.
        
        Args:
            hot_age_threshold_hours: Maximum age for HOT tier (hours).
                Files modified more recently than this are in HOT tier.
                Default: 24 hours (1 day)
            warm_age_threshold_hours: Maximum age for WARM tier (hours).
                Files modified between hot_age_threshold and warm_age_threshold
                are in WARM tier. Default: 168 hours (7 days)
            tier_capacities: Optional dictionary mapping Tier to capacity in GB.
                If provided, enforces capacity constraints. If not provided,
                tier assignments are purely age-based.
        
        Raises:
            ValueError: If warm_age_threshold_hours <= hot_age_threshold_hours
        """
        if warm_age_threshold_hours <= hot_age_threshold_hours:
            raise ValueError(
                f"warm_age_threshold_hours ({warm_age_threshold_hours}) must be "
                f"greater than hot_age_threshold_hours ({hot_age_threshold_hours})"
            )
        
        self.hot_age_threshold_hours = hot_age_threshold_hours
        self.warm_age_threshold_hours = warm_age_threshold_hours
        self.tier_capacities = tier_capacities or {}
        
        # Track state for each file
        self._file_tiers: Dict[str, Tier] = {}  # Current tier for each file
        self._file_sizes: Dict[str, float] = {}  # File size in GB
        self._file_created_at: Dict[str, datetime] = {}  # Creation timestamp
        self._file_modified_at: Dict[str, datetime] = {}  # Modification timestamp
        
        # Track tier occupancy
        self._tier_occupancy: Dict[Tier, float] = {
            Tier.HOT: 0.0,
            Tier.WARM: 0.0,
            Tier.COLD: 0.0
        }
    
    def add_file(
        self,
        file_id: str,
        created_at: datetime,
        modified_at: datetime,
        file_size: float
    ) -> None:
        """
        Add a file to tracking.
        
        Args:
            file_id: Unique file identifier
            created_at: File creation timestamp
            modified_at: File last modification timestamp
            file_size: File size in GB
            
        Raises:
            ValueError: If file_id already tracked or invalid timestamps
        """
        if file_id in self._file_tiers:
            raise ValueError(f"File {file_id} already tracked")
        
        if modified_at < created_at:
            raise ValueError(
                f"File {file_id}: modified_at cannot be before created_at"
            )
        
        self._file_created_at[file_id] = created_at
        self._file_modified_at[file_id] = modified_at
        self._file_sizes[file_id] = file_size
        
        # Initially assign based on age
        self._file_tiers[file_id] = Tier.COLD
    
    def get_tier(self, file_id: str) -> Optional[Tier]:
        """
        Get current tier for a file.
        
        Args:
            file_id: Unique file identifier
            
        Returns:
            Current Tier assignment or None if file not tracked
        """
        return self._file_tiers.get(file_id)
    
    def _determine_tier(self, file_id: str, reference_time: datetime) -> Tier:
        """
        Determine appropriate tier for a file based on age.
        
        Args:
            file_id: File identifier
            reference_time: Current time for age calculation
            
        Returns:
            Tier based on file age thresholds
            
        Raises:
            ValueError: If file not tracked
        """
        if file_id not in self._file_modified_at:
            raise ValueError(f"File {file_id} not tracked")
        
        modified_at = self._file_modified_at[file_id]
        age_hours = (reference_time - modified_at).total_seconds() / 3600.0
        
        # Determine tier based on age thresholds
        # Boundary transitions: at exactly the threshold, move to the next tier
        if age_hours < self.hot_age_threshold_hours:
            return Tier.HOT
        elif age_hours < self.warm_age_threshold_hours:
            return Tier.WARM
        else:
            return Tier.COLD
    
    def on_decision_interval(
        self,
        timestamp: datetime,
        file_ids: Optional[List[str]] = None
    ) -> List[MigrationDecision]:
        """
        Process decision interval - evaluate files for migration based on age.
        
        At each decision interval, files are re-evaluated to determine if they
        should be moved to a different tier based on their age. Files are moved
        to colder tiers as they age, and to hotter tiers if they are recently
        modified.
        
        Args:
            timestamp: Current simulation timestamp
            file_ids: List of file IDs to evaluate (None = all files)
            
        Returns:
            List of migration decisions for files requiring tier changes
        """
        if file_ids is None:
            file_ids = list(self._file_tiers.keys())
        
        decisions = []
        
        for file_id in file_ids:
            if file_id not in self._file_tiers:
                continue
            
            current_tier = self._file_tiers[file_id]
            target_tier = self._determine_tier(file_id, timestamp)
            
            # If tier changed, create migration decision
            if current_tier != target_tier:
                modified_at = self._file_modified_at[file_id]
                age_hours = (timestamp - modified_at).total_seconds() / 3600.0
                
                decision = MigrationDecision(
                    file_id=file_id,
                    action=MigrationAction.MIGRATE,
                    source_tier=current_tier,
                    dest_tier=target_tier,
                    rationale=(
                        f"File {file_id} age-based migration: "
                        f"age={age_hours:.1f}h, "
                        f"hot_threshold={self.hot_age_threshold_hours:.1f}h, "
                        f"warm_threshold={self.warm_age_threshold_hours:.1f}h"
                    )
                )
                
                self._execute_migration(file_id, current_tier, target_tier, timestamp)
                decisions.append(decision)
        
        return decisions
    
    def _execute_migration(
        self,
        file_id: str,
        source_tier: Tier,
        dest_tier: Tier,
        timestamp: datetime
    ) -> Tier:
        """
        Execute file migration between tiers.
        
        Updates tier assignment and occupancy tracking.
        
        Args:
            file_id: File to migrate
            source_tier: Source tier
            dest_tier: Destination tier
            timestamp: Migration timestamp
            
        Returns:
            Actual destination tier
        """
        file_size = self._file_sizes[file_id]
        
        # Remove from source tier
        self._tier_occupancy[source_tier] -= file_size
        
        # Add to destination tier
        self._tier_occupancy[dest_tier] += file_size
        self._file_tiers[file_id] = dest_tier
        
        return dest_tier
    
    def get_tier_occupancy(self) -> Dict[Tier, float]:
        """
        Get current occupancy for each tier.
        
        Returns:
            Dictionary mapping Tier to current occupancy in GB
        """
        return self._tier_occupancy.copy()
    
    def get_tier_utilization(self) -> Dict[Tier, float]:
        """
        Get current utilization percentage for each tier.
        
        Returns:
            Dictionary mapping Tier to utilization percentage (0-100)
        """
        utilization = {}
        for tier, capacity in self.tier_capacities.items():
            occupancy = self._tier_occupancy[tier]
            if capacity > 0:
                utilization[tier] = (occupancy / capacity) * 100
            else:
                utilization[tier] = 0.0
        return utilization
    
    def get_file_age_hours(self, file_id: str, reference_time: datetime) -> Optional[float]:
        """
        Get age of a file in hours (time since last modification).
        
        Args:
            file_id: File identifier
            reference_time: Time to measure age from
            
        Returns:
            Age in hours or None if file not tracked
        """
        if file_id not in self._file_modified_at:
            return None
        
        modified_at = self._file_modified_at[file_id]
        age_hours = (reference_time - modified_at).total_seconds() / 3600.0
        return age_hours
    
    def reset_file_state(self, file_id: str) -> None:
        """
        Reset state for a file.
        
        Useful for simulation scenarios or testing.
        
        Args:
            file_id: File identifier
        """
        if file_id in self._file_tiers:
            tier = self._file_tiers[file_id]
            size = self._file_sizes.get(file_id, 0.0)
            self._tier_occupancy[tier] -= size
            del self._file_tiers[file_id]
        
        if file_id in self._file_created_at:
            del self._file_created_at[file_id]
        if file_id in self._file_modified_at:
            del self._file_modified_at[file_id]
        if file_id in self._file_sizes:
            del self._file_sizes[file_id]
