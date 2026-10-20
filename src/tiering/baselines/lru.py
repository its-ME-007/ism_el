"""LRU (Least Recently Used) baseline tiering policy."""

from collections import OrderedDict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple


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


class LRUBaseline:
    """
    LRU (Least Recently Used) tiering policy.
    
    Tracks last access time for each file and promotes files to HOT tier
    on access. Files are demoted based on recency ranking when tier
    capacity limits are exceeded. Uses configurable tier capacity limits.
    
    Does not use semantic or future-access information, relying solely
    on access recency for tiering decisions.
    """
    
    def __init__(
        self,
        tier_capacities: Dict[Tier, float],
        recency_window_hours: float = 24.0,
        hot_capacity_files: Optional[int] = None
    ):
        """
        Initialize LRU baseline with configuration.
        
        Args:
            tier_capacities: Dictionary mapping Tier to capacity in GB
            recency_window_hours: Time window for considering "recent" access (hours)
            hot_capacity_files: Optional maximum number of files in HOT tier
        """
        self.tier_capacities = tier_capacities
        self.recency_window_hours = recency_window_hours
        self.hot_capacity_files = hot_capacity_files
        
        # Track state for each file
        self._file_tiers: Dict[str, Tier] = {}  # Current tier for each file
        self._file_sizes: Dict[str, float] = {}  # File size in GB
        self._last_access_time: Dict[str, datetime] = {}  # Last access timestamp
        
        # OrderedDict to track access order efficiently (LRU = least recently used)
        # Key: file_id, Value: last_access_time
        # Most recently accessed files are at the end
        self._access_order: OrderedDict[str, datetime] = OrderedDict()
        
        # Track tier occupancy
        self._tier_occupancy: Dict[Tier, float] = {
            Tier.HOT: 0.0,
            Tier.WARM: 0.0,
            Tier.COLD: 0.0
        }
        # Capacity evictions happen immediately on access. Retain their
        # records so the next decision interval can report those migrations.
        self._pending_migration_decisions: List[MigrationDecision] = []
    
    def get_tier(self, file_id: str) -> Optional[Tier]:
        """
        Get current tier for a file.
        
        Args:
            file_id: Unique file identifier
            
        Returns:
            Current Tier assignment or None if file not tracked
        """
        return self._file_tiers.get(file_id)
    
    def on_access(
        self,
        file_id: str,
        timestamp: datetime,
        file_size: Optional[float] = None
    ) -> Optional[Tier]:
        """
        Update tier state on access - promote file to HOT tier.
        
        When a file is accessed, it is promoted to the HOT tier.
        If the HOT tier is over capacity, the least recently used
        file is demoted to WARM.
        
        Args:
            file_id: Unique file identifier
            timestamp: Access timestamp
            file_size: File size in GB (required for first access)
            
        Returns:
            New tier if migration occurred, None otherwise
        """
        # Initialize file state if not seen before
        if file_id not in self._file_tiers:
            if file_size is None:
                raise ValueError(f"File size required for new file {file_id}")
            self._file_sizes[file_id] = file_size
            self._file_tiers[file_id] = Tier.COLD  # Start in COLD tier
            self._tier_occupancy[Tier.COLD] += file_size
            self._access_order[file_id] = timestamp
            self._last_access_time[file_id] = timestamp
        
        # Update access time and move to end of OrderedDict (most recently accessed)
        current_tier = self._file_tiers[file_id]
        
        self._last_access_time[file_id] = timestamp
        
        # Update access order - move to end (most recently used)
        if file_id in self._access_order:
            del self._access_order[file_id]
        self._access_order[file_id] = timestamp
        
        # Promote to HOT tier on access
        new_tier = Tier.HOT
        
        if current_tier != new_tier:
            # Execute migration from current tier to HOT
            self._execute_migration(file_id, current_tier, new_tier, timestamp)
            
            # Check if HOT tier is now over capacity and demote if needed
            hot_capacity = self.tier_capacities[Tier.HOT]
            hot_occupancy = self._tier_occupancy[Tier.HOT]
            
            if hot_occupancy > hot_capacity:
                # Need to demote LRU file from HOT
                hot_files = [
                    fid for fid in self._access_order.keys()
                    if self._file_tiers[fid] == Tier.HOT
                ]
                
                if hot_files:
                    # The first file in access_order is the least recently used
                    lru_file = hot_files[0]
                    if lru_file != file_id:  # Don't demote the file we just promoted
                        self._pending_migration_decisions.append(MigrationDecision(
                            file_id=lru_file,
                            action=MigrationAction.MIGRATE,
                            source_tier=Tier.HOT,
                            dest_tier=Tier.WARM,
                            rationale=(
                                f"HOT tier capacity exceeded after accessing {file_id}; "
                                f"demoted least recently used file {lru_file}"
                            )
                        ))
                        self._execute_migration(lru_file, Tier.HOT, Tier.WARM, timestamp)
            
            return new_tier
        
        return None
    
    def on_decision_interval(
        self,
        timestamp: datetime,
        file_ids: Optional[List[str]] = None
    ) -> List[MigrationDecision]:
        """
        Process decision interval - evaluate files for demotion based on recency.
        
        This method is called periodically to demote files based on recency.
        Files that haven't been accessed recently are moved to colder tiers
        to make room for frequently accessed files in the HOT tier.
        
        Args:
            timestamp: Current simulation timestamp
            file_ids: List of file IDs to evaluate (None = all files)
            
        Returns:
            List of migration decisions for files requiring demotion
        """
        if file_ids is None:
            file_ids = list(self._file_tiers.keys())
        
        decisions = self._pending_migration_decisions
        self._pending_migration_decisions = []
        
        # Sort files by recency to identify candidates for demotion
        recency_threshold = timestamp - timedelta(hours=self.recency_window_hours)
        
        # Check if HOT tier is over capacity
        hot_capacity = self.tier_capacities[Tier.HOT]
        hot_occupancy = self._tier_occupancy[Tier.HOT]
        
        # Collect files in HOT tier sorted by access recency (oldest first)
        hot_files = [
            (file_id, self._last_access_time[file_id])
            for file_id in file_ids
            if file_id in self._file_tiers and self._file_tiers[file_id] == Tier.HOT
        ]
        hot_files.sort(key=lambda x: x[1])  # Sort by access time, oldest first
        
        # Demote files from HOT if capacity exceeded
        for file_id, last_access in hot_files:
            if hot_occupancy <= hot_capacity:
                break
            
            file_size = self._file_sizes[file_id]
            
            # Demote to WARM tier
            decision = MigrationDecision(
                file_id=file_id,
                action=MigrationAction.MIGRATE,
                source_tier=Tier.HOT,
                dest_tier=Tier.WARM,
                rationale=(
                    f"HOT tier capacity exceeded (occupancy={hot_occupancy:.2f}GB, "
                    f"capacity={hot_capacity:.2f}GB). "
                    f"File {file_id} not accessed for {(timestamp - last_access).total_seconds() / 3600:.1f}h"
                )
            )
            
            self._execute_migration(file_id, Tier.HOT, Tier.WARM, timestamp)
            hot_occupancy -= file_size
            decisions.append(decision)
        
        # Demote files from WARM to COLD based on recency
        warm_files = [
            (file_id, self._last_access_time[file_id])
            for file_id in file_ids
            if file_id in self._file_tiers and self._file_tiers[file_id] == Tier.WARM
        ]
        warm_files.sort(key=lambda x: x[1])  # Sort by access time, oldest first
        
        warm_capacity = self.tier_capacities[Tier.WARM]
        warm_occupancy = self._tier_occupancy[Tier.WARM]
        
        for file_id, last_access in warm_files:
            if warm_occupancy <= warm_capacity:
                break
            
            # Check if file is outside recency window
            if last_access < recency_threshold:
                file_size = self._file_sizes[file_id]
                
                decision = MigrationDecision(
                    file_id=file_id,
                    action=MigrationAction.MIGRATE,
                    source_tier=Tier.WARM,
                    dest_tier=Tier.COLD,
                    rationale=(
                        f"File {file_id} outside recency window "
                        f"({(timestamp - last_access).total_seconds() / 3600:.1f}h old). "
                        f"WARM tier occupancy={warm_occupancy:.2f}GB, capacity={warm_capacity:.2f}GB"
                    )
                )
                
                self._execute_migration(file_id, Tier.WARM, Tier.COLD, timestamp)
                warm_occupancy -= file_size
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
            Actual destination tier (may differ if capacity exceeded)
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
    
    def get_file_access_time(self, file_id: str) -> Optional[datetime]:
        """
        Get last access time for a file.
        
        Args:
            file_id: File identifier
            
        Returns:
            Last access timestamp or None if file not tracked
        """
        return self._last_access_time.get(file_id)
    
    def reset_file_state(self, file_id: str) -> None:
        """
        Reset access state for a file.
        
        Useful for simulation scenarios or testing.
        
        Args:
            file_id: File identifier
        """
        if file_id in self._access_order:
            del self._access_order[file_id]
        if file_id in self._last_access_time:
            del self._last_access_time[file_id]
