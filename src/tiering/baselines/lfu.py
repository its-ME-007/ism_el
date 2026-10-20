"""LFU (Least Frequently Used) baseline tiering policy."""

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple
import heapq


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


class LFUBaseline:
    """
    LFU (Least Frequently Used) tiering policy.
    
    Tracks access frequency counts for each file and promotes frequently
    accessed files to HOT tier while demoting rarely accessed files to
    COLD tier. Implements configurable frequency counting windows with
    sliding window for frequency decay.
    """
    
    def __init__(
        self,
        tier_capacities: Dict[Tier, float],
        frequency_window_hours: float = 24.0,
        hot_threshold: int = 10,
        cold_threshold: int = 2,
        window_decay_enabled: bool = True
    ):
        """
        Initialize LFU baseline with configuration.
        
        Args:
            tier_capacities: Dictionary mapping Tier to capacity in GB
            frequency_window_hours: Time window for frequency counting (hours)
            hot_threshold: Access count threshold for HOT tier
            cold_threshold: Access count threshold for COLD tier
            window_decay_enabled: Whether to apply frequency decay outside window
        """
        self.tier_capacities = tier_capacities
        self.frequency_window_hours = frequency_window_hours
        self.hot_threshold = hot_threshold
        self.cold_threshold = cold_threshold
        self.window_decay_enabled = window_decay_enabled
        
        # Track state for each file
        self._file_tiers: Dict[str, Tier] = {}  # Current tier for each file
        self._file_sizes: Dict[str, float] = {}  # File size in GB
        self._access_frequencies: Dict[str, Counter] = defaultdict(
            lambda: Counter()
        )  # Timestamped access counts
        self._last_access_time: Dict[str, datetime] = {}  # Last access timestamp
        
        # Track tier occupancy
        self._tier_occupancy: Dict[Tier, float] = {
            Tier.HOT: 0.0,
            Tier.WARM: 0.0,
            Tier.COLD: 0.0
        }
    
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
        Update frequency count and potentially promote on access.
        
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
        
        # Update access frequency
        self._access_frequencies[file_id][timestamp] = (
            self._access_frequencies[file_id].get(timestamp, 0) + 1
        )
        self._last_access_time[file_id] = timestamp
        
        # Check if promotion is needed
        new_tier = self._determine_tier(file_id, timestamp)
        current_tier = self._file_tiers[file_id]
        
        if new_tier != current_tier:
            return self._execute_migration(file_id, current_tier, new_tier)
        
        return None
    
    def on_decision_interval(
        self,
        timestamp: datetime,
        file_ids: Optional[List[str]] = None
    ) -> List[MigrationDecision]:
        """
        Process decision interval - evaluate all files for demotion.
        
        This method is called periodically (e.g., hourly) to reassess
        tier assignments based on recent frequency data and apply
        frequency decay for files with old access patterns.
        
        Args:
            timestamp: Current simulation timestamp
            file_ids: List of file IDs to evaluate (None = all files)
            
        Returns:
            List of migration decisions for files requiring demotion
        """
        if file_ids is None:
            file_ids = list(self._file_tiers.keys())
        
        decisions = []
        
        for file_id in file_ids:
            if file_id not in self._file_tiers:
                continue
            
            current_tier = self._file_tiers[file_id]
            new_tier = self._determine_tier(file_id, timestamp)
            
            if new_tier != current_tier:
                decision = MigrationDecision(
                    file_id=file_id,
                    action=MigrationAction.MIGRATE,
                    source_tier=current_tier,
                    dest_tier=new_tier,
                    rationale=(
                        f"Frequency trend changed from {current_tier} to {new_tier} "
                        f"(frequency={self._get_current_frequency(file_id, timestamp)})"
                    )
                )
                self._execute_migration(file_id, current_tier, new_tier)
                decisions.append(decision)
            
            # Apply frequency decay to old entries
            if self.window_decay_enabled:
                self._apply_frequency_decay(file_id, timestamp)
        
        return decisions
    
    def _determine_tier(self, file_id: str, reference_time: datetime) -> Tier:
        """
        Determine appropriate tier for a file based on frequency.
        
        Uses a sliding window approach: counts accesses within the
        frequency_window_hours and applies decay to older accesses.
        
        Args:
            file_id: File identifier
            reference_time: Reference timestamp for window calculation
            
        Returns:
            Recommended Tier
        """
        frequency = self._get_current_frequency(file_id, reference_time)
        
        # Tier assignment based on frequency thresholds
        if frequency >= self.hot_threshold:
            return Tier.HOT
        elif frequency >= self.cold_threshold:
            return Tier.WARM
        else:
            return Tier.COLD
    
    def _get_current_frequency(
        self,
        file_id: str,
        reference_time: datetime
    ) -> int:
        """
        Get access frequency within the current window.
        
        Counts accesses in the sliding window ending at reference_time.
        Applies exponential decay to older accesses if window_decay is enabled.
        
        Args:
            file_id: File identifier
            reference_time: Reference time for window calculation
            
        Returns:
            Current frequency score
        """
        window_start = reference_time - timedelta(hours=self.frequency_window_hours)
        
        # Get access frequency counter for this file
        freq_counter = self._access_frequencies.get(file_id, Counter())
        
        frequency = 0.0
        
        for timestamp, count in freq_counter.items():
            if timestamp > window_start:
                # Within window: full weight
                frequency += count
            elif self.window_decay_enabled:
                # Outside window and decay enabled: apply exponential decay
                time_diff = (reference_time - timestamp).total_seconds()
                window_seconds = self.frequency_window_hours * 3600
                
                # Decay factor: e^(-time_diff / window)
                # This gives weight ~0.37 at 1 window age, ~0.14 at 2 windows, etc.
                decay_weight = 2.718281828 ** (-time_diff / window_seconds)
                frequency += count * decay_weight
            else:
                # Outside window and decay disabled: count all accesses
                frequency += count
        
        return int(frequency)
    
    def _apply_frequency_decay(
        self,
        file_id: str,
        reference_time: datetime
    ) -> None:
        """
        Apply frequency decay to old entries.
        
        Removes very old entries from the frequency counter to avoid
        unbounded growth of the counter dictionary.
        
        Only applies decay if window_decay_enabled is True.
        
        Args:
            file_id: File identifier
            reference_time: Current reference time
        """
        if not self.window_decay_enabled:
            return
        
        freq_counter = self._access_frequencies.get(file_id)
        if not freq_counter:
            return
        
        # Remove entries older than 3x the window
        decay_threshold = reference_time - timedelta(
            hours=3 * self.frequency_window_hours
        )
        
        old_timestamps = [
            ts for ts in list(freq_counter.keys())
            if ts < decay_threshold
        ]
        
        for ts in old_timestamps:
            del freq_counter[ts]
    
    def _execute_migration(
        self,
        file_id: str,
        source_tier: Tier,
        dest_tier: Tier
    ) -> Tier:
        """
        Execute file migration between tiers.
        
        Updates tier assignment and occupancy tracking. May trigger
        cascading demotions if destination tier becomes over capacity.
        
        Args:
            file_id: File to migrate
            source_tier: Source tier
            dest_tier: Destination tier
            
        Returns:
            Actual destination tier (may differ if capacity exceeded)
        """
        file_size = self._file_sizes[file_id]
        
        # Remove the file from its actual source exactly once. If the target
        # tier is full, the incoming file cascades to a colder tier; it was
        # never resident in the full tier and must not be subtracted from it.
        self._tier_occupancy[source_tier] -= file_size

        tier_order = [Tier.HOT, Tier.WARM, Tier.COLD]
        destination_index = tier_order.index(dest_tier)
        for candidate_tier in tier_order[destination_index:]:
            available_capacity = (
                self.tier_capacities[candidate_tier]
                - self._tier_occupancy[candidate_tier]
            )
            if candidate_tier == Tier.COLD or file_size <= available_capacity:
                self._tier_occupancy[candidate_tier] += file_size
                self._file_tiers[file_id] = candidate_tier
                return candidate_tier

        # COLD is treated as unbounded above, so execution cannot reach here.
        raise RuntimeError(f"Unable to place file {file_id} in a storage tier")
    
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
    
    def reset_file_state(self, file_id: str) -> None:
        """
        Reset frequency state for a file.
        
        Useful for simulation scenarios or testing.
        
        Args:
            file_id: File identifier
        """
        if file_id in self._access_frequencies:
            self._access_frequencies[file_id].clear()
