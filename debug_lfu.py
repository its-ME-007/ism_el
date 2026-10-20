from src.tiering.baselines.lfu import LFUBaseline, Tier
from datetime import datetime, timedelta

capacities = {Tier.HOT: 10.0, Tier.WARM: 50.0, Tier.COLD: 500.0}
baseline = LFUBaseline(
    tier_capacities=capacities,
    frequency_window_hours=24.0,
    hot_threshold=2,
    cold_threshold=1
)
base_time = datetime(2024, 1, 1, 12, 0)

# First access
print("=== FIRST ACCESS ===")
baseline.on_access("file1", base_time, file_size=1.0)
tier = baseline.get_tier("file1")
occupancy = baseline.get_tier_occupancy()
freq = baseline._get_current_frequency("file1", base_time)
print(f"Tier: {tier}, Occupancy: {occupancy}, Frequency: {freq}")

# Second access  
print("\n=== SECOND ACCESS ===")
baseline.on_access("file1", base_time + timedelta(hours=1), file_size=1.0)
tier = baseline.get_tier("file1")
occupancy = baseline.get_tier_occupancy()
freq = baseline._get_current_frequency("file1", base_time + timedelta(hours=1))
print(f"Tier: {tier}, Occupancy: {occupancy}, Frequency: {freq}")

# Check internal frequency counter
print(f"\nInternal frequency counter: {baseline._access_frequencies.get('file1', {})}")
