"""Tests for configuration system (loader, validation, seed manager)."""

import pytest
import yaml
import json
from pathlib import Path
from datetime import datetime
import tempfile
import numpy as np
import random

from src.tiering.config.loader import ConfigLoader, load_config_with_defaults, ConfigSnapshot
from src.tiering.config.validation import ConfigValidator, ConfigurationValidator, validate_config_file
from src.tiering.config.seed_manager import SeedManager, DeterministicRNG, set_global_seed


# ============================================================================
# Configuration Loader Tests
# ============================================================================


class TestConfigLoader:
    """Tests for YAML configuration loading."""
    
    def test_load_existing_config(self):
        """Test loading an existing configuration file."""
        loader = ConfigLoader(Path("configs"))
        config = loader.load("default")
        
        assert isinstance(config, dict)
        assert "seed" in config
        assert "dataset" in config
        assert "workload" in config
        assert "simulation" in config
    
    def test_load_nonexistent_config(self):
        """Test that loading nonexistent config raises error."""
        loader = ConfigLoader(Path("configs"))
        
        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent_config_xyz")
    
    def test_config_caching(self):
        """Test that configurations are cached."""
        loader = ConfigLoader(Path("configs"))
        
        config1 = loader.load("default")
        config2 = loader.load("default")
        
        # Should be the same object from cache
        assert config1 is config2
    
    def test_force_reload_config(self):
        """Test forcing config reload despite cache."""
        loader = ConfigLoader(Path("configs"))
        
        config1 = loader.load("default")
        config2 = loader.load("default", force_reload=True)
        
        # Different objects after reload
        assert config1 is not config2
        # But same content
        assert config1 == config2
    
    def test_load_yaml_normalization(self):
        """Test that .yaml extension is optional."""
        loader = ConfigLoader(Path("configs"))
        
        config1 = loader.load("default.yaml")
        config2 = loader.load("default")
        
        assert config1 == config2
    
    def test_cache_management(self):
        """Test cache clearing."""
        loader = ConfigLoader(Path("configs"))
        
        loader.load("default")
        cached = loader.get_cached_configs()
        assert len(cached) > 0
        
        loader.clear_cache()
        cached = loader.get_cached_configs()
        assert len(cached) == 0


class TestConfigSnapshot:
    """Tests for configuration snapshots."""
    
    def test_save_and_load_snapshot(self):
        """Test saving and loading config snapshots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_dir = Path(tmpdir)
            snapshot = ConfigSnapshot(snapshot_dir)
            
            config = {
                "seed": 42,
                "model": {"max_depth": 6}
            }
            metadata = {"git_commit": "abc123", "version": "1.0"}
            
            path = snapshot.save_snapshot(config, "test_exp", metadata)
            
            assert path.exists()
            
            loaded = snapshot.load_snapshot(path.name)
            
            assert loaded["experiment_id"] == "test_exp"
            assert loaded["configuration"] == config
            assert loaded["metadata"] == metadata


# ============================================================================
# Configuration Validation Tests
# ============================================================================


class TestConfigValidator:
    """Tests for configuration validation."""
    
    def test_validate_dataset_config(self):
        """Test validation of dataset configuration."""
        validator = ConfigurationValidator()
        
        valid_config = {
            "num_files": 1000,
            "seed": 42,
            "category_distribution": {
                "Finance": 0.2,
                "Legal": 0.8
            }
        }
        
        assert validator.validate_dataset_config(valid_config)
    
    def test_invalid_dataset_config(self):
        """Test that invalid configs are rejected."""
        validator = ConfigurationValidator()
        
        invalid_config = {
            "num_files": -1,  # Invalid: negative
            "seed": 42
        }
        
        with pytest.raises(Exception):  # jsonschema.ValidationError
            validator.validate_dataset_config(invalid_config)
    
    def test_validate_tier_config(self):
        """Test validation of tier configuration."""
        validator = ConfigurationValidator()
        
        valid_config = {
            "tier": "HOT",
            "latency_ms": 1.0,
            "cost_per_gb_month": 0.02,
            "capacity_gb": 1000.0,
            "throughput_mbps": 1000.0
        }
        
        assert validator.validate_tier_config(valid_config)
    
    def test_validate_migration_config(self):
        """Test validation of migration configuration."""
        validator = ConfigurationValidator()
        
        valid_config = {
            "confidence_threshold": 0.7,
            "hysteresis_count": 3,
            "cooldown_hours": 6.0
        }
        
        assert validator.validate_migration_config(valid_config)


# ============================================================================
# Seed Manager Tests
# ============================================================================


class TestSeedManager:
    """Tests for reproducible seed management."""
    
    def test_seed_setting(self):
        """Test that seed is properly set."""
        manager = SeedManager(base_seed=42)
        manager.set_seed(42)
        
        assert manager.is_deterministic()
        assert manager.base_seed == 42
    
    def test_reproducibility(self):
        """Test that seeding produces reproducible results."""
        # First run
        manager1 = SeedManager(base_seed=123)
        manager1.set_seed(123)
        values1 = [np.random.random() for _ in range(10)]
        
        # Second run with same seed
        manager2 = SeedManager(base_seed=123)
        manager2.set_seed(123)
        values2 = [np.random.random() for _ in range(10)]
        
        assert np.allclose(values1, values2)
    
    def test_different_seeds_different_results(self):
        """Test that different seeds produce different results."""
        manager1 = SeedManager(base_seed=123)
        manager1.set_seed(123)
        values1 = [np.random.random() for _ in range(10)]
        
        manager2 = SeedManager(base_seed=456)
        manager2.set_seed(456)
        values2 = [np.random.random() for _ in range(10)]
        
        assert not np.allclose(values1, values2)
    
    def test_derive_component_seed(self):
        """Test deriving different seeds for components."""
        manager = SeedManager(base_seed=42)
        
        seed1 = manager.derive_seed("component_1")
        seed2 = manager.derive_seed("component_2")
        
        # Different components should get different seeds
        assert seed1 != seed2
        
        # Same component should get same seed (deterministic)
        seed1_again = manager.derive_seed("component_1", offset=0)
        assert seed1 == seed1_again
    
    def test_seed_log(self):
        """Test seed logging."""
        manager = SeedManager(base_seed=42)
        manager.set_seed(42)
        
        manager.set_component_seed("dataset_gen", 100)
        manager.set_component_seed("model_train", 200)
        
        log = manager.get_seed_log()
        
        assert "python_random" in log
        assert "numpy" in log
        assert "dataset_gen" in log
        assert "model_train" in log
    
    def test_seed_snapshot(self):
        """Test saving and loading seed snapshots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot_path = Path(tmpdir) / "seed_log.json"
            
            # Save snapshot
            manager1 = SeedManager(base_seed=42)
            manager1.set_seed(42)
            manager1.save_seed_log(snapshot_path)
            
            assert snapshot_path.exists()
            
            # Load snapshot
            manager2 = SeedManager.from_seed_log(snapshot_path)
            
            assert manager2.base_seed == 42
            assert manager2.is_deterministic()


class TestDeterministicRNG:
    """Tests for deterministic RNG wrapper."""
    
    def test_randint_generation(self):
        """Test integer generation."""
        rng = DeterministicRNG(seed=42)
        
        values1 = [rng.randint(0, 100) for _ in range(10)]
        
        rng2 = DeterministicRNG(seed=42)
        values2 = [rng2.randint(0, 100) for _ in range(10)]
        
        assert values1 == values2
    
    def test_uniform_generation(self):
        """Test uniform distribution generation."""
        rng = DeterministicRNG(seed=42)
        
        values1 = [rng.uniform(0.0, 1.0) for _ in range(10)]
        
        rng2 = DeterministicRNG(seed=42)
        values2 = [rng2.uniform(0.0, 1.0) for _ in range(10)]
        
        assert np.allclose(values1, values2)
    
    def test_normal_distribution(self):
        """Test normal distribution generation."""
        rng = DeterministicRNG(seed=42)
        
        values1 = [rng.normal(0.0, 1.0) for _ in range(10)]
        
        rng2 = DeterministicRNG(seed=42)
        values2 = [rng2.normal(0.0, 1.0) for _ in range(10)]
        
        assert np.allclose(values1, values2)
    
    def test_choice_reproducibility(self):
        """Test choice reproducibility."""
        rng = DeterministicRNG(seed=42)
        choices1 = rng.choice([1, 2, 3, 4, 5], size=10)
        
        rng2 = DeterministicRNG(seed=42)
        choices2 = rng2.choice([1, 2, 3, 4, 5], size=10)
        
        assert np.array_equal(choices1, choices2)
    
    def test_shuffle_reproducibility(self):
        """Test shuffle reproducibility."""
        rng = DeterministicRNG(seed=42)
        arr1 = np.array([1, 2, 3, 4, 5])
        rng.shuffle(arr1)
        
        rng2 = DeterministicRNG(seed=42)
        arr2 = np.array([1, 2, 3, 4, 5])
        rng2.shuffle(arr2)
        
        assert np.array_equal(arr1, arr2)


# ============================================================================
# Integration Tests
# ============================================================================


class TestConfigIntegration:
    """Integration tests for configuration system."""
    
    def test_load_and_validate_default_config(self):
        """Test loading and validating default configuration."""
        loader = ConfigLoader(Path("configs"))
        config = loader.load("default")
        
        validator = ConfigurationValidator()
        
        # Validate different sections
        if "dataset" in config:
            validator.validate_dataset_config(config["dataset"])
        
        if "workload" in config:
            validator.validate_workload_config(config["workload"])
        
        if "simulation" in config:
            validator.validate_simulation_config(config["simulation"])
    
    def test_seed_management_with_config(self):
        """Test using seed from configuration."""
        loader = ConfigLoader(Path("configs"))
        config = loader.load("default")
        
        seed = config.get("seed", 42)
        manager = SeedManager(base_seed=seed)
        manager.set_seed(seed)
        
        assert manager.is_deterministic()
        
        # Generate some random numbers
        val1 = np.random.random()
        val2 = np.random.random()
        
        # Reset with same seed and verify reproducibility
        manager2 = SeedManager(base_seed=seed)
        manager2.set_seed(seed)
        val1_again = np.random.random()
        val2_again = np.random.random()
        
        assert val1 == val1_again
        assert val2 == val2_again


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
