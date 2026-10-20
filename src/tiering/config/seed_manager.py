"""
Seed management system for reproducible random number generation.

This module provides utilities to manage and apply random seeds consistently
across all stochastic components in the system (numpy, random, torch, etc.).
"""

import random
import numpy as np
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class SeedManager:
    """
    Manages random seeds for reproducibility across all components.
    
    Ensures deterministic behavior by seeding:
    - Python's random module
    - NumPy's random number generator
    - PyTorch (if available)
    - Other random sources
    """
    
    def __init__(self, base_seed: int = 42):
        """
        Initialize seed manager.
        
        Args:
            base_seed: Base seed for reproducibility
        """
        self.base_seed = base_seed
        self._is_set = False
        self._seed_log: Dict[str, int] = {}
    
    def set_seed(self, seed: Optional[int] = None) -> None:
        """
        Set random seed globally.
        
        Seeds all random number generators in:
        - Python's random module
        - NumPy
        - PyTorch (if available)
        - PyTorch CUDA operations (if available)
        
        Args:
            seed: Seed to use. Defaults to base_seed if None
        """
        if seed is None:
            seed = self.base_seed
        
        # Set Python random seed
        random.seed(seed)
        self._seed_log['python_random'] = seed
        
        # Set NumPy seed
        np.random.seed(seed)
        self._seed_log['numpy'] = seed
        
        # Try to set PyTorch seed if available
        try:
            import torch
            torch.manual_seed(seed)
            self._seed_log['torch'] = seed
            
            # Set CUDA seed if GPU available
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                self._seed_log['torch_cuda'] = seed
                logger.info("Set PyTorch CUDA seed")
            
            logger.debug("Set PyTorch seed")
        except ImportError:
            logger.debug("PyTorch not available, skipping torch seeding")
        except Exception as e:
            logger.warning(f"Error setting torch seed: {e}")
        
        # Try to set TensorFlow seed if available
        try:
            import tensorflow as tf
            tf.random.set_seed(seed)
            self._seed_log['tensorflow'] = seed
            logger.debug("Set TensorFlow seed")
        except ImportError:
            logger.debug("TensorFlow not available, skipping tensorflow seeding")
        except Exception as e:
            logger.warning(f"Error setting tensorflow seed: {e}")
        
        self._is_set = True
        logger.info(f"Set global random seed to {seed}")
    
    def set_component_seed(self, component_name: str, seed: int) -> None:
        """
        Set seed for a specific component.
        
        Useful for deriving different seeds for different components
        from a base seed.
        
        Args:
            component_name: Name of component (e.g., 'dataset_generator')
            seed: Seed for this component
        """
        # This method is mainly for logging/tracking
        # Components should call set_seed() with their own seed
        self._seed_log[component_name] = seed
        logger.info(f"Assigned seed {seed} to component: {component_name}")
    
    def derive_seed(self, component_name: str, offset: int = 0) -> int:
        """
        Derive a seed for a specific component from the base seed.
        
        Ensures different components get different but reproducible seeds.
        
        Args:
            component_name: Name of component
            offset: Additional offset to apply
            
        Returns:
            Derived seed for this component
        """
        # Use a simple hash-based approach to derive seeds
        component_hash = hash(component_name) % 100000
        derived_seed = (self.base_seed + component_hash + offset) % (2**31)
        
        self.set_component_seed(component_name, derived_seed)
        logger.debug(f"Derived seed {derived_seed} for {component_name}")
        
        return derived_seed
    
    def get_seed_log(self) -> Dict[str, int]:
        """
        Get log of all seeds set.
        
        Returns:
            Dictionary mapping component names to their seeds
        """
        return self._seed_log.copy()
    
    def is_deterministic(self) -> bool:
        """
        Check if deterministic mode is enabled.
        
        Returns:
            True if seeds have been set
        """
        return self._is_set
    
    def save_seed_log(self, output_file: Path) -> None:
        """
        Save seed log to JSON file for reproducibility tracking.
        
        Args:
            output_file: Path to save seed log
        """
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        log_data = {
            'base_seed': self.base_seed,
            'is_deterministic': self._is_set,
            'component_seeds': self._seed_log
        }
        
        with open(output_file, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"Saved seed log to {output_file}")
    
    @classmethod
    def from_seed_log(cls, seed_log_file: Path) -> 'SeedManager':
        """
        Create a SeedManager from a saved seed log for exact reproduction.
        
        Args:
            seed_log_file: Path to seed log JSON file
            
        Returns:
            New SeedManager with settings from the log
        """
        with open(seed_log_file, 'r') as f:
            log_data = json.load(f)
        
        manager = cls(base_seed=log_data['base_seed'])
        
        if log_data['is_deterministic']:
            manager.set_seed(log_data['base_seed'])
        
        logger.info(f"Loaded SeedManager from {seed_log_file}")
        return manager


def set_global_seed(seed: int) -> None:
    """
    Convenience function to set global seed.
    
    Args:
        seed: Seed to set globally
    """
    manager = SeedManager(seed)
    manager.set_seed(seed)


def enable_deterministic_mode() -> SeedManager:
    """
    Enable fully deterministic mode with default seed.
    
    Returns:
        SeedManager instance
    """
    manager = SeedManager(base_seed=42)
    manager.set_seed()
    logger.info("Enabled deterministic mode with default seed 42")
    return manager


def enable_deterministic_mode_with_seed(seed: int) -> SeedManager:
    """
    Enable fully deterministic mode with specific seed.
    
    Args:
        seed: Seed to use
        
    Returns:
        SeedManager instance
    """
    manager = SeedManager(base_seed=seed)
    manager.set_seed(seed)
    logger.info(f"Enabled deterministic mode with seed {seed}")
    return manager


# ============================================================================
# Random Number Generator Wrappers
# ============================================================================


class DeterministicRNG:
    """
    Wrapper for deterministic random number generation.
    
    Provides a consistent interface for generating random numbers
    in a reproducible manner.
    """
    
    def __init__(self, seed: int):
        """
        Initialize with seed.
        
        Args:
            seed: Random seed
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
    
    def randint(self, low: int, high: int, size: Optional[int] = None) -> Any:
        """Generate random integers."""
        if size is None:
            return self.rng.randint(low, high)
        return self.rng.randint(low, high, size=size)
    
    def uniform(self, low: float = 0.0, high: float = 1.0, size: Optional[int] = None) -> Any:
        """Generate random floats from uniform distribution."""
        if size is None:
            return self.rng.uniform(low, high)
        return self.rng.uniform(low, high, size=size)
    
    def normal(self, loc: float = 0.0, scale: float = 1.0, size: Optional[int] = None) -> Any:
        """Generate random floats from normal distribution."""
        if size is None:
            return self.rng.normal(loc, scale)
        return self.rng.normal(loc, scale, size=size)
    
    def choice(self, a, size: Optional[int] = None, replace: bool = True) -> Any:
        """Choose random elements from array."""
        if size is None:
            return self.rng.choice(a, replace=replace)
        return self.rng.choice(a, size=size, replace=replace)
    
    def shuffle(self, x):
        """Shuffle array in-place."""
        self.rng.shuffle(x)
    
    def reset(self):
        """Reset RNG to initial seed."""
        self.rng = np.random.RandomState(self.seed)


# ============================================================================
# Configuration Integration
# ============================================================================


def apply_seeds_from_config(config: Dict[str, Any]) -> SeedManager:
    """
    Apply seeds from a configuration dictionary.
    
    Looks for 'seed' or 'seeds' keys in configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SeedManager instance
    """
    seed = config.get('seed', config.get('seeds', {}).get('base', 42))
    
    manager = SeedManager(base_seed=seed)
    manager.set_seed(seed)
    
    # Apply component-specific seeds if provided
    component_seeds = config.get('seeds', {})
    for component, seed_value in component_seeds.items():
        if component != 'base':
            manager.set_component_seed(component, seed_value)
    
    logger.info(f"Applied seeds from configuration with base seed {seed}")
    return manager
