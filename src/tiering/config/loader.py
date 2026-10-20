"""
Configuration loading system for YAML-based configuration management.

This module provides utilities to load, validate, and manage YAML configurations
for the Context-Aware Predictive Storage Tiering system.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and caches YAML configuration files."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Path to configuration directory. Defaults to ./configs
        """
        self.config_dir = config_dir or Path("configs")
        if not self.config_dir.exists():
            self.config_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created config directory at {self.config_dir}")
        
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    def load(self, config_file: str, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load a YAML configuration file.
        
        Args:
            config_file: Name of config file (with or without .yaml extension)
            force_reload: Force reload even if cached
            
        Returns:
            Parsed configuration dictionary
            
        Raises:
            FileNotFoundError: If config file not found
            yaml.YAMLError: If YAML parsing fails
        """
        # Normalize filename
        if not config_file.endswith('.yaml') and not config_file.endswith('.yml'):
            config_file = config_file + '.yaml'
        
        # Check cache
        if config_file in self._cache and not force_reload:
            logger.debug(f"Loaded config from cache: {config_file}")
            return self._cache[config_file]
        
        # Load from disk
        config_path = self.config_dir / config_file
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                config = {}
            
            # Cache the configuration
            self._cache[config_file] = config
            logger.info(f"Loaded configuration from {config_path}")
            
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {config_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration {config_path}: {e}")
            raise
    
    def load_all(self, pattern: str = "*.yaml") -> Dict[str, Dict[str, Any]]:
        """
        Load all configuration files matching a pattern.
        
        Args:
            pattern: Glob pattern for config files (default: "*.yaml")
            
        Returns:
            Dictionary mapping config names to their contents
        """
        configs = {}
        for config_path in self.config_dir.glob(pattern):
            try:
                config_name = config_path.stem
                configs[config_name] = self.load(config_path.name)
            except Exception as e:
                logger.warning(f"Failed to load config {config_path.name}: {e}")
        
        return configs
    
    def save(self, config: Dict[str, Any], config_file: str) -> Path:
        """
        Save configuration to a YAML file.
        
        Args:
            config: Configuration dictionary
            config_file: Name of output file (with or without .yaml extension)
            
        Returns:
            Path to saved file
        """
        # Normalize filename
        if not config_file.endswith('.yaml') and not config_file.endswith('.yml'):
            config_file = config_file + '.yaml'
        
        config_path = self.config_dir / config_file
        
        with open(config_path, 'w') as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved configuration to {config_path}")
        return config_path
    
    def clear_cache(self):
        """Clear all cached configurations."""
        self._cache.clear()
        logger.debug("Cleared configuration cache")
    
    def get_cached_configs(self) -> Dict[str, str]:
        """Get list of currently cached configurations."""
        return list(self._cache.keys())


def load_config_with_defaults(
    config_file: str,
    defaults: Dict[str, Any],
    config_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Load a configuration file with defaults for missing keys.
    
    Args:
        config_file: Name of config file to load
        defaults: Dictionary of default values
        config_dir: Path to config directory
        
    Returns:
        Merged configuration with defaults applied
    """
    loader = ConfigLoader(config_dir)
    config = loader.load(config_file)
    
    # Merge with defaults (config values take precedence)
    merged = {**defaults, **config}
    
    return merged


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Later dictionaries take precedence over earlier ones.
    
    Args:
        *configs: Variable number of configuration dictionaries
        
    Returns:
        Merged configuration
    """
    result = {}
    for config in configs:
        result.update(config)
    return result


def load_nested_config(
    config: Dict[str, Any],
    keys: list,
    default: Any = None
) -> Any:
    """
    Load a nested value from a configuration dictionary.
    
    Args:
        config: Configuration dictionary
        keys: List of keys representing the path (e.g., ['model', 'xgboost', 'max_depth'])
        default: Default value if key path not found
        
    Returns:
        Value at the specified path, or default if not found
        
    Example:
        >>> config = {'model': {'xgboost': {'max_depth': 6}}}
        >>> load_nested_config(config, ['model', 'xgboost', 'max_depth'])
        6
    """
    current = config
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current


class ConfigSnapshot:
    """Creates and manages snapshots of configurations for reproducibility."""
    
    def __init__(self, snapshot_dir: Optional[Path] = None):
        """
        Initialize snapshot manager.
        
        Args:
            snapshot_dir: Directory to store snapshots. Defaults to results/logs
        """
        self.snapshot_dir = snapshot_dir or Path("results/logs/configs")
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
    
    def save_snapshot(
        self,
        config: Dict[str, Any],
        experiment_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save a configuration snapshot with timestamp and experiment ID.
        
        Args:
            config: Configuration to snapshot
            experiment_id: Unique experiment identifier
            metadata: Optional metadata to include (e.g., git commit, model version)
            
        Returns:
            Path to saved snapshot
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        snapshot_name = f"{experiment_id}_{timestamp}.yaml"
        
        # Prepare snapshot data
        snapshot_data = {
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {},
            'configuration': config
        }
        
        snapshot_path = self.snapshot_dir / snapshot_name
        with open(snapshot_path, 'w') as f:
            yaml.safe_dump(snapshot_data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved configuration snapshot to {snapshot_path}")
        return snapshot_path
    
    def load_snapshot(self, snapshot_file: str) -> Dict[str, Any]:
        """
        Load a configuration snapshot.
        
        Args:
            snapshot_file: Name of snapshot file
            
        Returns:
            Snapshot data including original configuration
        """
        snapshot_path = self.snapshot_dir / snapshot_file
        
        with open(snapshot_path, 'r') as f:
            snapshot_data = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration snapshot from {snapshot_path}")
        return snapshot_data


def expand_paths(config: Dict[str, Any], base_dir: Path = Path(".")) -> Dict[str, Any]:
    """
    Expand relative paths in configuration to absolute paths.
    
    Processes any config values ending with '_path' or '_dir' to convert
    relative paths to absolute paths based on base_dir.
    
    Args:
        config: Configuration dictionary
        base_dir: Base directory for resolving relative paths
        
    Returns:
        Configuration with expanded paths
    """
    expanded = {}
    
    for key, value in config.items():
        if isinstance(value, str) and (key.endswith('_path') or key.endswith('_dir')):
            # Convert relative path to absolute
            path = Path(value)
            if not path.is_absolute():
                path = (base_dir / path).resolve()
            expanded[key] = str(path)
        elif isinstance(value, dict):
            # Recursively expand nested dicts
            expanded[key] = expand_paths(value, base_dir)
        else:
            expanded[key] = value
    
    return expanded
