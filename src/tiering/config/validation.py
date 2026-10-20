"""
Configuration validation system using JSON schemas.

This module provides utilities to validate configurations against JSON schemas
to ensure type safety and completeness.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import jsonschema
from jsonschema import Draft7Validator, ValidationError
import logging

logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates configurations against JSON schemas."""
    
    def __init__(self, schema_dir: Optional[Path] = None):
        """
        Initialize validator.
        
        Args:
            schema_dir: Path to directory containing JSON schema files.
                       Defaults to ./configs/schemas
        """
        self.schema_dir = schema_dir or Path("configs/schemas")
        self.schema_dir.mkdir(parents=True, exist_ok=True)
        self._schema_cache: Dict[str, Dict[str, Any]] = {}
    
    def load_schema(self, schema_name: str, force_reload: bool = False) -> Dict[str, Any]:
        """
        Load a JSON schema file.
        
        Args:
            schema_name: Name of schema file (with or without .json extension)
            force_reload: Force reload even if cached
            
        Returns:
            Parsed schema dictionary
            
        Raises:
            FileNotFoundError: If schema file not found
            json.JSONDecodeError: If JSON parsing fails
        """
        # Normalize filename
        if not schema_name.endswith('.json'):
            schema_name = schema_name + '.json'
        
        # Check cache
        if schema_name in self._schema_cache and not force_reload:
            logger.debug(f"Loaded schema from cache: {schema_name}")
            return self._schema_cache[schema_name]
        
        # Load from disk
        schema_path = self.schema_dir / schema_name
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        
        try:
            with open(schema_path, 'r') as f:
                schema = json.load(f)
            
            # Cache the schema
            self._schema_cache[schema_name] = schema
            logger.info(f"Loaded schema from {schema_path}")
            
            return schema
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON schema {schema_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading schema {schema_path}: {e}")
            raise
    
    def validate(
        self,
        config: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> bool:
        """
        Validate a configuration against a schema.
        
        Args:
            config: Configuration dictionary to validate
            schema: JSON schema dictionary
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        validator = Draft7Validator(schema)
        
        # Collect all errors
        errors = list(validator.iter_errors(config))
        
        if errors:
            error_messages = [str(e.message) for e in errors]
            error_text = "\n".join(error_messages)
            logger.error(f"Configuration validation failed:\n{error_text}")
            
            # Raise the first error
            raise errors[0]
        
        logger.info("Configuration validation passed")
        return True
    
    def validate_by_name(
        self,
        config: Dict[str, Any],
        schema_name: str
    ) -> bool:
        """
        Validate a configuration by loading schema by name.
        
        Args:
            config: Configuration to validate
            schema_name: Name of schema file to use
            
        Returns:
            True if valid
            
        Raises:
            ValidationError: If validation fails
        """
        schema = self.load_schema(schema_name)
        return self.validate(config, schema)
    
    def get_validation_errors(
        self,
        config: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> List[str]:
        """
        Get all validation errors without raising.
        
        Args:
            config: Configuration to validate
            schema: JSON schema
            
        Returns:
            List of error messages
        """
        validator = Draft7Validator(schema)
        errors = list(validator.iter_errors(config))
        return [e.message for e in errors]
    
    def clear_cache(self):
        """Clear schema cache."""
        self._schema_cache.clear()
        logger.debug("Cleared schema cache")


def create_schema(
    properties: Dict[str, Any],
    required: Optional[List[str]] = None,
    description: str = "",
    additional_properties: bool = True
) -> Dict[str, Any]:
    """
    Create a JSON schema dictionary programmatically.
    
    Args:
        properties: Dictionary of property definitions
        required: List of required properties
        description: Schema description
        additional_properties: Whether to allow additional properties
        
    Returns:
        JSON schema dictionary
    """
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": properties,
        "additionalProperties": additional_properties
    }
    
    if description:
        schema["description"] = description
    
    if required:
        schema["required"] = required
    
    return schema


def save_schema(schema: Dict[str, Any], schema_file: str, schema_dir: Optional[Path] = None):
    """
    Save a JSON schema to file.
    
    Args:
        schema: Schema dictionary
        schema_file: Output filename (with or without .json extension)
        schema_dir: Directory to save schema. Defaults to ./configs/schemas
    """
    schema_dir = schema_dir or Path("configs/schemas")
    schema_dir.mkdir(parents=True, exist_ok=True)
    
    # Normalize filename
    if not schema_file.endswith('.json'):
        schema_file = schema_file + '.json'
    
    schema_path = schema_dir / schema_file
    
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2)
    
    logger.info(f"Saved schema to {schema_path}")


# ============================================================================
# Predefined Schemas
# ============================================================================


def get_dataset_config_schema() -> Dict[str, Any]:
    """Get JSON schema for dataset configuration."""
    return create_schema(
        properties={
            "num_files": {
                "type": "integer",
                "minimum": 1,
                "description": "Number of files to generate"
            },
            "category_distribution": {
                "type": "object",
                "description": "Distribution over file categories",
                "additionalProperties": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1
                }
            },
            "size_distribution": {
                "type": "object",
                "properties": {
                    "mean_bytes": {
                        "type": "integer",
                        "minimum": 0
                    },
                    "std_bytes": {
                        "type": "integer",
                        "minimum": 0
                    }
                }
            },
            "seed": {
                "type": "integer",
                "description": "Random seed for reproducibility"
            }
        },
        required=["num_files"],
        description="Configuration schema for dataset generation"
    )


def get_workload_config_schema() -> Dict[str, Any]:
    """Get JSON schema for workload configuration."""
    return create_schema(
        properties={
            "duration_hours": {
                "type": "number",
                "minimum": 0.1,
                "description": "Simulation duration in hours"
            },
            "pattern_distribution": {
                "type": "object",
                "properties": {
                    "hot": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1
                    },
                    "warm": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1
                    },
                    "cold": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1
                    },
                    "seasonal": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1
                    }
                }
            },
            "seed": {
                "type": "integer",
                "description": "Random seed for reproducibility"
            }
        },
        required=["duration_hours"],
        description="Configuration schema for workload generation"
    )


def get_simulation_config_schema() -> Dict[str, Any]:
    """Get JSON schema for simulation configuration."""
    return create_schema(
        properties={
            "duration_hours": {
                "type": "number",
                "minimum": 0.1,
                "description": "Simulation duration in hours"
            },
            "decision_interval_hours": {
                "type": "number",
                "minimum": 0.01,
                "description": "Interval between tiering decisions in hours"
            },
            "seed": {
                "type": "integer",
                "description": "Random seed for reproducibility"
            }
        },
        required=["duration_hours", "decision_interval_hours"],
        description="Configuration schema for simulation"
    )


def get_tier_config_schema() -> Dict[str, Any]:
    """Get JSON schema for tier configuration."""
    return create_schema(
        properties={
            "tier": {
                "type": "string",
                "enum": ["HOT", "WARM", "COLD"],
                "description": "Tier identifier"
            },
            "latency_ms": {
                "type": "number",
                "minimum": 0,
                "description": "Average access latency in milliseconds"
            },
            "cost_per_gb_month": {
                "type": "number",
                "minimum": 0,
                "description": "Storage cost per GB per month"
            },
            "capacity_gb": {
                "type": "number",
                "minimum": 0.1,
                "description": "Maximum capacity in GB"
            },
            "throughput_mbps": {
                "type": "number",
                "minimum": 0,
                "description": "Throughput in MB/s"
            }
        },
        required=["tier", "latency_ms", "cost_per_gb_month", "capacity_gb", "throughput_mbps"],
        description="Configuration schema for storage tier"
    )


def get_migration_config_schema() -> Dict[str, Any]:
    """Get JSON schema for migration controller configuration."""
    return create_schema(
        properties={
            "confidence_threshold": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Minimum confidence for migration execution"
            },
            "hysteresis_count": {
                "type": "integer",
                "minimum": 1,
                "description": "Number of decision points to sustain prediction before migrating"
            },
            "cooldown_hours": {
                "type": "number",
                "minimum": 0,
                "description": "Mandatory waiting period between migrations in hours"
            }
        },
        required=["confidence_threshold", "hysteresis_count", "cooldown_hours"],
        description="Configuration schema for migration controller"
    )


# ============================================================================
# Configuration Validation Utilities
# ============================================================================


class ConfigurationValidator:
    """High-level configuration validation with predefined schemas."""
    
    def __init__(self):
        """Initialize with predefined schemas."""
        self.validator = ConfigValidator()
        self._predefined_schemas = {
            'dataset': get_dataset_config_schema(),
            'workload': get_workload_config_schema(),
            'simulation': get_simulation_config_schema(),
            'tier': get_tier_config_schema(),
            'migration': get_migration_config_schema()
        }
    
    def validate_dataset_config(self, config: Dict[str, Any]) -> bool:
        """Validate dataset configuration."""
        return self.validator.validate(config, self._predefined_schemas['dataset'])
    
    def validate_workload_config(self, config: Dict[str, Any]) -> bool:
        """Validate workload configuration."""
        return self.validator.validate(config, self._predefined_schemas['workload'])
    
    def validate_simulation_config(self, config: Dict[str, Any]) -> bool:
        """Validate simulation configuration."""
        return self.validator.validate(config, self._predefined_schemas['simulation'])
    
    def validate_tier_config(self, config: Dict[str, Any]) -> bool:
        """Validate tier configuration."""
        return self.validator.validate(config, self._predefined_schemas['tier'])
    
    def validate_migration_config(self, config: Dict[str, Any]) -> bool:
        """Validate migration controller configuration."""
        return self.validator.validate(config, self._predefined_schemas['migration'])


def validate_config_file(
    config_file: str,
    schema_name: str,
    config_dir: Optional[Path] = None,
    schema_dir: Optional[Path] = None
) -> bool:
    """
    Validate a configuration file against a schema.
    
    Args:
        config_file: Name of config file to validate
        schema_name: Name of schema to validate against
        config_dir: Configuration directory
        schema_dir: Schema directory
        
    Returns:
        True if valid
        
    Raises:
        ValidationError: If validation fails
    """
    from .loader import ConfigLoader
    
    config_loader = ConfigLoader(config_dir)
    config = config_loader.load(config_file)
    
    validator = ConfigValidator(schema_dir)
    return validator.validate_by_name(config, schema_name)
