"""
Migration utilities for configuration system.

This module provides functions to migrate between different versions of the
configuration system and handle backward compatibility.
"""
from __future__ import annotations

import json
import os
import logging
from pathlib import Path
from typing import Any, Dict, Union, TypeVar
import yaml

from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

# Type variable for generic model validation
T = TypeVar('T')

# Configuration versions
CURRENT_CONFIG_VERSION = "1.0.0"

def get_config_version(config_data: Dict[str, Any]) -> str:
    """
    Get the version of the configuration data.
    
    Args:
        config_data: Configuration data dictionary
        
    Returns:
        str: Version string or '0.0.0' if not specified
    """
    return config_data.get('version', '0.0.0')

def upgrade_config(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upgrade configuration to the current version.
    
    Args:
        config_data: Configuration data to upgrade
        
    Returns:
        Dict[str, Any]: Upgraded configuration data
        
    Raises:
        ConfigurationError: If the configuration cannot be upgraded
    """
    version = get_config_version(config_data)
    
    if version == CURRENT_CONFIG_VERSION:
        return config_data
    
    # Add upgrade paths here as needed
    if version < '1.0.0':
        config_data = _upgrade_to_v1_0_0(config_data)
    
    # After all upgrades, set to current version
    config_data['version'] = CURRENT_CONFIG_VERSION
    return config_data

def _upgrade_to_v1_0_0(config_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Upgrade configuration to version 1.0.0.
    
    This handles the transition from the old flat structure to the new nested structure.
    """
    logger.info("Upgrading configuration to version 1.0.0")
    
    # Create a new config dictionary with the new structure
    new_config = {'version': '1.0.0'}
    
    # Map old keys to new nested structure
    key_mapping = {
        # LLM
        'llm_provider': 'llm.provider',
        'llm_model': 'llm.model',
        'llm_api_key': 'llm.api_key',
        'llm_base_url': 'llm.base_url',
        'llm_timeout': 'llm.timeout',
        'llm_max_retries': 'llm.max_retries',
        'llm_temperature': 'llm.temperature',
        
        # Cache
        'cache_enabled': 'cache.enabled',
        'cache_dir': 'cache.directory',
        'cache_ttl': 'cache.ttl',
        'cache_max_size': 'cache.max_size',
        
        # Logging
        'log_level': 'logging.level',
        'log_file': 'logging.file',
        'log_format': 'logging.format',
        'log_max_size': 'logging.max_size',
        'log_backup_count': 'logging.backup_count',
        'log_json_format': 'logging.json_format',
        
        # App
        'debug': 'debug',
        'environment': 'environment',
        'max_concurrent_tasks': 'max_concurrent_tasks',
        'request_timeout': 'request_timeout',
        'enable_experimental_features': 'enable_experimental_features',
        'data_dir': 'data_dir',
    }
    
    # Convert flat structure to nested
    for old_key, new_key in key_mapping.items():
        if old_key in config_data:
            # Handle nested keys
            keys = new_key.split('.')
            current = new_config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = config_data[old_key]
    
    # Handle any special cases or transformations
    if 'llm' in new_config and 'api_key' in new_config['llm']:
        new_config['llm']['api_key'] = str(new_config['llm']['api_key'])
    
    return new_config

def migrate_from_dict(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Migrate configuration from a dictionary to the current version.
    
    Args:
        config_dict: Dictionary containing configuration values
        
    Returns:
        Dict[str, Any]: Migrated configuration dictionary
        
    Raises:
        ConfigurationError: If the configuration cannot be migrated
    """
    if not config_dict:
        return {}
    
    try:
        # Make a copy to avoid modifying the input
        config_data = config_dict.copy()
        
        # Upgrade the configuration to the current version
        upgraded_config = upgrade_config(config_data)
        
        logger.info("Configuration migration completed successfully")
        return upgraded_config
        
    except Exception as e:
        error_msg = f"Failed to migrate configuration: {e}"
        logger.error(error_msg, exc_info=True)
        raise ConfigurationError(error_msg) from e

def migrate_from_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Migrate configuration from a file to the current version.
    
    Args:
        config_path: Path to the configuration file (YAML or JSON)
        
    Returns:
        Dict[str, Any]: Migrated configuration dictionary
        
    Raises:
        ConfigurationError: If the file cannot be read or parsed
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    
    try:
        # Read the file based on its extension
        suffix = config_path.suffix.lower()
        
        if suffix in ('.yaml', '.yml'):
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
        elif suffix == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        else:
            raise ConfigurationError(
                f"Unsupported config file format: {suffix}. "
                "Expected .yaml, .yml, or .json"
            )
        
        # Migrate the configuration
        return migrate_from_dict(config_data)
        
    except (yaml.YAMLError, json.JSONDecodeError) as e:
        raise ConfigurationError(f"Failed to parse configuration file: {e}") from e
    except Exception as e:
        raise ConfigurationError(f"Failed to read configuration file: {e}") from e

def migrate_from_env() -> Dict[str, Any]:
    """
    Migrate configuration from environment variables to a dictionary.
    
    This is useful for containerized environments where configuration
    is passed via environment variables.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    config = {}
    
    # LLM Configuration
    if 'MYJOBS_LLM_PROVIDER' in os.environ:
        config['llm'] = config.get('llm', {})
        config['llm']['provider'] = os.environ['MYJOBS_LLM_PROVIDER']
    
    if 'MYJOBS_LLM_MODEL' in os.environ:
        config['llm'] = config.get('llm', {})
        config['llm']['model'] = os.environ['MYJOBS_LLM_MODEL']
    
    if 'MYJOBS_LLM_API_KEY' in os.environ:
        config['llm'] = config.get('llm', {})
        config['llm']['api_key'] = os.environ['MYJOBS_LLM_API_KEY']
    
    # Add other environment variables as needed...
    
    # Set the version
    config['version'] = CURRENT_CONFIG_VERSION
    
    return config
