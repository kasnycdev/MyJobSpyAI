"""Utility functions for configuration management."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

import yaml
from pydantic import BaseModel

logger = logging.getLogger('config')

T = TypeVar('T', bound=BaseModel)


def load_yaml_config(file_path: Union[str, Path], model: Type[T]) -> T:
    """Load configuration from a YAML file into a Pydantic model.

    Args:
        file_path: Path to the YAML configuration file.
        model: Pydantic model class to load the configuration into.

    Returns:
        An instance of the provided Pydantic model populated with the configuration.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
        pydantic.ValidationError: If the configuration does not match the model.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    try:
        with open(file_path, 'r') as f:
            config_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {file_path}: {e}")
        raise

    return model.model_validate(config_data)


def save_yaml_config(file_path: Union[str, Path], config: BaseModel) -> None:
    """Save a Pydantic model to a YAML file.

    Args:
        file_path: Path where the YAML file should be saved.
        config: Pydantic model instance to save.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, 'w') as f:
        yaml.dump(
            config.model_dump(exclude_none=True, exclude_unset=True),
            f,
            default_flow_style=False,
            sort_keys=False,
        )


def load_env_file(env_file: Union[str, Path]) -> Dict[str, str]:
    """Load environment variables from a .env file.

    Args:
        env_file: Path to the .env file.

    Returns:
        Dictionary of environment variables.
    """
    env_file = Path(env_file)
    if not env_file.exists():
        logger.warning(f"Environment file not found: {env_file}")
        return {}

    env_vars = {}
    with open(env_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue

            # Split on first '=' only
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()

            # Remove quotes if present
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]

            env_vars[key] = value

    return env_vars


def update_environment(env_vars: Dict[str, str]) -> None:
    """Update os.environ with the given environment variables.

    Args:
        env_vars: Dictionary of environment variables to set.
    """
    for key, value in env_vars.items():
        os.environ[key] = value


def get_environment_config(prefix: str = '') -> Dict[str, str]:
    """Get environment variables with the given prefix.

    Args:
        prefix: Optional prefix to filter environment variables.

    Returns:
        Dictionary of environment variables with the prefix removed from the keys.
    """
    prefix = prefix.upper()
    if not prefix.endswith('_'):
        prefix += '_'

    config = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            config_key = key[len(prefix) :].lower()
            config[config_key] = value

    return config


def deep_update(target: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a dictionary with another dictionary.

    Args:
        target: Dictionary to update.
        source: Dictionary with updates to apply.

    Returns:
        The updated target dictionary.
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            target[key] = deep_update(target[key], value)
        else:
            target[key] = value
    return target


def load_settings(
    model: Type[T],
    config_file: Optional[Union[str, Path]] = None,
    env_file: Optional[Union[str, Path]] = None,
    env_prefix: str = '',
) -> T:
    """Load settings from multiple sources with precedence:
    1. Environment variables (with prefix)
    2. .env file
    3. YAML config file
    4. Default values from the model

    Args:
        config_file: Path to the YAML configuration file.
        env_file: Path to the .env file.
        model: Pydantic model class for the settings.
        env_prefix: Prefix for environment variables.

    Returns:
        An instance of the provided Pydantic model populated with the configuration.
    """
    # Start with default values from the model
    settings = model()

    # Load from YAML config file if provided
    if config_file is not None and Path(config_file).exists():
        try:
            file_settings = load_yaml_config(config_file, model)
            settings = file_settings
        except Exception as e:
            logger.warning(f"Error loading config file {config_file}: {e}")

    # Load from .env file if provided
    env_vars = {}
    if env_file is not None and Path(env_file).exists():
        try:
            env_vars = load_env_file(env_file)
            update_environment(env_vars)  # Update os.environ for Pydantic
        except Exception as e:
            logger.warning(f"Error loading .env file {env_file}: {e}")

    # Apply environment variable overrides with the specified prefix
    env_config = get_environment_config(env_prefix)
    if env_config:
        # Convert environment variables to nested dict structure
        nested_config = {}
        for key, value in env_config.items():
            keys = key.lower().split('__')
            current = nested_config
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value

        # Update settings with environment variables
        settings = model.model_validate(
            deep_update(settings.model_dump(), nested_config)
        )

    return settings
