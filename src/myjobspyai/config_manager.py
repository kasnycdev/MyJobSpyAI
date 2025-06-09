"""
Configuration manager for MyJobSpy AI.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field, model_validator

from .config import AppConfig, LLMProviderConfig, DatabaseConfig, APIConfig, AnalysisConfig
from .utils.env import get_env_config

logger = logging.getLogger(__name__)


class ConfigManager:
    """Central configuration manager."""

    def __init__(self):
        """Initialize the configuration manager."""
        self._config = None
        self._env_config = None
        self._config_path = None

    async def load(
        self,
        config_path: Optional[Union[str, Path]] = None,
        env_override: bool = True,
    ) -> None:
        """Load configuration from file and environment.

        Args:
            config_path: Path to the configuration file.
            env_override: Whether to override config with environment variables.
        """
        # Load environment config first
        if env_override:
            self._env_config = get_env_config()

        # Load config file
        if config_path:
            config_path = Path(config_path).expanduser().resolve()
        else:
            # Try default locations
            default_paths = [
                Path("config.yaml"),
                Path("~/.myjobspyai/config.yaml").expanduser(),
                Path("/etc/myjobspyai/config.yaml"),
            ]

            for path in default_paths:
                if path.exists():
                    config_path = path
                    break

        if config_path and config_path.exists():
            try:
                with open(config_path, "r") as f:
                    config_data = yaml.safe_load(f)
                self._config = AppConfig.model_validate(config_data)
                self._config_path = config_path
            except Exception as e:
                logger.error(f"Failed to load config from {config_path}: {e}")
                raise
        else:
            logger.warning("No config file found, using defaults")
            self._config = AppConfig()

        # Apply environment overrides if enabled
        if env_override and self._env_config:
            self._apply_env_overrides()

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        if not self._config or not self._env_config:
            return

        # Apply overrides
        for key, value in self._env_config.dict().items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

    def get_config(self) -> AppConfig:
        """Get the current configuration."""
        if not self._config:
            raise RuntimeError("Configuration not loaded")
        return self._config

    def get_provider_config(self, provider_name: str) -> Optional[LLMProviderConfig]:
        """Get configuration for a specific provider."""
        if not self._config:
            raise RuntimeError("Configuration not loaded")
        return self._config.providers.get(provider_name)

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        if not self._config:
            raise RuntimeError("Configuration not loaded")
        return self._config.database

    def get_api_config(self) -> APIConfig:
        """Get API configuration."""
        if not self._config:
            raise RuntimeError("Configuration not loaded")
        return self._config.api

    def get_analysis_config(self) -> AnalysisConfig:
        """Get analysis configuration."""
        if not self._config:
            raise RuntimeError("Configuration not loaded")
        return self._config.analysis

    def save(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Save the current configuration to a file.

        Args:
            config_path: Path to save the configuration to. If None, uses the current config path.
        """
        if not self._config:
            raise RuntimeError("Configuration not loaded")

        if config_path is None:
            config_path = self._config_path
            if not config_path:
                config_path = Path("config.yaml")

        config_path = Path(config_path).expanduser().resolve()
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert Path objects to strings
        config_dict = self._config.model_dump()
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

# Global instance
config_manager = ConfigManager()

async def load_config(
    config_path: Optional[Union[str, Path]] = None,
    env_override: bool = True,
) -> None:
    """Load configuration using the global config manager."""
    await config_manager.load(config_path, env_override)

async def get_config() -> AppConfig:
    """Get the current configuration."""
    return config_manager.get_config()
