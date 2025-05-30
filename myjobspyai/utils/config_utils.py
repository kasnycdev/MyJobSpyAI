"""Configuration management utilities for MyJobSpyAI."""
from __future__ import annotations

import json
import logging
import os
import yaml
from pathlib import Path
from typing import Any, Dict, TypeVar
from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    ConfigDict,
    DirectoryPath,
    Field,
    model_validator,
)

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for generic model type
T = TypeVar("T", bound="BaseModel")

# Load environment variables from .env file if it exists
load_dotenv()

class LLMConfig(BaseModel):
    """
    Configuration for LLM provider settings.
    
    This class handles configuration for various LLM providers with proper validation
    and default values. It supports both cloud-based (OpenAI, Anthropic) and self-hosted
    (Ollama) LLM providers.
    """
    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for provider-specific config
        env_prefix="LLM_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        validate_assignment=True,
        frozen=False,
    )

    # Core LLM settings
    provider: str = Field(
        default="ollama",
        description="LLM provider to use (openai, anthropic, ollama, etc.)",
        pattern=r"^(openai|anthropic|ollama|huggingface)$"
    )
    
    model: str = Field(
        default="llama3:instruct",
        description="Model name or identifier to use with the provider"
    )
    
    api_key: str | None = Field(
        default=None,
        description=(
            "API key for the LLM provider. Required for cloud providers "
            "like OpenAI and Anthropic."
        ),
        exclude=True  # Don't include in string representation
    )
    
    base_url: str | None = Field(
        default=None,
        description=(
            "Base URL for self-hosted LLM APIs. "
            "Example: http://localhost:11434 for Ollama"
        ),
        pattern=r"^https?://[^\s/$.?#].[^\s]*$"
    )
    
    # Generation parameters
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description=(
            "Controls randomness in the model's output. "
            "Higher values make the output more random, while lower values "
            "make it more deterministic."
        )
    )
    
    max_tokens: int = Field(
        default=2048,
        gt=0,
        le=32768,
        description=(
            "Maximum number of tokens to generate. "
            "Be aware of the model's context window limit."
        )
    )
    
    timeout: int = Field(
        default=30,
        gt=0,
        le=300,
        description=(
            "Timeout in seconds for API requests. "
            "Increase this for larger models or slower connections."
        )
    )
    
    # Provider-specific configurations
    openai: dict[str, Any] = Field(
        default_factory=dict,
        description="OpenAI-specific configuration options"
    )
    
    ollama: dict[str, Any] = Field(
        default_factory=dict,
        description="Ollama-specific configuration options"
    )
    
    anthropic: dict[str, Any] = Field(
        default_factory=dict,
        description="Anthropic-specific configuration options"
    )
    
    huggingface: dict[str, Any] = Field(
        default_factory=dict,
        description="HuggingFace-specific configuration options"
    )
    
    @model_validator(mode='after')
    def validate_provider_config(self) -> 'LLMConfig':
        """Validate provider-specific configuration."""
        provider = self.provider.lower()
        
        # Validate required fields for each provider
        if provider in ['openai', 'anthropic'] and not self.api_key:
            raise ValueError(
                f"API key is required for {provider} provider. "
                f"Set the {provider.upper()}_API_KEY environment variable or provide it in the config."
            )
            
        # Set default base_url for Ollama if not provided
        if provider == 'ollama' and not self.base_url:
            self.base_url = "http://localhost:11434"
            logger.info("Using default Ollama base URL: %s", self.base_url)
        
        # Validate model format for known providers
        if provider == 'openai' and not self.model.startswith(('gpt-', 'text-')):
            logger.warning(
                "Unrecognized OpenAI model format: %s. "
                "Expected format: gpt-4, gpt-3.5-turbo, etc.", self.model
            )
        
        return self
    
    def get_provider_config(self) -> dict[str, Any]:
        """Get the configuration for the current provider."""
        provider = self.provider.lower()
        base_config = {
            'provider': provider,
            'model': self.model,
            'api_key': self.api_key,
            'base_url': self.base_url,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'timeout': self.timeout,
        }
        
        # Add provider-specific config
        provider_config = getattr(self, provider, {}) or {}
        return {**base_config, **provider_config}

class FilterConfig(BaseModel):
    """Configuration for job filtering."""
    model_config = ConfigDict(
        extra="ignore",
        env_prefix="FILTER_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    salary_min: int | None = Field(
        default=None,
        ge=0,
        description="Minimum salary threshold"
    )
    salary_max: int | None = Field(
        default=None,
        gt=0,
        description="Maximum salary threshold"
    )
    work_models: list[str] = Field(
        default_factory=lambda: ["remote", "hybrid", "onsite"],
        description="List of acceptable work models"
    )
    title_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords to filter job titles"
    )

class LoggingConfig(BaseModel):
    """Configuration for logging."""
    model_config = ConfigDict(
        extra="ignore",
        env_prefix="LOG_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    file: str | None = Field(
        default=None,
        description="Path to log file (if None, logs to console only)"
    )
    max_bytes: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        gt=0,
        description="Maximum log file size in bytes"
    )
    backup_count: int = Field(
        default=5,
        gt=0,
        description="Number of backup log files to keep"
    )

class AppConfig(BaseModel):
    """Main application configuration."""
    model_config = ConfigDict(
        extra="ignore",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    app_name: str = Field(
        default="MyJobSpyAI",
        description="Name of the application"
    )
    version: str = Field(
        default="1.0.0",
        description="Application version"
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode"
    )
    cache_dir: DirectoryPath = Field(
        default=Path("cache"),
        description="Directory for caching data"
    )
    output_dir: DirectoryPath = Field(
        default=Path("output"),
        description="Directory for output files"
    )
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="LLM configuration"
    )
    filter: FilterConfig = Field(
        default_factory=FilterConfig,
        description="Filter configuration"
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration"
    )

    @model_validator(mode='before')
    @classmethod
    def set_defaults_based_on_env(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Set default values based on environment variables."""
        if values is None:
            values = {}
            
        # Initialize nested configs if not provided
        if 'llm' not in values:
            values['llm'] = {}
        if 'filter' not in values:
            values['filter'] = {}
        if 'logging' not in values:
            values['logging'] = {}
            
        # Ensure required LLM model is set
        if 'llm' in values and values['llm'] and not values['llm'].get('model'):
            # If model is not set in llm config, try to get it from provider-specific config
            provider = values['llm'].get('provider', 'ollama')
            if provider in values and 'model' in values[provider]:
                values['llm']['model'] = values[provider]['model']
            else:
                raise ValueError("No model specified in LLM configuration")
            
        # Ensure cache and output directories exist
        if 'cache_dir' in values and values['cache_dir']:
            cache_dir = Path(values['cache_dir'])
            cache_dir.mkdir(parents=True, exist_ok=True)
            
        if 'output_dir' in values and values['output_dir']:
            output_dir = Path(values['output_dir'])
            output_dir.mkdir(parents=True, exist_ok=True)
            
        return values

class ConfigManager:
    """Manages application configuration with support for YAML and environment variables."""
    
    _instance = None
    _config: AppConfig | None = None
    _config_path: Path | None = None
    _initialized: bool = False
    
    def __new__(cls) -> 'ConfigManager':
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: str | Path | None = None) -> None:
        if not self._initialized:
            self._config_path = Path(config_path) if config_path else None
            self._config = self._load_config()
            self._initialized = True
            
    def _deep_merge(self, base: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two dictionaries, with update taking precedence.
        
        Args:
            base: The base dictionary to merge into
            update: The dictionary with updates to apply
            
        Returns:
            A new dictionary with the merged contents
        """
        result = base.copy()
        
        for key, value in update.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Recursively merge nested dictionaries
                result[key] = self._deep_merge(result[key], value)
            else:
                # Add or update the key
                result[key] = value
                
        return result
    
    def _load_config(self) -> AppConfig:
        """
        Load and validate configuration from multiple sources with proper precedence.
        
        Configuration is loaded in the following order of precedence:
        1. Environment variables (with MYJOBS_ prefix, highest precedence)
        2. YAML config file (from provided path or standard locations)
        3. Default values from Pydantic models (lowest precedence)
        
        Returns:
            AppConfig: The loaded and validated configuration.
            
        Raises:
            ValueError: If the configuration is invalid or cannot be loaded.
        """
        def _parse_env_vars(prefix: str = 'MYJOBS_') -> Dict[str, Any]:
            """Parse environment variables with the given prefix into a nested dict."""
            result = {}
            prefix_len = len(prefix)
            
            for key, value in os.environ.items():
                if key.startswith(prefix):
                    # Convert MYJOBS_LLM__MODEL to llm.model
                    parts = key[prefix_len:].lower().split('__')
                    current = result
                    
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    
                    # Try to parse JSON values, fall back to string
                    try:
                        current[parts[-1]] = json.loads(value)
                    except (json.JSONDecodeError, TypeError):
                        # Handle boolean strings
                        if value.lower() in ('true', 'false'):
                            current[parts[-1]] = value.lower() == 'true'
                        # Handle null/None
                        elif value.lower() == 'null':
                            current[parts[-1]] = None
                        # Handle numbers
                        elif value.isdigit():
                            current[parts[-1]] = int(value)
                        elif value.replace('.', '', 1).isdigit() and value.count('.') < 2:
                            current[parts[-1]] = float(value)
                        # Default to string
                        else:
                            current[parts[-1]] = value
            
            return result
        
        def _load_yaml_config(path: Path) -> Dict[str, Any]:
            """Safely load YAML configuration from a file."""
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
                logger.info("Loaded configuration from %s", path)
                return config
            except yaml.YAMLError as e:
                logger.error("Invalid YAML in config file %s: %s", path, e)
                raise ValueError(f"Invalid YAML in config file {path}: {e}") from e
            except Exception as e:
                logger.error("Failed to load config from %s: %s", path, e)
                raise
        
        # 1. Start with default configuration
        default_config = AppConfig().model_dump()
        
        # 2. Load YAML configuration if available
        config_data = {}
        config_paths = []
        
        # Add user-specified path if provided
        if self._config_path:
            config_paths.append(Path(self._config_path).expanduser().resolve())
        
        # Add standard config locations
        config_paths.extend([
            Path("config.yaml").resolve(),
            Path("config/config.yaml").resolve(),
            Path("myjobspyai/config/config.yaml").resolve(),
            Path("~/.config/myjobspyai/config.yaml").expanduser().resolve(),
            Path("/etc/myjobspyai/config.yaml"),
        ])
        
        # Try each config path until we find one that works
        for path in config_paths:
            if path.exists() and path.is_file():
                try:
                    config_data = _load_yaml_config(path)
                    self._config_path = path  # Save the path that worked
                    break
                except Exception as e:
                    logger.debug("Skipping config at %s: %s", path, e)
        
        # 3. Load environment variables
        env_config = _parse_env_vars()
        
        # 4. Merge configurations with proper precedence
        #   1. Environment variables (highest precedence)
        #   2. YAML config file
        #   3. Default values (lowest precedence)
        merged_config = self._deep_merge(default_config, config_data)
        merged_config = self._deep_merge(merged_config, env_config)
        
        # 5. Create and validate the final configuration
        try:
            config = AppConfig.model_validate(merged_config)
            
            # Log configuration summary (without sensitive data)
            log_config = config.model_dump(mode='json')
            
            # Redact sensitive information
            def redact_sensitive_data(config_dict: Dict[str, Any]) -> None:
                """Recursively redact sensitive data for logging."""
                sensitive_keys = {'api_key', 'password', 'secret', 'token'}
                
                for key, value in list(config_dict.items()):
                    if isinstance(value, dict):
                        redact_sensitive_data(value)
                    elif isinstance(value, str) and any(s in key.lower() for s in sensitive_keys):
                        config_dict[key] = '***REDACTED***'
            
            redact_sensitive_data(log_config)
            
            # Log configuration summary
            logger.info("Configuration loaded successfully")
            logger.debug("Effective configuration: %s", 
                        json.dumps(log_config, indent=2, default=str))
            
            return config
            
        except Exception as e:
            logger.error("Invalid configuration: %s", e, exc_info=True)
            
            # Provide more detailed error information
            if hasattr(e, 'errors') and isinstance(e.errors, list):
                for error in e.errors():
                    logger.error("Configuration error - %s: %s", 
                                ' -> '.join(str(loc) for loc in error['loc']), 
                                error['msg'])
            
            # Only fall back to defaults if explicitly allowed
            if os.environ.get('MYJOBS_ALLOW_DEFAULTS', '').lower() in ('1', 'true', 'yes'):
                logger.warning("Falling back to default configuration due to error")
                return AppConfig()
            
            raise ValueError("Invalid configuration. See logs for details.") from e
    
    @property
    def config(self) -> AppConfig:
        """
        Get the current configuration.
        
        Returns:
            AppConfig: The current configuration.
            
        Raises:
            RuntimeError: If the configuration cannot be loaded.
        """
        if self._config is None:
            try:
                self._config = self._load_config()
            except Exception as e:
                logger.critical("Failed to load configuration: %s", e, exc_info=True)
                raise RuntimeError(f"Failed to load configuration: {e}") from e
        return self._config
    
    def update_config(self, **updates: Any) -> None:
        """
        Update configuration with new values.
        
        Args:
            **updates: Key-value pairs to update in the configuration.
            
        Raises:
            ValueError: If the updates result in an invalid configuration.
        """
        if self._config is None:
            self._config = self._load_config()
        
        try:
            # Create a new config with the updates
            updated_config = self._config.model_copy(update=updates)
            # Validate the updated config
            updated_config.model_validate(updated_config.model_dump())
            # Only update if validation succeeds
            self._config = updated_config
            logger.info("Configuration updated successfully")
        except Exception as e:
            logger.error("Failed to update configuration: %s", e, exc_info=True)
            raise ValueError(f"Invalid configuration update: {e}") from e
    
    def save_config(self, path: str | Path | None = None) -> Path:
        """
        Save current configuration to a YAML file.
        
        Args:
            path: Path to save the configuration to. If None, uses the default config path.
            
        Returns:
            Path: The path where the configuration was saved.
            
        Raises:
            ValueError: If no path is provided and no default config path is set.
            IOError: If the configuration cannot be saved to the file.
        """
        save_path = Path(path) if path else self._config_path
        if not save_path:
            raise ValueError("No path provided and no default config path set")
        
        if self._config is None:
            self._config = self._load_config()
        
        try:
            # Ensure parent directory exists
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert model to dict and save as YAML
            config_dict = self._config.model_dump(
                mode='json',
                exclude_unset=True,
                exclude_none=True,
                by_alias=True
            )
            
            # Use a temporary file for atomic write
            temp_path = save_path.with_suffix('.tmp')
            with open(temp_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(
                    config_dict,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                    encoding='utf-8'
                )
            
            # Replace the original file atomically
            if save_path.exists():
                save_path.unlink()
            temp_path.rename(save_path)
            
            logger.info("Configuration saved to %s", save_path)
            return save_path
            
        except Exception as e:
            logger.error("Failed to save configuration to %s: %s", save_path, e, exc_info=True)
            if 'temp_path' in locals() and temp_path.exists():
                temp_path.unlink(missing_ok=True)
            raise IOError(f"Failed to save configuration: {e}") from e

# Global config manager instance
config_manager = ConfigManager()

# Shortcut to access config
config = config_manager.config

def get_config() -> AppConfig:
    """Get the current configuration."""
    return config_manager.config

def update_config(**updates: Any) -> None:
    """Update configuration with new values."""
    config_manager.update_config(**updates)

def save_config(path: str | Path | None = None) -> None:
    """Save current configuration to a YAML file."""
    config_manager.save_config(path)
