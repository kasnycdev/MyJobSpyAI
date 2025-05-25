"""
Centralized configuration management for MyJobSpyAI.

This module provides a single source of truth for all application settings,
with support for environment variables, .env files, and direct configuration.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, TypeVar
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    HttpUrl,
    ValidationError as PydanticValidationError,
)
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Type variable for generic model validation
T = TypeVar('T')

# Set up logging
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
load_dotenv()

class ValidationError(Exception):
    """Base exception for configuration validation errors."""

class MissingRequiredFieldError(ValidationError):
    """Raised when a required configuration field is missing."""

class InvalidValueError(ValidationError):
    """Raised when a configuration value is invalid."""

class ConfigurationError(Exception):
    """Raised when there is a problem with the configuration."""

def validate_required_fields(model: type[BaseModel], data: Dict[str, Any]) -> None:
    """Validate that all required fields are present in the data."""
    errors = []
    for field_name, field in model.model_fields.items():
        if field.is_required() and field_name not in data:
            errors.append(f"Missing required field: {field_name}")
    
    if errors:
        raise MissingRequiredFieldError("\n".join(errors))

class LLMConfig(BaseModel):
    """Configuration for LLM providers."""
    provider: str = Field(
        default="openai",
        description="LLM provider to use (openai, ollama, gemini, etc.)",
        examples=["openai", "ollama", "gemini"]
    )
    model: str = Field(
        default="gpt-4",
        description="Model name to use",
        examples=["gpt-4", "llama2", "gemini-pro"]
    )
    api_key: Optional[str] = Field(
        default=None,
        description=(
            "API key for the provider. Required for cloud-based providers "
            "like OpenAI and Anthropic."
        ),
        exclude=True  # Don't include in string representation
    )
    base_url: Optional[HttpUrl] = Field(
        default=None,
        description=(
            "Base URL for self-hosted instances. "
            "Example: http://localhost:11434 for Ollama"
        )
    )
    timeout: float = Field(
        default=30.0,
        gt=0,
        description=(
            "Request timeout in seconds. Increase this for slower connections "
            "or larger models."
        )
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description=(
            "Maximum number of retry attempts for failed requests. "
            "Set to 0 to disable retries."
        )
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description=(
            "Sampling temperature. Higher values (closer to 2.0) make output "
            "more random, while lower values (closer to 0) make it more "
            "deterministic."
        )
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=8192,
        description=(
            "Maximum number of tokens to generate. "
            "Leave as None to use the model's default."
        )
    )

    @field_validator('api_key', mode='before')
    @classmethod
    def validate_api_key_required(cls, v: Optional[str], info: Any) -> Optional[str]:
        """Validate that API key is provided when required."""
        if v is None and info.field_name == 'api_key':
            provider = info.data.get('provider', '').lower()
            if provider in ['openai', 'anthropic']:
                raise ValueError(
                    f"API key is required for {provider} provider. "
                    f"Please set the MYJOBS_LLM_API_KEY environment variable or "
                    "add it to your configuration file."
                )
        return v
    
    @model_validator(mode='after')
    def validate_model_requirements(self) -> 'LLMConfig':
        """Validate model-specific requirements."""
        provider = self.provider.lower()
        model = self.model.lower()
        
        # Check for required base_url for self-hosted models
        if provider == 'ollama' and not self.base_url:
            self.base_url = "http://localhost:11434"
            logger.warning(
                "Using default Ollama base URL: http://localhost:11434. "
                "Set LLM_BASE_URL to override."
            )
        
        # Validate model names for known providers
        if provider == 'openai' and not model.startswith(('gpt-', 'text-')):
            logger.warning(
                f"Unknown OpenAI model: {model}. "
                f"Expected format: gpt-4, gpt-3.5-turbo, etc."
            )
        
        return self


class CacheConfig(BaseModel):
    """Configuration for caching responses and embeddings."""
    enabled: bool = Field(
        default=True,
        description=(
            "Enable or disable caching. When enabled, API responses and "
            "embeddings will be cached to improve performance."
        )
    )
    directory: Path = Field(
        default=Path("cache/"),
        description=(
            "Base directory for cache files. Will be created if it doesn't exist. "
            "Can be an absolute path or relative to the project root."
        )
    )
    ttl: int = Field(
        default=3600,  # 1 hour
        ge=0,
        description=(
            "Default time-to-live for cache entries in seconds. "
            "Set to 0 to disable cache expiration."
        )
    )
    max_size: int = Field(
        default=1000,
        gt=0,
        description=(
            "Maximum number of items to keep in the cache. "
            "Older items will be evicted when the limit is reached."
        )
    )
    
    @field_validator('directory')
    @classmethod
    def validate_directory(cls, v: Path) -> Path:
        """Ensure cache directory exists and is writable."""
        try:
            v = v.absolute()
            v.mkdir(parents=True, exist_ok=True)
            return v
        except OSError as e:
            raise ValueError(f"Failed to create cache directory {v}: {e}") from e
            # Test if directory is writable
            test_file = v / ".write_test"
            test_file.touch()
            test_file.unlink()
            
            return v
        except (OSError, PermissionError) as e:
            raise ValueError(f"Cannot write to cache directory {v}: {e}")
            
    def get_cache_path(self, key: str) -> Path:
        """Get the full path to a cache file."""
        # Create a safe filename from the key
        import hashlib
        import base64
        
        # Create a hash of the key
        key_hash = hashlib.sha256(key.encode()).digest()
        # Use base64 encoding for a shorter, filesystem-safe string
        safe_key = base64.urlsafe_b64encode(key_hash).decode()[:16]
        
        return self.directory / f"cache_{safe_key}.pkl"

class LoggingConfig(BaseModel):
    """Configuration for application logging."""
    level: str = Field(
        default="INFO",
        description=(
            "Logging level. One of: DEBUG, INFO, WARNING, ERROR, CRITICAL. "
            "Logs at this level and above will be emitted."
        ),
        examples=["INFO", "DEBUG", "WARNING"]
    )
    file: Optional[Path] = Field(
        default=None,
        description=(
            "Path to log file. If None, logs will only be output to the console. "
            "If the file exists, new logs will be appended."
        ),
        examples=["logs/app.log", "/var/log/myjobspyai.log"]
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description=(
            "Log message format using Python's logging formatter syntax. "
            "Common placeholders: %(asctime)s, %(name)s, %(levelname)s, %(message)s"
        )
    )
    max_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        gt=0,
        description="Maximum log file size in bytes before rotation"
    )
    backup_count: int = Field(
        default=5,
        ge=0,
        description=(
            "Number of backup log files to keep. Set to 0 to disable log rotation. "
            "Only used when file logging is enabled."
        )
    )
    json_format: bool = Field(
        default=False,
        description=(
            "Whether to output logs in JSON format. Useful for log aggregation "
            "systems like ELK or CloudWatch Logs."
        )
    )
    
    @field_validator('level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate that the log level is valid."""
        import logging
        v_upper = v.upper()
        if v_upper not in logging._nameToLevel:  # type: ignore
            raise ValueError(
                f"Invalid log level: {v}. "
                f"Must be one of: {', '.join(logging._nameToLevel.keys())}"  # type: ignore
            )
        return v_upper
    
    @field_validator('file', mode='before')
    @classmethod
    def validate_log_file(cls, v: Optional[Union[str, Path]]) -> Optional[Path]:
        """Convert string paths to Path objects."""
        if v is None:
            return None
        return Path(v)

class AppConfig(BaseSettings):
    """
    Main application configuration.
    
    This class defines all configurable parameters for the application,
    with support for loading from multiple sources in the following order:
    1. Environment variables (with MYJOBS_ prefix)
    2. .env file
    3. Default values
    
    Environment variables should be prefixed with MYJOBS_ and use double
    underscores for nested fields. For example:
    - MYJOBS_DEBUG=true
    - MYJOBS_LLM__PROVIDER=openai
    - MYJOBS_LLM__MODEL=gpt-4
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        env_prefix="myjobs_",
        extra="ignore",
        validate_default=True,
        case_sensitive=False,
        frozen=False,
        arbitrary_types_allowed=True,
    )
    
    # Core application settings
    debug: bool = Field(
        default=False,
        description=(
            "Enable debug mode. When True, more verbose logging will be "
            "enabled and sensitive information may be exposed in logs."
        )
    )
    
    environment: str = Field(
        default="production",
        description=(
            "Runtime environment. Affects things like logging levels and "
            "default configurations. Should be one of: development, staging, production"
        ),
        examples=["development", "staging", "production"]
    )
    
    # Component configurations
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Configuration for the Language Model provider"
    )
    
    cache: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Caching configuration for API responses and embeddings"
    )
    
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration"
    )
    
    # Performance settings
    max_concurrent_tasks: int = Field(
        default=10,
        ge=1,
        le=100,
        description=(
            "Maximum number of concurrent tasks or API requests. "
            "Adjust based on your system's capabilities and rate limits."
        )
    )
    
    request_timeout: float = Field(
        default=60.0,
        gt=0,
        le=600,  # 10 minutes max
        description=(
            "Default timeout in seconds for HTTP requests. "
            "Increase this if you have a slow connection or are working with large models."
        )
    )
    
    # Feature flags
    enable_experimental_features: bool = Field(
        default=False,
        description=(
            "Enable experimental features. These features may be unstable "
            "or change without notice in future releases."
        )
    )
    
    # Paths
    data_dir: Path = Field(
        default=Path("data"),
        description=(
            "Base directory for application data. Will be created if it doesn't exist."
        )
    )
    
    # Add __repr__ that hides sensitive information
    def __repr_args__(self) -> list[tuple[str, Any]]:
        """Custom repr that hides sensitive information."""
        # Get all fields
        fields = self.model_dump()
        
        # Redact sensitive information
        if 'llm' in fields and 'api_key' in fields['llm'] and fields['llm']['api_key'] is not None:
            fields['llm']['api_key'] = '***REDACTED***'
            
        return list(fields.items())
    
    @model_validator(mode='after')
    def validate_config(self) -> 'AppConfig':
        """Run additional validation after model initialization."""
        # Set environment-specific defaults
        if self.environment == 'development':
            if self.logging.level == 'INFO':
                self.logging.level = 'DEBUG'
            
            if self.llm.provider == 'openai' and not self.llm.api_key:
                logger.warning(
                    "Running in development mode without an OpenAI API key. "
                    "Some features may not work correctly."
                )
        
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        return self
    
    @classmethod
    def load(cls, config_path: Optional[Path] = None, **overrides) -> 'AppConfig':
        """Load configuration from a file with overrides."""
        config_data = {}
        
        # Load from file if path is provided
        if config_path and config_path.exists():
            import yaml
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
        
        # Apply overrides
        config_data.update(overrides)
        
        # Create and validate config
        return cls(**config_data)

# Global configuration instance
settings = AppConfig()

def configure(config_path: Optional[Union[str, Path]] = None, **overrides) -> AppConfig:
    """
    Configure application settings.
    
    This is the main entry point for configuring the application.
    It loads settings from multiple sources in the following order:
    1. Default values (defined in the model)
    2. Environment variables (with MYJOBS_ prefix)
    3. .env file (if it exists)
    4. Configuration file (YAML/JSON, if provided)
    5. Explicit overrides (passed as kwargs)
    
    Args:
        config_path: Path to a configuration file (YAML or JSON)
        **overrides: Settings to override (takes highest precedence)
        
    Returns:
        AppConfig: The configured settings instance
        
    Example:
        ```python
        # Load from default locations
        configure()
        
        # Load from a specific config file
        configure("config/production.yaml")
        
        # Override specific settings
        configure(debug=True, max_concurrent_tasks=5)
        
        # Load from file with overrides
        configure("config/staging.yaml", debug=True)
        ```
    """
    global settings
    
    try:
        # Convert string path to Path object
        config_path = Path(config_path) if config_path else None
        
        # Initialize with default values
        config_data: Dict[str, Any] = {}
        
        # Load from configuration file if provided
        if config_path and config_path.exists():
            suffix = config_path.suffix.lower()
            
            if suffix in ('.yaml', '.yml'):
                import yaml
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_data = yaml.safe_load(f) or {}
                    config_data.update(file_data)
            elif suffix == '.json':
                import json
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    config_data.update(file_data)
            else:
                logger.warning(
                    f"Unsupported config file format: {suffix}. "
                    "Expected .yaml, .yml, or .json"
                )
        
        # Apply overrides (highest precedence)
        config_data.update(overrides)
        
        # Create and validate the config
        settings = AppConfig.model_validate(config_data)
        
        # Log the configuration source
        source = f"file: {config_path}" if config_path else "default values"
        logger.info(f"Loaded configuration from {source}")
        
        # Log important configuration details (safely)
        logger.debug("Configuration loaded successfully")
        
        return settings
        
    except PydanticValidationError as e:
        logger.error(f"Configuration validation error: {e}")
        raise ConfigurationError(f"Invalid configuration: {e}") from e
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        raise ConfigurationError(f"Failed to load configuration: {e}") from e

# Initialize with default settings
configure()

# Add a reload function for convenience
def reload_config(config_path: Optional[Union[str, Path]] = None, **overrides) -> None:
    """
    Reload the configuration from the specified source.
    
    This is a convenience function that calls configure() and updates
    the global settings object.
    
    Args:
        config_path: Path to a configuration file (YAML or JSON)
        **overrides: Settings to override
    """
    global settings
    settings = configure(config_path, **overrides)

# Add a context manager for temporary configuration changes
class temporary_config:
    """
    Context manager for temporary configuration changes.
    
    Example:
        ```python
        with temporary_config(debug=True):
            # Code that uses the temporary configuration
            pass  # Original config is restored when the block exits
        ```
    """
    def __init__(self, **overrides):
        self.overrides = overrides
        self._original_settings = None
        
    def __enter__(self):
        global settings
        self._original_settings = settings.model_dump()
        settings = settings.model_copy(update=self.overrides)
        return settings
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        global settings
        if self._original_settings is not None:
            settings = AppConfig.model_validate(self._original_settings)
