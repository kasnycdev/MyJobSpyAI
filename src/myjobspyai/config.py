"""Unified configuration management for MyJobSpy AI using Pydantic v2."""

import logging
import os
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Type, Union

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_serializer,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    InitSettingsSource,
    PydanticBaseSettingsSource,
    SecretsSettingsSource,
    SettingsConfigDict,
)

logger = logging.getLogger(__name__)

# Default directories
DEFAULT_CONFIG_DIR = Path("~/.config/myjobspyai").expanduser()
DEFAULT_DATA_DIR = Path("~/.local/share/myjobspyai").expanduser()
DEFAULT_CACHE_DIR = Path("~/.cache/myjobspyai").expanduser()

# Ensure default directories exist
for directory in [DEFAULT_CONFIG_DIR, DEFAULT_DATA_DIR, DEFAULT_CACHE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


class DatabaseConfig(BaseModel):
    """Database configuration."""

    url: str = Field(
        default="sqlite:///" + str(DEFAULT_DATA_DIR / "myjobspyai.db"),
        description="Database connection URL",
    )
    echo: bool = Field(
        default=False, description="Enable SQLAlchemy engine echo output"
    )
    pool_size: int = Field(default=5, description="Connection pool size")
    max_overflow: int = Field(
        default=10, description="Maximum overflow for connection pool"
    )


class LLMProviderConfig(BaseModel):
    """Configuration for an LLM provider."""

    name: str = Field(..., description="Name of the LLM provider")
    enabled: bool = Field(True, description="Whether this provider is enabled")
    type: str = Field(
        "langchain",
        description="Type of provider (e.g., 'openai', 'ollama', 'langchain'). Defaults to 'langchain' as it's the most common type.",
    )
    model: str = Field("gpt-3.5-turbo", description="Default model to use")
    api_key: Optional[str] = Field(None, description="API key for the provider")
    base_url: Optional[str] = Field(None, description="Base URL for the API")
    timeout: int = Field(60, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries")
    temperature: float = Field(0.7, description="Default temperature")
    max_tokens: Optional[int] = Field(
        None, description="Maximum number of tokens to generate"
    )

    # Common additional fields that might be in the config
    class_name: Optional[str] = Field(
        None, description="Fully qualified class name for the provider"
    )
    streaming: Optional[bool] = Field(
        False, description="Whether to enable streaming responses"
    )
    top_p: Optional[float] = Field(1.0, description="Nucleus sampling parameter")
    frequency_penalty: Optional[float] = Field(
        0.0, description="Frequency penalty for text generation"
    )
    presence_penalty: Optional[float] = Field(
        0.0, description="Presence penalty for text generation"
    )

    class Config:
        extra = "allow"  # Allow provider-specific settings

    @model_validator(mode='before')
    @classmethod
    def validate_provider_config(cls, data):
        if not isinstance(data, dict):
            return data

        # Log the raw data being validated
        logger.debug(f"Validating LLM provider config: {data}")

        # Set default values for required fields if not present
        if 'type' not in data:
            data['type'] = 'langchain'
            logger.debug(f"Set default type to 'langchain' for provider")

        if 'model' not in data:
            if data.get('type') == 'ollama':
                data['model'] = 'llama2'
                logger.debug(f"Set default model 'llama2' for ollama provider")
            else:
                data['model'] = 'gpt-3.5-turbo'
                logger.debug(
                    f"Set default model 'gpt-3.5-turbo' for {data.get('type')} provider"
                )

        # Log the final config being validated
        logger.debug(f"Final LLM provider config after validation: {data}")
        return data


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field("INFO", description="Logging level")
    file: Optional[Path] = Field(
        DEFAULT_DATA_DIR / "logs" / "myjobspyai.log", description="Path to log file"
    )
    file_level: str = Field("DEBUG", description="Logging level for file output")
    max_size: int = Field(
        10 * 1024 * 1024, description="Maximum log file size in bytes"
    )
    backup_count: int = Field(5, description="Number of backup log files to keep")
    format: str = Field(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s",
        description="Log message format",
    )

    @field_validator("file", mode="before")
    @classmethod
    def resolve_log_file(cls, v: Optional[Union[str, Path]]) -> Optional[Path]:
        if v is None:
            return None
        return Path(v).expanduser().resolve()


class APIConfig(BaseModel):
    """API configuration."""

    enabled: bool = Field(True, description="Enable the API server")
    host: str = Field("0.0.0.0", description="Host to bind the API server to")
    port: int = Field(8000, description="Port to run the API server on")
    debug: bool = Field(False, description="Enable debug mode")
    cors_origins: List[str] = Field(["*"], description="Allowed CORS origins")
    api_key: Optional[str] = Field(None, description="API key for authentication")
    rate_limit: str = Field("100/minute", description="Rate limiting configuration")


class AnalysisConfig(BaseModel):
    """Analysis configuration."""

    # Cover letter generation settings
    enable_cover_letter_generation: bool = Field(
        True,
        description="Enable automatic cover letter generation during resume analysis",
    )
    default_cover_letter_style: str = Field(
        "PROFESSIONAL",
        description="Default style for generated cover letters (PROFESSIONAL, CREATIVE, TECHNICAL, etc.)",
    )
    default_cover_letter_tone: str = Field(
        "professional",
        description="Default tone for generated cover letters (professional, enthusiastic, formal, etc.)",
    )
    max_cover_letter_length: int = Field(
        1000, description="Maximum length for generated cover letters in characters"
    )

    # Training recommendations settings
    enable_training_recommendations: bool = Field(
        True,
        description="Enable training resource recommendations during resume analysis",
    )
    max_training_recommendations: int = Field(
        5, description="Maximum number of training recommendations to generate"
    )
    training_resources_file: Optional[Path] = Field(
        None,
        description="Path to a JSON file containing training resources. If not provided, uses built-in resources.",
    )

    # Performance settings
    enable_caching: bool = Field(True, description="Enable caching of analysis results")
    cache_ttl_seconds: int = Field(
        3600, description="Time-to-live for cached analysis results in seconds"
    )


class AppConfig(BaseSettings):
    """Main application configuration.

    Loads configuration from the following sources in order:
    1. Default values defined in this class
    2. Environment variables (with MYJOBSPYAI_ prefix)
    3. YAML configuration file
    """

    class Config:
        env_prefix = "MYJOBSPYAI_"
        env_nested_delimiter = "__"
        case_sensitive = False
        env_file = None  # Load .env file manually to control loading
        env_file_encoding = "utf-8"
        extra = "ignore"
        validate_default = True
        json_encoders = {
            Path: str,
        }

    # Application metadata
    name: str = Field("MyJobSpy AI", description="Application name")
    version: str = Field("0.1.0", description="Application version")
    debug: bool = Field(False, description="Enable debug mode")
    environment: str = Field("production", description="Runtime environment")

    # Core directories
    data_dir: Path = Field(
        DEFAULT_DATA_DIR, description="Directory for application data"
    )
    cache_dir: Path = Field(DEFAULT_CACHE_DIR, description="Directory for cached data")
    config_dir: Path = Field(
        DEFAULT_CONFIG_DIR, description="Directory for configuration files"
    )

    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)
    default_llm_provider: str = Field(
        "ollama", description="Default LLM provider to use when none is specified"
    )
    llm_providers: Dict[str, Any] = Field(
        default_factory=dict, description="Configured LLM providers"
    )

    # Store the raw provider configs for reference
    _raw_llm_providers: Dict[str, Any] = PrivateAttr(default_factory=dict)

    @model_validator(mode='before')
    @classmethod
    def validate_llm_providers(cls, data):
        """Validate and process LLM providers configuration.

        Supports multiple configuration formats:
        1. Direct llm_providers at root level with default_llm_provider at root
        2. Nested under llm.providers with llm.default_provider
        3. Direct providers under llm_providers at root
        """
        logger.debug(
            f"[LLM VALIDATION] Starting LLM providers validation. Data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}"
        )

        if not isinstance(data, dict):
            logger.warning(
                "Configuration data is not a dictionary, cannot validate LLM providers"
            )
            data['llm_providers'] = {}
            return data

        # Debug: Log the raw data structure
        logger.debug(f"[LLM VALIDATION] Raw data keys: {list(data.keys())}")
        if 'llm' in data:
            logger.debug(
                f"[LLM VALIDATION] Found 'llm' key with type: {type(data['llm']).__name__}"
            )
        if 'llm_providers' in data:
            logger.debug(
                f"[LLM VALIDATION] Found 'llm_providers' key with type: {type(data['llm_providers']).__name__}"
            )

        # Get the LLM section if it exists
        llm_section = data.get('llm', {}) or {}
        if not isinstance(llm_section, dict):
            llm_section = {}

        # Check for providers in different locations
        raw_providers = None
        config_source = "unknown"

        # 1. Check llm.providers first (nested format)
        if 'providers' in llm_section and isinstance(llm_section['providers'], dict):
            raw_providers = llm_section['providers']
            config_source = "llm.providers"
            logger.debug("[LLM VALIDATION] Using LLM providers from llm.providers")
        # 2. Check root llm_providers
        elif 'llm_providers' in data and isinstance(data['llm_providers'], dict):
            raw_providers = data['llm_providers']
            config_source = "root llm_providers"
            logger.debug("[LLM VALIDATION] Using LLM providers from root llm_providers")
            logger.debug(
                f"[LLM VALIDATION] Providers found: {list(raw_providers.keys())}"
            )
        # 3. Check if llm_providers is at root level directly
        elif any(k in data for k in ['llm_providers', 'default_llm_provider']):
            # This handles the case where llm_providers might be at root level
            raw_providers = {}
            for k, v in data.items():
                if k == 'llm_providers' and isinstance(v, dict):
                    raw_providers.update(v)
                    config_source = "root level"
                    logger.debug("[LLM VALIDATION] Using LLM providers from root level")
                    logger.debug(
                        f"[LLM VALIDATION] Providers found: {list(raw_providers.keys())}"
                    )
                    break

        if not raw_providers:
            logger.warning("No valid LLM providers found in configuration")
            data['llm_providers'] = {}
            return data

        if not isinstance(raw_providers, dict):
            logger.warning(
                f"LLM providers should be a dictionary, got {type(raw_providers).__name__}"
            )
            data['llm_providers'] = {}
            return data

        logger.info(
            f"Found {len(raw_providers)} LLM providers in {config_source}: {list(raw_providers.keys())}"
        )

        # Store the raw provider configs for reference
        processed_providers = {}

        for provider_name, provider_config in raw_providers.items():
            if not isinstance(provider_config, dict):
                logger.warning(
                    f"Provider config for {provider_name} is not a dictionary, skipping"
                )
                continue

            try:
                # Make a copy to avoid modifying the original
                provider_config = provider_config.copy()

                # Handle nested config structure (for langchain providers)
                if 'config' in provider_config and isinstance(
                    provider_config['config'], dict
                ):
                    # Flatten the config for LLMProviderConfig
                    flat_config = provider_config.copy()
                    flat_config.update(provider_config['config'])
                    provider_config = flat_config

                # Ensure the provider has a name
                if 'name' not in provider_config:
                    provider_config['name'] = provider_name

                # Ensure required fields have defaults
                if 'enabled' not in provider_config:
                    provider_config['enabled'] = True

                # Skip disabled providers
                if not provider_config.get('enabled', True):
                    logger.debug(f"Skipping disabled provider: {provider_name}")
                    continue

                # Log provider config (without sensitive data)
                safe_config = provider_config.copy()
                if 'api_key' in safe_config and safe_config['api_key']:
                    safe_config['api_key'] = '***REDACTED***'
                logger.debug(
                    f"Creating LLMProviderConfig for {provider_name}: {safe_config}"
                )

                # Create LLMProviderConfig instance
                provider_instance = LLMProviderConfig(**provider_config)
                processed_providers[provider_name] = provider_instance

                logger.info(
                    f"Successfully loaded LLM provider: {provider_name} ({getattr(provider_instance, 'type', 'unknown')})"
                )

            except Exception as e:
                logger.error(
                    f"Failed to create LLM provider {provider_name}: {str(e)}",
                    exc_info=True,
                )
                continue

        # Store the raw providers in the data for later use
        data['_raw_llm_providers'] = raw_providers

        # Update the data with processed providers
        data['llm_providers'] = processed_providers

        if not processed_providers:
            logger.warning("No valid LLM providers were successfully loaded")
        else:
            logger.info(
                f"Successfully processed {len(processed_providers)} LLM providers: {list(processed_providers.keys())}"
            )

        # Set default provider
        default_provider = None

        # Check default provider in this order:
        # 1. llm.default_provider
        # 2. root default_llm_provider
        # 3. First enabled provider if only one exists
        # 4. First provider if any exist

        if 'default_provider' in llm_section:
            default_provider = llm_section['default_provider']
            logger.debug(
                f"Using default provider from llm.default_provider: {default_provider}"
            )
        elif 'default_llm_provider' in data:
            default_provider = data['default_llm_provider']
            logger.debug(
                f"Using default provider from root default_llm_provider: {default_provider}"
            )

        # If we have a default provider, validate it exists
        if default_provider and default_provider in processed_providers:
            data['default_llm_provider'] = default_provider
            logger.info(f"Set default LLM provider to: {default_provider}")
        elif processed_providers:
            # If default not found but we have providers, use the first one
            first_provider = next(iter(processed_providers.keys()))
            data['default_llm_provider'] = first_provider
            logger.warning(
                f"Default LLM provider '{default_provider}' not found or not specified, using first available: {first_provider}"
            )

        return data

    def model_post_init(self, __context):
        """Post-initialization hook to ensure LLM providers are properly set."""
        super().model_post_init(__context)

        # If we have raw providers but no processed ones, try to process them
        if hasattr(self, '_raw_llm_providers') and not self.llm_providers:
            logger.warning(
                "No LLM providers found in config, attempting to process raw providers..."
            )
            processed = {}
            for name, config in self._raw_llm_providers.items():
                try:
                    processed[name] = LLMProviderConfig(**config)
                except Exception as e:
                    logger.error(f"Failed to process LLM provider {name}: {str(e)}")

            if processed:
                logger.info(
                    f"Successfully processed {len(processed)} LLM providers in post-init"
                )
                self.llm_providers = processed

    # Ensure paths are absolute and expanded
    @model_validator(mode="after")
    def resolve_paths(self) -> 'AppConfig':
        """Resolve all paths to absolute paths."""
        if self.data_dir:
            self.data_dir = Path(self.data_dir).expanduser().resolve()
        if self.cache_dir:
            self.cache_dir = Path(self.cache_dir).expanduser().resolve()
        if self.config_dir:
            self.config_dir = Path(self.config_dir).expanduser().resolve()

        # Ensure logging file path is resolved
        if self.logging.file:
            self.logging.file = Path(self.logging.file).expanduser().resolve()
            self.logging.file.parent.mkdir(parents=True, exist_ok=True)

        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        return self

    @classmethod
    def load_defaults(cls) -> "AppConfig":
        """Load default configuration."""
        return cls()

    @classmethod
    def from_file(cls, config_path: Optional[Path] = None) -> "AppConfig":
        """Load configuration from a YAML file.

        Args:
            config_path: Path to the configuration file. If None, uses the default location.

        Returns:
            Loaded AppConfig instance.
        """
        logger.debug(f"AppConfig.from_file called with config_path: {config_path}")

        if config_path is None:
            config_path = DEFAULT_CONFIG_DIR / "config.yaml"
            logger.debug(f"Using default config path: {config_path}")

        config_path = Path(config_path).expanduser().resolve()
        logger.debug(f"Resolved config path: {config_path}")

        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return cls()

        try:
            logger.debug(f"Reading YAML file: {config_path}")
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}
                logger.debug(f"Raw YAML data: {config_data}")

            # Convert string paths to Path objects
            logger.debug("Converting string paths to Path objects...")
            config_data = cls._convert_strings_to_paths(config_data)

            # Load environment variables from .env file if present
            dotenv_path = config_path.parent / ".env"
            if dotenv_path.exists():
                logger.debug(f"Loading environment variables from: {dotenv_path}")
                from dotenv import load_dotenv

                load_dotenv(dotenv_path)

            # Process environment variables in the config
            config_data = cls._process_environment_variables(config_data)

            logger.debug("Creating AppConfig instance from config data...")
            config = cls.model_validate(config_data)

            # Log configuration
            logger.debug(
                f"Successfully created AppConfig instance. Default LLM provider: {getattr(config, 'default_llm_provider', 'not set')}"
            )
            if hasattr(config, 'llm_providers') and config.llm_providers:
                logger.debug(
                    f"Configured LLM providers: {list(config.llm_providers.keys())}"
                )
                for provider_name, provider in config.llm_providers.items():
                    if hasattr(provider, 'enabled') and provider.enabled:
                        logger.debug(f"  - {provider_name} (enabled): {provider.model}")
                    else:
                        logger.debug(f"  - {provider_name} (disabled)")
            else:
                logger.warning("No LLM providers configured in the configuration file")

            return config

        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}", exc_info=True)
            return cls()

    @classmethod
    def _process_environment_variables(
        cls, config_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process environment variables in the configuration.

        Args:
            config_data: The configuration data dictionary

        Returns:
            Processed configuration with environment variables resolved
        """
        if not isinstance(config_data, dict):
            return config_data

        result = {}
        for key, value in config_data.items():
            if isinstance(value, dict):
                # Recursively process nested dictionaries
                result[key] = cls._process_environment_variables(value)
            elif (
                isinstance(value, str)
                and value.startswith('${')
                and value.endswith('}')
            ):
                # Handle environment variable substitution
                env_var = value[2:-1]  # Remove ${ and }
                default = ''
                if ':' in env_var:
                    # Handle default value: ${VAR:default}
                    env_var, default = env_var.split(':', 1)

                # Special handling for API keys to avoid logging sensitive data
                if (
                    'key' in key.lower()
                    or 'secret' in key.lower()
                    or 'token' in key.lower()
                ):
                    log_value = '[REDACTED]' if os.getenv(env_var) else 'not set'
                    logger.debug(
                        f"Resolving sensitive environment variable {key} from ${env_var} (value: {log_value})"
                    )
                else:
                    logger.debug(
                        f"Resolving environment variable {key} from ${env_var}"
                    )

                result[key] = os.getenv(env_var, default)

                # If the environment variable is required but not set, log a warning
                if not result[key] and not default:
                    logger.warning(
                        f"Environment variable ${env_var} is not set and has no default value"
                    )
            else:
                result[key] = value

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""

        # Convert Path objects to strings for serialization
        def serialize_paths(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: serialize_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [serialize_paths(item) for item in obj]
            elif hasattr(obj, 'model_dump'):
                return serialize_paths(obj.model_dump())
            return obj

        return serialize_paths(self.model_dump())

    def save(self, config_path: Optional[Path] = None) -> None:
        """Save configuration to a YAML file.

        Args:
            config_path: Path to save the configuration to. If None, uses the default location.
        """
        if config_path is None:
            config_path = DEFAULT_CONFIG_DIR / "config.yaml"

        config_path = Path(config_path).expanduser().resolve()
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            config_data = self.to_dict()
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    config_data, f, default_flow_style=False, sort_keys=False
                )
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")
            raise

    @staticmethod
    def _convert_strings_to_paths(config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively convert string paths to Path objects."""
        result = {}
        for key, value in config_data.items():
            if isinstance(value, dict):
                result[key] = AppConfig._convert_strings_to_paths(value)
            elif isinstance(value, list):
                result[key] = [
                    AppConfig._convert_strings_to_paths(v) if isinstance(v, dict) else v
                    for v in value
                ]
            elif isinstance(value, str) and "_dir" in key:
                result[key] = Path(value).expanduser().resolve()
            elif key == "file" and isinstance(value, str):
                result[key] = Path(value).expanduser().resolve()
            else:
                result[key] = value
        return result


# Global configuration instance
config = AppConfig.load_defaults()


def load_config(config_path: Optional[Union[str, Path]] = None) -> AppConfig:
    """Load configuration from file and update the global config.

    Args:
        config_path: Path to the configuration file. If None, uses the default location.

    Returns:
        The loaded AppConfig instance.
    """
    global config

    if config_path is None:
        config_path = DEFAULT_CONFIG_DIR / "config.yaml"

    config_path = Path(config_path).expanduser().resolve()

    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return config

    try:
        logger.info(f"Loading configuration from {config_path}")

        # Load the raw YAML data first
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        logger.debug(f"Raw config data: {config_data}")

        # Process environment variables in the config data
        config_data = AppConfig._process_environment_variables(config_data)

        # Convert string paths to Path objects
        config_data = AppConfig._convert_strings_to_paths(config_data)

        # Update the global config with the loaded data
        for field, value in config_data.items():
            if field == 'llm_providers':
                # Handle LLM providers specially
                if not isinstance(value, dict):
                    logger.warning(f"Invalid LLM providers configuration: {value}")
                    continue

                # Store raw providers for reference
                config._raw_llm_providers = value.copy()

                # Process each provider
                processed_providers = {}
                for provider_name, provider_data in value.items():
                    if not isinstance(provider_data, dict):
                        logger.warning(
                            f"Invalid provider config for {provider_name}: {provider_data}"
                        )
                        continue

                    try:
                        # Ensure the provider has a name
                        provider_data = provider_data.copy()
                        if 'name' not in provider_data:
                            provider_data['name'] = provider_name

                        # Create and store the provider config
                        processed_providers[provider_name] = LLMProviderConfig(
                            **provider_data
                        )
                        logger.debug(f"Loaded LLM provider: {provider_name}")
                    except Exception as e:
                        logger.error(
                            f"Failed to load LLM provider {provider_name}: {e}",
                            exc_info=True,
                        )

                # Update the providers
                config.llm_providers = processed_providers
                logger.info(f"Loaded {len(processed_providers)} LLM providers")

            else:
                # Update other fields
                try:
                    setattr(config, field, value)
                except Exception as e:
                    logger.warning(f"Failed to set config field {field}: {e}")

        logger.info(f"Successfully loaded configuration from {config_path}")
        logger.info(
            f"Default LLM provider: {getattr(config, 'default_llm_provider', 'not set')}"
        )
        logger.info(
            f"Available LLM providers: {list(getattr(config, 'llm_providers', {}).keys())}"
        )

        return config
    except Exception as e:
        logger.error(
            f"Failed to load configuration from {config_path}: {e}", exc_info=True
        )
        return config


def save_config(config_path: Optional[Union[str, Path]] = None) -> None:
    """Save the current configuration to a file.

    Args:
        config_path: Path to save the configuration to. If None, uses the current config path.
    """
    if config_path is not None:
        config_path = Path(config_path).expanduser().resolve()

    config.save(config_path)

    # Job sources
    job_sources: Dict[str, JobSource] = Field(
        default_factory=dict, description="Configured job sources"
    )

    # Provider configuration
    provider: ProviderConfig = Field(
        default_factory=lambda: ProviderConfig(type="ollama"),
        description="Configuration for the analysis provider",
    )

    # Logging
    logging_level: str = "INFO"
    log_file: Optional[Path] = None

    # Web server
    web_host: str = "0.0.0.0"
    web_port: int = 8000
    web_debug: bool = False

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: InitSettingsSource,
        env_settings: EnvSettingsSource,
        dotenv_settings: DotEnvSettingsSource,
        file_secret_settings: SecretsSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )

    @field_validator("data_dir", "log_file", mode='before')
    @classmethod
    def resolve_paths(cls, v: Optional[Union[str, Path]], info: Any) -> Optional[Path]:
        """Resolve file paths to absolute paths."""
        if v is None:
            return None

        path = Path(str(v)).expanduser().resolve()
        if path != path.parent and not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> "AppConfig":
        """Load configuration from a YAML file.

        Args:
            file_path: Path to the YAML configuration file.

        Returns:
            An instance of AppConfig with settings from the file.
        """
        file_path = Path(file_path).expanduser().resolve()
        with open(file_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        return cls(**config_data)

    def to_yaml(self, file_path: Union[str, Path]) -> None:
        """Save configuration to a YAML file.

        Args:
            file_path: Path to save the YAML configuration to.
        """
        file_path = Path(file_path).expanduser().resolve()
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert Path objects to strings for YAML serialization
        config_dict = self.dict()
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)

        with open(file_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        """Update configuration from a dictionary.

        Args:
            data: Dictionary of configuration values to update.
        """
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)


def load_config(config_path: Optional[Union[str, Path]] = None) -> AppConfig:
    """Load application configuration.

    Args:
        config_path: Optional path to a YAML config file.

    Returns:
        Loaded AppConfig instance.
    """
    import yaml

    # Default config paths to check
    default_paths = [
        Path("config.yaml"),
        Path("~/.myjobspyai/config.yaml").expanduser(),
        Path("/etc/myjobspyai/config.yaml"),
    ]

    # If config_path is provided, use it exclusively
    if config_path:
        config_path = Path(config_path).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)
            return AppConfig.model_validate(config_data)

    # Otherwise, try default paths
    for path in default_paths:
        if path.exists():
            with open(path, "r") as f:
                config_data = yaml.safe_load(f)
                return AppConfig.model_validate(config_data)

    # If no config file found, use defaults
    return AppConfig()


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to the config.

    Args:
        config: The configuration dictionary.

    Returns:
        Updated configuration dictionary.
    """
    # Example: MYJOBSPYAI_LLM_API_KEY -> llm.api_key
    prefix = "MYJOBSPYAI_"

    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue

        # Convert ENV_VAR to config path (e.g., LLM_API_KEY -> llm.api_key)
        path = key[len(prefix) :].lower().split("_")

        # Navigate to the target in the config
        current = config
        for part in path[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the value
        current[path[-1]] = value

    return config
