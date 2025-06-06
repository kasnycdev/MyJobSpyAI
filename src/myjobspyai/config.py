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
        ..., description="Type of provider (e.g., 'openai', 'ollama', 'langchain')"
    )
    model: str = Field("gpt-3.5-turbo", description="Default model to use")
    api_key: Optional[str] = Field(None, description="API key for the provider")
    base_url: Optional[str] = Field(None, description="Base URL for the API")
    timeout: int = Field(600, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries")
    temperature: float = Field(0.7, description="Default temperature")
    max_tokens: Optional[int] = Field(
        None, description="Maximum number of tokens to generate"
    )

    class Config:
        extra = "allow"  # Allow provider-specific settings


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
    llm_providers: Dict[str, LLMProviderConfig] = Field(
        default_factory=dict, description="Configured LLM providers"
    )

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
        if config_path is None:
            config_path = DEFAULT_CONFIG_DIR / "config.yaml"

        config_path = Path(config_path).expanduser().resolve()

        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return cls()

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f) or {}

            # Convert string paths to Path objects
            config_data = cls._convert_strings_to_paths(config_data)

            # Load environment variables from .env file if present
            dotenv_path = config_path.parent / ".env"
            if dotenv_path.exists():
                from dotenv import load_dotenv

                load_dotenv(dotenv_path)

            return cls(**config_data)

        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return cls()

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
    if config_path is not None:
        config_path = Path(config_path).expanduser().resolve()

    config = AppConfig.from_file(config_path)
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
