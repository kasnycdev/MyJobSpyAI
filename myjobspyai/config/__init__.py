"""
Centralized configuration management for MyJobSpyAI.

This module provides a single source of truth for all application settings,
with support for environment variables, .env files, and direct configuration.
"""
from __future__ import annotations

import logging
import yaml
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
        default="ollama",
        description="LLM provider to use (openai, ollama, gemini, etc.)",
        examples=["ollama", "openai", "gemini"],
        pattern=r"^(openai|ollama|gemini)$"
    )
    model: str = Field(
        ...,  # Required field, no default
        description="Model name to use (must be specified in config.yaml)",
        examples=["deepseek-r1:1.5b", "llama3:instruct", "llama2", "gemini-pro"]
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
            "Controls randomness in the model's output. Higher values (e.g., 0.8) make the output "
            "more random, while lower values (e.g., 0.2) make it more deterministic."
        )
    )
    max_tokens: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "Maximum number of tokens to generate. Note that the model's context window "
            "may impose additional limits."
        )
    )
    top_p: float = Field(
        default=1.0,
        gt=0.0,
        le=1.0,
        description=(
            "Nucleus sampling parameter. The model considers the smallest set of tokens "
            "whose cumulative probability is at least top_p."
        )
    )
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description=(
            "Penalty for using frequent tokens. Positive values decrease the likelihood "
            "of repeating tokens that appear frequently in the training data."
        )
    )
    presence_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description=(
            "Penalty for using new tokens. Positive values increase the likelihood "
            "of the model generating new tokens rather than repeating existing ones."
        )
    )
    
    # Provider-specific configurations
    openai: Dict[str, Any] = Field(
        default_factory=dict,
        description="OpenAI-specific configuration"
    )
    ollama: Dict[str, Any] = Field(
        default_factory=dict,
        description="Ollama-specific configuration"
    )
    gemini: Dict[str, Any] = Field(
        default_factory=dict,
        description="Gemini-specific configuration"
    )
    
    # Streaming configuration
    streaming: Dict[str, Any] = Field(
        default_factory=dict,
        description="Streaming configuration for the LLM provider"
    )
    
    @model_validator(mode='after')
    def validate_provider_config(self) -> 'LLMConfig':
        """Validate provider-specific configuration."""
        # Ensure the selected provider is valid
        if self.provider not in ["openai", "ollama", "gemini"]:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
            
        # Ensure API key is provided for cloud providers
        if self.provider in ["openai", "gemini"] and not self.api_key:
            env_var = "OPENAI_API_KEY" if self.provider == "openai" else "GOOGLE_API_KEY"
            raise ValueError(
                f"API key is required for {self.provider} provider. "
                f"Please set the {env_var} environment variable or provide it in the config."
            )
            
        # Ensure model is specified
        if not self.model:
            raise ValueError("LLM model must be specified in the configuration")
            
        return self
    
    class Config:
        json_schema_extra = {
            "example": {
                "provider": "ollama",
                "model": "llama3:instruct",
                "base_url": "http://localhost:11434",
                "timeout": 300,
                "max_retries": 3,
                "temperature": 0.7,
                "max_tokens": 4096,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "streaming": {
                    "enabled": True,
                    "chunk_size": 128,
                    "timeout": 30,
                    "buffer_size": 5
                },
                "ollama": {
                    "keep_alive": "5m",
                    "num_ctx": 512,
                    "num_thread": 2,
                    "gpu_layers": 1
                }
            }
        }

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

class AnalysisConfig(BaseModel):
    """Configuration for text analysis and processing.
    
    This configuration controls how job postings and resumes are processed and analyzed.
    """
    # Text chunking settings
    chunk_size: int = Field(
        default=3000,
        gt=0,
        description="Size of text chunks for processing (in characters)."
    )
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Overlap between chunks (in characters)."
    )
    semantic_chunking: bool = Field(
        default=True,
        description="Whether to use semantic chunking when possible."
    )
    max_parallel_chunks: int = Field(
        default=5,
        gt=0,
        le=20,
        description="Maximum number of chunks to process in parallel."
    )
    
    # Document loaders configuration
    document_loaders: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for document loaders."
    )
    
    # Prompts configuration
    prompts: Dict[str, str] = Field(
        default_factory=dict,
        description="Paths to prompt templates."
    )
    
    # Suitability analysis
    suitability: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for job suitability analysis."
    )
    
    # Resume analysis
    resume: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for resume analysis."
    )
    
    # Job analysis
    job: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for job analysis."
    )
    
    # Matching configuration
    matching: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for job-resume matching."
    )
    
    # Prompt engineering
    prompt_engineering: Dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for prompt engineering."
    )
    
    # Text analysis
    min_section_length: int = Field(
        default=100,
        gt=0,
        description="Minimum length of text sections for analysis (in characters)."
    )
    max_section_length: int = Field(
        default=1000,
        gt=0,
        description="Maximum length of text sections for analysis (in characters)."
    )
    extract_entities: bool = Field(
        default=True,
        description="Whether to extract named entities from text."
    )
    extract_keyphrases: bool = Field(
        default=True,
        description="Whether to extract keyphrases from text."
    )
    analyze_sentiment: bool = Field(
        default=False,
        description="Whether to perform sentiment analysis on text."
    )
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score (0-1) for considering two pieces of text as matching."
    )
    
    class Config:
        extra = "allow"
        json_schema_extra = {
            "example": {
                "chunk_size": 3000,
                "chunk_overlap": 200,
                "semantic_chunking": True,
                "max_parallel_chunks": 5,
                "min_section_length": 100,
                "max_section_length": 1000,
                "extract_entities": True,
                "extract_keyphrases": True,
                "analyze_sentiment": False,
                "similarity_threshold": 0.7
            }
        }


class ModelLoggingConfig(BaseModel):
    """Configuration for model-specific logging."""
    
    enabled: bool = Field(
        default=True,
        description="Enable model logging"
    )
    log_dir: Path = Field(
        default=Path("logs/models"),
        description="Directory to store model logs"
    )
    log_inputs: bool = Field(
        default=False,
        description="Log model inputs (be cautious with sensitive data)"
    )
    log_outputs: bool = Field(
        default=True,
        description="Log model outputs"
    )
    log_latency: bool = Field(
        default=True,
        description="Log model inference latency"
    )
    log_errors: bool = Field(
        default=True,
        description="Log model errors and exceptions"
    )
    log_mode: str = Field(
        default="overwrite",
        description="Log file mode: 'overwrite' to replace file each run, 'append' to add to existing file",
        pattern=r"^(overwrite|append)$"
    )
    max_log_size_mb: int = Field(
        default=100,
        ge=1,
        description="Maximum log file size in MB before rotation (only used when log_mode is 'append')"
    )
    backup_count: int = Field(
        default=5,
        ge=0,
        description="Number of backup log files to keep (only used when log_mode is 'append')"
    )
    
    class Config:
        extra = "forbid"
        json_schema_extra = {
            "example": {
                "enabled": True,
                "log_dir": "logs/models",
                "log_inputs": False,
                "log_outputs": True,
                "log_latency": True,
                "log_errors": True,
                "max_log_size_mb": 100,
                "backup_count": 5
            }
        }


class OutputConfig(BaseModel):
    """Configuration for output files and formats."""
    
    # General output files
    scraped_jobs_file: Path = Field(
        default=Path("data/scraped_jobs.json"),
        description="Path to save scraped jobs data"
    )
    analysis_results_file: Path = Field(
        default=Path("data/analysis_results.json"),
        description="Path to save analysis results"
    )
    
    # Application logging
    log_file: Path = Field(
        default=Path("logs/application.log"),
        description="Path to the application log file"
    )
    error_file: Path = Field(
        default=Path("logs/error.log"),
        description="Path to the error log file"
    )
    debug_file: Path = Field(
        default=Path("logs/debug.log"),
        description="Path to the debug log file"
    )
    
    # Model-specific logging
    model_logging: ModelLoggingConfig = Field(
        default_factory=ModelLoggingConfig,
        description="Configuration for model-specific logging"
    )
    
    # Per-model logging overrides
    model_log_overrides: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="""
        Per-model logging configuration overrides.
        Key: model name or pattern
        Value: Dictionary of ModelLoggingConfig fields to override
        """,
        example={
            "gpt-4": {"log_inputs": False, "log_outputs": True},
            "llama3": {"enabled": True, "log_dir": "logs/models/llama3"}
        }
    )
    
    class Config:
        extra = "forbid"
        json_schema_extra = {
            "example": {
                "scraped_jobs_file": "data/scraped_jobs.json",
                "analysis_results_file": "data/analysis_results.json",
                "log_file": "logs/application.log",
                "error_file": "logs/error.log",
                "debug_file": "logs/debug.log",
                "model_logging": {
                    "enabled": True,
                    "log_dir": "logs/models",
                    "log_inputs": False,
                    "log_outputs": True,
                    "log_latency": True,
                    "log_errors": True,
                    "max_log_size_mb": 100,
                    "backup_count": 5
                },
                "model_log_overrides": {
                    "gpt-4": {"log_inputs": False, "log_outputs": True},
                    "llama3": {"enabled": True, "log_dir": "logs/models/llama3"}
                }
            }
        }


class ScrapingConfig(BaseModel):
    """Configuration for job scraping parameters.
    
    This configuration maps to the parameters accepted by JobSpy's scrape_jobs function.
    """
    
    # Basic search parameters
    default_sites: list[str] = Field(
        default_factory=lambda: ["linkedin", "indeed", "zip_recruiter", "glassdoor", "google", "naukri"],
        description="List of job sites to search. Supported values: 'linkedin', 'indeed', 'zip_recruiter', 'glassdoor', 'google', 'naukri', 'bayt'"
    )
    
    default_results_limit: int = Field(
        default=10, 
        ge=1, 
        le=1000,
        description="Maximum number of results to return per site (1-1000)"
    )
    
    default_days_old: Optional[int] = Field(
        default=30, 
        ge=0,
        description="Maximum age of job postings in days. Set to None for no age limit."
    )
    
    default_country_indeed: str = Field(
        default="usa",
        description="Default country for Indeed searches (e.g., 'usa', 'uk', 'ca')"
    )
    
    # Location parameters
    distance: int = Field(
        default=50,
        ge=0,
        description="Search radius in miles from the specified location"
    )
    
    is_remote: bool = Field(
        default=False,
        description="Whether to only show remote jobs"
    )
    
    # Job type and filters
    job_type: Optional[str] = Field(
        default=None,
        description="Filter by job type (e.g., 'fulltime', 'parttime', 'contract', 'internship')"
    )
    
    easy_apply: Optional[bool] = Field(
        default=None,
        description="Whether to only show jobs with easy apply"
    )
    
    # Advanced configuration
    proxies: Optional[list[str]] = Field(
        default=None,
        description="List of proxy servers to use for scraping (e.g., ['http://user:pass@proxy:port'])"
    )
    
    ca_cert: Optional[str] = Field(
        default=None,
        description="Path to CA certificate file for SSL verification with proxies"
    )
    
    # LinkedIn-specific parameters
    linkedin_company_ids: list[int] = Field(
        default_factory=list,
        description="List of LinkedIn company IDs to filter job results"
    )
    
    linkedin_fetch_description: bool = Field(
        default=False,
        description="Whether to fetch full job descriptions for LinkedIn (slower but more detailed)"
    )
    
    # Google-specific parameters
    google_search_term: Optional[str] = Field(
        default=None,
        description="Custom search term to use for Google job search"
    )
    
    # Result handling
    offset: int = Field(
        default=0,
        ge=0,
        description="Number of results to skip (for pagination)"
    )
    
    description_format: str = Field(
        default="markdown",
        pattern=r"^(markdown|html)$",
        description="Format for job descriptions ('markdown' or 'html')"
    )
    
    enforce_annual_salary: bool = Field(
        default=False,
        description="Whether to convert all salary information to annual amounts"
    )
    
    # Logging and debugging
    verbose: int = Field(
        default=0,
        ge=0,
        le=2,
        description="Verbosity level (0=errors only, 1=info, 2=debug)"
    )
    
    # Processing configuration
    chunk_size: int = Field(
        default=3000,
        gt=0,
        description="Size of text chunks for processing (in characters)."
    )
    
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        description="Overlap between chunks (in characters)."
    )
    
    semantic_chunking: bool = Field(
        default=True,
        description=(
            "Whether to use semantic chunking (more accurate but slower). "
            "If False, uses simple text splitting."
        )
    )
    
    max_parallel_chunks: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of chunks to process in parallel."
    )
    
    class Config:
        extra = "forbid"  # Prevent extra fields not defined in the model
        json_schema_extra = {
            "example": {
                "default_sites": ["linkedin", "indeed"],
                "default_results_limit": 20,
                "default_days_old": 30,
                "default_country_indeed": "usa",
                "distance": 25,
                "is_remote": True,
                "job_type": "fulltime",
                "verbose": 1,
                "chunk_size": 3000,
                "chunk_overlap": 200,
                "semantic_chunking": True,
                "max_parallel_chunks": 5
            }
        }

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
            "Behavior with existing files depends on log_mode setting."
        ),
        examples=["logs/app.log", "/var/log/myjobspyai.log"]
    )
    
    log_mode: str = Field(
        default="overwrite",
        description=(
            "Log file mode: 'overwrite' to replace file each run, 'append' to add to existing file. "
            "When 'overwrite', max_size and backup_count are ignored."
        ),
        pattern=r"^(overwrite|append)$"
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
        description=(
            "Maximum log file size in bytes before rotation. "
            "Only used when log_mode is 'append'."
        )
    )
    
    backup_count: int = Field(
        default=5,
        ge=0,
        description=(
            "Number of backup log files to keep. "
            "Only used when log_mode is 'append'."
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
    """Main application configuration.
    
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
    
    # Model configuration
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
    
    # Application settings
    debug: bool = Field(
        default=False,
        description=(
            "Enable debug mode. When True, more verbose logging will be "
            "enabled and sensitive information may be exposed in logs."
        )
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        pattern=r"^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$"
    )
    environment: str = Field(
        default="production",
        description=(
            "Runtime environment. Affects things like logging levels and "
            "default configurations. Should be one of: development, staging, production"
        ),
        pattern=r"^(development|staging|production)$"
    )
    
    # LLM configuration
    llm: LLMConfig = Field(
        default_factory=LLMConfig,
        description="Configuration for the Language Model provider"
    )
    
    # Caching configuration
    cache: CacheConfig = Field(
        default_factory=CacheConfig,
        description="Caching configuration for API responses and embeddings"
    )
    
    # Logging configuration
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration"
    )
    
    # Analysis configuration
    analysis: AnalysisConfig = Field(
        default_factory=AnalysisConfig,
        description="Configuration for text analysis and processing"
    )
    
    # Scraping configuration
    scraping: ScrapingConfig = Field(
        default_factory=ScrapingConfig,
        description="Configuration for job scraping"
    )
    
    # Output configuration
    output: OutputConfig = Field(
        default_factory=OutputConfig,
        description="Configuration for output files and formats"
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
    
    # Request settings
    request_timeout: float = Field(
        default=60.0,
        gt=0,
        le=600,  # 10 minutes max
        description=(
            "Default timeout in seconds for HTTP requests. "
            "Increase this if you have a slow connection or are working with large models."
        )
    )
    
    # Paths
    data_dir: Path = Field(
        default=Path("data"),
        description=(
            "Base directory for application data. Will be created if it doesn't exist."
        )
    )
    
    @model_validator(mode='after')
    def validate_config_values(self) -> 'AppConfig':
        """Validate configuration values after model initialization."""
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate LLM provider configuration
        if not self.llm.provider:
            raise ValueError("LLM provider must be specified in the configuration")
            
        if not self.llm.model:
            raise ValueError("LLM model must be specified in the configuration")
        
        # Validate analysis configuration
        if self.analysis.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")
            
        if self.analysis.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be greater than or equal to 0")
            
        if self.analysis.chunk_overlap >= self.analysis.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
            
        if self.analysis.max_parallel_chunks <= 0:
            raise ValueError("max_parallel_chunks must be greater than 0")
        
        return self
    
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

def configure(config_path: Optional[Union[str, Path]] = None, **overrides) -> AppConfig:
    """Configure the application settings.
    
    Args:
        config_path: Path to the configuration file. If None, looks for config.yaml in the project root.
        **overrides: Configuration overrides.
        
    Returns:
        The configured AppConfig instance.
    """
    global settings
    
    # Default config file path
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config.yaml"
    else:
        config_path = Path(config_path)
    
    # Load configuration from file if it exists
    config_data = {}
    if config_path.exists():
        logger.info(f"Loading configuration from {config_path}")
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f) or {}
    else:
        logger.warning(f"Configuration file not found at {config_path}, using default settings")
    
    # Apply overrides
    if overrides:
        config_data.update(overrides)
    
    # Create and validate config
    settings = AppConfig(**config_data)
    
    # Log the loaded configuration
    logger.info("Configuration loaded successfully")
    logger.debug("LLM provider: %s", settings.llm.provider)
    logger.debug("LLM model: %s", settings.llm.model)
    
    # Check if we have Ollama-specific settings
    if hasattr(settings.llm, 'ollama') and settings.llm.ollama:
        logger.debug("Ollama config: %s", settings.llm.ollama)
        if hasattr(settings.llm.ollama, 'model'):
            logger.debug("Ollama model: %s", settings.llm.ollama.model)
    
    return settings

# Global configuration instance (will be initialized when configure() is called)
settings: Optional[AppConfig] = None

# Initialize with default configuration if not already configured
if settings is None:
    configure()

def _deep_update(d: Dict[str, Any], u: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a dictionary with another dictionary."""
    for k, v in u.items():
        if isinstance(v, dict) and k in d and isinstance(d[k], dict):
            d[k] = _deep_update(d[k], v)
        else:
            d[k] = v
    return d

def parse_cli_args() -> Dict[str, Any]:
    """Parse command line arguments for configuration overrides.
    
    Returns:
        Dict[str, Any]: Parsed command line arguments as a dictionary
    """
    import argparse
    import json
    from typing import List, Tuple
    
    def parse_key_value(parts: List[str]) -> Tuple[str, Any]:
        """Parse a key=value pair into a key and value."""
        if '=' not in parts[0]:
            # Handle boolean flags (e.g., --debug)
            key = parts[0].lstrip('-')
            return key, True
            
        key, value = parts[0].split('=', 1)
        key = key.lstrip('-')
        
        # Try to parse as JSON first (for complex values)
        try:
            return key, json.loads(value)
        except (json.JSONDecodeError, ValueError):
            # Handle other simple types
            if value.lower() == 'true':
                return key, True
            elif value.lower() == 'false':
                return key, False
            elif value.isdigit():
                return key, int(value)
            try:
                return key, float(value)
            except ValueError:
                return key, value
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='MyJobSpyAI Configuration')
    
    # Add common arguments
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--log-level', type=str, help='Set log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    
    # Parse known args first to get the config file path
    args, remaining_argv = parser.parse_known_args()
    
    # Parse remaining arguments as key=value pairs
    cli_args = {}
    i = 0
    while i < len(remaining_argv):
        if remaining_argv[i].startswith('--'):
            key, value = parse_key_value([remaining_argv[i][2:]] + remaining_argv[i+1:i+2])
            # If we consumed a value, skip the next argument
            if '=' not in remaining_argv[i]:
                i += 1
            cli_args[key] = value
        i += 1
    
    # Add parsed args to the result
    if args.config:
        cli_args['config'] = args.config
    if args.debug:
        cli_args['debug'] = True
    if args.log_level:
        cli_args['log_level'] = args.log_level.upper()
    
    return cli_args

def configure(config_path: Optional[Union[str, Path]] = None, **overrides) -> AppConfig:
    """
    Configure application settings with the following precedence order (highest to lowest):
    1. Command-line arguments (passed as **overrides or parsed from sys.argv)
    2. Explicit function arguments (config_path and **overrides)
    3. Configuration file (YAML/JSON, from config_path or --config argument)
    4. Environment variables (from .env file or system environment with MYJOBS_ prefix)
    5. Hard-coded default values (lowest precedence)
    
    Args:
        config_path: Path to a configuration file (YAML or JSON). If None, will check --config argument.
        **overrides: Settings to override (takes precedence over config file but not CLI args)
        
    Returns:
        AppConfig: The configured settings instance
        
    Example:
        ```python
        # Load with command-line overrides (highest precedence)
        # python script.py --debug llm.provider=ollama llm.model=llama3:instruct
        
        # Load from a specific config file
        # python script.py --config config/production.yaml
        
        # Combine file with overrides
        # python script.py --config config/staging.yaml --debug
        
        # Or programmatically:
        configure(debug=True, llm={"provider": "ollama"})
        ```
    """
    global settings
    
    try:
        # 1. Parse command line arguments (highest precedence)
        cli_args = parse_cli_args()
        
        # 2. Use config_path from CLI args if not provided
        if config_path is None and 'config' in cli_args:
            config_path = cli_args.pop('config')
        
        # 3. Load from configuration file if provided
        file_data: Dict[str, Any] = {}
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                suffix = config_path.suffix.lower()
                try:
                    if suffix in ('.yaml', '.yml'):
                        import yaml
                        with open(config_path, 'r', encoding='utf-8') as f:
                            file_data = yaml.safe_load(f) or {}
                    elif suffix == '.json':
                        import json
                        with open(config_path, 'r', encoding='utf-8') as f:
                            file_data = json.load(f) or {}
                    else:
                        logger.warning(
                            f"Unsupported config file format: {suffix}. "
                            "Expected .yaml, .yml, or .json"
                        )
                except Exception as e:
                    logger.warning(f"Failed to load config file {config_path}: {e}")
        
        # 4. Merge configurations in order of increasing precedence:
        #    a. Start with file data (lowest in our merge order)
        merged_data = {**file_data}
        
        #    b. Apply explicit function arguments (medium precedence)
        if overrides:
            merged_data = _deep_update(merged_data, overrides)
        
        #    c. Apply CLI arguments (highest precedence)
        if cli_args:
            # Convert dot notation to nested dict (e.g., 'llm.provider' -> {'llm': {'provider': value}})
            cli_nested = {}
            for key, value in cli_args.items():
                if '.' in key:
                    # Handle nested keys (e.g., 'llm.provider')
                    parts = key.split('.')
                    current = cli_nested
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = value
                else:
                    cli_nested[key] = value
            
            # Apply CLI overrides
            merged_data = _deep_update(merged_data, cli_nested)
        
        # 5. Create and validate the config
        settings = AppConfig.model_validate(merged_data)
        
        # Log the configuration source
        source = f"file: {config_path}" if config_path and Path(config_path).exists() else "default values"
        logger.info(f"Loaded configuration from {source}")
        if cli_args:
            logger.debug(f"CLI overrides: {cli_args}")
        
        # Log important configuration details
        logger.debug(f"Active LLM provider: {settings.llm.provider}")
        logger.debug(f"Active model: {settings.llm.model}")
        if settings.llm.base_url:
            logger.debug(f"Base URL: {settings.llm.base_url}")
        
        return settings
        
    except PydanticValidationError as e:
        logger.error(f"Configuration validation error: {e}")
        raise ConfigurationError(f"Invalid configuration: {e}") from e
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}", exc_info=True)
        raise ConfigurationError(f"Failed to load configuration: {e}") from e

# Default config file path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"

# Initialize with default settings and load config.yaml if it exists
if DEFAULT_CONFIG_PATH.exists():
    configure(DEFAULT_CONFIG_PATH)
else:
    logger.warning(f"Default config file not found at {DEFAULT_CONFIG_PATH}. Using default settings.")
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
