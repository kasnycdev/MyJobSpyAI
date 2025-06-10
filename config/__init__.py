"""Configuration management for MyJobSpyAI."""

import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment types."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LoggingConfig(BaseModel):
    """Logging configuration."""

    log_dir: str = "logs"
    log_level: str = "INFO"
    log_file_mode: str = "a"
    rolling_strategy: str = "size"
    max_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    files: Dict[str, Dict[str, str]] = Field(
        default_factory=lambda: {
            "app": {"path": "app.log", "level": "INFO"},
            "debug": {"path": "debug.log", "level": "DEBUG"},
            "error": {"path": "error.log", "level": "WARNING"},
            "llm": {"path": "llm.log", "level": "INFO"},
        }
    )


class DatabaseConfig(BaseModel):
    """Database configuration."""

    host: str = "localhost"
    port: int = 5432
    name: str = "myjobspyai"
    user: str = "postgres"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    echo: bool = False


class ScrapingConfig(BaseModel):
    """Web scraping configuration."""

    timeouts: Dict[str, int] = Field(
        default_factory=lambda: {
            "default": 30,
            "naukri": 10,
            "indeed": 20,
            "linkedin": 25,
        }
    )
    retry_attempts: int = 3
    retry_delay: int = 5
    user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


class LLMConfig(BaseModel):
    """LLM configuration."""

    provider: str = "openai"
    model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    api_key: str = ""


class APIConfig(BaseModel):
    """API configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    secret_key: str = "your-secret-key-here"
    cors_origins: list[str] = ["http://localhost:3000"]


class Settings(BaseSettings):
    """Application settings."""

    environment: Environment = Environment.DEVELOPMENT
    debug: bool = False

    # Nested configs
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    scraping: ScrapingConfig = Field(default_factory=ScrapingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    model_config = SettingsConfigDict(
        env_nested_delimiter='__',
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
    )

    @classmethod
    def from_yaml(cls, file_path: Union[str, Path]) -> 'Settings':
        """Load settings from a YAML file."""
        with open(file_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data)

    def to_yaml(self, file_path: Union[str, Path]) -> None:
        """Save settings to a YAML file."""
        with open(file_path, 'w') as f:
            yaml.dump(self.model_dump(exclude_none=True), f, default_flow_style=False)


# Global settings instance
settings = Settings()


def load_settings() -> Settings:
    """Load settings with overrides from environment variables."""
    return Settings()
