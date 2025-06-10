"""Configuration validation for LLM providers.

This module provides configuration validation for LLM providers using Pydantic.
It ensures that provider configurations are valid before they are used to
initialize provider instances.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from myjobspyai.services.llm.exceptions import LLMProviderConfigError


class BaseProviderConfig(BaseModel):
    """Base configuration model for LLM providers.

    This class defines common configuration options that are supported by most
    LLM providers. Provider-specific configurations should inherit from this
    class and add provider-specific fields.
    """

    model: str = Field(
        ...,
        description=(
            "The model to use for text generation. "
            "Example: 'gpt-4', 'claude-2', etc."
        ),
        min_length=1,
    )
    provider: str = Field(
        ...,
        description="The name of the LLM provider",
        min_length=1,
    )
    api_key: Optional[str] = Field(
        None,
        description=(
            "API key for the provider. " "Can also be set via environment variables."
        ),
        min_length=1,
    )
    base_url: Optional[str] = Field(
        None,
        description=(
            "Base URL for the provider's API. "
            "Used for self-hosted or custom endpoints."
        ),
        min_length=1,
    )
    timeout: int = Field(
        60,
        description="Request timeout in seconds",
        gt=0,
        le=600,
    )
    max_retries: int = Field(
        3,
        description="Maximum number of retries for failed requests",
        ge=0,
        le=10,
    )
    request_timeout: int = Field(
        60,
        description="Timeout for individual requests in seconds",
        gt=0,
        le=600,
    )
    connection_pool_size: int = Field(
        10,
        description="Maximum number of concurrent connections in the connection pool",
        gt=0,
        le=100,
    )
    temperature: float = Field(
        0.7,
        description="Sampling temperature for text generation",
        ge=0.0,
        le=2.0,
    )
    max_tokens: int = Field(
        1000,
        description=(
            "Maximum number of tokens to generate. " "Must be between 1 and 8192."
        ),
        gt=0,
        le=8192,
    )

    @field_validator("*", mode="before")
    @classmethod
    def validate_not_none(cls, v: Any, info: ValidationInfo) -> Any:
        """Validate that required fields are not None."""
        if v is None and info.field_name in cls.model_fields:
            field = cls.model_fields[info.field_name]
            if field.is_required():
                raise ValueError(f"{info.field_name} must not be None")
        return v


class OpenAIConfig(BaseProviderConfig):
    """Configuration for OpenAI provider."""

    organization: Optional[str] = Field(
        None,
        description="OpenAI organization ID",
        min_length=1,
    )
    project: Optional[str] = Field(
        None,
        description="OpenAI project ID",
        min_length=1,
    )


class AnthropicConfig(BaseProviderConfig):
    """Configuration for Anthropic provider."""

    max_retries: int = Field(
        5,
        description="Maximum number of retries for failed requests",
        ge=0,
        le=10,
    )


def validate_config(provider_name: str, config: dict[str, Any]) -> dict[str, Any]:
    """Validate provider configuration.

    Args:
        provider_name: Name of the provider (e.g., 'openai', 'anthropic')
        config: Configuration dictionary to validate

    Returns:
        Validated configuration as a dictionary

    Raises:
        LLMProviderConfigError: If the configuration is invalid
    """
    try:
        # Create a copy to avoid modifying the original
        config = config.copy()

        # Set provider name if not specified
        if "provider" not in config:
            config["provider"] = provider_name.lower()

        # Select the appropriate config class based on provider
        config_class = {
            "openai": OpenAIConfig,
            "anthropic": AnthropicConfig,
        }.get(provider_name.lower(), BaseProviderConfig)

        # Validate and convert the config
        validated = config_class(**config).model_dump(exclude_none=True)
        return validated

    except Exception as e:
        raise LLMProviderConfigError(
            f"Invalid configuration for provider '{provider_name}': {e}"
        ) from e
