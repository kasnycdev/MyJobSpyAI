"""Factory for creating LLM provider instances.

This module provides a factory pattern for creating and managing different
LLM (Large Language Model) provider instances. It handles provider registration,
instantiation, and error handling in a consistent way.
"""

import logging
from typing import Any, TypeVar

from myjobspyai.services.llm.base import BaseLLMProvider, LLMError
from myjobspyai.services.llm.config import validate_config
from myjobspyai.services.llm.exceptions import LLMProviderConfigError

T = TypeVar('T', bound=BaseLLMProvider)

# Import providers conditionally to avoid circular imports
ANTHROPIC_AVAILABLE = False
OPENAI_AVAILABLE = False

# Try to import Anthropic provider
try:
    from myjobspyai.services.llm.providers.anthropic import (  # noqa: F401
        AnthropicClient,
    )

    ANTHROPIC_AVAILABLE = True
except ImportError:
    pass

# Try to import OpenAI provider
try:
    from myjobspyai.services.llm.providers.openai import OpenAIClient  # noqa: F401

    OPENAI_AVAILABLE = True
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Registry of available LLM providers
_PROVIDER_REGISTRY: dict[str, type[BaseLLMProvider]] = {}

# Register available providers
if OPENAI_AVAILABLE:
    _PROVIDER_REGISTRY["openai"] = OpenAIClient  # type: ignore
if ANTHROPIC_AVAILABLE:
    _PROVIDER_REGISTRY["anthropic"] = AnthropicClient  # type: ignore


def register_provider(name: str, provider_class: type[BaseLLMProvider]) -> None:
    """Register a new LLM provider.

    Args:
        name: Name of the provider (e.g., 'openai', 'anthropic')
        provider_class: The provider class to register

    Raises:
        ValueError: If the provider name is already registered
    """
    name_lower = name.lower()
    if name_lower in _PROVIDER_REGISTRY:
        raise ValueError(f"Provider '{name}' is already registered")
    _PROVIDER_REGISTRY[name_lower] = provider_class


def get_available_providers() -> list[str]:
    """Get a list of available provider names.

    Returns:
        List of provider names
    """
    return list(_PROVIDER_REGISTRY.keys())


def create_provider(
    name: str,
    config: dict[str, Any],
    validate: bool = True,
    provider_class: type[T] | None = None,
) -> T:
    """Create an instance of the specified LLM provider.

    Args:
        name: Name of the provider (e.g., 'openai', 'anthropic')
        config: Configuration dictionary for the provider
        validate: Whether to validate the configuration
        provider_class: Optional provider class to use instead of looking up by name

    Returns:
        An instance of the specified LLM provider

    Raises:
        LLMProviderConfigError: If the configuration is invalid
        LLMError: If the provider is not found or initialization fails
    """
    if not config or not isinstance(config, dict):
        raise LLMProviderConfigError("Configuration must be a non-empty dictionary")

    provider_name = name.lower()

    # Validate configuration if requested
    if validate:
        try:
            config = validate_config(provider_name, config)
        except Exception as e:
            raise LLMProviderConfigError(
                f"Invalid configuration for provider '{provider_name}': {e}"
            ) from e

    # Get provider class if not provided
    if provider_class is None:
        if provider_name not in _PROVIDER_REGISTRY:
            available = ", ".join(get_available_providers())
            raise LLMError(
                f"Unknown LLM provider: {name}. Available providers: {available}"
            )
        provider_class = _PROVIDER_REGISTRY[provider_name]

    try:
        return provider_class(config)
    except Exception as e:
        raise LLMError(f"Failed to initialize {name} provider: {e}") from e


# Register built-in providers if available
try:
    if OPENAI_AVAILABLE:
        register_provider("openai", OpenAIClient)  # type: ignore
    if ANTHROPIC_AVAILABLE:
        register_provider("anthropic", AnthropicClient)  # type: ignore
except Exception as e:
    logger.warning("Failed to register built-in providers: %s", e)
