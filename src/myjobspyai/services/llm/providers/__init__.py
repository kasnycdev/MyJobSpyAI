"""LLM provider registry and factory."""

from __future__ import annotations

from myjobspyai.services.llm.base import BaseLLMProvider

# Registry mapping provider names to provider classes
_provider_registry: dict[str, type[BaseLLMProvider]] = {}


def register_provider(name: str, provider_class: type[BaseLLMProvider]) -> None:
    """Register a provider class with the given name.

    Args:
        name: The name to register the provider under.
        provider_class: The provider class to register.

    Raises:
        ValueError: If a provider with the same name is already registered.
    """
    if name in _provider_registry:
        raise ValueError(f"Provider with name '{name}' is already registered")
    _provider_registry[name] = provider_class


def get_provider_class(name: str) -> type[BaseLLMProvider]:
    """Get a provider class by name.

    Args:
        name: The name of the provider to get.

    Returns:
        The provider class.

    Raises:
        ValueError: If no provider with the given name is registered.
    """
    name = name.lower()
    if name not in _provider_registry:
        raise ValueError(f"No provider registered with name '{name}'")
    return _provider_registry[name]


def list_providers() -> list[str]:
    """List all registered provider names.

    Returns:
        A list of registered provider names.
    """
    return list(_provider_registry.keys())


__all__ = ["register_provider", "get_provider_class", "list_providers"]
