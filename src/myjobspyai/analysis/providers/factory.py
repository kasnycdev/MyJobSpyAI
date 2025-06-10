"""Provider factory for creating and managing LLM providers."""

import importlib
import logging
from enum import Enum
from typing import Any, Dict, Optional, Type, TypeVar, Union, cast

from myjobspyai import config

from .base import BaseProvider

# Type variable for provider classes
T = TypeVar('T', bound=BaseProvider)

logger = logging.getLogger(__name__)


class ProviderType(str, Enum):
    """Enumeration of supported provider types.

    This enum defines the standard provider types that are supported out of the box.
    Additional provider types can be registered at runtime.
    """


class ProviderFactory:
    """Singleton factory for creating and managing LLM providers.

    This class follows the singleton pattern to ensure there's only one instance
    managing all provider registrations. It supports dynamic loading of providers
    from configuration and provides a consistent interface for creating provider
    instances.
    """

    _instance: Optional['ProviderFactory'] = None
    _providers: Dict[str, Type[BaseProvider]] = {}
    _initialized: bool = False

    def __new__(cls) -> 'ProviderFactory':
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super(ProviderFactory, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """Initialize the factory with default providers."""
        if self._initialized:
            return

        self._initialized = True

        # Register built-in providers
        try:
            from .langchain_provider import LangChainProvider

            self.register(ProviderType.LANGCHAIN, LangChainProvider)
            logger.info(
                "Registered built-in providers: %s", ", ".join(self._providers.keys())
            )
        except ImportError as e:
            logger.warning("Failed to register built-in providers: %s", e)

    @property
    def available_providers(self) -> Dict[str, Type[BaseProvider]]:
        """Get a dictionary of all registered provider classes.

        Returns:
            A dictionary mapping provider names to their corresponding classes.
        """
        return self._providers.copy()

    @property
    def provider_names(self) -> list[str]:
        """Get a list of all registered provider names."""
        return list(self._providers.keys())

    def register(self, name: Union[str, ProviderType], provider_class: Type[T]) -> None:
        """Register a new provider type.

        Args:
            name: Name of the provider type (can be a string or ProviderType enum)
            provider_class: Provider class that implements BaseProvider

        Raises:
            ValueError: If the provider class is invalid or already registered
        """
        if not (
            isinstance(provider_class, type)
            and issubclass(provider_class, BaseProvider)
        ):
            raise ValueError(
                f"Provider class must be a subclass of BaseProvider, got {provider_class}"
            )

        name_str = name.value if isinstance(name, ProviderType) else str(name).lower()

        if name_str in self._providers:
            if self._providers[name_str] is not provider_class:
                logger.warning("Overriding existing provider: %s", name_str)
            else:
                logger.debug("Provider %s is already registered", name_str)
                return

        self._providers[name_str] = provider_class
        logger.info("Registered provider: %s -> %s", name_str, provider_class.__name__)

    def get_provider_class(self, name: Union[str, ProviderType]) -> Type[BaseProvider]:
        """Get a provider class by name.

        Args:
            name: Name of the provider (can be a string or ProviderType enum)

        Returns:
            The provider class

        Raises:
            ValueError: If the provider is not found
        """
        name_str = name.value if isinstance(name, ProviderType) else str(name).lower()

        if name_str not in self._providers:
            # Try to dynamically import the provider
            try:
                module_name = f"{__package__}.{name_str}_provider"
                module = importlib.import_module(module_name)
                provider_class = getattr(module, f"{name_str.capitalize()}Provider")
                self.register(name_str, provider_class)
                return provider_class
            except (ImportError, AttributeError) as e:
                raise ValueError(f"Unknown provider: {name_str}") from e

        return self._providers[name_str]

    def create(
        self, provider_name: str, config: Optional[Dict[str, Any]] = None, **kwargs
    ) -> BaseProvider:
        """Create a provider instance.

        Args:
            provider_name: Name of the provider to create
            config: Provider configuration (optional, can be None for defaults)
            **kwargs: Additional arguments to pass to the provider

        Returns:
            Initialized provider instance

        Raises:
            ValueError: If the provider cannot be created
        """
        try:
            provider_class = self.get_provider_class(provider_name)

            # Merge config with defaults if needed
            if config is None:
                config = {}

            # Add provider name to config if not present
            if "name" not in config:
                config["name"] = provider_name

            # Create and return the provider instance
            instance = provider_class(config=config, **kwargs)
            logger.info("Created provider instance: %s", provider_name)
            return instance

        except Exception as e:
            logger.exception("Failed to create provider %s: %s", provider_name, str(e))
            raise ValueError(
                f"Failed to create provider {provider_name}: {str(e)}"
            ) from e

    def create_from_config(
        self, config: Dict[str, Dict[str, Any]]
    ) -> Dict[str, BaseProvider]:
        """Create multiple provider instances from a configuration dictionary.

        Args:
            config: Dictionary mapping provider names to their configurations

        Returns:
            Dictionary mapping provider names to provider instances

        Example:
            ```python
            config = {
                "openai": {"api_key": "sk-...", "model": "gpt-4"},
                "langchain": {"model_name": "gpt-3.5-turbo"},
            }
            providers = factory.create_from_config(config)
            ```
        """
        providers = {}

        for name, provider_config in config.items():
            try:
                # Skip disabled providers
                if not provider_config.get("enabled", True):
                    logger.debug("Skipping disabled provider: %s", name)
                    continue

                provider = self.create(name, provider_config)
                providers[name] = provider

            except Exception as e:
                logger.error("Failed to create provider %s: %s", name, str(e))
                if config.get("raise_errors", True):
                    raise

        return providers

    def load_from_settings(self) -> Dict[str, BaseProvider]:
        """Load and create providers from application settings.

        This loads provider configurations from the global config object
        and creates provider instances accordingly.

        Returns:
            Dictionary mapping provider names to provider instances
        """
        if not hasattr(config, 'llm_providers'):
            logger.warning("No LLM providers found in config")
            return {}

        return self.create_from_config(config.llm_providers)


# Public API
__all__ = [
    "ProviderType",
    "ProviderFactory",
    "BaseProvider",
]
