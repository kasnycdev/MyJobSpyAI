
"""Factory for creating and managing LLM providers with Pydantic v2 configuration."""
from __future__ import annotations

import asyncio
import importlib
import logging
import os
from typing import Any, Dict, Optional, Type, TypeVar

from myjobspyai.analysis.providers.base import BaseProvider
from myjobspyai.analysis.config import (
    ProviderFactoryConfig,
    DEFAULT_FACTORY_CONFIG,
)

# Map provider names to their classes
PROVIDER_CLASSES = {
    'openai': 'myjobspyai.analysis.providers.openai.OpenAIClient',
    'ollama': 'myjobspyai.analysis.providers.ollama.OllamaClient',
    'gemini': 'myjobspyai.analysis.providers.gemini.GeminiClient',
}

# Environment variable names for API keys
ENV_VAR_MAP = {
    'openai': 'OPENAI_API_KEY',
    'gemini': 'GOOGLE_API_KEY',
}

def _get_provider(provider_name: str, config: Dict[str, Any]) -> Type[BaseProvider]:
    """Get the provider class for a given provider name.
    
    Args:
        provider_name: Name of the provider
        config: Provider configuration
        
    Returns:
        The provider class
        
    Raises:
        ProviderNotSupported: If the provider is not supported
    """
    # Try to get from predefined classes first
    if provider_name in PROVIDER_CLASSES:
        class_ref = PROVIDER_CLASSES[provider_name]
        if isinstance(class_ref, str):
            module_path, class_name = class_ref.rsplit('.', 1)
            module = importlib.import_module(module_path)
            provider_class = getattr(module, class_name)
        else:
            # Handle case where class is passed directly (for testing)
            provider_class = class_ref
        
        if not issubclass(provider_class, BaseProvider):
            raise ValueError(f"Provider class {provider_class.__name__} is not a subclass of BaseProvider")
        return provider_class
    
    # If not found, try to import dynamically
    try:
        module = importlib.import_module(f'.providers.{provider_name}', package='myjobspyai.analysis')
        provider_class = getattr(module, f'{provider_name.capitalize()}Client')
        if not issubclass(provider_class, BaseProvider):
            raise ValueError(f"Provider class {provider_class.__name__} is not a subclass of BaseProvider")
        return provider_class
    except (ImportError, AttributeError) as e:
        raise ProviderNotSupported(f"Provider '{provider_name}' is not supported") from e

class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass

class ProviderNotConfigured(LLMError):
    """Raised when a provider is not properly configured."""
    pass

class ProviderNotSupported(LLMError):
    """Raised when a provider is not supported."""
    pass

logger = logging.getLogger(__name__)

# Default configuration for providers
DEFAULT_CONFIG = DEFAULT_FACTORY_CONFIG

# Type variable for provider classes
T = TypeVar('T', bound=BaseProvider)

class ProviderFactory:
    """Factory for creating and managing LLM providers with Pydantic v2 configuration."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the factory with configuration.
        
        Args:
            config: Optional configuration that will be merged with defaults
        """
        # Load and validate configuration
        self._config = DEFAULT_FACTORY_CONFIG.model_copy(deep=True)
        if config:
            self._config = self._config.model_validate({
                **self._config.model_dump(exclude_none=True),
                **config
            })
        
        self._providers: Dict[str, BaseProvider] = {}
    
    @property
    def config(self) -> ProviderFactoryConfig:
        """Get the current configuration."""
        return self._config
    
    def get_provider_config(
        self,
        provider_name: str,
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get configuration for a provider with optional overrides.
        
        Args:
            provider_name: Name of the provider
            overrides: Optional configuration overrides
            
        Returns:
            Validated provider configuration as a dictionary
            
        Raises:
            ValueError: If the configuration is invalid
        """
        # Get base config from factory config
        config = self._config.get_provider_config(provider_name, overrides or {})
        
        # Process environment variables in the config
        processed_config = {}
        for key, value in config.items():
            if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                # Extract environment variable name
                env_var = value[2:-1]
                # Get the value from environment or keep the original value if not found
                processed_config[key] = os.getenv(env_var, value)
            else:
                processed_config[key] = value
        
        # Load API key from environment if not set
        if provider_name in ENV_VAR_MAP and not processed_config.get('api_key'):
            if api_key := os.getenv(ENV_VAR_MAP[provider_name]):
                processed_config['api_key'] = api_key
        
        # Ensure provider name is set in the config
        processed_config['provider'] = provider_name
        
        # Validate the configuration using the appropriate provider config class
        from myjobspyai.analysis.config import ProviderConfig
        try:
            # Create a provider config instance to validate the configuration
            # Remove provider from processed_config to avoid duplicate argument
            provider = processed_config.pop('provider', provider_name)
            provider_config = ProviderConfig.create(provider, **processed_config)
            
            # Get the validated config as a dict
            validated_config = provider_config.model_dump(exclude_none=True)
            
            # Preserve any extra fields that weren't in the model
            for key in processed_config:
                if key not in validated_config:
                    validated_config[key] = processed_config[key]
                    
            # Ensure provider is set in the returned config
            validated_config['provider'] = provider
            return validated_config
            
        except Exception as e:
            logger.error("Invalid provider configuration: %s", str(e))
            raise ValueError(f"Invalid configuration for provider '{provider_name}': {str(e)}") from e
    
    async def create_provider(
        self,
        provider_name: str,
        config_overrides: Optional[Dict[str, Any]] = None,
        provider_class: Optional[Type[T]] = None,
    ) -> T:
        """Create a new provider instance.
        
        Args:
            provider_name: Name of the provider ('openai', 'ollama', 'gemini')
            config_overrides: Optional configuration overrides
            provider_class: Optional provider class to use instead of looking up by name
            
        Returns:
            An instance of the specified provider
            
        Raises:
            ValueError: If the provider is not found or configuration is invalid
        """
        # Get the provider configuration
        config = self.get_provider_config(provider_name, config_overrides)
        
        # Get the provider class if not specified
        if provider_class is None:
            provider_class = _get_provider(provider_name, config)
        
        # Create and return the provider instance
        if isinstance(provider_class, type):
            # It's a class, instantiate it with the config
            provider = provider_class(config)
            # Access to protected member is necessary for provider initialization
            if hasattr(provider, "_initialize_client"):
                # Handle both sync and async initialization
                # pylint: disable=protected-access
                init_client = provider._initialize_client  # noqa: SLF001
                if asyncio.iscoroutinefunction(init_client):
                    await init_client()
                else:
                    init_client()
            return provider
        else:
            # If it's not a class, raise an error
            raise ValueError(f"Invalid provider class type: {type(provider_class)}")
    
    async def get_or_create_provider(
        self,
        provider_name: str,
        config_overrides: Optional[Dict[str, Any]] = None,
        provider_class: Optional[Type[T]] = None,
    ) -> T:
        """Get an existing provider instance or create a new one.
        
        Args:
            provider_name: Name of the provider
            config_overrides: Optional configuration overrides
            provider_class: Optional provider class to use instead of looking up by name
            
        Returns:
            An existing or new provider instance
            
        Note:
            This is an async method and must be awaited.
        """
        # Check if we already have an instance of this provider
        if provider_name in self._providers:
            return self._providers[provider_name]
        
        # Create a new instance and cache it
        provider = await self.create_provider(
            provider_name=provider_name,
            config_overrides=config_overrides,
            provider_class=provider_class,
        )
        self._providers[provider_name] = provider
        return provider
    
    async def close(self) -> None:
        """Close all providers and release resources."""
        for provider in self._providers.values():
            try:
                await provider.close()
            except (RuntimeError, ConnectionError, OSError) as e:
                logger.warning(
                    "Error closing provider %s: %s",
                    provider.__class__.__name__,
                    e,
                    exc_info=True
                )
        self._providers.clear()
    
    async def __aenter__(self) -> 'ProviderFactory':
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit.
        
        Args:
            exc_type: The exception type if an exception was raised in the context
            exc_val: The exception value if an exception was raised in the context
            exc_tb: The traceback if an exception was raised in the context
            
        Returns:
            None
        """
        # Close all providers when exiting the context
        await self.close()

class _FactoryManager:
    """Manages the module-level factory instance."""
    _instance: Optional[ProviderFactory] = None
    
    @classmethod
    def get_factory(cls, config: Optional[Dict[str, Any]] = None) -> ProviderFactory:
        """Get or create the module-level provider factory.
        
        Args:
            config: Optional configuration to initialize the factory with.
                   If this is the first time creating the factory, this config
                   will be used as-is. For subsequent calls, it will be merged
                   with the existing configuration.
            
        Returns:
            The provider factory instance.
        """
        if cls._instance is None:
            # First time creation, use the config as-is
            cls._instance = ProviderFactory(config or {})
        elif config:
            # For existing instance, merge the new config with the existing one
            current_config = cls._instance.config.model_dump(exclude_none=True)
            merged_config = {**current_config, **config}
            cls._instance = ProviderFactory(merged_config)
        return cls._instance
    
    @classmethod
    def set_factory(cls, factory: ProviderFactory) -> None:
        """Set the module-level provider factory.
        
        Args:
            factory: The factory instance to set.
        """
        cls._instance = factory
    
    @classmethod
    async def close_factory(cls) -> None:
        """Close the module-level provider factory and release resources."""
        if cls._instance is not None:
            await cls._instance.close()
            cls._instance = None

# Module-level functions that use the _FactoryManager

def get_factory(config: Optional[Dict[str, Any]] = None) -> ProviderFactory:
    """Get or create the module-level provider factory.
    
    Args:
        config: Optional configuration to initialize the factory with.
        
    Returns:
        The provider factory instance.
    """
    return _FactoryManager.get_factory(config)

def set_factory(factory: ProviderFactory) -> None:
    """Set the module-level provider factory.
    
    Args:
        factory: The factory instance to set.
    """
    _FactoryManager.set_factory(factory)

async def close_factory() -> None:
    """Close the module-level provider factory and release resources."""
    await _FactoryManager.close_factory()
