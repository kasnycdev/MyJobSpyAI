"""Base classes for LLM providers."""
from __future__ import annotations

import abc
from typing import Any, Dict, Optional, TypeVar, Type

from tenacity import RetryCallState

# Type variable for the provider configuration
T = TypeVar('T')

class BaseProvider(abc.ABC):
    """Base class for all LLM providers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the provider with configuration.
        
        Args:
            config: Provider-specific configuration
        """
        self.config = config
    
    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Generate text using the provider's API.
        
        Args:
            prompt: The prompt to send to the model
            model: The model to use for generation
            temperature: Controls randomness (0-2)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional arguments to pass to the provider
            
        Returns:
            The generated text
        """
        pass
    
    @abc.abstractmethod
    async def close(self) -> None:
        """Close the client and release any resources."""
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        
    @classmethod
    def get_retry_config(cls, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get the retry configuration for the provider.
        
        Args:
            config: Optional configuration to override defaults
            
        Returns:
            Dictionary with retry configuration
        """
        defaults = {
            'stop': lambda state: state.attempt_number >= 3,
            'wait': lambda state: min(2 ** state.attempt_number, 30),
            'retry': lambda state: True,
            'reraise': True,
        }
        
        if config:
            defaults.update(config)
            
        return defaults

# Type alias for provider types
ProviderType = Type[BaseProvider]
