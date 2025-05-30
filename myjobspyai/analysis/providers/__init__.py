"""LLM provider implementations for the analyzer module."""
from typing import Dict, Type, Any

from .base import BaseProvider
from .openai import OpenAIClient
from .ollama import OllamaClient
from .gemini import GeminiClient
from .instructor_ollama import InstructorOllamaClient

# Re-export the provider classes
__all__ = [
    'BaseProvider',
    'OpenAIClient',
    'OllamaClient',
    'GeminiClient',
    'InstructorOllamaClient',
]

# Provider registry mapping provider names to their client classes
PROVIDER_REGISTRY: Dict[str, Type[BaseProvider]] = {
    'openai': OpenAIClient,
    'ollama': OllamaClient,
    'gemini': GeminiClient,
    'instructor_ollama': InstructorOllamaClient,
}

def get_provider(provider_name: str, config: Dict[str, Any]) -> BaseProvider:
    """Get an instance of the specified provider.
    
    Args:
        provider_name: Name of the provider ('openai', 'ollama', 'gemini')
        config: Configuration dictionary for the provider
        
    Returns:
        An instance of the specified provider
        
    Raises:
        ValueError: If the provider is not found
    """
    provider_name = provider_name.lower()
    if provider_name not in PROVIDER_REGISTRY:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Available providers: {list(PROVIDER_REGISTRY.keys())}"
        )
    
    return PROVIDER_REGISTRY[provider_name](config)
