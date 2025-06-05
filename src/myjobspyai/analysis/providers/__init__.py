"""
LLM Provider implementations for MyJobSpyAI.

This package contains implementations of various LLM providers that can be used
with the MyJobSpyAI application. Each provider implements a common interface
for generating text completions.

Available Providers:
    - LangChainProvider: For LangChain-compatible LLM backends
    - OpenAIProvider: For OpenAI API
    - OllamaProvider: For Ollama local models
    - GeminiProvider: For Google's Gemini models

Usage:
    ```python
    from myjobspyai.analysis.providers import (
        ProviderFactory, 
        ProviderType,
        LangChainProvider,
        SyncLangChainProvider
    )

    # Create a provider instance using the factory
    provider = ProviderFactory.create_provider(
        provider_type=ProviderType.LANGCHAIN,
        config={
            "class_name": "ChatOpenAI",
            "model_config": {
                "model_name": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1000
            },
            "system_message": "You are a helpful assistant."
        }
    )

    # Or create directly
    provider = LangChainProvider(
        config={
            "class_name": "ChatOpenAI",
            "model_config": {
                "model_name": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1000
            },
            "system_message": "You are a helpful assistant."
        },
        name="my-langchain-provider"
    )

    # Use the provider
    try:
        response = await provider.generate("Hello, world!")
        print(response)
    finally:
        await provider.close()

    # Or use the synchronous version
    sync_provider = SyncLangChainProvider(
        config={
            "class_name": "ChatOpenAI",
            "model_config": {
                "model_name": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1000
            },
            "system_message": "You are a helpful assistant."
        }
    )

    try:
        response = sync_provider.generate_sync("Hello, world!")
        print(response)
    finally:
        sync_provider.close_sync()
"""

from enum import Enum
from typing import Dict, Type, Any, Optional

from .base import BaseProvider, SyncProvider, ProviderError
from .factory import ProviderFactory
from .langchain_provider import LangChainProvider, SyncLangChainProvider, clean_json_string

# Export the ProviderType enum
class ProviderType(str, Enum):
    """Enumeration of supported provider types."""
    LANGCHAIN = "langchain"
    OPENAI = "openai"
    OLLAMA = "ollama"
    GEMINI = "gemini"

# Export the main classes
__all__ = [
    # Base classes
    'BaseProvider',
    'SyncProvider',
    'ProviderError',
    'ProviderType',
    'ProviderFactory',
    
    # Providers
    'LangChainProvider',
    'SyncLangChainProvider',
    
    # Utilities
    'clean_json_string',
]
