"""Analysis module for LLM-based job and resume analysis."""

import atexit
import asyncio

# Import core components
from .analyzer import ResumeAnalyzer
from .base import BaseAnalyzer, LLMMetrics
from .factory import (
    ProviderFactory,
    ProviderNotConfigured,
    ProviderNotSupported,
    get_factory,
    set_factory,
    close_factory,
)
from .providers import (
    OpenAIClient,
    OllamaClient,
    GeminiClient,
)

# Re-export public API
__all__ = [
    'ResumeAnalyzer',
    'BaseAnalyzer',
    'LLMMetrics',
    'ProviderFactory',
    'ProviderNotConfigured',
    'ProviderNotSupported',
    'get_factory',
    'set_factory',
    'close_factory',
    'OpenAIClient',
    'OllamaClient',
    'GeminiClient',
]

# Initialize the default factory on import
_factory = None

def get_default_factory():
    """Get or create the default provider factory.
    
    Returns:
        ProviderFactory: The default factory instance
    """
    global _factory
    if _factory is None:
        _factory = get_factory()
    return _factory

def _cleanup():
    """Clean up resources on module unload."""
    global _factory
    if _factory is not None:
        asyncio.run(close_factory(_factory))
        _factory = None

# Clean up on module unload
atexit.register(_cleanup)