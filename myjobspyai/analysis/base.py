"""Base analyzer module for LLM-based analysis.

This module provides the base classes and utilities for LLM-based analysis,
including metrics collection, retry logic, and provider management.
"""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC
from collections import defaultdict
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar

# Local imports
from myjobspyai.analysis.factory import get_factory
from myjobspyai.analysis.providers.base import BaseProvider

# Configure logger
logger = logging.getLogger(__name__)
model_output_logger = logging.getLogger(f"{__name__}.model_output")

# Type variables for generic type hints
T = TypeVar('T')
P = TypeVar('P')

class LLMMetrics:
    """Metrics for LLM calls."""
    
    def __init__(self):
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.total_duration = 0.0
        self.errors = {}
        self.calls_by_task = defaultdict(lambda: {"success": 0, "failure": 0, "total": 0})
    
    def record_success(self, task_name: str, duration: float) -> None:
        """Record a successful LLM call."""
        self.total_calls += 1
        self.successful_calls += 1
        self.total_duration += duration
        self.calls_by_task[task_name]["success"] += 1
        self.calls_by_task[task_name]["total"] += 1
    
    def record_failure(self, task_name: str, error: Exception) -> None:
        """Record a failed LLM call."""
        self.total_calls += 1
        self.failed_calls += 1
        error_name = error.__class__.__name__
        self.errors[error_name] = self.errors.get(error_name, 0) + 1
        self.calls_by_task[task_name]["failure"] += 1
        self.calls_by_task[task_name]["total"] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the metrics."""
        return {
            "total_calls": self.total_calls,
            "successful_calls": self.successful_calls,
            "failed_calls": self.failed_calls,
            "success_rate": self.successful_calls / self.total_calls if self.total_calls > 0 else 0,
            "avg_duration": self.total_duration / self.successful_calls if self.successful_calls > 0 else 0,
            "errors": dict(self.errors),
            "calls_by_task": dict(self.calls_by_task)
        }

class BaseAnalyzer(ABC):
    """Base class for LLM-based analysis with retry logic and metrics.
    
    This class provides a foundation for LLM-based analysis with built-in support for:
    - Multiple LLM providers (OpenAI, Ollama, Gemini)
    - Async/await patterns
    - Retry logic with exponential backoff
    - Performance metrics collection
    - Resource cleanup
    """
    
    def __init__(
        self, 
        provider: Optional[Any] = None,  # Can be a provider instance or string
        model: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the analyzer with configuration.
        
        Args:
            provider: Either a pre-initialized provider instance or a string with the provider name.
                     If None, uses the provider from application settings.
            model: The model to use for this analyzer.
                  If None, uses the model from application settings.
            config: Optional configuration overrides.
        """
        from myjobspyai import settings  # Lazy import to avoid circular imports
        
        # Get configuration from settings if not provided
        self.config = config or {}
        
        # Define valid providers
        valid_providers = ['ollama', 'openai', 'gemini']
        
        # Handle provider initialization
        if hasattr(provider, 'generate'):
            # Provider instance was passed directly
            self._provider_instance = provider
            self.provider = getattr(provider, 'provider_name', 'ollama')
            logger.debug("Using pre-initialized provider instance: %s", self.provider)
        else:
            # Provider name or None was passed
            self._provider_instance = None
            
            # Get provider from parameter or settings with fallback to 'ollama'
            provider_name = provider if isinstance(provider, str) else getattr(settings.llm, 'provider', 'ollama')
            
            # Validate the provider name
            if provider_name not in valid_providers:
                logger.warning(
                    f"Invalid provider '{provider_name}'. Falling back to 'ollama'. "
                    f"Valid providers are: {', '.join(valid_providers)}"
                )
                provider_name = 'ollama'
                
            self.provider = provider_name
            logger.debug("Using provider: %s", self.provider)
        
        # Set the model from the most specific to least specific source
        self.model = model or getattr(settings.llm, 'model', None)
        
        # If no model is specified, use the one from the provider config
        if self.model is None and 'model' in self.config:
            self.model = self.config['model']
            
        # If still no model, raise an error to ensure proper configuration
        if self.model is None:
            raise ValueError(
                "No model specified. Please provide a model name either through the 'model' parameter "
                "or in the configuration under 'llm.model' or provider-specific config."
            )
        
        # Log the configuration
        logger.debug(
            "Initializing BaseAnalyzer with provider='%s', model='%s'", 
            self.provider, 
            self.model
        )
        
        # Initialize metrics and state
        self.metrics = LLMMetrics()
        self._client: Optional[BaseProvider] = None
        self._client_initialized = False
        self._initialization_lock = asyncio.Lock()
    
    @property
    async def client(self) -> BaseProvider:
        """Get the LLM client, using the pre-initialized provider if available."""
        if self._provider_instance is not None:
            return self._provider_instance
            
        if self._client is None:
            await self.initialize()
            
        # If client is awaitable, await it
        if hasattr(self._client, '__await__'):
            self._client = await self._client
            
        return self._client
        
    async def initialize(self) -> None:
        """Initialize the LLM client if not already initialized."""
        if not self._client_initialized:
            async with self._initialization_lock:
                if not self._client_initialized:  # Double-checked locking
                    try:
                        await self._initialize_client()
                        # If client is awaitable, await it now
                        if hasattr(self._client, '__await__'):
                            self._client = await self._client
                        self._client_initialized = True
                        logger.debug("LLM client initialized successfully")
                    except Exception as e:
                        logger.error("Failed to initialize LLM client: %s", str(e), exc_info=True)
                        raise
    
    async def _initialize_client(self) -> None:
        """Initialize the LLM client using the provider factory.
        
        This method:
        1. Gets the provider factory instance
        2. Creates or gets a provider instance
        3. Verifies the provider is properly initialized
        """
        try:
            # Get the factory instance
            factory = get_factory()
            
            # Ensure we have a valid provider
            if not self.provider:
                raise ValueError("No LLM provider specified. Please set the 'provider' parameter or configure it in settings.")
            
            # Create provider configuration
            provider_config = {
                'provider': self.provider,
                'model': self.model,
                **self.config  # Allow config overrides
            }
            
            # Log the configuration we're using
            logger.debug(
                "Initializing provider with config: %s", 
                {k: v for k, v in provider_config.items() if k != 'api_key'}
            )
            
            # Get or create the provider
            self._client = await factory.get_or_create_provider(
                provider_name=self.provider,
                config_overrides=provider_config
            )
            
            # Verify the client was created
            if self._client is None:
                raise RuntimeError(f"Failed to initialize {self.provider} provider")
                
            logger.info(
                "Successfully initialized %s provider with model: %s", 
                self.provider, 
                self.model
            )
            
        except Exception as e:
            logger.error(
                "Failed to initialize %s provider: %s", 
                self.provider, 
                str(e),
                exc_info=True
            )
            raise
    
    async def generate(
        self,
        prompt: str,
        task_name: str = "generate",
        **kwargs
    ) -> str:
        """Generate text from the LLM with retry logic.
        
        Args:
            prompt: The prompt to send to the LLM
            task_name: Name of the task for metrics tracking
            **kwargs: Additional arguments to pass to the provider
            
        Returns:
            The generated text from the LLM
            
        Raises:
            RuntimeError: If the LLM call fails
        """
        start_time = time.time()
        
        try:
            # Get the client and generate the response
            client = await self.client
            # Prepare the config, ensuring model is included
            config = {
                'model': self.model,
                **{k: v for k, v in self.config.items() if k != 'model'},  # Don't override model from kwargs
                **kwargs
            }
            
            # Generate the response
            if hasattr(client, 'generate') and callable(client.generate):
                response = await client.generate(
                    prompt=prompt,
                    **config
                )
            else:
                raise RuntimeError(f"Client {client} does not have a valid generate method")
            
            # Record metrics
            duration = time.time() - start_time
            self.metrics.record_success(task_name, duration)
            
            # Log the response
            if model_output_logger.isEnabledFor(logging.DEBUG):
                model_output_logger.debug(
                    "Generated response for task '%s' (%.2fs):\n%s",
                    task_name, duration, response
                )
                
            return response
            
        except Exception as e:
            # Record failure metrics
            self.metrics.record_failure(task_name, e)
            logger.error(
                "Failed to generate text for task '%s' using %s: %s",
                task_name, self.provider, str(e),
                exc_info=isinstance(e, RuntimeError)
            )
            raise RuntimeError(
                f"Failed to generate text using {self.provider}: {str(e)}"
            ) from e
    
    async def close(self) -> None:
        """Clean up resources.
        
        This method:
        1. Closes the LLM client if it exists
        2. Resets the client state
        3. Can be called multiple times safely
        """
        if self._client is not None:
            try:
                if hasattr(self._client, 'close'):
                    if asyncio.iscoroutinefunction(self._client.close):
                        await self._client.close()
                    else:
                        self._client.close()
                logger.debug("Closed %s provider client", self.provider)
            except Exception as e:
                logger.warning(
                    "Error closing %s client: %s", 
                    self.provider, str(e),
                    exc_info=True
                )
            finally:
                self._client = None
                self._client_initialized = False
    
    async def __aenter__(self) -> 'BaseAnalyzer':
        """Async context manager entry.
        
        Returns:
            The analyzer instance.
        """
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit.
        
        Ensures resources are properly cleaned up when exiting the context.
        """
        await self.close()

def log_exception(message: str, exception: Exception) -> None:
    """Log an exception with context."""
    logger.error(
        "%s: %s\nTraceback: %s",
        message,
        str(exception),
        exc_info=True,
    )

def async_timed():
    """Decorator to measure execution time of async functions."""
    def wrapper(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            start_time = time.time()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.time() - start_time
                logger.debug(
                    "%s completed in %.4f seconds",
                    func.__name__,
                    duration,
                )
        return wrapped
    return wrapper
