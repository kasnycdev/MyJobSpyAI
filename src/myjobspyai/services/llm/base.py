"""Base module for LLM providers.

This module defines the base interface and common functionality for all LLM providers.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from typing import Any, Callable, TypeVar, Union

import aiohttp
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from myjobspyai.services.llm.exceptions import (
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
)
from myjobspyai.services.llm.tools import BaseTool
from myjobspyai.services.llm.types import LLMResponse

# Configure logger
logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30.0
DEFAULT_CONNECTION_POOL_SIZE = 10
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 1000
RATE_LIMIT_STATUS = 429

T = TypeVar('T', bound='BaseLLMProvider')


@dataclass
class LLMResponse:
    """Response from an LLM provider.

    Attributes:
        text: The generated text.
        model: The model used to generate the text.
        usage: Token usage information.
        metadata: Additional metadata from the provider.
    """

    text: str
    """The generated text."""

    model: str
    """The model used to generate the text."""

    usage: dict[str, int | None]
    """Token usage information."""

    metadata: dict[str, Any] | None = None
    """Additional metadata from the provider."""

    def __post_init__(self) -> None:
        """Initialize metadata if not provided."""
        if not hasattr(self, 'metadata') or self.metadata is None:
            self.metadata = {}


# Aliases for backward compatibility
LLMRequestError = LLMError
LLMProviderConfigError = LLMError


class BaseLLMProvider(ABC):
    """Base class for LLM providers.

    This class defines the interface that all LLM providers must implement.
    It includes connection management, retry logic, and common configuration.
    """

    # Class-level session for connection pooling
    _session: aiohttp.ClientSession | None = None
    _session_lock = asyncio.Lock()

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the provider with configuration.

        Args:
            config: Configuration dictionary for the provider.
                Should include at least 'model' and 'provider' keys.
                May include 'api_key', 'base_url', 'timeout', 'max_retries',
                'request_timeout', and 'connection_pool_size'.

        Raises:
            LLMProviderConfigError: If required configuration is missing or invalid.
        """
        self.config = config
        self.model = config.get("model")
        self.provider = config.get("provider")
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "")

        if not self.model or not self.provider:
            raise LLMProviderConfigError(
                "Both 'model' and 'provider' must be specified in config"
            )

        # Connection and retry configuration
        self.timeout = float(config.get("timeout", DEFAULT_TIMEOUT))
        self.max_retries = int(config.get("max_retries", DEFAULT_MAX_RETRIES))
        self.request_timeout = float(config.get("request_timeout", DEFAULT_TIMEOUT))
        self.connection_pool_size = int(
            config.get("connection_pool_size", DEFAULT_CONNECTION_POOL_SIZE)
        )

        # Generation parameters
        self.temperature = float(config.get("temperature", DEFAULT_TEMPERATURE))
        self.max_tokens = int(config.get("max_tokens", DEFAULT_MAX_TOKENS))

        # Connection state
        self._is_closed = False

    async def get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp ClientSession with connection pooling.

        Returns:
            An aiohttp ClientSession instance.

        Raises:
            LLMConnectionError: If session creation fails.
        """
        if self._session is None or self._session.closed:
            async with self._session_lock:
                if self._session is None or self._session.closed:
                    try:
                        timeout = aiohttp.ClientTimeout(total=self.request_timeout)
                        connector = aiohttp.TCPConnector(
                            limit=self.connection_pool_size,
                            enable_cleanup_closed=True,
                            force_close=False,
                        )
                        self._session = aiohttp.ClientSession(
                            timeout=timeout,
                            connector=connector,
                            raise_for_status=True,
                        )
                        logger.debug(
                            "Created new aiohttp session with connection pooling"
                        )
                    except Exception as e:
                        logger.error("Failed to create aiohttp session: %s", str(e))
                        raise LLMConnectionError(
                            f"Failed to create session: {e}"
                        ) from e
        return self._session

    def _get_retry_decorator(self):
        """Create a retry decorator with exponential backoff.

        Returns:
            A retry decorator configured for common LLM API failures.
        """
        return AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential_jitter(initial=1, jitter=2),
            retry=(
                retry_if_exception_type(
                    (LLMConnectionError, LLMRateLimitError, LLMTimeoutError)
                )
                | retry_if_exception_type(aiohttp.ClientError)
            ),
        )

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Make an HTTP request with retry logic and connection pooling.

        Args:
            method: HTTP method (get, post, etc.)
            endpoint: API endpoint (appended to base_url)
            **kwargs: Additional arguments to pass to aiohttp request.

        Returns:
            The parsed JSON response.

        Raises:
            LLMConnectionError: For connection-related errors.
            LLMRateLimitError: When rate limits are exceeded.
            LLMTimeoutError: When the request times out.
            LLMError: For other request failures.
        """

        @self._get_retry_decorator()
        async def _request():
            session = await self.get_session()
            url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

            try:
                async with session.request(method, url, **kwargs) as response:
                    if response.status == RATE_LIMIT_STATUS:
                        retry_after = int(response.headers.get('Retry-After', 1))
                        logger.warning(
                            "Rate limited. Retrying after %s seconds. Endpoint: %s",
                            retry_after,
                            endpoint,
                        )
                        await asyncio.sleep(retry_after)
                        raise LLMRateLimitError("Rate limit exceeded")

                    response.raise_for_status()
                    return await response.json()

            except asyncio.TimeoutError as e:
                logger.error("Request timed out: %s", e)
                raise LLMTimeoutError(
                    f"Request timed out after {self.request_timeout}s"
                ) from e

            except aiohttp.ClientResponseError as e:
                if e.status == RATE_LIMIT_STATUS:
                    raise LLMRateLimitError("Rate limit exceeded") from e
                logger.error("HTTP error %s: %s", e.status, e.message)
                raise LLMError(f"HTTP error {e.status}: {e.message}") from e
            except aiohttp.ClientError as e:
                logger.error("Connection error: %s", e)
                raise LLMConnectionError(f"Connection error: {e}") from e
            except Exception as e:
                logger.error("Unexpected error: %s", e)
                raise LLMError(f"Request failed: {e}") from e

        return await _request()

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str | LLMResponse:
        """Generate text from a prompt.

        Args:
            prompt: The input prompt.
            **kwargs: Additional provider-specific arguments.

        Returns:
            The generated text or an LLMResponse object.

        Raises:
            LLMError: If generation fails.
        """
        raise NotImplementedError("Subclasses must implement generate()")

    @abstractmethod
    async def generate_batch(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> list[str | LLMResponse]:
        """Generate text from multiple prompts.

        Args:
            prompts: List of input prompts.
            **kwargs: Additional provider-specific arguments.

        Returns:
            List of generated texts or LLMResponse objects.

        Raises:
            LLMError: If generation fails for any prompt.
        """
        raise NotImplementedError("Subclasses must implement generate_batch()")

    @abstractmethod
    async def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str | dict[str, Any], None]:
        """Stream generated text from a prompt.

        Args:
            prompt: The input prompt.
            **kwargs: Additional provider-specific arguments.

        Yields:
            Chunks of generated text or response objects.

        Raises:
            LLMError: If streaming fails.
        """
        # This method must be implemented by subclasses
        # The implementation should be an async generator that yields chunks
        # of generated text or response objects
        # Example implementation would look like:
        # async for chunk in self._stream_impl(prompt, **kwargs):
        #     yield chunk
        raise NotImplementedError("Subclasses must implement stream()")
        # This line is unreachable but makes type checkers happy
        if False:  # pylint: disable=using-constant-test
            yield None

    @abstractmethod
    async def get_embeddings(
        self,
        text: str | list[str],
        **kwargs: Any,
    ) -> list[float] | list[list[float]]:
        """Get embeddings for text.

        Args:
            text: Input text or list of texts to embed.
            **kwargs: Additional provider-specific arguments.

        Returns:
            A single embedding or list of embeddings.

        Raises:
            LLMError: If embedding generation fails.
        """
        # This method must be implemented by subclasses
        # The implementation should return either:
        # - A single embedding as a list of floats (for a single text input)
        # - A list of embeddings (for multiple text inputs)
        # Example implementation would look like:
        # if isinstance(text, str):
        #     return await self._get_single_embedding(text, **kwargs)
        # return await self._get_batch_embeddings(text, **kwargs)
        raise NotImplementedError("Subclasses must implement get_embeddings()")

    @abstractmethod
    async def close(self) -> None:
        """Close the provider and release resources.

        This method should be called when the provider is no longer needed
        to ensure proper cleanup of resources like connection pools.
        """
        if hasattr(self, '_session') and self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        self._is_closed = True
        logger.debug("LLM provider closed successfully")

    # Tool discovery and binding methods

    def get_available_tools(self) -> list[dict[str, Any]]:
        """Get a list of available tools that this provider can use.

        Returns:
            A list of tool schemas that can be used with this provider
        """
        # Default implementation returns an empty list
        return []

    def bind_tools(
        self,
        tools: Sequence[BaseTool | Callable | dict[str, Any]],
        **kwargs: Any,
    ) -> BaseLLMProvider:
        """Bind tools to this provider instance.

        Args:
            tools: A list of tools to bind. Can be:
                - A BaseTool instance
                - A callable (function) that will be converted to a tool
                - A dictionary with tool configuration
            **kwargs: Additional arguments to pass to the tool binding implementation

        Returns:
            A new provider instance with the tools bound

        Note:
            This method should be implemented by subclasses that support tool calling.
            The default implementation returns self without any changes.
        """
        # Default implementation does nothing
        return self

    def supports_tool_calling(self) -> bool:
        """Check if this provider supports tool calling.

        Returns:
            bool: True if the provider supports tool calling, False otherwise
        """
        return False

    async def __aenter__(self):
        """Support async context manager protocol."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Ensure resources are cleaned up when exiting context."""
        await self.close()

    def __str__(self) -> str:
        """Return a string representation of the provider."""
        return (
            f"{self.__class__.__name__}("
            f"provider='{self.provider}', "
            f"model='{self.model}'"
            ")"
        )

    def __repr__(self) -> str:
        """Return a detailed string representation of the provider."""
        return (
            f"{self.__class__.__name__}("
            f"provider='{self.provider}', "
            f"model='{self.model}', "
            f"temperature={self.temperature}, "
            f"max_tokens={self.max_tokens}"
            ")"
        )
