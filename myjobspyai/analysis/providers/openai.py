"""OpenAI provider implementation for LLM analysis."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Union, AsyncGenerator

import httpx
from openai import APIConnectionError, AsyncOpenAI, RateLimitError, APITimeoutError
from myjobspyai.utils import with_retry

from myjobspyai.exceptions import (
    LLMError,
    RateLimitExceeded,
    ConfigurationError,
)
from .base import BaseProvider

logger = logging.getLogger(__name__)

class OpenAIClient(BaseProvider):
    """Client for interacting with OpenAI's API."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the OpenAI client.
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        super().__init__(config)
        self._client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the OpenAI client."""
        api_key = self.config.get("api_key")
        api_base = self.config.get("api_base", "https://api.openai.com/v1")
        
        if not api_key:
            raise ConfigurationError("OpenAI API key not found in config")
        
        # Get streaming configuration
        stream_timeout = self.config.get("stream_timeout", 60.0)
        stream_chunk_size = self.config.get("stream_chunk_size", 1024)
        
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=httpx.Timeout(
                self.config.get("timeout", 30.0),
                connect=self.config.get("connect_timeout", 10.0),
            ),
            max_retries=self.config.get("max_retries", 3),
        )
        
        # Store streaming configuration
        self._stream_timeout = stream_timeout
        self._stream_chunk_size = stream_chunk_size
        self._stream_retry_delay = self.config.get("stream_retry_delay", 2.0)
        self._stream_max_retries = self.config.get("stream_max_retries", 3)
    
    @with_retry(
        max_attempts=3,
        base_delay=4.0,
        max_delay=10.0,
        exceptions=(
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            httpx.TimeoutException,
        ),
        reraise=True
    )
    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate text using the OpenAI API.
        
        Args:
            prompt: The prompt to send to the model
            model: The model to use for generation
            temperature: Controls randomness (0-2)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to use streaming mode
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            The generated text
            
        Raises:
            LLMError: If the API call fails
        """
        if stream:
            # Handle streaming mode
            return await self._stream_generate(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        
        try:
            response = await self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            if not response.choices or not response.choices[0].message.content:
                raise LLMError("Empty response from OpenAI API")
                
            return response.choices[0].message.content.strip()
            
        except RateLimitError as e:
            raise RateLimitExceeded(f"OpenAI rate limit exceeded: {str(e)}") from e
        except APIConnectionError as e:
            raise LLMError(f"Failed to connect to OpenAI API: {str(e)}") from e
        except APITimeoutError as e:
            raise LLMError(f"OpenAI API request timed out: {str(e)}") from e
        except Exception as e:
            raise LLMError(f"OpenAI API error: {str(e)}") from e
            
    async def _stream_generate(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate text using streaming mode.
        
        Args:
            model: The model to use for generation
            messages: List of message objects
            temperature: Controls randomness (0-2)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional arguments to pass to the API
            
        Yields:
            Chunks of generated text
            
        Raises:
            LLMError: If the streaming fails
        """
        retries = 0
        
        while retries < self._stream_max_retries:
            try:
                async for chunk in self._client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                    **kwargs
                ):
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                break
            except (APIConnectionError, RateLimitError, httpx.TimeoutException) as e:
                retries += 1
                if retries >= self._stream_max_retries:
                    raise LLMError(f"Streaming failed after {retries} attempts: {str(e)}") from e
                logger.warning(f"Streaming attempt {retries} failed, retrying in {self._stream_retry_delay} seconds...")
                await asyncio.sleep(self._stream_retry_delay)
            except Exception as e:
                raise LLMError(f"Streaming failed: {str(e)}") from e
    
    async def close(self) -> None:
        """Close the client and release resources."""
        if self._client is not None:
            await self._client.close()
            self._client = None
