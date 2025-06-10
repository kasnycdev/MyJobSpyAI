"""OpenAI provider implementation for the LLM service.

This module provides a concrete implementation of the BaseLLMProvider for the
OpenAI API, supporting text generation, chat completion, and embeddings.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

import openai
from openai.types.chat import ChatCompletionMessageParam

from myjobspyai.services.llm import factory
from myjobspyai.services.llm.base import BaseLLMProvider, LLMResponse
from myjobspyai.services.llm.exceptions import LLMError

# Configure logger
logger = logging.getLogger(__name__)

# Default values for OpenAI-specific parameters
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"
DEFAULT_EMBEDDING_MODEL = "text-embedding-ada-002"


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider for the LLM service.

    This class implements the BaseLLMProvider interface for the OpenAI API,
    supporting both chat and completion models, as well as embeddings.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """Initialize the OpenAI provider.

        Args:
            config: Configuration dictionary. Must include 'api_key' or have it
                   set in the environment as OPENAI_API_KEY.

        Raises:
            LLMError: If initialization fails.
        """
        # Set default model if not specified
        if "model" not in config:
            config["model"] = DEFAULT_OPENAI_MODEL

        # Initialize the base class
        super().__init__(config)

        # Set the API key
        self.api_key = config.get("api_key")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key must be provided in config or set as OPENAI_API_KEY"
            )

        # Set organization if provided
        self.organization = config.get("organization")

        # Initialize the OpenAI client
        self._initialize_client()

        logger.info("Initialized OpenAI provider with model: %s", self.model)

    def _initialize_client(self) -> None:
        """Initialize the OpenAI client with the provided configuration."""
        try:
            self.client = openai.AsyncOpenAI(
                api_key=self.api_key,
                organization=self.organization,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        except Exception as e:
            raise LLMError(f"Failed to initialize OpenAI client: {e}") from e

    async def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            **kwargs: Additional arguments to pass to the OpenAI API.

        Returns:
            LLMResponse containing the generated text and metadata.

        Raises:
            LLMError: If generation fails.
        """
        try:
            # Prepare messages for chat completion
            messages: list[ChatCompletionMessageParam] = [
                {"role": "user", "content": prompt}
            ]

            # Call the API
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )

            # Extract the generated text
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("No content in response")

            content = response.choices[0].message.content

            # Create response object
            return LLMResponse(
                text=content,
                model=self.model,
                request_id=response.id,
                usage=(
                    {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                    if response.usage
                    else None
                ),
            )

        except Exception as e:
            raise LLMError(f"OpenAI generation failed: {e}") from e

    async def generate_batch(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> list[LLMResponse]:
        """Generate text from multiple prompts in parallel.

        Args:
            prompts: List of prompts to generate text from.
            **kwargs: Additional arguments to pass to the OpenAI API.

        Returns:
            List of LLMResponse objects containing the generated text and metadata.

        Raises:
            LLMError: If batch generation fails.
        """
        try:
            # Create tasks for each prompt
            tasks = [self.generate(prompt, **kwargs) for prompt in prompts]

            # Run all tasks in parallel
            return await asyncio.gather(*tasks)

        except Exception as e:
            raise LLMError(f"OpenAI batch generation failed: {e}") from e

    async def stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncGenerator[str, None]:
        """Stream generated text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            **kwargs: Additional arguments to pass to the OpenAI API.

        Yields:
            Chunks of generated text as they become available.

        Raises:
            LLMError: If streaming fails.
        """
        try:
            # Prepare messages for chat completion
            messages: list[ChatCompletionMessageParam] = [
                {"role": "user", "content": prompt}
            ]

            # Call the API with streaming
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **kwargs,
            )

            # Stream the response
            async for chunk in stream:
                if not chunk.choices or not chunk.choices[0].delta.content:
                    continue
                yield chunk.choices[0].delta.content

        except Exception as e:
            raise LLMError(f"OpenAI streaming failed: {e}") from e

    async def get_embeddings(
        self,
        texts: list[str],
        model: str | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """Get embeddings for a list of texts.

        Args:
            texts: List of texts to get embeddings for.
            model: The embedding model to use. Defaults to the default embedding model.
            **kwargs: Additional arguments to pass to the OpenAI API.

        Returns:
            List of embeddings, one for each input text.

        Raises:
            LLMError: If embedding generation fails.
        """
        if not model:
            model = DEFAULT_EMBEDDING_MODEL

        try:
            # Call the API
            response = await self.client.embeddings.create(
                input=texts,
                model=model,
                **kwargs,
            )

            # Extract and return the embeddings
            return [item.embedding for item in response.data]
        except Exception as e:
            raise LLMError(f"Failed to get embeddings: {e}") from e

    async def close(self) -> None:
        """Close the OpenAI client and release resources.

        This should be called when the provider is no longer needed to ensure
        proper cleanup of resources.
        """
        if not hasattr(self, "client") or not self.client:
            return

        try:
            await self.client.close()
            logger.debug("Successfully closed OpenAI client")
        except Exception as e:
            logger.warning("Error closing OpenAI client: %s", e)


# Register the provider with the factory
factory.register_provider("openai", OpenAIProvider)
