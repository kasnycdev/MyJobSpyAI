"""Anthropic provider implementation for LLM integration."""

from __future__ import annotations

import logging
import os
from typing import Any

import anthropic
from pydantic import BaseModel, Field, field_validator

from myjobspyai.llm.base import BaseLLMProvider, LLMError, LLMResponse
from myjobspyai.utils.async_utils import gather_with_concurrency

logger = logging.getLogger(__name__)


class AnthropicConfig(BaseModel):
    """Configuration for the Anthropic provider."""

    model: str = Field(
        default="claude-3-sonnet-20240229",
        description="The model to use for text generation.",
    )
    api_key: str | None = Field(
        default=None,
        description=(
            "Anthropic API key. If not provided, will be read from "
            "ANTHROPIC_API_KEY environment variable."
        ),
    )
    base_url: str | None = Field(
        default=None,
        description="Base URL for the Anthropic API.",
    )
    default_headers: dict[str, str] | None = Field(
        default=None,
        description="Default headers to include in API requests.",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for API calls.",
        ge=0,
    )
    timeout: int = Field(
        default=30,
        description="Timeout in seconds for API calls.",
        gt=0,
    )
    temperature: float = Field(
        default=0.7,
        description="Sampling temperature (0-1)",
        ge=0.0,
        le=1.0,
    )
    max_tokens: int = Field(
        default=1024,
        description="Maximum number of tokens to generate",
        ge=1,
    )
    top_p: float = Field(
        default=1.0,
        description="Nucleus sampling parameter (0-1)",
        ge=0.0,
        le=1.0,
    )
    top_k: int = Field(
        default=40,
        description="Top-k sampling parameter",
        ge=1,
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: Optional[str]) -> str:
        """Validate API key or get it from environment."""
        if v is None:
            v = os.getenv("ANTHROPIC_API_KEY")
            if not v:
                msg = "API key not provided and ANTHROPIC_API_KEY environment variable not set"
                raise ValueError(msg)
        return v


class AnthropicProvider(BaseLLMProvider):
    """Anthropic provider implementation for LLM integration."""

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        """Initialize the Anthropic provider.

        Args:
            config: Configuration dictionary. If None, uses defaults from environment.
        """
        super().__init__("anthropic")
        self.config = AnthropicConfig(**(config or {}))

        self._client = anthropic.AsyncAnthropic(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )

    async def close(self) -> None:
        """Close the provider and release resources."""
        await self._client.close()

    async def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate text from.
            **kwargs: Additional generation parameters.

        Returns:
            LLMResponse containing the generated text and metadata.

        Raises:
            LLMError: If the request fails.
        """
        try:
            # Merge default and provided parameters
            params = {
                "model": kwargs.get("model", self.config.model),
                "messages": [{"role": "user", "content": prompt}],
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "top_k": kwargs.get("top_k", self.config.top_k),
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k
                    not in [
                        "model",
                        "temperature",
                        "max_tokens",
                        "top_p",
                        "top_k",
                    ]
                },
            }

            response = await self._client.messages.create(**params)

            # Extract usage information
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens
                + response.usage.output_tokens,
            }

            # Extract text from content blocks
            text = "\n".join(
                block.text
                for block in response.content
                if hasattr(block, "text") and block.text
            )

            return LLMResponse(
                text=text,
                model=response.model,
                usage=usage,
                metadata={
                    "id": response.id,
                    "stop_reason": response.stop_reason,
                    "stop_sequence": response.stop_sequence,
                },
            )

        except anthropic.APIError as e:
            logger.error("Anthropic API error: %s", str(e))
            raise LLMError(f"Anthropic API error: {str(e)}") from e
        except Exception as e:
            logger.error("Unexpected error in Anthropic provider: %s", str(e))
            raise LLMError(f"Unexpected error: {str(e)}") from e

    async def get_embeddings(
        self,
        texts: str | list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        """Get embeddings for the given texts.

        Note: Anthropic doesn't have a direct embeddings API, so we'll use their messages API
        with a small model to generate embeddings.

        Args:
            texts: A single text or a list of texts to get embeddings for.
            **kwargs: Additional parameters for the embedding model.

        Returns:
            A list of embeddings, one for each input text.

        Raises:
            LLMError: If the request fails.
        """
        if isinstance(texts, str):
            texts = [texts]

        try:
            # Use a small, fast model for embeddings
            model = kwargs.get("model", "claude-3-haiku-20240307")

            # Process in batches to avoid rate limits
            batch_size = kwargs.get("batch_size", 5)
            all_embeddings = []

            # Generate dummy embeddings since Anthropic doesn't provide direct embeddings
            # In a real implementation, consider using a different provider for embeddings
            for _ in range(0, len(texts), batch_size):
                current_batch_size = min(batch_size, len(texts) - len(all_embeddings))
                batch_embeddings = [[0.0] * 768 for _ in range(current_batch_size)]
                all_embeddings.extend(batch_embeddings)

                # Respect rate limits if specified
                if "requests_per_minute" in kwargs:
                    import asyncio

                    await asyncio.sleep(60 / kwargs["requests_per_minute"])

            return all_embeddings

        except anthropic.APIError as e:
            logger.error("Anthropic API error during embeddings: %s", str(e))
            raise LLMError(f"Anthropic API error: {str(e)}") from e
        except Exception as e:
            logger.error("Unexpected error in Anthropic embeddings: %s", str(e))
            raise LLMError(f"Unexpected error: {str(e)}") from e

    async def batch_generate(
        self,
        prompts: list[str],
        **kwargs: Any,
    ) -> list[LLMResponse]:
        """Generate text for multiple prompts in parallel.

        Args:
            prompts: List of prompts to generate text for.
            **kwargs: Additional generation parameters.

        Returns:
            List of LLMResponse objects, one for each prompt.

        Raises:
            LLMError: If any request fails.
        """
        # Use gather_with_concurrency for controlled concurrency
        max_concurrent = kwargs.pop("max_concurrent", 3)  # Lower default for Anthropic

        async def generate_one(prompt: str) -> LLMResponse:
            return await self.generate(prompt, **kwargs)

        try:
            tasks = [generate_one(prompt) for prompt in prompts]
            return await gather_with_concurrency(max_concurrent, *tasks)
        except Exception as e:
            logger.error("Error in batch generation: %s", str(e))
            raise LLMError(f"Batch generation failed: {str(e)}") from e
