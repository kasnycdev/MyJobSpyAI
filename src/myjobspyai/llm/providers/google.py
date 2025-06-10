"""Google provider implementation for LLM integration."""

from __future__ import annotations

import logging
import os
from typing import Any

import google.generativeai as genai
from pydantic import BaseModel, Field, field_validator

from myjobspyai.llm.base import BaseLLMProvider, LLMError, LLMResponse
from myjobspyai.utils.async_utils import gather_with_concurrency

logger = logging.getLogger(__name__)


class GoogleConfig(BaseModel):
    """Configuration for the Google provider."""

    model: str = Field(
        default="gemini-1.5-pro",
        description="The model to use for text generation.",
    )
    api_key: str | None = Field(
        default=None,
        description=(
            "Google API key. If not provided, will be read from "
            "GOOGLE_API_KEY environment variable."
        ),
    )
    temperature: float = Field(
        default=0.7,
        description="Sampling temperature (0-1)",
        ge=0.0,
        le=1.0,
    )
    max_output_tokens: int = Field(
        default=2048,
        description="Maximum number of tokens to generate",
        ge=1,
    )
    top_p: float = Field(
        default=0.95,
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
    def validate_api_key(cls, v: str | None) -> str:
        """Validate API key or get it from environment."""
        if v is None:
            v = os.getenv("GOOGLE_API_KEY")
            if not v:
                msg = "API key not provided and GOOGLE_API_KEY environment variable not set"
                raise ValueError(msg)
        return v


class GoogleProvider(BaseLLMProvider):
    """Google provider implementation for LLM integration."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """Initialize the Google provider.

        Args:
            config: Configuration dictionary. If None, uses defaults from environment.
        """
        super().__init__("google")
        self.config = GoogleConfig(**(config or {}))

        # Configure the Google API client
        genai.configure(api_key=self.config.api_key)

        # Initialize the model
        self._model = genai.GenerativeModel(self.config.model)
        self._chat = None

    async def close(self) -> None:
        """Close the provider and release resources."""
        # Google's API client doesn't require explicit cleanup
        self._chat = None

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
            # Start a new chat session if this is the first message
            if self._chat is None:
                self._chat = self._model.start_chat(history=[])

            # Prepare generation config
            gen_config = {
                "temperature": kwargs.get("temperature", self.config.temperature),
                "max_output_tokens": kwargs.get(
                    "max_output_tokens", self.config.max_output_tokens
                ),
                "top_p": kwargs.get("top_p", self.config.top_p),
                "top_k": kwargs.get("top_k", self.config.top_k),
            }

            # Filter out generation config from additional kwargs
            additional_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in ["temperature", "max_output_tokens", "top_p", "top_k"]
            }

            # Send the message and get the response
            response = await self._chat.send_message_async(
                prompt,
                generation_config=gen_config,
                **additional_kwargs,
            )

            # Extract the response text
            text = response.text

            # Get usage information
            usage = {
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
                "total_tokens": (
                    response.usage_metadata.prompt_token_count
                    + response.usage_metadata.candidates_token_count
                ),
            }

            return LLMResponse(
                text=text,
                model=self.config.model,
                usage=usage,
                metadata={
                    "safety_ratings": (
                        [
                            {
                                "category": rating.category.name,
                                "probability": rating.probability.name,
                            }
                            for rating in response.candidates[0].safety_ratings
                        ]
                        if (
                            hasattr(response, "candidates")
                            and response.candidates
                            and hasattr(response.candidates[0], "safety_ratings")
                        )
                        else {}
                    )
                },
            )

        except Exception as e:
            logger.error("Error in Google provider: %s", str(e))
            raise LLMError(f"Google API error: {str(e)}") from e

    async def get_embeddings(
        self,
        texts: str | list[str],
        **kwargs: Any,
    ) -> list[list[float]]:
        """Get embeddings for the given texts.

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
            model = kwargs.get("model", "models/embedding-001")
            batch_size = kwargs.get("batch_size", 100)

            # Process in batches to avoid rate limits
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = await genai.embed_content_async(
                    model=model,
                    content=batch,
                    task_type="retrieval_document",
                )

                # Extract embeddings from response
                embeddings = response.get("embedding", [])
                all_embeddings.extend(
                    embedding["values"] for embedding in embeddings  # type: ignore
                )

                # Respect rate limits if specified
                if "requests_per_minute" in kwargs:
                    import asyncio

                    await asyncio.sleep(60 / kwargs["requests_per_minute"])

            return all_embeddings

        except Exception as e:
            logger.error("Error in Google embeddings: %s", str(e))
            raise LLMError(f"Google embeddings error: {str(e)}") from e

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
        max_concurrent = kwargs.pop(
            "max_concurrent", 5
        )  # Default to 5 concurrent requests

        async def generate_one(prompt: str) -> LLMResponse:
            return await self.generate(prompt, **kwargs)

        try:
            tasks = [generate_one(prompt) for prompt in prompts]
            return await gather_with_concurrency(max_concurrent, *tasks)
        except Exception as e:
            logger.error("Error in batch generation: %s", str(e))
            raise LLMError(f"Batch generation failed: {str(e)}") from e
