"""OpenAI provider implementation for LLM integration."""

import logging
import os
from typing import Any, Dict, List, Optional, Union

from openai import AsyncOpenAI, OpenAIError
from pydantic import Field, field_validator

from myjobspyai.llm.base import BaseLLMProvider, LLMError, LLMResponse
from myjobspyai.utils.async_utils import gather_with_concurrency

logger = logging.getLogger(__name__)


class OpenAIConfig(BaseModel):
    """Configuration for the OpenAI provider."""

    model: str = Field(
        default="gpt-4-turbo-preview",
        description="The model to use for completions",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key. If not provided, will use OPENAI_API_KEY environment variable",
    )
    organization: Optional[str] = Field(
        default=None,
        description="OpenAI organization ID. If not provided, will use OPENAI_ORGANIZATION environment variable",
    )
    base_url: Optional[str] = Field(
        default=None,
        description="Base URL for the OpenAI API. Useful for proxy or self-hosted instances",
    )
    timeout: int = Field(
        default=60,
        description="Request timeout in seconds",
        ge=1,
        le=600,
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for failed requests",
        ge=0,
        le=10,
    )
    temperature: float = Field(
        default=0.7,
        description="Sampling temperature (0-2)",
        ge=0.0,
        le=2.0,
    )
    max_tokens: Optional[int] = Field(
        default=None,
        description="Maximum number of tokens to generate",
        ge=1,
    )
    top_p: float = Field(
        default=1.0,
        description="Nucleus sampling parameter (0-1)",
        ge=0.0,
        le=1.0,
    )

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: Optional[str]) -> str:
        """Validate API key or get it from environment."""
        if v is None:
            v = os.getenv("OPENAI_API_KEY")
            if not v:
                raise ValueError(
                    "OpenAI API key not provided and OPENAI_API_KEY environment variable not set"
                )
        return v


class OpenAIProvider(BaseLLMProvider):
    """OpenAI provider implementation for LLM integration."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the OpenAI provider.

        Args:
            config: Configuration dictionary. If None, uses defaults from environment.
        """
        super().__init__("openai")
        self.config = OpenAIConfig(**(config or {}))
        self._client = AsyncOpenAI(
            api_key=self.config.api_key,
            organization=self.config.organization,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
        )

    async def close(self) -> None:
        """Close the provider and release resources."""
        await self._client.close()

    async def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> LLMResponse:
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
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ["model", "temperature", "max_tokens", "top_p"]
                },
            }

            response = await self._client.chat.completions.create(**params)

            # Extract usage information
            usage = (
                {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
                if response.usage
                else {}
            )

            return LLMResponse(
                text=response.choices[0].message.content or "",
                model=response.model,
                usage=usage,
                metadata={
                    "id": response.id,
                    "created": response.created,
                    "finish_reason": response.choices[0].finish_reason,
                },
            )

        except OpenAIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            raise LLMError(f"OpenAI API error: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI provider: {str(e)}")
            raise LLMError(f"Unexpected error: {str(e)}") from e

    async def get_embeddings(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any,
    ) -> List[List[float]]:
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
            model = kwargs.get("model", "text-embedding-3-small")

            # Process in batches to avoid rate limits
            batch_size = kwargs.get("batch_size", 10)
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                response = await self._client.embeddings.create(
                    model=model,
                    input=batch,
                )

                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)

                # Respect rate limits if specified
                if "requests_per_minute" in kwargs:
                    import asyncio

                    await asyncio.sleep(60 / kwargs["requests_per_minute"])

            return all_embeddings

        except OpenAIError as e:
            logger.error(f"OpenAI embeddings error: {str(e)}")
            raise LLMError(f"OpenAI embeddings error: {str(e)}") from e
        except Exception as e:
            logger.error(f"Unexpected error in OpenAI embeddings: {str(e)}")
            raise LLMError(f"Unexpected error: {str(e)}") from e

    async def batch_generate(
        self,
        prompts: List[str],
        **kwargs: Any,
    ) -> List[LLMResponse]:
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
        max_concurrent = kwargs.pop("max_concurrent", 5)

        async def generate_one(prompt: str) -> LLMResponse:
            return await self.generate(prompt, **kwargs)

        try:
            tasks = [generate_one(prompt) for prompt in prompts]
            return await gather_with_concurrency(max_concurrent, *tasks)
        except Exception as e:
            logger.error(f"Error in batch generation: {str(e)}")
            raise LLMError(f"Batch generation failed: {str(e)}") from e
