"""Ollama LLM provider implementation using the enhanced HTTP client."""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Union, cast

from pydantic import BaseModel, Field, HttpUrl, field_validator, ValidationError

from myjobspyai.utils.http_client import HTTPClient, HTTPClientError
from myjobspyai.utils.async_utils import gather_with_concurrency
from myjobspyai.config import config as app_config
from myjobspyai.llm.base import BaseLLMProvider, LLMResponse, LLMError, LLMRequestError

logger = logging.getLogger(__name__)


class OllamaConfig(BaseModel):
    """Configuration for the Ollama provider."""

    base_url: str = Field(
        default="http://localhost:11434", description="Base URL for the Ollama API"
    )
    model: str = Field(
        default="llama3:instruct", description="Model to use for completions"
    )
    timeout: int = Field(default=600, description="Request timeout in seconds")
    max_retries: int = Field(
        default=3, description="Maximum number of retries for failed requests"
    )
    temperature: float = Field(default=0.7, description="Sampling temperature (0-2)")
    top_p: float = Field(default=0.9, description="Nucleus sampling parameter")
    max_tokens: Optional[int] = Field(
        default=None, description="Maximum number of tokens to generate"
    )
    stream: bool = Field(default=False, description="Whether to stream the response")

    @field_validator('temperature')
    def validate_temperature(cls, v):
        if v < 0.0 or v > 2.0:
            raise ValueError('Temperature must be between 0.0 and 2.0')
        return v

    @field_validator('top_p')
    def validate_top_p(cls, v):
        if v <= 0.0 or v > 1.0:
            raise ValueError('top_p must be between 0.0 (exclusive) and 1.0')
        return v


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider implementation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the Ollama provider.

        Args:
            config: Configuration for the provider. If None, uses defaults from config file.
        """
        super().__init__("ollama")

        # Load and validate config
        self.config = OllamaConfig(**(config or {}))

        # Initialize HTTP client
        self._http_client = HTTPClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
        )

    async def close(self) -> None:
        """Close the provider and release resources."""
        await self._http_client.close()

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
        # Merge default and provided parameters
        params = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": self.config.stream,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "num_predict": self.config.max_tokens,
            },
        }

        # Update with any additional parameters
        if "options" in kwargs:
            params["options"].update(kwargs["options"])
            del kwargs["options"]
        params.update(kwargs)

        try:
            if self.config.stream:
                # For streaming responses, collect the chunks
                full_response = []
                async with self._http_client.post(
                    "/api/generate",
                    json=params,
                    stream=True,
                ) as response:
                    data = await response.text()
                    # Process the streaming response
                    async for chunk in self._stream_response(data):
                        full_response.append(chunk)

                full_response_text = "".join(full_response)
                return LLMResponse(
                    text=full_response_text,
                    model=self.config.model,
                    usage={
                        "prompt_tokens": None,  # Not available in streaming mode
                        "completion_tokens": None,  # Not available in streaming mode
                        "total_tokens": None,  # Not available in streaming mode
                    },
                    metadata={
                        "streamed": True,
                    },
                )

            # Non-streaming response
            response = await self._http_client.post(
                "/api/generate",
                json=params,
            )
            data = await response.json()

            return LLMResponse(
                text=data.get("response", ""),
                model=self.config.model,
                usage={
                    "prompt_tokens": data.get("prompt_eval_count"),
                    "completion_tokens": data.get("eval_count"),
                    "total_tokens": None,
                },
                metadata={
                    "streamed": False,
                },
            )

        except HTTPClientError as e:
            raise LLMRequestError(f"Request to Ollama API failed: {e}") from e
        except Exception as e:
            raise LLMError(f"Error generating text: {e}") from e

    async def _stream_response(self, response: Any):
        """Handle streaming response from the Ollama API.
        
        Yields:
            str: Chunks of the generated text as they become available.
        """
        # If response is already a string, it's the raw response data
        if isinstance(response, str):
            for line in response.splitlines():
                if not line.strip():
                    continue

                try:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        yield chunk["response"]
                except json.JSONDecodeError:
                    continue

        # If response is a list, it's a list of chunks
        elif isinstance(response, list):
            for chunk in response:
                if "response" in chunk:
                    yield chunk["response"]

        # If response is a dict, it's a single chunk
        elif isinstance(response, dict) and "response" in response:
            yield response["response"]

        # If we get here, it's an unsupported response type
        else:
            raise LLMError(f"Unsupported response type: {type(response)}")

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

        # Ollama uses the same endpoint for single and batch embeddings
        params = {
            "model": self.config.model,
            "prompt": texts[0] if len(texts) == 1 else texts,
        }

        try:
            response, data = await self._http_client.post(
                "/api/embeddings",
                json_data=params,
            )

            if "embedding" in data:
                return [data["embedding"]]
            elif "embeddings" in data:
                return data["embeddings"]
            else:
                raise LLMError("Invalid response format from Ollama API")

        except HTTPClientError as e:
            raise LLMRequestError(f"Request to Ollama API failed: {e}") from e
        except Exception as e:
            raise LLMError(f"Error getting embeddings: {e}") from e

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
        """
        # Limit concurrency to avoid overwhelming the API
        max_concurrent = kwargs.pop("max_concurrent", 5)

        async def process_prompt(prompt: str) -> LLMResponse:
            return await self.generate(prompt, **kwargs)

        # Process prompts with limited concurrency
        tasks = [process_prompt(prompt) for prompt in prompts]
        return await gather_with_concurrency(max_concurrent, *tasks)
