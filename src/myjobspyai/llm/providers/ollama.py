"""Ollama LLM provider implementation using the enhanced HTTP client."""

import json
from typing import Any, Optional, Union
from urllib.parse import urlparse, urlunparse

from pydantic import BaseModel, Field, field_validator

from myjobspyai.llm.base import BaseLLMProvider, LLMError, LLMRequestError, LLMResponse
from myjobspyai.utils.async_utils import gather_with_concurrency
from myjobspyai.utils.http_client import HTTPClientError


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

    # Constants for validation
    MIN_TEMPERATURE: float = 0.0
    MAX_TEMPERATURE: float = 2.0
    MIN_TOP_P: float = 0.0
    MAX_TOP_P: float = 1.0
    DEFAULT_HTTP_PORT: int = 11434
    DEFAULT_HTTPS_PORT: int = 443

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        """Validate the temperature parameter.

        Args:
            value: The temperature value to validate.

        Returns:
            The validated temperature value.

        Raises:
            ValueError: If temperature is not between MIN_TEMPERATURE and MAX_TEMPERATURE.
        """
        min_temp = cls.MIN_TEMPERATURE
        max_temp = cls.MAX_TEMPERATURE
        if not min_temp <= value <= max_temp:
            msg = f"Temperature must be between {min_temp} and " f"{max_temp}"
            raise ValueError(msg)
        return value

    @field_validator("top_p")
    @classmethod
    def validate_top_p(cls, value: float) -> float:
        """Validate the top_p parameter.

        Args:
            value: The top_p value to validate.

        Returns:
            The validated top_p value.

        Raises:
            ValueError: If top_p is not between MIN_TOP_P and MAX_TOP_P.
        """
        min_p = cls.MIN_TOP_P
        max_p = cls.MAX_TOP_P
        if not min_p < value <= max_p:
            msg = f"top_p must be between {min_p} (exclusive) and " f"{max_p}"
            raise ValueError(msg)
        return value

    @field_validator("base_url", mode="before")
    def validate_base_url(cls, value: Any) -> str:
        """Validate and normalize the base URL.

        Args:
            value: The base URL to validate and normalize.

        Returns:
            The normalized base URL.

        Raises:
            ValueError: If the URL is invalid.
        """
        if not value:
            return "http://localhost:11434"

        if not isinstance(value, str):
            value = str(value)

        # Parse the URL
        parsed = urlparse(value)

        # Ensure scheme is present
        if not parsed.scheme:
            parsed = parsed._replace(scheme="http")

        # Set default port if not specified
        netloc = parsed.netloc
        if ":" not in netloc and parsed.scheme in ("http", "https"):
            port = (
                cls.DEFAULT_HTTP_PORT
                if parsed.scheme == "http"
                else cls.DEFAULT_HTTPS_PORT
            )
            netloc = f"{netloc}:{port}"

        # Reconstruct URL with normalized components
        normalized = parsed._replace(
            netloc=netloc,
            path=parsed.path.rstrip("/") or "",
            params="",
            query="",
            fragment="",
        )

        # Ensure the URL is valid by parsing it again
        try:
            parsed_normalized = urlparse(urlunparse(normalized))
            if not parsed_normalized.netloc:
                raise ValueError("Invalid URL: no network location")
            return urlunparse(parsed_normalized)
        except Exception as e:
            raise ValueError(f"Invalid URL: {e}") from e


class OllamaProvider(BaseLLMProvider):
    """Ollama LLM provider implementation."""

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        """Initialize the Ollama provider.

        Args:
            config: Configuration for the provider. If None, uses defaults from config file.
        """
        super().__init__("ollama")
        self.config = OllamaConfig(**(config or {}))
        self._http_client: Optional[Any] = None
        self._initialized = False

    async def _ensure_http_client(self) -> None:
        """Ensure the HTTP client is initialized.

        This method is idempotent and can be called multiple times.
        """
        if self._http_client is None or not self._initialized:
            from myjobspyai.utils.http_factory import get_http_client

            # Ensure string for Pydantic models
            base_url = str(self.config.base_url)
            self._http_client = await get_http_client(
                base_url=base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )
            self._initialized = True

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
        await self._ensure_http_client()

        # Merge default and provided parameters
        params: dict[str, Any] = {
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
                return await self._handle_streaming_response(params)
            return await self._handle_standard_response(params)

        except HTTPClientError as e:
            raise LLMRequestError(f"Request to Ollama API failed: {e}") from e
        except Exception as e:
            raise LLMError(f"Error generating text: {e}") from e

    async def _handle_streaming_response(self, params: dict[str, Any]) -> LLMResponse:
        """Handle a streaming response from the API.

        Args:
            params: The request parameters

        Returns:
            LLMResponse with the generated text and metadata
        """
        full_response: list[str] = []
        async with self._http_client.post(
            "/api/generate",
            json=params,
            stream=True,
        ) as response:
            data = await response.text()
            async for chunk in self._stream_response(data):
                full_response.append(chunk)

        return LLMResponse(
            text="".join(full_response),
            model=self.config.model,
            usage={
                "prompt_tokens": None,  # Not available in streaming mode
                "completion_tokens": None,
                "total_tokens": None,
            },
            metadata={"streamed": True},
        )

    async def _handle_standard_response(self, params: dict[str, Any]) -> LLMResponse:
        """Handle a standard (non-streaming) response from the API.

        Args:
            params: The request parameters

        Returns:
            LLMResponse with the generated text and metadata
        """
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
            metadata={"streamed": False},
        )

    async def _process_stream_chunk(self, chunk: Any) -> str:
        """Process a single chunk from the streaming response.

        Args:
            chunk: A chunk of data from the streaming response

        Returns:
            The extracted text from the chunk if valid, empty string otherwise
        """
        if isinstance(chunk, dict) and "response" in chunk:
            return chunk["response"]
        return ""

    async def _stream_response(self, response: Any) -> str:
        """Handle streaming response from the Ollama API.

        Args:
            response: The raw response from the API (str, list, or dict)

        Yields:
            str: Chunks of the generated text as they become available.

        Raises:
            LLMError: If the response type is not supported
        """
        if isinstance(response, str):
            # Process as newline-delimited JSON
            for line in response.splitlines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    yield await self._process_stream_chunk(chunk)
                except json.JSONDecodeError:
                    continue
        elif isinstance(response, list):
            # Process as list of chunks
            for chunk in response:
                yield await self._process_stream_chunk(chunk)
        elif isinstance(response, dict):
            # Process as single chunk
            yield await self._process_stream_chunk(response)
        else:
            raise LLMError(f"Unsupported response type: {type(response)}")

    async def get_embeddings(
        self,
        texts: Union[str, list[str]],
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
        await self._ensure_http_client()

        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        # Ollama uses the same endpoint for single and batch embeddings
        params = {
            "model": self.config.model,
            "prompt": texts[0] if len(texts) == 1 else texts,
        }

        try:
            response = await self._http_client.post(
                "/api/embeddings",
                json=params,
            )
            data = await response.json()

            if "embedding" in data:
                return [data["embedding"]]
            if "embeddings" in data:
                return data["embeddings"]

            raise LLMError("Invalid response format from Ollama API")

        except HTTPClientError as e:
            raise LLMRequestError(f"Request to Ollama API failed: {e}") from e
        except Exception as e:
            raise LLMError(f"Error getting embeddings: {e}") from e

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
        """
        # Limit concurrency to avoid overwhelming the API
        max_concurrent = kwargs.pop("max_concurrent", 5)

        async def process_prompt(prompt: str) -> LLMResponse:
            return await self.generate(prompt, **kwargs)

        # Process prompts with limited concurrency
        tasks = [process_prompt(prompt) for prompt in prompts]
        return await gather_with_concurrency(max_concurrent, *tasks)
