"""Base classes for LLM providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    text: str
    """The generated text."""

    model: str
    """The model used to generate the text."""

    usage: Dict[str, Optional[int]]
    """Token usage information."""

    metadata: Dict[str, Any] = None
    """Additional metadata from the provider."""

    def __post_init__(self) -> None:
        """Initialize metadata if not provided."""
        if self.metadata is None:
            self.metadata = {}


class LLMError(Exception):
    """Base exception for LLM provider errors."""

    pass


class LLMRequestError(LLMError):
    """Exception raised when an LLM API request fails."""

    pass


class BaseLLMProvider(ABC):
    """Base class for LLM providers."""

    def __init__(self, provider_name: str) -> None:
        """Initialize the provider.

        Args:
            provider_name: Name of the provider.
        """
        self.provider_name = provider_name

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

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
        return [await self.generate(prompt, **kwargs) for prompt in prompts]

    async def close(self) -> None:
        """Close the provider and release resources.

        Subclasses should override this if they need to clean up resources.
        """
        pass

    def __str__(self) -> str:
        """Return a string representation of the provider."""
        return f"{self.__class__.__name__}(provider={self.provider_name})"

    async def __aenter__(self):
        """Support async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting async context."""
        await self.close()
