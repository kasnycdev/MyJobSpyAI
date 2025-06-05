"""LangChain provider implementation for LLM integration."""

import os
import logging
from typing import Any, Dict, Optional

from myjobspyai.analysis.providers.langchain_provider import LangChainProvider as AnalysisLangChainProvider
from myjobspyai.analysis.providers.base import BaseProvider

logger = logging.getLogger(__name__)

class LangChainProvider(BaseProvider):
    """LangChain provider implementation for LLM integration."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the LangChain provider.
        
        Args:
            config: Configuration dictionary with the following keys:
                - provider: The underlying provider to use (e.g., 'openai', 'anthropic')
                - model: The model to use (e.g., 'gpt-4', 'claude-2')
                - temperature: Temperature for generation (0.0 to 1.0)
                - max_tokens: Maximum number of tokens to generate
                - api_key: API key for the provider (optional, can be set via environment variable)
        """
        self.config = config
        self.provider = config.get("provider", "openai").lower()
        self.model = config.get("model", "gpt-4")
        self.temperature = float(config.get("temperature", 0.7))
        self.max_tokens = int(config.get("max_tokens", 1000))
        
        # Set API key from config or environment variable
        self.api_key = config.get("api_key") or os.getenv(f"{self.provider.upper()}_API_KEY")
        if not self.api_key and self.provider == "openai":
            self.api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize the analysis LangChain provider
        self._provider = AnalysisLangChainProvider({
            "provider": self.provider,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "api_key": self.api_key,
        })
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: The prompt to generate text from
            **kwargs: Additional arguments to pass to the LLM
            
        Returns:
            Generated text
            
        Raises:
            Exception: If there's an error generating the response
        """
        try:
            # Use the analysis provider's generate method
            response = await self._provider.generate(
                prompt=prompt,
                model=kwargs.get("model", self.model),
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                **{k: v for k, v in kwargs.items() if k not in ["model", "temperature", "max_tokens"]}
            )
            return response.text if hasattr(response, 'text') else str(response)
        except Exception as e:
            logger.error(f"Error in LangChain provider generate: {str(e)}")
            raise
    
    async def close(self):
        """Close the provider's resources."""
        await self._provider.close()
        
    def __str__(self) -> str:
        """Return a string representation of the provider."""
        return f"LangChainProvider(provider={self.provider}, model={self.model})"


class SyncLangChainProvider:
    """Synchronous wrapper for the LangChain provider."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the synchronous LangChain provider.
        
        Args:
            config: Configuration dictionary for the provider
        """
        self._provider = LangChainProvider(config)
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt synchronously.
        
        Args:
            prompt: The prompt to generate text from
            **kwargs: Additional arguments to pass to the LLM
            
        Returns:
            Generated text
        """
        import asyncio
        
        try:
            return asyncio.run(self._provider.generate(prompt, **kwargs))
        except Exception as e:
            logger.error(f"Error in synchronous LangChain provider generate: {str(e)}")
            raise
    
    def close(self):
        """Close the provider's resources synchronously."""
        import asyncio
        
        try:
            asyncio.run(self._provider.close())
        except Exception as e:
            logger.error(f"Error closing LangChain provider: {str(e)}")
            raise
