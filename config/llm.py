"""LLM configuration and client for MyJobSpyAI."""
from typing import Dict, Any, Optional, Union
import logging
from functools import lru_cache
import openai
from openai import OpenAI
from . import settings

logger = logging.getLogger('llm')

class LLMClient:
    """Client for interacting with LLM providers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the LLM client with configuration."""
        self.config = config or settings.settings.llm.model_dump()
        self.client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the LLM client based on the provider."""
        try:
            if self.config['provider'] == 'openai':
                self.client = OpenAI(api_key=self.config['api_key'])
                logger.info(f"Initialized OpenAI client with model: {self.config['model']}")
            else:
                raise ValueError(f"Unsupported LLM provider: {self.config['provider']}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            raise

    @lru_cache(maxsize=100)
    def get_completion(self, prompt: str, **kwargs) -> str:
        """Get completion from the configured LLM provider."""
        try:
            if self.config['provider'] == 'openai':
                response = self.client.chat.completions.create(
                    model=self.config['model'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get('temperature', self.config.get('temperature', 0.7)),
                    max_tokens=kwargs.get('max_tokens', self.config.get('max_tokens', 2000)),
                    **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
                )
                return response.choices[0].message.content.strip()
            else:
                raise ValueError(f"Unsupported LLM provider: {self.config['provider']}")
        except Exception as e:
            logger.error(f"Error getting completion: {str(e)}")
            raise

    async def get_async_completion(self, prompt: str, **kwargs) -> str:
        """Get completion asynchronously."""
        try:
            if self.config['provider'] == 'openai':
                response = await self.client.chat.completions.create(
                    model=self.config['model'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get('temperature', self.config.get('temperature', 0.7)),
                    max_tokens=kwargs.get('max_tokens', self.config.get('max_tokens', 2000)),
                    **{k: v for k, v in kwargs.items() if k not in ['temperature', 'max_tokens']}
                )
                return response.choices[0].message.content.strip()
            else:
                raise ValueError(f"Unsupported LLM provider: {self.config['provider']}")
        except Exception as e:
            logger.error(f"Error getting async completion: {str(e)}")
            raise

# Singleton instance
_llm_client = None

def get_llm_client() -> LLMClient:
    """Get or create the LLM client singleton."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
