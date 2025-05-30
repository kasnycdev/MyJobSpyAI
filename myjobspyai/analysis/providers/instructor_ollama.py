"""Instructor integration with Ollama for structured outputs."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type, TypeVar, Generic

import instructor
from openai import OpenAI
from pydantic import BaseModel

from myjobspyai.exceptions import LLMError
from .base import BaseProvider

T = TypeVar('T', bound=BaseModel)
logger = logging.getLogger(__name__)

class InstructorOllamaClient(BaseProvider, Generic[T]):
    """Client for interacting with Ollama using Instructor for structured outputs.
    
    This client provides structured outputs using Pydantic models with Ollama's LLMs.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Instructor Ollama client.
        
        Args:
            config: Configuration dictionary with the following keys:
                - model: The model name to use (e.g., 'llama3')
                - base_url: The base URL for the Ollama API (default: 'http://localhost:11434/v1')
                - mode: The Instructor mode to use (default: instructor.Mode.JSON)
        """
        super().__init__(config)
        
        self.model = config.get('model', 'llama3')
        self.base_url = config.get('base_url', 'http://localhost:11434/v1')
        self.mode = config.get('mode', instructor.Mode.JSON)
        
        # Initialize the OpenAI client with Ollama's API
        self._client = instructor.from_openai(
            OpenAI(
                base_url=self.base_url,
                api_key="ollama",  # Required but unused
            ),
            mode=self.mode
        )
        
        logger.debug(f"Initialized InstructorOllamaClient with model: {self.model}")
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        response_model: Optional[Type[T]] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> T:
        """Generate structured output from the LLM.
        
        Args:
            prompt: The prompt to send to the model
            model: Override the default model for this request
            response_model: Pydantic model for structured output
            temperature: Controls randomness (0-2)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            An instance of the response_model with the generated data
            
        Raises:
            LLMError: If the generation fails
        """
        if response_model is None:
            raise ValueError("response_model is required for structured generation")
            
        try:
            model_to_use = model or self.model
            
            # Create the completion with structured output
            response = self._client.chat.completions.create(
                model=model_to_use,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                response_model=response_model,
                **kwargs
            )
            
            return response
            
        except Exception as e:
            error_msg = f"Failed to generate structured output: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise LLMError(error_msg) from e
    
    async def close(self) -> None:
        """Close the client and release any resources."""
        if hasattr(self, '_client') and self._client is not None:
            try:
                self._client.close()
            except Exception as e:
                logger.warning(f"Error closing InstructorOllamaClient: {str(e)}")
            finally:
                self._client = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
