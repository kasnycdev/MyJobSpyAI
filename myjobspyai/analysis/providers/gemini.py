"""Google Gemini provider implementation for LLM analysis."""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional, Union, AsyncGenerator

import google.generativeai as genai
from myjobspyai.utils import with_retry

from myjobspyai.exceptions import LLMError, RateLimitExceeded, ConfigurationError
from .base import BaseProvider

logger = logging.getLogger(__name__)

class GeminiClient(BaseProvider):
    """Client for interacting with Google's Gemini API."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Gemini client.
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        super().__init__(config)
        self._model = None
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the Gemini client with configuration."""
        api_key = self.config.get("api_key") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ConfigurationError("Google API key not found in config or environment")
        
        # Configure the Gemini client
        genai.configure(api_key=api_key)
        
        # Set default model
        self._model_name = self.config.get("model", "gemini-pro")
    
    @with_retry(
        max_attempts=3,
        base_delay=4.0,
        max_delay=10.0,
        exceptions=(Exception,),
        reraise=True
    )
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate text using the Gemini API.
        
        Args:
            prompt: The prompt to send to the model
            model: The model to use for generation (defaults to config)
            temperature: Controls randomness (0-2)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to use streaming mode
            **kwargs: Additional arguments to pass to the API
            
        Returns:
            The generated text (non-streaming) or an async generator of chunks (streaming)
            
        Raises:
            LLMError: If the API call fails
        """
        if stream:
            return await self._stream_generate(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        
        try:
            # Use provided model or default from config
            model_name = model or self._model_name
            
            # Create the model instance if not already created
            if self._model is None or model_name != self._model_name:
                self._model = genai.GenerativeModel(model_name)
                self._model_name = model_name
            
            # Generate content
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                        **kwargs
                    }
                )
            )
            
            if not response.text:
                raise LLMError("Empty response from Gemini API")
                
            return response.text.strip()
            
        except Exception as e:
            error_msg = str(e).lower()
            if "quota" in error_msg or "rate limit" in error_msg:
                raise RateLimitExceeded(f"Gemini rate limit exceeded: {str(e)}") from e
            elif "model" in error_msg and "not found" in error_msg:
                raise LLMError(f"Model not available: {str(e)}") from e
            elif "invalid api key" in error_msg or "permission denied" in error_msg:
                raise LLMError(f"Gemini authentication failed: {str(e)}") from e
            else:
                raise LLMError(f"Gemini API error: {str(e)}") from e
    
    async def close(self) -> None:
        """Clean up resources."""
        # No explicit cleanup needed for Gemini client
        pass
