"""Ollama provider implementation for LLM analysis using the official Python module."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, Optional, Union, AsyncIterator

import httpx
from ollama import AsyncClient
from myjobspyai.utils import with_retry

from myjobspyai.exceptions import LLMError, RateLimitExceeded
from .base import BaseProvider

logger = logging.getLogger(__name__)

class OllamaClient(BaseProvider):
    """Client for interacting with Ollama's API using the official Python module."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Ollama client with enhanced configuration.
        
        Args:
            config: Configuration dictionary with API keys and settings
        """
        # Extract provider-specific config if nested under 'ollama' key
        ollama_config = config.get('ollama', {})
        if not isinstance(ollama_config, dict):
            ollama_config = {}
            
        # Merge top-level config with provider-specific config
        merged_config = {**config, **ollama_config}
        
        # Ensure we have a model specified
        if 'model' not in merged_config:
            raise ValueError("No model specified in Ollama configuration. Please provide a 'model' in the config.")
            
        super().__init__(merged_config)
        
        # Clean and validate the model name
        model_name = str(merged_config['model']).strip("'\"")  # Remove any surrounding quotes
        if not model_name:
            raise ValueError("Model name cannot be empty in Ollama configuration.")
            
        self.model_name = model_name
        logger.debug("OllamaClient initialized with model: %s (from config: %s)", 
                   self.model_name, merged_config.get('model'))
        self.base_url = str(merged_config.get("base_url", "http://localhost:11434")).rstrip('/')
        self.timeout = float(merged_config.get("timeout", 600.0))  # Increased default timeout to 10 minutes
        self.connect_timeout = float(merged_config.get("connect_timeout", 60.0))  # Increased connect timeout to 60s
        self.max_retries = int(merged_config.get("max_retries", 5))  # Increased max retries
        self.keep_alive = str(merged_config.get("keep_alive", "5m"))  # Keep model loaded for 5 minutes
        self.request_timeout = float(merged_config.get("request_timeout", 300.0))  # Per-request timeout
        self._client = None
        self._session = None  # For connection pooling
        self._last_used = 0  # Track last API call time
        self._model_loaded = None  # Track currently loaded model
        
        # Configure logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.setLevel(logging.DEBUG if merged_config.get("debug", False) else logging.INFO)
        
        self.logger.debug("Initialized Ollama client with model: %s", self.model_name)
    
    async def _verify_model_available(self, model: str) -> str:
        """Verify if the specified model is available on the Ollama server.
        
        Args:
            model: The model name to verify
            
        Returns:
            The verified model name (might be different if an alias was found)
            
        Raises:
            LLMError: If the model is not found or there's an error checking
        """
        if not model:
            raise ValueError("Model name cannot be empty")
            
        self.logger.debug("Verifying model: %s", model)
        
        try:
            # First try direct model info - most reliable check
            try:
                self.logger.debug("Attempting direct model info check for: %s", model)
                model_info = await self._client.show(model)
                if model_info:
                    model_name = getattr(model_info, 'name', model)
                    self.logger.debug("Verified model via direct check: %s", model_name)
                    return model_name
            except Exception as e:
                self.logger.debug("Direct model check failed, falling back to list: %s", str(e))
            
            # Get list of all available models for more detailed error reporting
            self.logger.debug("Fetching list of all available models")
            try:
                models = await self._client.list()
                self.logger.debug("Raw models response: %s", str(models)[:500])  # Log first 500 chars
                
                # Extract model names from different response formats
                model_list = []
                
                # Handle case where models is an object with 'models' attribute
                if hasattr(models, 'models'):
                    model_objects = models.models
                    if model_objects and hasattr(model_objects[0], 'name'):
                        model_list = [str(m.name) for m in model_objects]
                    else:
                        model_list = [str(m) for m in model_objects]
                # Handle case where models is a dict with 'models' key
                elif isinstance(models, dict) and 'models' in models:
                    model_objects = models['models']
                    model_list = [str(m.get('name', m)) for m in model_objects]
                # Handle case where models is a list
                elif isinstance(models, (list, tuple)):
                    model_list = [str(m.get('name', m) if hasattr(m, 'get') else m) for m in models]
                
                # Clean up the model list
                model_list = [m.strip() for m in model_list if m and isinstance(m, str)]
                self.logger.debug("Available Ollama models: %s", model_list)
                
                if not model_list:
                    self.logger.warning("No models found on the Ollama server")
                    return model  # Return original and let the API call fail
                
                # Try different matching strategies
                target_model = model.strip().lower()
                
                # 1. Try exact match (case-insensitive)
                exact_matches = [m for m in model_list if m.lower() == target_model]
                if exact_matches:
                    self.logger.debug("Found exact match: %s", exact_matches[0])
                    return exact_matches[0]
                
                # 2. Try matching model name without tag/version
                base_model = target_model.split(':')[0].strip()
                base_matches = [m for m in model_list if m.lower().split(':')[0] == base_model]
                
                if base_matches:
                    # If only one match, use it
                    if len(base_matches) == 1:
                        self.logger.debug("Found base model match: %s", base_matches[0])
                        return base_matches[0]
                    # If multiple matches, prefer the one with the same tag if specified
                    if ':' in target_model:
                        tag = target_model.split(':', 1)[1]
                        tag_matches = [m for m in base_matches if ':' in m and m.split(':', 1)[1].lower() == tag]
                        if tag_matches:
                            self.logger.debug("Found tag match: %s", tag_matches[0])
                            return tag_matches[0]
                    # Otherwise return the first match
                    self.logger.debug("Using first base model match: %s", base_matches[0])
                    return base_matches[0]
                
                # 3. Try partial match (in case of small differences in naming)
                partial_matches = [m for m in model_list if base_model in m.lower()]
                if partial_matches:
                    self.logger.debug("Found partial match: %s", partial_matches[0])
                    return partial_matches[0]
                
                # If we get here, the model wasn't found
                error_msg = (
                    f"Model '{model}' not found in Ollama.\n"
                    f"Available models:\n- " + "\n- ".join(sorted(model_list)) + "\n"
                    f"\nTry running: ollama pull {model}"
                )
                self.logger.error(error_msg)
                raise LLMError(error_msg)
                
            except Exception as e:
                self.logger.error("Error listing models: %s", str(e), exc_info=True)
                raise LLMError(f"Failed to list models from Ollama: {str(e)}") from e
                
        except Exception as e:
            if not isinstance(e, LLMError):
                error_msg = f"Error verifying model '{model}': {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                raise LLMError(error_msg) from e
            raise  # Re-raise LLMError as is
    
    async def _initialize_client(self) -> 'OllamaClient':
        """Initialize the Ollama client with enhanced configuration and connection handling."""
        if self._client is not None:
            return self
            
        # Ensure base_url is properly formatted
        base_url = self.base_url
        if not base_url.startswith(('http://', 'https://')):
            base_url = f"http://{base_url}"
            
        # Get the model from config (should be set by now)
        model = self.config.get('model')
        if not model:
            raise ValueError("No model specified in Ollama configuration")
            
        self.logger.debug(
            "Initializing Ollama client with config: base_url=%s, model=%s, timeout=%.1f, connect_timeout=%.1f, max_retries=%d",
            base_url, model, self.timeout, self.connect_timeout, self.max_retries
        )
        
        try:
            # Initialize the Ollama client with basic configuration
            self._client = AsyncClient(
                host=base_url,
                timeout=self.timeout,
            )
            
            # Verify the model is available and update the model name if needed
            verified_model = await self._verify_model_available(model)
            if verified_model != model:
                self.logger.info("Using model alias: %s (requested: %s)", verified_model, model)
                self.config['model'] = verified_model
                self.model_name = verified_model
            
            self._model_loaded = verified_model
            self.logger.info("Ollama client initialized successfully with model: %s", self._model_loaded)
            return self
            
        except Exception as e:
            error_msg = f"Failed to initialize Ollama client: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            # Clean up if initialization fails
            if hasattr(self, '_client') and self._client:
                if hasattr(self._client, 'close'):
                    await self._client.close()
                elif hasattr(self._client, 'aclose'):
                    await self._client.aclose()
            self._client = None
            raise LLMError(error_msg) from e
            
    def __await__(self):
        """Allow the client to be awaited."""
        return self._await_impl().__await__()
    
    async def _await_impl(self):
        """Implementation of the awaitable protocol."""
        if self._client is None:
            await self._initialize_client()
        return self
    
    async def _verify_connection(self) -> None:
        """Verify connection to Ollama server with retries and detailed diagnostics."""
        max_retries = 3
        retry_delay = 2.0  # seconds
        
        for attempt in range(1, max_retries + 1):
            try:
                # Simple health check endpoint
                health_url = f"{self.base_url}/api/health"
                self.logger.debug("Checking Ollama server health at %s (attempt %d/%d)", 
                                health_url, attempt, max_retries)
                
                async with httpx.AsyncClient(timeout=10.0) as client:
                    response = await client.get(health_url)
                    response.raise_for_status()
                    
                # If we get here, health check passed
                self.logger.debug("Ollama server is healthy")
                
                # Also verify we can list models
                models = await self._client.list()
                if not models.get('models'):
                    self.logger.warning("No models found in Ollama instance")
                else:
                    self.logger.debug("Verified %d models available", len(models['models']))
                
                return  # Success!
                
            except httpx.HTTPStatusError as e:
                error_msg = f"Ollama server returned error: {e.response.status_code} {e.response.text}"
                if attempt < max_retries:
                    self.logger.warning("%s - Retrying... (attempt %d/%d)", 
                                      error_msg, attempt, max_retries)
                    await asyncio.sleep(retry_delay * attempt)
                    continue
                self.logger.error(error_msg, exc_info=True)
                raise ConnectionError(error_msg) from e
                
            except (httpx.RequestError, httpx.TimeoutException) as e:
                error_msg = f"Failed to connect to Ollama server: {str(e)}"
                if attempt < max_retries:
                    self.logger.warning("%s - Retrying in %.1fs (attempt %d/%d)", 
                                      error_msg, retry_delay * attempt, attempt, max_retries)
                    await asyncio.sleep(retry_delay * attempt)
                    continue
                self.logger.error(error_msg, exc_info=True)
                raise ConnectionError(error_msg) from e
                
            except Exception as e:
                error_msg = f"Unexpected error verifying Ollama connection: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                if attempt >= max_retries:
                    raise ConnectionError(error_msg) from e
                await asyncio.sleep(retry_delay * attempt)
    
    async def _verify_model_availability(self, model: str) -> None:
        """Verify that the specified model is available in Ollama.
        
        Args:
            model: Model name to verify
        """
        if not model:
            return
            
        try:
            # If we already have a client, use it to check models
            if self._client is not None:
                models = await self._client.list()
                base_model = model.split(':')[0]  # Handle tags like 'llama2:13b'
                
                # Handle different response formats
                available_models = []
                
                # Handle case where models is a dict with 'models' key
                if isinstance(models, dict) and 'models' in models:
                    available_models = [
                        m.get('name') if isinstance(m, dict) else str(m) 
                        for m in models['models']
                        if m is not None
                    ]
                # Handle case where models has a 'models' attribute
                elif hasattr(models, 'models'):
                    model_list = models.models
                    available_models = [
                        getattr(m, 'name', str(m)) 
                        for m in model_list
                        if m is not None
                    ]
                # Handle case where models is a list
                elif isinstance(models, (list, tuple)):
                    available_models = [
                        getattr(m, 'name', str(m)) 
                        for m in models
                        if m is not None
                    ]
                
                # Convert all model names to strings and filter out any None values
                available_models = [str(m) for m in available_models if m is not None]
                
                # Check if either the exact model or base model is available
                if not (any(m.startswith(f"{base_model}:") for m in available_models) or 
                       any(m == base_model for m in available_models)):
                    raise LLMError(
                        f"Model '{model}' not found in Ollama. "
                        f"Available models: {available_models}"
                    )
                logger.debug("Verified model availability: %s", model)
        except Exception as e:
            if not isinstance(e, LLMError):
                logger.warning("Failed to verify model availability: %s", str(e), exc_info=True)
    
    @with_retry(
        max_attempts=5,  # Increased from 3 to 5
        base_delay=1.0,
        max_delay=30.0,  # Increased max delay for longer retry cycles
        exceptions=(
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
            httpx.NetworkError,
            httpx.TimeoutException,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpx.ReadError,
            httpx.WriteError,
            httpx.RemoteProtocolError,
        ),
        reraise=True
    )
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = False,
        timeout: Optional[float] = None,
        **kwargs
    ) -> Union[str, AsyncIterator[str]]:
        """Generate text using the Ollama API.
        
        Args:
            prompt: The prompt to generate text from
            model: The model to use for generation (defaults to config model)
            temperature: Controls randomness (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            The generated text or an async generator for streaming responses
            
        Raises:
            LLMError: If there's an error generating text
        """
        # Ensure client is initialized and get model
        if self._client is None:
            await self._initialize_client()
            
        # Use the model from config if not provided
        model = model or self.config.get('model')
        if not model:
            raise LLMError("No model specified and no default model in config")
            
        # Handle streaming case
        if stream:
            return await self._stream_generate(
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
        
        try:
            # Verify model is available
            await self._verify_model_availability(model)
            
            # Verify model availability with timeout
            try:
                await asyncio.wait_for(self._verify_model_availability(model), timeout=30.0)
            except asyncio.TimeoutError:
                self.logger.warning("Model verification timed out, attempting to continue...")
                # Continue anyway as the model might still be usable
            
            # Prepare generation options with enhanced defaults
            options = {
                'temperature': max(0.0, min(1.0, float(temperature))),  # Clamp to 0-1 range
                'num_predict': max(1, min(10000, int(max_tokens))),  # Clamp to reasonable range
                'top_p': 0.9,
                'repeat_penalty': 1.1,
                'top_k': 40,
                'stop': ['\n###', '\n\n'],
                **{k: v for k, v in kwargs.items() if v is not None and k not in ['model', 'prompt', 'stream']}
            }
            
            # Log the request (without the full prompt in production)
            self.logger.debug("Sending request to Ollama API")
            self.logger.debug("Model: %s, Options: %s", model, options)
            
            # Calculate timeout - use min of remaining timeout and configured timeout
            request_timeout = min(
                float(timeout) if timeout is not None else float('inf'),
                self.request_timeout
            )
            
            # Make the request with timeout
            start_time = time.time()
            response = await asyncio.wait_for(
                self._client.generate(
                    model=model,
                    prompt=prompt,
                    options=options,
                    stream=False
                ),
                timeout=request_timeout
            )
            
            # Log response time
            elapsed = time.time() - start_time
            self.logger.debug("Received Ollama API response in %.2fs: %s", elapsed, response)
            
            # Handle different response formats
            if not response:
                raise LLMError("Empty response from Ollama API")
                
            # Handle case where response is already a string
            if isinstance(response, str):
                response_text = response.strip()
            # Handle case where response is a dictionary with 'response' key
            elif isinstance(response, dict) and 'response' in response:
                response_text = response['response']
            # Handle case where response is an object with 'response' attribute
            elif hasattr(response, 'response'):
                response_text = response.response
            # Handle case where response is an object with 'message' attribute (some API versions)
            elif hasattr(response, 'message'):
                response_text = response.message
            # Handle case where response is an object with 'content' attribute
            elif hasattr(response, 'content'):
                response_text = response.content
            else:
                # Try to get the response text using common attributes
                for attr in ['text', 'output', 'result']:
                    if hasattr(response, attr):
                        response_text = getattr(response, attr)
                        break
                else:
                    # Last resort: convert to string and clean up
                    response_text = str(response).replace('\n', ' ').strip()
            
            if not response_text:
                error_msg = response.get('error', 'No response content') if isinstance(response, dict) else 'Empty response content'
                raise LLMError(f"Ollama API error: {error_msg}")
                
            # Update last used timestamp
            self._last_used = time.time()
            
            return response_text.strip()
            
        except Exception as e:
            if hasattr(e, 'status_code') and getattr(e, 'status_code') == 429:
                raise RateLimitExceeded(f"Ollama rate limit exceeded: {str(e)}") from e
            raise LLMError(f"Ollama API error: {str(e)}") from e

    async def _stream_generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> AsyncIterator[str]:
        """Generate text using streaming mode with enhanced error handling and configuration.
        
        Args:
            prompt: The prompt to send to the model
            model: The model to use for generation
            temperature: Controls randomness (0-2)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional arguments to pass to the API
            
        Yields:
            Chunks of generated text
            
        Raises:
            LLMError: If the streaming fails
            RateLimitExceeded: If rate limit is exceeded
        """
        try:
            # Ensure client is initialized
            if self._client is None:
                await self._initialize_client()
                
            # Verify model is available
            await self._verify_model_availability(model)
            
            # Prepare generation options
            options = {
                'temperature': temperature,
                'num_predict': max_tokens,
                **{k: v for k, v in kwargs.items() if v is not None}
            }
            
            # Make the streaming request
            stream = await self._client.generate(
                model=model,
                prompt=prompt,
                options=options,
                stream=True
            )
            
            # Stream the response
            async for chunk in stream:
                if 'response' in chunk:
                    yield chunk['response']
                    
        except Exception as e:
            if hasattr(e, 'status_code') and getattr(e, 'status_code') == 429:
                raise RateLimitExceeded(f"Ollama rate limit exceeded: {str(e)}") from e
            raise LLMError(f"Ollama streaming error: {str(e)}") from e
    
    async def close(self) -> None:
        """Close the client and release any resources."""
        if self._client is not None:
            try:
                # The AsyncClient might have resources to clean up
                if hasattr(self._client, 'close'):
                    await self._client.close()
                self.logger.debug("Ollama client closed successfully")
            except Exception as e:
                self.logger.warning("Error closing Ollama client: %s", str(e))
            finally:
                self._client = None
                self._model_loaded = None
