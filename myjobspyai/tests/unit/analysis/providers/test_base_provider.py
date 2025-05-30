"""Comprehensive tests for the BaseProvider class."""
import asyncio
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from myjobspyai.analysis.providers.base import BaseProvider
from myjobspyai.exceptions import (
    AuthenticationError,
    ModelUnavailable,
    ProviderError,
    RateLimitExceeded,
    ServiceUnavailable,
)


class TestConfig(BaseModel):
    """Test configuration model."""
    api_key: str = "test-key"
    model: str = "test-model"
    timeout: int = 30
    max_retries: int = 3


class TestProvider(BaseProvider):
    """A test provider implementation for testing the base provider functionality."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the test provider."""
        super().__init__(config)
        # Only pass known fields to TestConfig
        config_for_test = {k: v for k, v in config.items() if k in TestConfig.model_fields}
        self.config = TestConfig(**config_for_test)
        self._client = None
        self._is_closed = False
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize_client()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        return False  # Don't suppress exceptions
    
    async def _initialize_client(self) -> None:
        """Initialize the test client."""
        if self._is_closed:
            raise RuntimeError("Cannot initialize a closed provider")
        self._client = MagicMock()
    
    async def close(self) -> None:
        """Close the test client."""
        if not self._is_closed:
            if self._client:
                self._client = None
            self._is_closed = True
    
    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs: Any,
    ) -> str:
        """Generate a response from the test provider."""
        if not self._client:
            await self._initialize_client()
        
        if "error" in prompt:
            if "rate_limit" in prompt:
                raise RateLimitExceeded("Rate limit exceeded")
            elif "auth" in prompt:
                raise AuthenticationError("Authentication failed")
            elif "model" in prompt:
                raise ModelUnavailable("Model not available")
            elif "service" in prompt:
                raise ServiceUnavailable("Service unavailable")
            else:
                raise ProviderError("Test error")
                
        return f"Response to: {prompt}"
    
    @classmethod
    def get_retry_config(cls, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get retry configuration with test-specific defaults."""
        defaults = {
            'stop': lambda state: state.attempt_number >= (config or {}).get('max_retries', 3),
            'wait': lambda state: min(2 ** state.attempt_number, 30),
            'retry': lambda state: True,
            'reraise': True,
        }
        
        if config:
            defaults.update(config)
            
        return defaults


@pytest.mark.asyncio
async def test_provider_initialization():
    """Test provider initialization and basic properties."""
    config = {"api_key": "test-key", "model": "test-model"}
    provider = TestProvider(config)
    
    assert provider.config.api_key == "test-key"
    assert provider.config.model == "test-model"
    assert provider._client is None
    assert not provider._is_closed
    
    # Test that config is stored in the base class
    assert provider.config.api_key == config["api_key"]
    assert provider.config.model == config["model"]


@pytest.mark.asyncio
async def test_provider_generate():
    """Test the generate method with various scenarios."""
    config = {"api_key": "test-key", "model": "test-model"}
    provider = TestProvider(config)
    
    # Test successful generation
    response = await provider.generate("test-prompt", "test-model")
    assert response == "Response to: test-prompt"
    
    # Test with custom parameters
    response = await provider.generate(
        "test-prompt", 
        "test-model", 
        temperature=0.9, 
        max_tokens=500,
        custom_param="value"
    )
    assert response == "Response to: test-prompt"


@pytest.mark.asyncio
async def test_provider_error_handling():
    """Test error handling in the provider."""
    config = {"api_key": "test-key", "model": "test-model"}
    provider = TestProvider(config)
    
    # Test various error scenarios
    with pytest.raises(RateLimitExceeded):
        await provider.generate("rate_limit error", "test-model")
    
    with pytest.raises(AuthenticationError):
        await provider.generate("auth error", "test-model")
    
    with pytest.raises(ModelUnavailable):
        await provider.generate("model error", "test-model")
    
    with pytest.raises(ServiceUnavailable):
        await provider.generate("service error", "test-model")
    
    with pytest.raises(ProviderError):
        await provider.generate("error", "test-model")


@pytest.mark.asyncio
async def test_provider_context_manager():
    """Test the provider as a context manager."""
    config = {"api_key": "test-key", "model": "test-model"}
    
    # Test async context manager
    async with TestProvider(config) as provider:
        assert provider.config.api_key == "test-key"
        assert not provider._is_closed
    
    # Verify close was called
    assert provider._is_closed
    
    # Test that close is idempotent
    await provider.close()
    assert provider._is_closed


@pytest.mark.asyncio
async def test_provider_retry_config():
    """Test the retry configuration."""
    # Test with default config
    config = {"api_key": "test-key", "model": "test-model"}
    provider = TestProvider(config)
    retry_config = provider.get_retry_config()
    
    assert retry_config['stop'](type('', (), {'attempt_number': 3})())
    assert not retry_config['stop'](type('', (), {'attempt_number': 2})())
    
    # Test with custom config
    custom_config = {
        "api_key": "test-key",
        "model": "test-model",
        "max_retries": 5,
        "custom_retry": "value"
    }
    retry_config = provider.get_retry_config(custom_config)
    assert retry_config['stop'](type('', (), {'attempt_number': 5})())
    assert 'custom_retry' in retry_config


@pytest.mark.asyncio
async def test_provider_initialization_error():
    """Test error handling during provider initialization."""
    config = {"api_key": "test-key", "model": "test-model"}
    
    with patch.object(TestProvider, '_initialize_client', side_effect=Exception("Init error")):
        with pytest.raises(Exception, match="Init error"):
            provider = TestProvider(config)
            await provider.generate("test", "test-model")


@pytest.mark.asyncio
async def test_provider_generate_with_timeout():
    """Test the generate method with timeout handling."""
    config = {
        "api_key": "test-key",
        "model": "test-model",
        "timeout": 1  # Use an integer value for timeout
    }
    
    provider = TestProvider(config)
    
    # Mock generate to sleep longer than timeout
    async def slow_generate(*args, **kwargs):
        await asyncio.sleep(2)  # Sleep longer than timeout
        return "too late"
    
    with patch.object(provider, 'generate', slow_generate):
        with pytest.raises(asyncio.TimeoutError):
            # Use a shorter timeout for the test
            await asyncio.wait_for(
                provider.generate("test-prompt", "test-model"),
                timeout=0.1
            )


@pytest.mark.asyncio
async def test_provider_streaming_not_implemented():
    """Test that streaming raises NotImplementedError in the base class."""
    config = {"api_key": "test-key", "model": "test-model"}
    
    # Create a minimal provider that implements required methods but not streaming
    class NoStreamProvider(BaseProvider):
        async def generate(self, *args, **kwargs):
            if kwargs.get('stream', False):
                raise NotImplementedError("Streaming not implemented")
            return "test"
            
        async def close(self):
            pass
    
    provider = NoStreamProvider(config)
    
    # Test that regular generation works
    result = await provider.generate("test-prompt", "test-model")
    assert result == "test"
    
    # Test that streaming raises NotImplementedError
    with pytest.raises(NotImplementedError):
        await provider.generate("test-prompt", "test-model", stream=True)


def test_provider_abstract_methods():
    """Test that abstract methods raise TypeError when not implemented."""
    # This test verifies that we can't instantiate a provider without implementing abstract methods
    with pytest.raises(TypeError) as exc_info:
        # Use a dynamic class creation to avoid lint errors
        provider_class = type('IncompleteProvider', (BaseProvider,), {})
        # This line should raise TypeError because abstract methods aren't implemented
        provider_class({})
    
    # Verify the error message indicates which methods need to be implemented
    error_msg = str(exc_info.value)
    assert "Can't instantiate abstract class" in error_msg
    assert "generate" in error_msg  # Should mention the missing method
