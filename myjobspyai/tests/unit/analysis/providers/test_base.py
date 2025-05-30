"""Tests for the base provider interface."""
import asyncio
import os
from typing import Any, Dict, Generator, Optional, TypeVar, AsyncGenerator
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pytest import fixture, mark
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


class TestProvider(BaseProvider):
    """Test provider implementation for testing the base class."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the test provider."""
        self.config = TestConfig(**config)
        self._client = None
    
    async def _initialize_client(self) -> None:
        """Initialize the test client."""
        self._client = MagicMock()
    
    async def close(self) -> None:
        """Close the test client."""
        if self._client:
            self._client = None
    
    async def _generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Generate a response from the test provider."""
        if not self._client:
            await self._initialize_client()
        
        # Simulate different error scenarios based on prompt
        if "rate_limit" in prompt:
            raise RateLimitExceeded("Rate limit exceeded")
        elif "auth_error" in prompt:
            raise AuthenticationError("Authentication failed")
        elif "model_unavailable" in prompt:
            raise ModelUnavailable("Model not found")
        elif "service_unavailable" in prompt:
            raise ServiceUnavailable("Service unavailable")
        elif "provider_error" in prompt:
            raise ProviderError("Provider error occurred")
        
        return f"Response to: {prompt}"


@pytest.mark.asyncio
async def test_base_provider_initialization():
    """Test base provider initialization."""
    config = {"api_key": "test-key", "model": "test-model"}
    provider = TestProvider(config)
    
    assert provider.config.api_key == "test-key"
    assert provider.config.model == "test-model"
    assert provider._client is None
    
    # Cleanup
    await provider.close()


@pytest.mark.asyncio
async def test_base_provider_generate():
    """Test base provider generate method."""
    config = {"api_key": "test-key", "model": "test-model"}
    provider = TestProvider(config)
    
    # Test successful generation
    response = await provider.generate("Test prompt")
    assert response == "Response to: Test prompt"
    assert provider._client is not None
    
    # Cleanup
    await provider.close()


@pytest.mark.asyncio
async def test_base_provider_retry_logic():
    """Test base provider retry logic."""
    config = {
        "api_key": "test-key",
        "model": "test-model",
        "max_retries": 3,
        "initial_retry_delay": 0.1,
    }
    
    # Test with rate limit error (should retry)
    with patch("asyncio.sleep") as mock_sleep:
        provider = TestProvider(config)
        
        with pytest.raises(RateLimitExceeded):
            await provider.generate("rate_limit")
        
        # Should have retried 3 times (initial + 3 retries)
        assert mock_sleep.call_count == 3
    
    # Cleanup
    await provider.close()


@pytest.mark.asyncio
async def test_base_provider_error_handling():
    """Test base provider error handling."""
    config = {
        "api_key": "test-key",
        "model": "test-model",
        "max_retries": 0,  # Disable retries for this test
    }
    
    provider = TestProvider(config)
    
    # Test authentication error
    with pytest.raises(AuthenticationError):
        await provider.generate("auth_error")
    
    # Test model unavailable error
    with pytest.raises(ModelUnavailable):
        await provider.generate("model_unavailable")
    
    # Test service unavailable error
    with pytest.raises(ServiceUnavailable):
        await provider.generate("service_unavailable")
    
    # Test generic provider error
    with pytest.raises(ProviderError):
        await provider.generate("provider_error")
    
    # Cleanup
    await provider.close()


@pytest.mark.asyncio
async def test_base_provider_context_manager():
    """Test base provider context manager."""
    config = {"api_key": "test-key", "model": "test-model"}
    
    async with TestProvider(config) as provider:
        response = await provider.generate("Test prompt")
        assert response == "Response to: Test prompt"
        assert provider._client is not None
    
    # Client should be closed after context manager exits
    assert provider._client is None


@pytest.mark.asyncio
async def test_base_provider_timeout():
    """Test base provider timeout handling."""
    config = {
        "api_key": "test-key",
        "model": "test-model",
        "timeout": 0.1,  # Very short timeout for testing
    }
    
    # Create a provider with a slow _generate method
    class SlowTestProvider(TestProvider):
        async def _generate(self, prompt: str, **kwargs: Any) -> str:
            await asyncio.sleep(1)  # Sleep longer than the timeout
            return "This should timeout"
    
    provider = SlowTestProvider(config)
    
    with pytest.raises(TimeoutError):
        await provider.generate("Test prompt")
    
    # Cleanup
    await provider.close()


@pytest.mark.asyncio
async def test_base_provider_streaming_not_implemented():
    """Test that streaming raises NotImplementedError if not implemented."""
    config = {"api_key": "test-key", "model": "test-model"}
    provider = TestProvider(config)
    
    with pytest.raises(NotImplementedError):
        async for _ in provider.generate_stream("Test prompt"):
            pass
    
    # Cleanup
    await provider.close()


@pytest.mark.asyncio
async def test_base_provider_streaming():
    """Test base provider streaming implementation."""
    # Create a provider with streaming implementation
    class StreamingTestProvider(TestProvider):
        async def generate_stream(
            self,
            prompt: str,
            **kwargs: Any,
        ) -> AsyncGenerator[str, None]:
            """Generate a streaming response."""
            if not self._client:
                await self._initialize_client()
            
            chunks = ["Hello, ", "world!"]
            for chunk in chunks:
                yield chunk
                await asyncio.sleep(0.01)  # Simulate network delay
    
    config = {"api_key": "test-key", "model": "test-model"}
    provider = StreamingTestProvider(config)
    
    # Test streaming
    chunks = []
    async for chunk in provider.generate_stream("Stream test"):
        chunks.append(chunk)
    
    assert chunks == ["Hello, ", "world!"]
    
    # Cleanup
    await provider.close()


@pytest.mark.asyncio
async def test_base_provider_validate_config():
    """Test base provider config validation."""
    # Test with missing required field
    with pytest.raises(ValueError):
        TestProvider({})
    
    # Test with invalid config type
    with pytest.raises(TypeError):
        TestProvider("invalid-config")  # type: ignore
    
    # Test with extra fields (should be ignored)
    provider = TestProvider({"api_key": "test-key", "model": "test-model", "extra": "field"})
    assert not hasattr(provider.config, "extra")
    await provider.close()


@pytest.mark.asyncio
async def test_base_provider_async_context_manager():
    """Test base provider async context manager."""
    config = {"api_key": "test-key", "model": "test-model"}
    
    async with TestProvider(config) as provider:
        assert provider._client is not None
        response = await provider.generate("Test prompt")
        assert response == "Response to: Test prompt"
    
    # Client should be closed after context manager exits
    assert provider._client is None
