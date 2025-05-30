"""Tests for the provider factory."""
import asyncio
from typing import Any, Dict, TypeVar
import pytest
import pytest_asyncio

from myjobspyai.analysis.factory import ProviderFactory
from myjobspyai.analysis.providers import (
    OpenAIClient,
    OllamaClient,
    GeminiClient,
)

# Type variable for provider classes
T = TypeVar('T', bound='BaseProvider')

# Mark all tests in this file as async
pytestmark = pytest.mark.asyncio

# Test fixtures
@pytest.fixture
def provider_config() -> Dict[str, Any]:
    """Return a basic provider configuration."""
    return {
        "providers": {
            "openai": {
                "api_key": "test-api-key",
                "model": "gpt-4",
                "timeout": 30.0,
                "connect_timeout": 10.0,
                "max_retries": 3
            },
            "ollama": {
                "base_url": "http://localhost:11434",
                "model": "llama2",
                "timeout": 300.0,
                "connect_timeout": 30.0
            },
            "gemini": {
                "api_key": "test-gemini-key",
                "model": "gemini-pro"
            }
        }
    }

# Tests
class TestProviderFactory:
    """Test cases for ProviderFactory."""
    
    async def test_get_provider_config(self, provider_config):
        """Test getting provider configuration."""
        factory = ProviderFactory(provider_config)
        config = factory.get_provider_config("openai")
        assert config["api_key"] == "test-api-key"
        assert config["model"] == "gpt-4"
        assert config["timeout"] == 30.0
        assert config["connect_timeout"] == 10.0
        assert config["max_retries"] == 3
    
    async def test_get_provider_config_with_overrides(self, provider_config):
        """Test getting provider configuration with overrides."""
        factory = ProviderFactory(provider_config)
        overrides = {"api_key": "override-key", "model": "gpt-3.5-turbo"}
        config = factory.get_provider_config("openai", overrides)
        
        assert config["api_key"] == "override-key"
        assert config["model"] == "gpt-3.5-turbo"
        assert config["timeout"] == 30.0  # Original value
        assert config["connect_timeout"] == 10.0
        assert config["max_retries"] == 3
    
    async def test_get_provider_config_env_var(self, provider_config, monkeypatch):
        """Test getting provider configuration with environment variable."""
        monkeypatch.setenv("OPENAI_API_KEY", "env-api-key")
        factory = ProviderFactory(provider_config)
        config = factory.get_provider_config("openai")
        assert config["api_key"] == "env-api-key"
    
    async def test_create_provider(self, provider_config, monkeypatch):
        """Test creating a provider instance."""
        # Create a mock provider class
        class MockOpenAIProvider:
            def __init__(self, config):
                self.config = config
                self._client = None
            
            async def _initialize_client(self):
                pass

        # Monkeypatch the provider factory to use our mock
        monkeypatch.setattr(
            "myjobspyai.analysis.factory._get_provider",
            lambda _, config: MockOpenAIProvider
        )
        
        factory = ProviderFactory(provider_config)
        provider = await factory.get_or_create_provider("openai")
        assert isinstance(provider, MockOpenAIProvider)
        assert provider.config["api_key"] == "test-api-key"
        assert provider.config["model"] == "gpt-4"
    
    async def test_create_unknown_provider(self, provider_config):
        """Test creating an unknown provider raises an error."""
        factory = ProviderFactory(provider_config)
        with pytest.raises(ValueError, match="Unknown provider: unknown"):
            await factory.get_or_create_provider("unknown")
    
    async def test_concurrent_initialization(self, provider_config):
        """Test that concurrent initialization is handled correctly."""
        factory = ProviderFactory(provider_config)
        
        # Create multiple tasks to initialize the same provider
        tasks = [
            factory.get_or_create_provider("openai") 
            for _ in range(5)
        ]
        providers = await asyncio.gather(*tasks)
        
        # All providers should be the same instance
        assert all(p is providers[0] for p in providers)
    
    async def test_multiple_providers(self, provider_config):
        """Test creating multiple different provider types."""
        factory = ProviderFactory(provider_config)
        
        # Test OpenAI provider
        openai_provider = await factory.get_or_create_provider("openai")
        assert isinstance(openai_provider, OpenAIClient)
        
        # Test Ollama provider
        ollama_provider = await factory.get_or_create_provider("ollama")
        assert isinstance(ollama_provider, OllamaClient)
        
        # Test Gemini provider
        gemini_provider = await factory.get_or_create_provider("gemini")
        assert isinstance(gemini_provider, GeminiClient)
