"""
Tests for the ProviderFactory class.

This module contains tests to verify that the ProviderFactory
can create different types of LLM providers correctly.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from myjobspyai.analysis.providers.base import BaseProvider

# Import the modules to test
from myjobspyai.analysis.providers.factory import ProviderFactory, ProviderType
from myjobspyai.analysis.providers.langchain_provider import LangChainProvider

# Test configurations
TEST_OPENAI_CONFIG = {
    "type": "openai",
    "model": "gpt-3.5-turbo",
    "api_key": "test-api-key",
    "base_url": "http://test-openai-api",
}

TEST_OLLAMA_CONFIG = {
    "type": "ollama",
    "model": "llama2",
    "base_url": "http://test-ollama-api",
}

TEST_LANGCHAIN_CONFIG = {
    "type": "langchain",
    "class": "langchain_community.chat_models.ChatOpenAI",
    "params": {
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 100,
        "openai_api_key": "test-api-key",
    },
}


def test_provider_factory_registration():
    """Test that provider types are correctly registered in the factory."""
    factory = ProviderFactory()

    # Check that the default providers are registered
    assert "langchain" in factory.providers
    assert factory.providers["langchain"] == LangChainProvider

    # Create a test provider class
    class TestProvider(BaseProvider):
        async def generate(self, prompt, **kwargs):
            return "Test response"

    # Register a test provider
    factory.register_provider("test_provider", TestProvider)

    # Verify it was registered
    assert "test_provider" in factory.providers
    assert factory.providers["test_provider"] == TestProvider


def test_create_openai_provider():
    """Test creating an OpenAI provider."""
    factory = ProviderFactory()

    # Create a mock provider class that inherits from BaseProvider
    class MockOpenAIProvider(BaseProvider):
        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            self.provider_type = "openai"

        async def generate(self, prompt, **kwargs):
            return f"Response to: {prompt}"

    # Register the mock provider
    factory.register_provider("openai", MockOpenAIProvider)

    # Create the provider instance
    provider = factory.create_provider(
        provider_type="openai", config=TEST_OPENAI_CONFIG, name="test_openai"
    )

    # Verify the provider was created correctly
    assert provider is not None
    assert isinstance(provider, BaseProvider)
    assert provider.provider_name == "test_openai"
    assert provider.provider_type == "openai"


def test_create_ollama_provider():
    """Test creating an Ollama provider."""
    factory = ProviderFactory()

    # Create a mock provider class that inherits from BaseProvider
    class MockOllamaProvider(BaseProvider):
        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            self.provider_type = "ollama"

        async def generate(self, prompt, **kwargs):
            return f"Response to: {prompt}"

    # Register the mock provider
    factory.register_provider("ollama", MockOllamaProvider)

    # Create the provider instance
    provider = factory.create_provider(
        provider_type="ollama", config=TEST_OLLAMA_CONFIG, name="test_ollama"
    )

    # Verify the provider was created correctly
    assert provider is not None
    assert isinstance(provider, BaseProvider)
    assert provider.provider_name == "test_ollama"
    assert provider.provider_type == "ollama"


def test_create_langchain_provider():
    """Test creating a LangChain provider."""
    factory = ProviderFactory()
    # Mock the import and class creation
    with patch("importlib.import_module") as mock_import_module, \
         patch.object(factory, '_providers', {"langchain": LangChainProvider}):
        # Create a mock module with the ChatOpenAI class
        mock_module = MagicMock()
        mock_chat_openai = MagicMock()
        mock_module.ChatOpenAI = mock_chat_openai
        mock_import_module.return_value = mock_module

        # Create the provider
        provider = factory.create_provider(
            provider_type="langchain",
            config=TEST_LANGCHAIN_CONFIG,
            name="test_langchain",
        )

        assert provider is not None
        assert isinstance(provider, LangChainProvider)
        assert provider.provider_name == "test_langchain"
        assert hasattr(provider, 'provider_type')  # Check if provider_type is set

        # Verify the LLM was created with the correct parameters
        mock_chat_openai.assert_called_once_with(
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100,
            openai_api_key="test-api-key",
        )


def test_register_custom_provider():
    """Test registering a custom provider class."""
    factory = ProviderFactory()

    # Define a custom provider class
    class CustomProvider(BaseProvider):
        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            self.custom_init = True

        async def generate(self, prompt, **kwargs):
            return f"Custom response to: {prompt}"

    # Register the custom provider
    factory.register_provider("custom", CustomProvider)

    # Create an instance of the custom provider
    provider = factory.create_provider(
        provider_type="custom", config={"param": "value"}, name="test_custom"
    )

    assert provider is not None
    assert isinstance(provider, CustomProvider)
    assert provider.provider_name == "test_custom"
    assert hasattr(provider, 'provider_type')  # Check if provider_type is set
    assert hasattr(provider, "custom_init")
    assert provider.custom_init is True


def test_create_provider_invalid_type():
    """Test creating a provider with an invalid type raises an error."""
    factory = ProviderFactory()
    with pytest.raises(ValueError) as exc_info:
        factory.create_provider(
            provider_type="invalid_type", config={}, name="test_invalid"
        )

    assert "Unknown provider type: invalid_type" in str(exc_info.value)


@pytest.mark.asyncio
async def test_provider_generate_method():
    """Test that the generate method works correctly on a provider."""
    factory = ProviderFactory()

    # Create a test provider with a mock generate method
    class TestProvider(BaseProvider):
        async def generate(self, prompt, **kwargs):
            return f"Response to: {prompt}"

    # Register the test provider
    factory.register_provider("test", TestProvider)

    # Create an instance and test the generate method
    provider = factory.create_provider(
        provider_type="test", config={}, name="test_generate"
    )

    response = await provider.generate("Test prompt")
    assert response == "Response to: Test prompt"


def test_provider_factory_singleton():
    """Test that the ProviderFactory follows the singleton pattern."""
    # Clear the _instance to test singleton behavior
    ProviderFactory._instance = None

    # Create two instances
    factory1 = ProviderFactory()
    factory2 = ProviderFactory()

    # They should be the same instance
    assert factory1 is factory2

    # The _providers dictionary should be shared
    factory1._providers["test_singleton"] = "test_value"
    assert "test_singleton" in factory2._providers
    assert factory2._providers["test_singleton"] == "test_value"


if __name__ == "__main__":
    # Run the tests
    import sys

    sys.exit(pytest.main([__file__] + sys.argv[1:]))
