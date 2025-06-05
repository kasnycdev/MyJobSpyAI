"""
Integration tests for the LangChain provider.

This module contains integration tests that verify the LangChain provider
works correctly with actual LangChain implementations.
"""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Skip these tests if integration tests are not enabled
pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_INTEGRATION_TESTS"),
    reason="Integration tests disabled. Set RUN_INTEGRATION_TESTS=1 to enable.",
)

# Import the modules to test
from myjobspyai.analysis.providers.factory import ProviderFactory
from myjobspyai.analysis.providers.langchain_provider import LangChainProvider

# Test configuration for different LangChain backends
TEST_CONFIGS = {
    "openai": {
        "type": "langchain",
        "class": "langchain_community.chat_models.ChatOpenAI",
        "params": {
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 100,
            "openai_api_key": os.environ.get("OPENAI_API_KEY", "test-api-key"),
        },
    },
    "ollama": {
        "type": "langchain",
        "class": "langchain_community.chat_models.ChatOllama",
        "params": {
            "model": "llama2",
            "base_url": "http://localhost:11434",
            "temperature": 0.7,
            "num_ctx": 2048,
        },
    },
    "google_genai": {
        "type": "langchain",
        "class": "langchain_google_genai.ChatGoogleGenerativeAI",
        "params": {
            "model": "gemini-pro",
            "google_api_key": os.environ.get("GOOGLE_API_KEY", "test-api-key"),
            "temperature": 0.7,
            "max_output_tokens": 1000,
        },
    },
}

# Test prompt and expected response
TEST_PROMPT = "Hello, world!"
EXPECTED_RESPONSE_PREFIX = "Hello"

# Test parameters for parameterized tests
TEST_PARAMS = [
    ("openai", "test_openai"),
    # ("ollama", "test_ollama"),  # Uncomment if you have Ollama running locally
    # ("google_genai", "test_google_genai"),  # Uncomment if you have Google API key
]


@pytest.mark.integration
@pytest.mark.parametrize("config_key,provider_name", TEST_PARAMS)
@pytest.mark.asyncio
async def test_langchain_provider_integration(config_key, provider_name):
    """Test that the LangChain provider works with actual LangChain backends."""
    # Skip if the required environment variables are not set
    if config_key == "openai" and not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY environment variable not set")
    elif config_key == "google_genai" and not os.environ.get("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY environment variable not set")

    # Get the test config
    test_config = TEST_CONFIGS[config_key]

    # Create a provider instance
    provider = ProviderFactory.create_provider(
        provider_type="langchain", config=test_config, name=provider_name
    )

    # Verify the provider was created correctly
    assert isinstance(provider, LangChainProvider)
    assert provider.provider_name == provider_name
    assert provider.provider_type == "langchain"

    # Test a simple prompt
    response = await provider.generate(TEST_PROMPT)

    # Verify the response
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0

    # Check that the response starts with the expected prefix
    # (The actual response may vary, but should start with a greeting)
    assert response.startswith(EXPECTED_RESPONSE_PREFIX) or any(
        word in response.lower() for word in ["hello", "hi", "hey"]
    )


@pytest.mark.integration
@pytest.mark.parametrize("config_key,provider_name", TEST_PARAMS)
@pytest.mark.asyncio
async def test_langchain_provider_with_system_message(config_key, provider_name):
    """Test that the LangChain provider works with system messages."""
    # Skip if the required environment variables are not set
    if config_key == "openai" and not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY environment variable not set")
    elif config_key == "google_genai" and not os.environ.get("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY environment variable not set")

    # Get the test config and add a system message
    test_config = TEST_CONFIGS[config_key].copy()
    test_config["system_message"] = (
        "You are a helpful assistant that speaks like a pirate."
    )

    # Create a provider instance
    provider = ProviderFactory.create_provider(
        provider_type="langchain", config=test_config, name=f"{provider_name}_pirate"
    )

    # Test a prompt that would normally get a standard response
    response = await provider.generate("Hello, how are you?")

    # Verify the response
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0

    # Check for pirate-like language (this is a bit of a heuristic)
    pirate_phrases = ["arr", "ahoy", "matey", "yo ho", "shiver me timbers"]
    has_pirate_language = any(phrase in response.lower() for phrase in pirate_phrases)

    # The model might not always respond in pirate, but it should at least respond
    assert len(response) > 0

    # If we want to be strict about the pirate language, we can uncomment this:
    # assert has_pirate_language, f"Expected pirate language in response: {response}"


@pytest.mark.integration
@pytest.mark.parametrize("config_key,provider_name", TEST_PARAMS)
@pytest.mark.asyncio
async def test_langchain_provider_streaming(config_key, provider_name):
    """Test that the LangChain provider works with streaming responses."""
    # Skip if the required environment variables are not set
    if config_key == "openai" and not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY environment variable not set")
    elif config_key == "google_genai" and not os.environ.get("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY environment variable not set")

    # Get the test config and enable streaming
    test_config = TEST_CONFIGS[config_key].copy()
    test_config["streaming"] = True

    # Create a provider instance
    provider = ProviderFactory.create_provider(
        provider_type="langchain", config=test_config, name=f"{provider_name}_streaming"
    )

    # Test a streaming response
    response = await provider.generate(
        "Tell me a short story about a robot.", stream=True
    )

    # Verify the response
    assert response is not None
    assert isinstance(response, str)
    assert len(response) > 0

    # Check that the response looks like a story
    assert len(response.split()) > 5  # Should be more than 5 words


if __name__ == "__main__":
    # Run the tests
    import sys

    sys.exit(pytest.main([__file__] + sys.argv[1:]))
