"""
Tests for the LangChain provider integration.

This module contains tests to verify that the LangChain provider
is properly integrated with the application and works as expected.
"""

import os
import sys
import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the modules to test
from myjobspyai.analysis.providers.factory import ProviderFactory
from myjobspyai.analysis.providers.langchain_provider import LangChainProvider

# Test prompt and response
TEST_PROMPT = "Hello, world!"
TEST_RESPONSE = "Hello! How can I assist you today?"


@pytest.fixture
def mock_chat_openai():
    """Mock the ChatOpenAI class from langchain_community."""
    with patch("langchain_community.chat_models.ChatOpenAI") as mock_class:
        # Create a mock instance
        mock_instance = AsyncMock()

        # Mock the generate method
        mock_instance.agenerate = AsyncMock(
            return_value=MagicMock(generations=[[MagicMock(text=TEST_RESPONSE)]])
        )

        # Mock the ainvoke method for direct calls
        mock_instance.ainvoke = AsyncMock(return_value=MagicMock(content=TEST_RESPONSE))

        # Return the mock instance when the class is instantiated
        mock_class.return_value = mock_instance

        # Make the class itself callable and return the mock instance
        mock_class.return_value = mock_instance

        yield mock_class  # Yield the mock class instead of the instance


@pytest.mark.asyncio
async def test_langchain_provider_initialization(mock_chat_openai, mock_config):
    """Test that the LangChain provider can be initialized correctly."""
    # Reset the mock to ensure we're starting fresh
    mock_chat_openai.reset_mock()

    # Create a provider instance
    provider = LangChainProvider(config=mock_config)

    # Check that the LLM instance is set
    assert provider.llm is not None

    # Check that the LLM was initialized with the correct parameters
    mock_chat_openai.assert_called_once_with(
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=100,
        openai_api_key="test-api-key",
    )


@pytest.mark.asyncio
async def test_langchain_provider_generate(mock_chat_openai, mock_config):
    """Test that the generate method works correctly."""
    # Create a provider instance
    provider = LangChainProvider(config=mock_config)

    # Call the generate method
    response = await provider.generate(prompt=TEST_PROMPT)

    # Check that the response is correct
    assert response == TEST_RESPONSE

    # Check that the LLM was called with the correct parameters
    provider.llm.agenerate.assert_called_once()


@pytest.mark.asyncio
async def test_langchain_provider_error_handling(mock_chat_openai, mock_config):
    """Test that errors are properly handled."""
    # Reset the mock to ensure we're starting fresh
    mock_chat_openai.reset_mock()

    # Get the mock instance and set up the side effect
    mock_instance = mock_chat_openai.return_value
    mock_instance.agenerate.side_effect = Exception("Test error")

    # Create a provider instance
    provider = LangChainProvider(config=mock_config)

    # Call the generate method and expect an exception
    with pytest.raises(Exception) as exc_info:
        await provider.generate(prompt=TEST_PROMPT)

    # Check that the error message is correct
    assert "Test error" in str(exc_info.value)

    # Verify that the LLM was called
    mock_instance.agenerate.assert_called_once()


def test_provider_factory_langchain(mock_config):
    """Test that the provider factory can create a LangChain provider."""
    # Create a provider using the factory instance
    factory = ProviderFactory()
    provider = factory.create_provider(
        provider_type="langchain", config=mock_config, name="test_provider"
    )

    # Check that the provider is a LangChainProvider
    assert isinstance(provider, LangChainProvider)


@pytest.mark.asyncio
async def test_langchain_provider_with_system_message(mock_chat_openai, mock_config):
    """Test that the provider works with system messages."""
    # Create a test config with a system message
    test_config = mock_config.copy()
    test_config["system_message"] = "You are a helpful assistant."

    # Reset the mock to ensure we're starting fresh
    mock_chat_openai.reset_mock()

    # Get the mock instance
    mock_instance = mock_chat_openai.return_value

    # Create a provider instance
    provider = LangChainProvider(config=test_config)

    # Call the generate method
    response = await provider.generate(prompt=TEST_PROMPT)

    # Check that the response is correct
    assert response == TEST_RESPONSE

    # Verify that the LLM was called with the system message
    assert mock_instance.agenerate.called, "The generate method was not called"


if __name__ == "__main__":
    # Run the tests
    import sys

    sys.exit(pytest.main([__file__] + sys.argv[1:]))
