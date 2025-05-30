"""Tests for the Gemini provider."""
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from myjobspyai.analysis.providers.gemini import GeminiClient
from myjobspyai.exceptions import AuthenticationError, ModelUnavailable, RateLimitExceeded


@pytest.mark.asyncio
async def test_gemini_generate_success():
    """Test successful text generation with Gemini."""
    # Setup
    config = {
        "api_key": "test-key",
        "model": "gemini-pro",
    }
    
    # Mock the response
    mock_response = MagicMock()
    mock_response.text = "Test response from Gemini"
    
    # Mock the model
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response
    
    # Mock the Gemini client
    with patch("google.generativeai.GenerativeModel", return_value=mock_model), \
         patch("google.generativeai.configure"):
        client = GeminiClient(config)
        response = await client.generate("Test prompt")
        
        # Assertions
        assert response == "Test response from Gemini"
        mock_model.generate_content.assert_called_once_with(
            "Test prompt",
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 1000,
            },
        )


@pytest.mark.asyncio
async def test_gemini_generate_with_custom_params():
    """Test text generation with custom parameters."""
    # Setup
    config = {
        "api_key": "test-key",
        "model": "gemini-pro",
    }
    
    # Mock the response
    mock_response = MagicMock()
    mock_response.text = "Test response from Gemini"
    
    # Mock the model
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response
    
    # Test with custom parameters
    with patch("google.generativeai.GenerativeModel", return_value=mock_model), \
         patch("google.generativeai.configure"):
        client = GeminiClient(config)
        response = await client.generate(
            "Test prompt",
            temperature=0.9,
            max_tokens=500,
            top_p=0.9,
            top_k=40,
        )
        
        # Assertions
        assert response == "Test response from Gemini"
        mock_model.generate_content.assert_called_once_with(
            "Test prompt",
            generation_config={
                "temperature": 0.9,
                "max_output_tokens": 500,
                "top_p": 0.9,
                "top_k": 40,
            },
        )


@pytest.mark.asyncio
async def test_gemini_generate_rate_limit():
    """Test rate limit handling with Gemini."""
    # Setup
    config = {
        "api_key": "test-key",
        "model": "gemini-pro",
    }
    
    # Mock the model to raise a rate limit error
    mock_model = MagicMock()
    mock_model.generate_content.side_effect = Exception("429 Quota exceeded")
    
    # Test
    with patch("google.generativeai.GenerativeModel", return_value=mock_model), \
         patch("google.generativeai.configure"):
        client = GeminiClient(config)
        
        with pytest.raises(RateLimitExceeded):
            await client.generate("Test prompt")


@pytest.mark.asyncio
async def test_gemini_model_unavailable():
    """Test handling of unavailable model with Gemini."""
    # Setup
    config = {
        "api_key": "test-key",
        "model": "nonexistent-model",
    }
    
    # Mock the model to raise a model not found error
    mock_model = MagicMock()
    mock_model.generate_content.side_effect = Exception("Model not found")
    
    # Test
    with patch("google.generativeai.GenerativeModel", return_value=mock_model), \
         patch("google.generativeai.configure"):
        client = GeminiClient(config)
        
        with pytest.raises(ModelUnavailable):
            await client.generate("Test prompt")


@pytest.mark.asyncio
async def test_gemini_authentication_error():
    """Test authentication error handling with Gemini."""
    # Setup
    config = {
        "api_key": "invalid-key",
        "model": "gemini-pro",
    }
    
    # Mock the model to raise an authentication error
    mock_model = MagicMock()
    mock_model.generate_content.side_effect = Exception("Invalid API key")
    
    # Test
    with patch("google.generativeai.GenerativeModel", return_value=mock_model), \
         patch("google.generativeai.configure"):
        client = GeminiClient(config)
        
        with pytest.raises(AuthenticationError):
            await client.generate("Test prompt")


def test_gemini_environment_variable():
    """Test that the API key is loaded from environment variables."""
    # Setup - no API key in config
    config = {
        "model": "gemini-pro",
    }
    
    # Set the environment variable
    os.environ["GOOGLE_API_KEY"] = "env-test-key"
    
    # Mock the model
    mock_model = MagicMock()
    mock_model.generate_content.return_value = MagicMock(text="Test response")
    
    # Test
    with patch("google.generativeai.GenerativeModel", return_value=mock_model), \
         patch("google.generativeai.configure") as mock_configure:
        client = GeminiClient(config)
        
        # Assertions
        mock_configure.assert_called_once_with(api_key="env-test-key")
    
    # Cleanup
    del os.environ["GOOGLE_API_KEY"]


@pytest.mark.asyncio
async def test_gemini_close():
    """Test closing the Gemini client."""
    # Setup
    config = {
        "api_key": "test-key",
        "model": "gemini-pro",
    }
    
    # Test - Gemini client doesn't need explicit cleanup
    with patch("google.generativeai.GenerativeModel"), \
         patch("google.generativeai.configure"):
        client = GeminiClient(config)
        await client.close()  # Should not raise
