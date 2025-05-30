"""Tests for the Ollama provider."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from myjobspyai.analysis.providers.ollama import OllamaClient
from myjobspyai.exceptions import ModelUnavailable, RateLimitExceeded


@pytest.mark.asyncio
async def test_ollama_generate_success():
    """Test successful text generation with Ollama."""
    # Setup
    config = {
        "base_url": "http://localhost:11434",
        "model": "llama2",
        "timeout": 300.0,
        "connect_timeout": 30.0,
    }
    
    # Mock the response
    mock_response = {
        "model": "llama2",
        "created_at": "2023-01-01T00:00:00.000000Z",
        "response": "Test response from Ollama",
        "done": True,
        "total_duration": 500000000,
        "load_duration": 100000000,
        "prompt_eval_count": 10,
        "eval_count": 20,
        "eval_duration": 400000000,
    }
    
    # Mock the HTTP client
    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=MagicMock(
        status_code=200,
        json=MagicMock(return_value=mock_response),
    ))
    
    # Test
    with patch("httpx.AsyncClient", return_value=mock_client):
        client = OllamaClient(config)
        response = await client.generate("Test prompt")
        
        # Assertions
        assert response == "Test response from Ollama"
        mock_client.post.assert_awaited_once_with(
            "/api/generate",
            json={
                "model": "llama2",
                "prompt": "Test prompt",
                "temperature": 0.7,
                "max_tokens": 1000,
            },
        )


@pytest.mark.asyncio
async def test_ollama_generate_streaming():
    """Test streaming text generation with Ollama."""
    # Setup
    config = {
        "base_url": "http://localhost:11434",
        "model": "llama2",
    }
    
    # Create a mock streaming response
    chunks = [
        b'{"model":"llama2","created_at":"2023-01-01T00:00:00.000000Z","response":"Hello","done":false}\n',
        b'{"model":"llama2","created_at":"2023-01-01T00:00:01.000000Z","response":", ","done":false}\n',
        b'{"model":"llama2","created_at":"2023-01-01T00:00:02.000000Z","response":"world!","done":true}\n',
    ]
    
    # Mock the HTTP client
    mock_client = MagicMock()
    mock_client.post.return_value = MagicMock(
        status_code=200,
        aiter_lines=MagicMock(return_value=chunks),
    )
    
    # Test
    with patch("httpx.AsyncClient", return_value=mock_client):
        client = OllamaClient(config)
        response = await client.generate("Test prompt", stream=True)
        
        # Assertions
        assert response == "Hello, world!"


@pytest.mark.asyncio
async def test_ollama_generate_rate_limit():
    """Test rate limit handling with Ollama."""
    # Setup
    config = {
        "base_url": "http://localhost:11434",
        "model": "llama2",
    }
    
    # Mock the HTTP client to raise a rate limit error
    mock_client = MagicMock()
    mock_client.post = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "Rate limit exceeded",
            request=MagicMock(),
            response=MagicMock(status_code=429),
        )
    )
    
    # Test
    with patch("httpx.AsyncClient", return_value=mock_client):
        client = OllamaClient(config)
        
        with pytest.raises(RateLimitExceeded):
            await client.generate("Test prompt")


@pytest.mark.asyncio
async def test_ollama_model_unavailable():
    """Test handling of unavailable model with Ollama."""
    # Setup
    config = {
        "base_url": "http://localhost:11434",
        "model": "nonexistent-model",
    }
    
    # Mock the HTTP client to return 404 for model check
    mock_client = MagicMock()
    mock_client.get = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "Model not found",
            request=MagicMock(),
            response=MagicMock(status_code=404),
        )
    )
    
    # Test
    with patch("httpx.AsyncClient", return_value=mock_client):
        client = OllamaClient(config)
        
        with pytest.raises(ModelUnavailable):
            await client.generate("Test prompt")


@pytest.mark.asyncio
async def test_ollama_verify_connection():
    """Test connection verification with Ollama."""
    # Setup
    config = {
        "base_url": "http://localhost:11434",
        "model": "llama2",
    }
    
    # Mock the HTTP client
    mock_client = MagicMock()
    mock_client.get = AsyncMock(return_value=MagicMock(status_code=200))
    
    # Test
    with patch("httpx.AsyncClient", return_value=mock_client):
        client = OllamaClient(config)
        await client._verify_connection()  # Should not raise
        
        # Assertions
        mock_client.get.assert_awaited_once_with("/api/tags")


@pytest.mark.asyncio
async def test_ollama_close():
    """Test closing the Ollama client."""
    # Setup
    config = {
        "base_url": "http://localhost:11434",
        "model": "llama2",
    }
    
    # Mock the HTTP client
    mock_client = MagicMock()
    mock_client.aclose = AsyncMock()
    
    # Test
    with patch("httpx.AsyncClient", return_value=mock_client):
        client = OllamaClient(config)
        await client.close()
        
        # Assertions
        mock_client.aclose.assert_awaited_once()
