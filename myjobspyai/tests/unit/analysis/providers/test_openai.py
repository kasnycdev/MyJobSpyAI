"""Tests for the OpenAI provider."""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage

from myjobspyai.analysis.providers.openai import OpenAIClient


@pytest.mark.asyncio
async def test_openai_generate_success():
    """Test successful text generation with OpenAI."""
    # Setup
    config = {
        "api_key": "test-key",
        "model": "gpt-4",
        "timeout": 30.0,
        "connect_timeout": 10.0,
    }
    
    # Mock the OpenAI client
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=ChatCompletion(
            id="test-id",
            model="gpt-4",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant", content="Test response"
                    ),
                    finish_reason="stop",
                )
            ],
            created=1234567890,
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=20, total_tokens=30
            ),
        )
    )
    
    # Test
    with patch("openai.AsyncOpenAI", return_value=mock_client):
        client = OpenAIClient(config)
        response = await client.generate("Test prompt")
        
        # Assertions
        assert response == "Test response"
        mock_client.chat.completions.create.assert_awaited_once_with(
            model="gpt-4",
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=0.7,
            max_tokens=1000,
        )


@pytest.mark.asyncio
async def test_openai_generate_rate_limit():
    """Test rate limit handling with OpenAI."""
    # Setup
    config = {
        "api_key": "test-key",
        "model": "gpt-4",
    }
    
    # Mock the OpenAI client to raise a rate limit error
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=httpx.HTTPStatusError(
            "Rate limit exceeded",
            request=MagicMock(),
            response=MagicMock(status_code=429),
        )
    )
    
    # Test
    with patch("openai.AsyncOpenAI", return_value=mock_client):
        client = OpenAIClient(config)
        
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.generate("Test prompt")
            
        assert exc_info.value.response.status_code == 429


@pytest.mark.asyncio
async def test_openai_generate_streaming():
    """Test streaming text generation with OpenAI."""
    # Setup
    config = {
        "api_key": "test-key",
        "model": "gpt-4",
    }
    
    # Create a mock streaming response
    mock_chunk = MagicMock()
    mock_chunk.choices = [
        MagicMock(delta=MagicMock(role="assistant"), finish_reason=None),
        MagicMock(delta=MagicMock(content="Hello, "), finish_reason=None),
        MagicMock(delta=MagicMock(content="world!"), finish_reason="stop"),
    ]
    
    # Mock the OpenAI client
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_chunk)
    
    # Test
    with patch("openai.AsyncOpenAI", return_value=mock_client):
        client = OpenAIClient(config)
        response = await client.generate("Test prompt", stream=True)
        
        # Assertions
        assert response == "Hello, world!"


@pytest.mark.asyncio
async def test_openai_generate_with_custom_params():
    """Test text generation with custom parameters."""
    # Setup
    config = {
        "api_key": "test-key",
        "model": "gpt-4-turbo",
    }
    
    # Mock the OpenAI client
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=ChatCompletion(
            id="test-id",
            model="gpt-4-turbo",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant", content="Test response"
                    ),
                    finish_reason="stop",
                )
            ],
            created=1234567890,
            usage=CompletionUsage(
                prompt_tokens=10, completion_tokens=20, total_tokens=30
            ),
        )
    )
    
    # Test with custom parameters
    with patch("openai.AsyncOpenAI", return_value=mock_client):
        client = OpenAIClient(config)
        response = await client.generate(
            "Test prompt",
            model="gpt-4-turbo",
            temperature=0.9,
            max_tokens=500,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.5,
        )
        
        # Assertions
        assert response == "Test response"
        mock_client.chat.completions.create.assert_awaited_once_with(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=0.9,
            max_tokens=500,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.5,
        )


@pytest.mark.asyncio
async def test_openai_close():
    """Test closing the OpenAI client."""
    # Setup
    config = {
        "api_key": "test-key",
        "model": "gpt-4",
    }
    
    # Mock the OpenAI client
    mock_client = MagicMock()
    mock_client.close = AsyncMock()
    
    # Test
    with patch("openai.AsyncOpenAI", return_value=mock_client):
        client = OpenAIClient(config)
        await client.close()
        
        # Assertions
        mock_client.close.assert_awaited_once()
