"""Tests for the LLM provider system."""

import json
from unittest import TestCase, mock
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from myjobspyai.llm.base import LLMResponse, LLMError, LLMRequestError
from myjobspyai.llm.providers.ollama import OllamaProvider, OllamaConfig


class TestOllamaConfig(TestCase):
    """Tests for OllamaConfig."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = OllamaConfig()
        self.assertEqual(config.base_url, "http://localhost:11434")
        self.assertEqual(config.model, "llama3:instruct")
        self.assertEqual(config.timeout, 600)
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.top_p, 0.9)
        self.assertIsNone(config.max_tokens)
        self.assertFalse(config.stream)

    def test_validation(self):
        """Test validation of config values."""
        # Test valid config
        config = OllamaConfig(
            base_url="http://example.com",
            model="llama3:8b",
            temperature=0.5,
            max_tokens=1000,
        )
        self.assertEqual(config.base_url, "http://example.com")
        self.assertEqual(config.model, "llama3:8b")
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.max_tokens, 1000)

        # Test invalid temperature
        with self.assertRaises(ValidationError):
            OllamaConfig(temperature=-1.0)
        with self.assertRaises(ValidationError):
            OllamaConfig(temperature=2.1)

        # Test invalid top_p
        with self.assertRaises(ValidationError):
            OllamaConfig(top_p=0.0)
        with self.assertRaises(ValidationError):
            OllamaConfig(top_p=1.1)


class TestOllamaProvider(TestCase):
    """Tests for OllamaProvider."""

    def setUp(self):
        """Set up test environment."""
        self.config = {
            "base_url": "http://localhost:11434",
            "model": "llama3:instruct",
            "timeout": 60,
            "max_retries": 2,
        }
        self.provider = OllamaProvider(self.config)

    async def asyncSetUp(self):
        """Async setup."""
        await self.provider._http_client.__aenter__()

    async def asyncTearDown(self):
        """Async teardown."""
        await self.provider.close()

    @pytest.mark.asyncio
    async def test_generate_success(self):
        """Test successful text generation."""
        # Mock the HTTP client
        mock_response = {
            "model": "llama3:instruct",
            "response": "Hello, world!",
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 20,
        }

        with patch.object(
            self.provider._http_client,
            "post",
            new_callable=AsyncMock,
            return_value=(
                MagicMock(status=200),
                mock_response,
            ),
        ) as mock_post:
            response = await self.provider.generate("Hello")

            # Verify the response
            self.assertIsInstance(response, LLMResponse)
            self.assertEqual(response.text, "Hello, world!")
            self.assertEqual(response.model, "llama3:instruct")
            self.assertEqual(response.usage["prompt_tokens"], 10)
            self.assertEqual(response.usage["completion_tokens"], 20)

            # Verify the request
            mock_post.assert_awaited_once()
            args, kwargs = mock_post.await_args
            self.assertEqual(args[0], "/api/generate")
            self.assertEqual(kwargs["json_data"]["prompt"], "Hello")
            self.assertEqual(kwargs["json_data"]["model"], "llama3:instruct")

    @pytest.mark.asyncio
    async def test_generate_streaming(self):
        """Test streaming text generation."""
        # Create a streaming response
        chunks = [
            b'{"model":"llama3:instruct","response":"Hello","done":false}\n',
            b'{"model":"llama3:instruct","response":", ","done":false}\n',
            b'{"model":"llama3:instruct","response":"world!","done":true,"prompt_eval_count":10,"eval_count":6}\n',
        ]

        # Mock the HTTP client to return a streaming response
        mock_response = MagicMock()
        mock_response.__aiter__.return_value = chunks

        with patch.object(
            self.provider._http_client,
            "post",
            new_callable=AsyncMock,
            return_value=(
                MagicMock(status=200),
                mock_response,
            ),
        ):
            # Enable streaming
            self.provider.config.stream = True

            # Call the method
            response = await self.provider.generate("Hello")

            # Verify the response
            self.assertIsInstance(response, LLMResponse)
            self.assertEqual(response.text, "Hello, world!")
            self.assertEqual(response.model, "llama3:instruct")
            self.assertTrue(response.metadata.get("streamed", False))

    @pytest.mark.asyncio
    async def test_generate_error(self):
        """Test error handling during text generation."""
        # Mock the HTTP client to raise an error
        with patch.object(
            self.provider._http_client,
            "post",
            new_callable=AsyncMock,
            side_effect=Exception("API error"),
        ):
            with self.assertRaises(LLMError):
                await self.provider.generate("Hello")

    @pytest.mark.asyncio
    async def test_get_embeddings(self):
        """Test getting embeddings."""
        # Mock the HTTP client
        mock_response = {
            "embedding": [0.1, 0.2, 0.3],
        }

        with patch.object(
            self.provider._http_client,
            "post",
            new_callable=AsyncMock,
            return_value=(
                MagicMock(status=200),
                mock_response,
            ),
        ) as mock_post:
            # Test with single text
            embeddings = await self.provider.get_embeddings("Hello, world!")
            self.assertEqual(len(embeddings), 1)
            self.assertEqual(len(embeddings[0]), 3)

            # Test with multiple texts
            mock_response = {
                "embeddings": [
                    [0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                ]
            }
            mock_post.return_value = (MagicMock(status=200), mock_response)

            embeddings = await self.provider.get_embeddings(["Hello", "World"])
            self.assertEqual(len(embeddings), 2)
            self.assertEqual(len(embeddings[0]), 3)
            self.assertEqual(len(embeddings[1]), 3)

    @pytest.mark.asyncio
    async def test_batch_generate(self):
        """Test batch generation of multiple prompts."""
        # Mock the generate method
        with patch.object(
            self.provider,
            "generate",
            side_effect=[
                LLMResponse(
                    "Response 1",
                    "llama3:instruct",
                    {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
                ),
                LLMResponse(
                    "Response 2",
                    "llama3:instruct",
                    {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
                ),
            ],
        ) as mock_generate:
            # Call batch_generate
            prompts = ["Prompt 1", "Prompt 2"]
            responses = await self.provider.batch_generate(prompts, max_concurrent=2)

            # Verify the responses
            self.assertEqual(len(responses), 2)
            self.assertEqual(responses[0].text, "Response 1")
            self.assertEqual(responses[1].text, "Response 2")

            # Verify generate was called with the right arguments
            self.assertEqual(mock_generate.await_count, 2)
            self.assertEqual(mock_generate.await_args_list[0].args[0], "Prompt 1")
            self.assertEqual(mock_generate.await_args_list[1].args[0], "Prompt 2")
