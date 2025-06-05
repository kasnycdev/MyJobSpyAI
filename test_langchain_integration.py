"""
Integration test for the LangChain provider.

This script tests the basic functionality of the LangChain provider.
"""

import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Import the provider
from myjobspyai.llm.providers import LangChainProvider


class TestLangChainProvider(unittest.IsolatedAsyncioTestCase):
    """Test cases for the LangChain provider."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "provider": "openai",
            "model": "gpt-4-turbo-preview",
            "api_key": "test-api-key",
            "temperature": 0.7,
            "max_tokens": 100,
            "provider_config": {}
        }

    @patch('myjobspyai.llm.providers.langchain_chat.ChatOpenAI')
    async def test_initialization(self, mock_chat_openai):
        """Test provider initialization."""
        # Mock the chat model
        mock_model = AsyncMock()
        mock_chat_openai.return_value = mock_model

        # Initialize the provider
        provider = LangChainProvider(self.config)

        # Verify the model was initialized with the correct parameters
        mock_chat_openai.assert_called_once_with(
            model="gpt-4-turbo-preview",
            temperature=0.7,
            max_tokens=100,
            api_key="test-api-key",
            streaming=False
        )

        # Clean up
        await provider.close()

    @patch('myjobspyai.llm.providers.langchain_chat.ChatOpenAI')
    async def test_generate(self, mock_chat_openai):
        """Test text generation."""
        # Mock the chat model and its response
        mock_model = AsyncMock()
        mock_chat_openai.return_value = mock_model

        # Mock the response from the model
        mock_response = MagicMock()
        mock_response.generations = [[MagicMock(text="Test response")]]
        mock_model.agenerate.return_value = mock_response

        # Initialize the provider
        provider = LangChainProvider(self.config)

        try:
            # Test generate
            response = await provider.generate("Test prompt")

            # Verify the response
            self.assertEqual(response.text, "Test response")
            self.assertEqual(response.model, "gpt-4-turbo-preview")
            mock_model.agenerate.assert_called_once()

        finally:
            await provider.close()

    @patch('myjobspyai.llm.providers.langchain_chat.ChatOpenAI')
    async def test_generate_stream(self, mock_chat_openai):
        """Test streaming text generation."""
        # Mock the chat model and its streaming response
        mock_model = AsyncMock()
        mock_chat_openai.return_value = mock_model

        # Mock the streaming response
        async def mock_astream(*args, **kwargs):
            yield "Chunk"
            yield " 1"
            yield " 2"

        mock_model.astream = mock_astream

        # Initialize the provider with streaming enabled
        self.config["streaming"] = True
        provider = LangChainProvider(self.config)

        try:
            # Test streaming
            chunks = []
            async for chunk in provider.generate_stream("Test prompt"):
                chunks.append(chunk.text)

            # Verify the response
            self.assertEqual(len(chunks), 3)
            self.assertEqual("".join(chunks), "Chunk 1 2")

        finally:
            await provider.close()

if __name__ == "__main__":
    unittest.main()
