"""Tests for the HTTP client."""

import asyncio
import json
from unittest import TestCase, mock
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from myjobspyai.utils.http_client import HTTPClient, HTTPClientError, HTTPRequestError


class TestHTTPClient(TestCase):
    """Tests for HTTPClient."""

    def setUp(self):
        """Set up test environment."""
        self.base_url = "http://example.com/api"
        self.client = HTTPClient(
            base_url=self.base_url,
            timeout=30,
            max_retries=3,
            headers={"User-Agent": "test"},
        )

    async def asyncSetUp(self):
        """Async setup."""
        await self.client.__aenter__()

    async def asyncTearDown(self):
        """Async teardown."""
        await self.client.close()

    @pytest.mark.asyncio
    async def test_get_success(self):
        """Test successful GET request."""
        # Mock the session
        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"key": "value"})

        with patch.object(
            self.client._session,
            "get",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_get:
            # Make the request
            response, data = await self.client.get("/test")

            # Verify the response
            self.assertEqual(response.status, 200)
            self.assertEqual(data, {"key": "value"})

            # Verify the request
            mock_get.assert_awaited_once()
            args, kwargs = mock_get.await_args
            self.assertEqual(args[0], f"{self.base_url}/test")
            self.assertEqual(kwargs["headers"]["User-Agent"], "test")

    @pytest.mark.asyncio
    async def test_post_json(self):
        """Test POST request with JSON data."""
        # Mock the session
        mock_response = MagicMock()
        mock_response.status = 201
        mock_response.json = AsyncMock(return_value={"id": 123})

        with patch.object(
            self.client._session,
            "post",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_post:
            # Make the request
            response, data = await self.client.post(
                "/items",
                json_data={"name": "test"},
                headers={"X-Custom": "value"},
            )

            # Verify the response
            self.assertEqual(response.status, 201)
            self.assertEqual(data, {"id": 123})

            # Verify the request
            mock_post.assert_awaited_once()
            args, kwargs = mock_post.await_args
            self.assertEqual(args[0], f"{self.base_url}/items")
            self.assertEqual(kwargs["json"], {"name": "test"})
            self.assertEqual(kwargs["headers"]["X-Custom"], "value")
            self.assertEqual(kwargs["headers"]["Content-Type"], "application/json")

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """Test that the client retries on failure."""
        # Mock the session to fail twice then succeed
        mock_response_success = MagicMock()
        mock_response_success.status = 200
        mock_response_success.json = AsyncMock(return_value={"status": "ok"})

        mock_response_error = MagicMock()
        mock_response_error.status = 500
        mock_response_error.json = AsyncMock(
            return_value={"error": "Internal Server Error"}
        )

        with patch.object(
            self.client._session,
            "get",
            side_effect=[
                mock_response_error,
                mock_response_error,
                mock_response_success,
            ],
        ) as mock_get:
            # Make the request
            response, data = await self.client.get("/test-retry")

            # Verify the response
            self.assertEqual(response.status, 200)
            self.assertEqual(data, {"status": "ok"})

            # Verify it was retried 3 times (2 failures + 1 success)
            self.assertEqual(mock_get.await_count, 3)

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """Test that the client gives up after max retries."""
        # Mock the session to always fail
        mock_response = MagicMock()
        mock_response.status = 500
        mock_response.json = AsyncMock(return_value={"error": "Internal Server Error"})

        with patch.object(
            self.client._session,
            "get",
            return_value=mock_response,
        ) as mock_get:
            # Make the request
            with self.assertRaises(HTTPRequestError) as cm:
                await self.client.get("/test-failure")

            # Verify the error
            self.assertIn("Max retries (3) exceeded", str(cm.exception))

            # Verify it was retried 3 times
            self.assertEqual(mock_get.await_count, 3)

    @pytest.mark.asyncio
    async def test_timeout(self):
        """Test request timeout handling."""
        # Mock the session to simulate a timeout
        with patch.object(
            self.client._session,
            "get",
            side_effect=asyncio.TimeoutError("Request timed out"),
        ) as mock_get:
            # Make the request
            with self.assertRaises(HTTPRequestError) as cm:
                await self.client.get("/test-timeout", timeout=1)

            # Verify the error
            self.assertIn("Request timed out", str(cm.exception))

            # Verify it was retried
            self.assertEqual(mock_get.await_count, 3)

    @pytest.mark.asyncio
    async def test_download_file(self):
        """Test file download."""
        # Create a temporary file
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(b"test content")
            temp_path = temp_file.name

        try:
            # Mock the session to return our temporary file
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.content.read = AsyncMock(return_value=b"test content")

            with patch.object(
                self.client._session,
                "get",
                return_value=mock_response,
            ) as mock_get:
                # Download the file
                output_path = "downloaded_file.txt"
                await self.client.download_file("/file", output_path)

                # Verify the file was downloaded
                self.assertTrue(os.path.exists(output_path))
                with open(output_path, "rb") as f:
                    self.assertEqual(f.read(), b"test content")

                # Clean up
                os.unlink(output_path)

        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
