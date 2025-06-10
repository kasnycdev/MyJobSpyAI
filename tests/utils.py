"""
Test utilities and helpers for the MyJobSpyAI test suite.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

import aiohttp
import pytest
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)


async def read_test_data(file_name: str) -> str:
    """Read test data from the test_data directory.

    Args:
        file_name: Name of the file to read from the test_data directory.

    Returns:
        The contents of the file as a string.
    """
    test_data_dir = Path(__file__).parent / "test_data"
    file_path = test_data_dir / file_name
    return file_path.read_text(encoding="utf-8")


async def read_json_test_data(file_name: str) -> Dict[str, Any]:
    """Read and parse JSON test data from the test_data directory.

    Args:
        file_name: Name of the JSON file to read.

    Returns:
        The parsed JSON data as a dictionary.
    """
    content = await read_test_data(file_name)
    return json.loads(content)


async def create_model_from_file(
    model_class: Type[T], file_name: str, **overrides: Any
) -> T:
    """Create a model instance from a JSON test data file.

    Args:
        model_class: The Pydantic model class to instantiate.
        file_name: Name of the JSON file containing the test data.
        **overrides: Field values to override in the test data.

    Returns:
        An instance of the model class populated with the test data.
    """
    data = await read_json_test_data(file_name)
    data.update(overrides)
    return model_class(**data)


class AsyncContextManagerMock:
    """A mock that works as an async context manager."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.entered = False
        self.exited = False

    async def __aenter__(self):
        self.entered = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.exited = True
        return False


class MockResponse:
    """Mock response for testing HTTP requests."""

    def __init__(
        self,
        status: int,
        json_data: Optional[Dict[str, Any]] = None,
        text: str = "",
        headers: Optional[Dict[str, str]] = None,
    ):
        self.status = status
        self._json_data = json_data or {}
        self._text = text
        self.headers = headers or {}
        self.url = "http://example.com"

    async def json(self) -> Dict[str, Any]:
        """Return the JSON response."""
        return self._json_data

    async def text(self) -> str:
        """Return the response text."""
        return self._text or json.dumps(self._json_data)

    def raise_for_status(self) -> None:
        """Raise an HTTPError if status is 4xx or 5xx."""
        if 400 <= self.status < 600:
            raise aiohttp.ClientResponseError(
                status=self.status,
                request_info=aiohttp.RequestInfo(
                    url=self.url,
                    method="GET",
                    headers={},
                    real_url=self.url,
                ),
                history=(),
            )


# Common test fixtures


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_http_response():
    """Fixture to create a mock HTTP response."""

    def _create_mock_response(
        status: int = 200,
        json_data: Optional[Dict[str, Any]] = None,
        text: str = "",
        headers: Optional[Dict[str, str]] = None,
    ) -> MockResponse:
        return MockResponse(status, json_data, text, headers)

    return _create_mock_response
