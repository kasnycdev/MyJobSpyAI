"""
Configuration and fixtures for tests.
"""

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Fixtures for testing the configuration system ---


@pytest.fixture
def temp_config_file() -> str:
    """Create a temporary config file for testing."""
    config_data = {
        "debug": True,
        "environment": "test",
        "log_level": "DEBUG",
        "data_dir": "/tmp/test_data",
        "cache_dir": "/tmp/test_cache",
        "config_dir": "/tmp/test_config",
        "database": {
            "url": "sqlite:///test.db",
            "echo": True,
            "pool_size": 5,
            "max_overflow": 10,
        },
        "logging": {
            "level": "DEBUG",
            "file": "/tmp/test.log",
            "max_size": 10485760,  # 10MB
            "backup_count": 5,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        },
        "api": {
            "enabled": True,
            "host": "0.0.0.0",
            "port": 8000,
            "debug": True,
            "cors_origins": ["*"],
            "api_key": "test-api-key",
            "rate_limit": "100/minute",
        },
        "llm_providers": {
            "ollama": {
                "enabled": True,
                "type": "ollama",
                "config": {
                    "base_url": "http://localhost:11434",
                    "model": "llama3:instruct",
                    "timeout": 60,
                    "max_retries": 3,
                    "temperature": 0.7,
                    "top_p": 0.9,
                },
            },
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        import yaml

        yaml.dump(config_data, f)

    yield f.name

    # Clean up
    if os.path.exists(f.name):
        os.unlink(f.name)


@pytest.fixture
def mock_llm_config() -> Dict[str, Any]:
    """Return a mock LLM configuration for testing."""
    return {
        "type": "ollama",
        "config": {
            "base_url": "http://localhost:11434",
            "model": "llama3:instruct",
            "timeout": 60,
            "max_retries": 3,
            "temperature": 0.7,
            "top_p": 0.9,
        },
    }


# --- Fixtures for testing the HTTP client ---


@pytest.fixture
async def http_client() -> AsyncGenerator:
    """Create an HTTP client for testing."""
    client = HTTPClient(
        base_url="http://example.com/api",
        timeout=30,
        max_retries=3,
        headers={"User-Agent": "test"},
    )

    # Start the client
    await client.__aenter__()

    yield client

    # Clean up
    await client.close()


# --- Fixtures for testing the LLM provider ---


@pytest.fixture
async def ollama_provider() -> AsyncGenerator:
    """Create an Ollama provider for testing."""
    provider = OllamaProvider(
        {
            "base_url": "http://localhost:11434",
            "model": "llama3:instruct",
            "timeout": 60,
            "max_retries": 3,
            "temperature": 0.7,
            "top_p": 0.9,
        }
    )

    # Start the provider
    await provider._http_client.__aenter__()

    yield provider

    # Clean up
    await provider.close()


# --- Fixtures for testing the LangChain provider ---


@pytest.fixture
def mock_config():
    """Return a mock configuration for testing."""
    return {
        "type": "langchain",
        "class": "langchain_community.chat_models.ChatOpenAI",
        "params": {
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 100,
            "openai_api_key": "test-api-key",
        },
    }


@pytest.fixture
def mock_provider(mock_config):
    """Create a mock provider with the given config."""
    from analysis.providers.langchain_provider import LangChainProvider

    return LangChainProvider(mock_config)


@pytest.fixture
def mock_otel():
    """Mock OpenTelemetry components."""
    with patch("opentelemetry.trace") as mock_trace, patch(
        "opentelemetry.metrics"
    ) as mock_metrics, patch("opentelemetry.trace.status") as mock_status:

        # Mock tracer
        mock_tracer = MagicMock()
        mock_span = MagicMock()
        mock_span_context = MagicMock()
        mock_span_context.trace_id = 12345
        mock_span.context = mock_span_context
        mock_tracer.start_span.return_value = mock_span
        mock_tracer.start_as_current_span.return_value.__enter__.return_value = (
            mock_span
        )
        mock_trace.get_tracer.return_value = mock_tracer

        # Mock meter
        mock_meter = MagicMock()
        mock_counter = MagicMock()
        mock_histogram = MagicMock()
        mock_meter.create_counter.return_value = mock_counter
        mock_meter.create_histogram.return_value = mock_histogram
        mock_metrics.get_meter.return_value = mock_meter

        # Mock status
        mock_status.StatusCode = MagicMock()
        mock_status.Status = MagicMock()

        yield {
            "tracer": mock_tracer,
            "span": mock_span,
            "meter": mock_meter,
            "counter": mock_counter,
            "histogram": mock_histogram,
            "status": mock_status,
        }


# Add any other test fixtures here


# Configure pytest to use asyncio for async tests
def pytest_configure(config):
    """Configure pytest for asyncio tests."""
    config.option.asyncio_mode = "auto"
