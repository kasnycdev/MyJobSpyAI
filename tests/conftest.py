"""
Common fixtures for configuration tests.

This module provides shared fixtures and utilities for testing the configuration system.
"""
from __future__ import annotations

import pytest
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

@pytest.fixture
def temp_env_file() -> Path:
    """Create a temporary .env file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.env') as f:
        env_path = Path(f.name)
    yield env_path
    env_path.unlink(missing_ok=True)

@pytest.fixture
def temp_config_file() -> Path:
    """Create a temporary configuration file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as f:
        config_path = Path(f.name)
    yield config_path
    config_path.unlink(missing_ok=True)

@pytest.fixture
def reset_env() -> None:
    """Reset environment variables to their original state after test."""
    original_env = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(original_env)
    load_dotenv()
