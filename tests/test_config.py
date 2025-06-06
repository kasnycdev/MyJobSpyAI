"""Tests for the configuration system."""

import os
import shutil
import tempfile
from pathlib import Path
from unittest import TestCase, mock

import pytest
import yaml

from myjobspyai.config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_CONFIG_DIR,
    DEFAULT_DATA_DIR,
    APIConfig,
    AppConfig,
    DatabaseConfig,
    LLMProviderConfig,
    LoggingConfig,
    load_config,
    save_config,
)


class TestDatabaseConfig(TestCase):
    """Tests for DatabaseConfig."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = DatabaseConfig()
        self.assertEqual(config.url, f"sqlite:///{DEFAULT_DATA_DIR}/myjobspyai.db")
        self.assertFalse(config.echo)
        self.assertEqual(config.pool_size, 5)
        self.assertEqual(config.max_overflow, 10)

    def test_custom_values(self):
        """Test custom values are set correctly."""
        config = DatabaseConfig(
            url="postgresql://user:pass@localhost/db",
            echo=True,
            pool_size=10,
            max_overflow=20,
        )
        self.assertEqual(config.url, "postgresql://user:pass@localhost/db")
        self.assertTrue(config.echo)
        self.assertEqual(config.pool_size, 10)
        self.assertEqual(config.max_overflow, 20)


class TestLLMProviderConfig(TestCase):
    """Tests for LLMProviderConfig."""

    def test_required_fields(self):
        """Test that name and type are required."""
        with self.assertRaises(ValueError):
            LLMProviderConfig()  # Missing name and type

        with self.assertRaises(ValueError):
            LLMProviderConfig(name="test")  # Missing type

        with self.assertRaises(ValueError):
            LLMProviderConfig(type="test_type")  # Missing name

    def test_default_values(self):
        """Test default values are set correctly."""
        config = LLMProviderConfig(name="test", type="test_type")
        self.assertEqual(config.name, "test")
        self.assertEqual(config.type, "test_type")
        self.assertTrue(config.enabled)
        self.assertEqual(config.timeout, 600)
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.temperature, 0.7)
        self.assertIsNone(config.api_key)
        self.assertIsNone(config.base_url)


class TestLoggingConfig(TestCase):
    """Tests for LoggingConfig."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = LoggingConfig()
        self.assertEqual(config.level, "INFO")
        self.assertEqual(config.file, DEFAULT_DATA_DIR / "logs" / "myjobspyai.log")
        self.assertEqual(config.max_size, 10485760)  # 10MB in bytes
        self.assertEqual(config.backup_count, 5)
        self.assertIn("%(asctime)s", config.format)


class TestAPIConfig(TestCase):
    """Tests for APIConfig."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = APIConfig()
        self.assertTrue(config.enabled)
        self.assertEqual(config.host, "0.0.0.0")
        self.assertEqual(config.port, 8000)
        self.assertFalse(config.debug)
        self.assertEqual(config.cors_origins, ["*"])
        self.assertIsNone(config.api_key)
        self.assertEqual(config.rate_limit, "100/minute")


class TestAppConfig(TestCase):
    """Tests for AppConfig."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = AppConfig()
        self.assertFalse(config.debug)
        self.assertEqual(config.environment, "production")
        self.assertEqual(config.logging.level, "INFO")
        self.assertEqual(config.data_dir, DEFAULT_DATA_DIR)
        self.assertEqual(config.cache_dir, DEFAULT_CACHE_DIR)
        self.assertEqual(config.config_dir, DEFAULT_CONFIG_DIR)
        self.assertIsInstance(config.database, DatabaseConfig)
        self.assertIsInstance(config.logging, LoggingConfig)
        self.assertIsInstance(config.api, APIConfig)
        self.assertEqual(config.llm_providers, {})

    def test_environment_variables(self):
        """Test that environment variables override defaults."""
        with mock.patch.dict(
            os.environ,
            {
                "MYJOBSPYAI_DEBUG": "true",
                "MYJOBSPYAI_ENVIRONMENT": "development",
                "MYJOBSPYAI_LOGGING__LEVEL": "DEBUG",
                "MYJOBSPYAI_DATA_DIR": "/custom/data",
                "MYJOBSPYAI_DATABASE__URL": "sqlite:///test.db",
            },
            clear=True,
        ):
            config = AppConfig()
            self.assertTrue(config.debug)
            self.assertEqual(config.environment, "development")
            self.assertEqual(config.logging.level, "DEBUG")
            self.assertEqual(str(config.data_dir), "/custom/data")
            self.assertEqual(config.database.url, "sqlite:///test.db")


class TestConfigLoading(TestCase):
    """Tests for config loading and saving."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.config_file = self.test_dir / "config.yaml"
        self.data_dir = self.test_dir / "data"
        self.cache_dir = self.test_dir / "cache"
        self.config_dir = self.test_dir / "config"

        # Create necessary directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        """Clean up test environment."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_load_default_config(self):
        """Test loading default config when no file exists."""
        # Create a minimal config file with data_dir set to our test directory
        config_data = {
            "data_dir": str(self.data_dir),
            "cache_dir": str(self.cache_dir),
            "config_dir": str(self.config_dir),
        }
        with open(self.config_file, "w") as f:
            import yaml

            yaml.dump(config_data, f)

        config = load_config(self.config_file)
        self.assertIsInstance(config, AppConfig)
        self.assertEqual(str(config.data_dir), str(self.data_dir.resolve()))

    def test_save_and_load_config(self):
        """Test saving and loading a config."""
        # Create a custom config
        config = AppConfig(
            debug=True,
            environment="test",
            data_dir=str(self.data_dir),
            cache_dir=str(self.cache_dir),
            config_dir=str(self.config_dir),
            database=DatabaseConfig(url="sqlite:///test.db"),
            logging=LoggingConfig(level="DEBUG"),
            llm_providers={
                "openai": {"name": "openai", "type": "openai", "model": "gpt-4"}
            },
        )

        # Save config
        config.save(self.config_file)
        self.assertTrue(self.config_file.exists())

        # Load config
        loaded_config = load_config(self.config_file)
        self.assertEqual(loaded_config.debug, True)
        self.assertEqual(loaded_config.environment, "test")
        self.assertEqual(loaded_config.database.url, "sqlite:///test.db")
        self.assertEqual(loaded_config.logging.level, "DEBUG")
        self.assertIn("openai", loaded_config.llm_providers)
        self.assertEqual(loaded_config.llm_providers["openai"].model, "gpt-4")

    def test_invalid_config_file(self):
        """Test handling of invalid config file."""
        # Create a file with completely invalid YAML
        with open(self.config_file, "w") as f:
            f.write("invalid yaml: file")

        # Should not raise an exception, but return default config
        config = load_config(self.config_file)
        self.assertIsInstance(config, AppConfig)
        self.assertIsInstance(config, AppConfig)

        # Should log an error
