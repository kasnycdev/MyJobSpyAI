"""Tests for the configuration system."""

import os
import tempfile
from pathlib import Path
from unittest import TestCase, mock

import pytest
import yaml

from myjobspyai.config_new import (
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
        """Test that name is required."""
        with self.assertRaises(ValueError):
            LLMProviderConfig()  # Missing name

    def test_default_values(self):
        """Test default values are set correctly."""
        config = LLMProviderConfig(name="test")
        self.assertEqual(config.name, "test")
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
        self.assertEqual(config.log_level, "INFO")
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
                "MYJOBSPYAI_LOG_LEVEL": "DEBUG",
                "MYJOBSPYAI_DATA_DIR": "/custom/data",
                "MYJOBSPYAI_DATABASE__URL": "sqlite:///test.db",
            },
        ):
            config = AppConfig()
            self.assertTrue(config.debug)
            self.assertEqual(config.environment, "development")
            self.assertEqual(config.log_level, "DEBUG")
            self.assertEqual(Path(config.data_dir), Path("/custom/data").resolve())
            self.assertEqual(config.database.url, "sqlite:///test.db")


class TestConfigLoading(TestCase):
    """Tests for config loading and saving."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_dir = Path(self.temp_dir.name) / "config"
        self.config_dir.mkdir()
        self.config_file = self.config_dir / "config.yaml"

    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    def test_load_default_config(self):
        """Test loading default config when no file exists."""
        # Config file doesn't exist yet
        self.assertFalse(self.config_file.exists())

        # Load config - should create default config
        config = load_config(self.config_file)

        # Verify default values
        self.assertIsInstance(config, AppConfig)
        self.assertFalse(config.debug)
        self.assertEqual(config.environment, "production")

        # Verify file was created
        self.assertTrue(self.config_file.exists())

    def test_save_and_load_config(self):
        """Test saving and loading a config."""
        # Create a config with custom values
        config = AppConfig(
            debug=True,
            environment="test",
            log_level="DEBUG",
            data_dir=Path("/custom/data"),
            llm_providers={
                "ollama": {
                    "name": "ollama",
                    "enabled": True,
                    "model": "llama3:instruct",
                }
            },
        )

        # Save the config
        save_config(config, self.config_file)

        # Verify file was created
        self.assertTrue(self.config_file.exists())

        # Load the config
        loaded_config = load_config(self.config_file)

        # Verify values were preserved
        self.assertTrue(loaded_config.debug)
        self.assertEqual(loaded_config.environment, "test")
        self.assertEqual(loaded_config.log_level, "DEBUG")
        self.assertEqual(loaded_config.data_dir, Path("/custom/data").resolve())
        self.assertIn("ollama", loaded_config.llm_providers)
        self.assertEqual(
            loaded_config.llm_providers["ollama"].model, "llama3:instruct"
        )

    def test_invalid_config_file(self):
        """Test handling of invalid config file."""
        # Create an invalid YAML file
        self.config_file.write_text("invalid: yaml: file:")

        # Loading should fall back to defaults
        with self.assertLogs(level="ERROR") as cm:
            config = load_config(self.config_file)

        # Should still return a valid config with defaults
        self.assertIsInstance(config, AppConfig)

        # Should log an error
        self.assertIn("Error loading config", "\n".join(cm.output))
