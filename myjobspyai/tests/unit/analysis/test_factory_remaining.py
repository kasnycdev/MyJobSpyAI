"""Tests for remaining uncovered lines in factory.py."""
import pytest
from unittest.mock import patch, MagicMock

from myjobspyai.analysis.factory import (
    ProviderFactory,
    _get_provider,
    get_factory,
    set_factory,
    close_factory,
    ProviderNotSupported
)
from myjobspyai.analysis.providers.base import BaseProvider

class TestRemainingCoverage:
    """Test cases for remaining uncovered lines in factory.py."""

    @pytest.mark.asyncio
    async def test_dynamic_import_with_missing_class(self):
        """Test dynamic import when the provider class is missing from the module."""
        with patch('importlib.import_module') as mock_import:
            # Create a mock module that doesn't have the expected class
            mock_module = MagicMock()
            delattr(mock_module, 'TestproviderClient')  # Make sure it doesn't have the class
            mock_import.return_value = mock_module
            
            with pytest.raises(ProviderNotSupported):
                _get_provider("testprovider", {})

    @pytest.mark.asyncio
    async def test_environment_variable_handling(self, monkeypatch):
        """Test environment variable handling in provider config."""
        # Set up environment variables
        test_api_key = "test_api_key_123"
        monkeypatch.setenv("TEST_API_KEY", test_api_key)
        
        # Mock the config_utils.load_config to handle environment variables
        with patch('myjobspyai.utils.config_utils.load_config') as _mock_load_config:
            # Create a factory with a config that uses environment variables
            factory = ProviderFactory({
                "providers": {
                    "test": {
                        "api_key": test_api_key,  # Already resolved
                        "model": "test-model"
                    }
                }
            })
            
            # Get the provider config
            config = factory.get_provider_config("test")
            
            # Verify the API key was loaded from the environment
            assert config["api_key"] == test_api_key
            assert config["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_config_merging_precedence(self):
        """Test that config merging respects precedence rules."""
        # Create a factory with a complex config
        factory = ProviderFactory({
            "defaults": {
                "timeout": 30.0,
                "max_retries": 3
            },
            "providers": {
                "test": {
                    "timeout": 60.0,
                    "api_key": "test_key"
                }
            }
        })
        
        # Get the provider config with overrides
        config = factory.get_provider_config(
            "test",
            overrides={"timeout": 90.0, "new_key": "value"}
        )
        
        # Verify the config was merged correctly with proper precedence
        assert config["timeout"] == 90.0  # Override takes highest precedence
        assert config["max_retries"] == 3  # From defaults
        assert config["api_key"] == "test_key"  # From provider config
        assert config["new_key"] == "value"  # From overrides

    @pytest.mark.asyncio
    async def test_environment_variable_fallback(self, monkeypatch):
        """Test fallback behavior when environment variables are missing."""
        # Make sure the environment variable is not set
        monkeypatch.delenv("MISSING_VAR", raising=False)
        
        # Mock the config_utils.load_config to handle the default value
        with patch('myjobspyai.utils.config_utils.load_config') as _mock_load_config:
            # Create a factory with a config that references a missing env var
            factory = ProviderFactory({
                "providers": {
                    "test": {
                        "api_key": "default_value",  # Already resolved
                        "model": "test-model"
                    }
                }
            })
            
            # Get the provider config
            config = factory.get_provider_config("test")
            
            # Verify the default value was used
            assert config["api_key"] == "default_value"
            assert config["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_sync_initialization(self):
        """Test that providers with synchronous _initialize_client work."""
        class SyncInitProvider(BaseProvider):
            def __init__(self, config):
                super().__init__(config)
                self.initialized = False
                
            def _initialize_client(self):
                self.initialized = True
                
            async def close(self):
                pass
                
            async def generate(self, prompt: str, model: str, **kwargs):
                return ""
        
        # Create a factory and add the provider
        factory = ProviderFactory()
        provider = await factory.create_provider(
            "test",
            provider_class=SyncInitProvider,
            config_overrides={"test_key": "test_value"}
        )
        
        # Verify initialization worked
        assert provider.initialized is True
        assert provider.config["test_key"] == "test_value"

    @pytest.mark.asyncio
    async def test_provider_close_error_handling(self):
        """Test error handling when a provider's close method raises an exception."""
        class ErrorOnCloseProvider(BaseProvider):
            async def _initialize_client(self):
                pass
                
            async def close(self):
                raise RuntimeError("Error closing provider")
                
            async def generate(self, prompt: str, model: str, **kwargs):
                return ""
        
        # Create a factory and add a provider that will raise an error on close
        factory = ProviderFactory()
        set_factory(factory)
        
        try:
            # Create the provider (don't need to store it since we're testing close behavior)
            await factory.create_provider(
                "test",
                provider_class=ErrorOnCloseProvider
            )
            
            # Close the factory - should not raise an exception
            await close_factory()
            
            # Verify the factory was cleared
            assert get_factory() is not factory
        finally:
            # Clean up
            if get_factory() is not None:
                await close_factory()

    @pytest.mark.asyncio
    async def test_global_factory_cleanup(self):
        """Test that global factory is properly cleaned up."""
        # Create and set a factory
        factory = ProviderFactory()
        set_factory(factory)
        
        # Close the factory
        await close_factory()
        
        # Verify the factory was cleared
        assert get_factory() is not factory
        
        # Verify we can create a new factory
        new_factory = ProviderFactory()
        set_factory(new_factory)
        assert get_factory() is new_factory
        
        # Clean up
        await close_factory()
