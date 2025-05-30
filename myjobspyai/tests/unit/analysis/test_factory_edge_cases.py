"""Tests for edge cases in the provider factory."""
from unittest.mock import patch, MagicMock
import pytest

from myjobspyai.analysis.factory import (
    ProviderFactory,
    ProviderNotSupported,
    _get_provider,
    get_factory,
    set_factory,
    close_factory
)
from myjobspyai.analysis.providers.base import BaseProvider

class TestEdgeCases:
    """Test cases for edge cases in the provider factory."""

    @pytest.mark.asyncio
    async def test_provider_dynamic_import(self):
        """Test dynamic import of provider modules."""
        # Create a real provider class for testing
        class TestProvider(BaseProvider):
            async def _initialize_client(self):
                pass
                
            async def close(self):
                pass
                
            async def generate(self, prompt: str, model: str, **kwargs):
                return ""
        
        # Mock the importlib.import_module to simulate a successful import
        with patch('importlib.import_module') as mock_import:
            # Set up the mock to return a module with our test provider class
            mock_module = MagicMock()
            mock_module.TestproviderClient = TestProvider
            mock_import.return_value = mock_module

            # Call _get_provider with a provider that needs to be imported
            provider_class = _get_provider("testprovider", {})
            
            # Verify the import was attempted
            mock_import.assert_called_once_with('.providers.testprovider', package='myjobspyai.analysis')
            assert provider_class == TestProvider

    @pytest.mark.asyncio
    async def test_unsupported_provider_error(self):
        """Test error message for unsupported providers."""
        with pytest.raises(ProviderNotSupported) as exc_info:
            _get_provider("nonexistent_provider", {})
        
        assert "Provider 'nonexistent_provider' is not supported" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_environment_variable_loading(self, monkeypatch):
        """Test loading of environment variables in provider config."""
        # Set up environment variables
        test_api_key = "test_api_key_123"
        monkeypatch.setenv("OPENAI_API_KEY", test_api_key)
        
        # Create a factory with a config that uses environment variables
        factory = ProviderFactory({
            "providers": {
                "openai": {
                    "model": "gpt-4"
                }
            }
        })
        
        # Get the provider config
        config = factory.get_provider_config("openai")
        
        # Verify the API key was loaded from the environment
        assert config["api_key"] == test_api_key

    @pytest.mark.asyncio
    async def test_invalid_provider_class_type(self):
        """Test error handling for invalid provider class types."""
        # Test with a class that's not a subclass of BaseProvider
        class NotAProvider:
            pass
            
        # Mock the import to return our invalid class
        with patch('importlib.import_module') as mock_import:
            mock_module = MagicMock()
            mock_module.TestClient = NotAProvider
            mock_import.return_value = mock_module
            
            # This will raise a ValueError because the class is not a subclass of BaseProvider
            with pytest.raises(ValueError, match="is not a subclass of BaseProvider"):
                _get_provider("test", {})

    @pytest.mark.asyncio
    async def test_provider_close_with_error(self):
        """Test error handling when closing a provider raises an exception."""
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
