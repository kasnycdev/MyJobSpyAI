"""Enhanced tests for the provider factory module with Pydantic v2 configuration."""
from unittest.mock import MagicMock, patch
import pytest

from myjobspyai.analysis.factory import (
    ProviderFactory,
    ProviderNotSupported,
    get_factory,
    set_factory,
    close_factory,
    _get_provider
)
from myjobspyai.analysis.providers.base import BaseProvider


class TestProvider(BaseProvider):
    """Test provider implementation for testing."""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.initialized = False
        self.closed = False
        self.api_key = config.get("api_key")
    
    async def _initialize_client(self):
        self.initialized = True
    
    async def close(self) -> None:
        """Mark the provider as closed."""
        if not self.closed:
            self.closed = True
            await super().close()
    
    async def generate(self, prompt: str, model: str, **kwargs) -> str:
        return f"Response to: {prompt}"


@pytest.fixture(name="provider_factory")
def fixture_provider_factory():
    """Create a test provider factory with test configuration."""
    return ProviderFactory({
        "defaults": {
            "timeout": 30.0,
            "max_retries": 3,
            "temperature": 0.7
        },
        "providers": {
            "test": {
                "model": "test-model",
                "api_key": "test-api-key"
            }
        },
        "environment": "test"
    })


@pytest.fixture(name="mock_provider_class")
def fixture_mock_provider_class():
    """Return a mock provider class."""
    class MockProvider(BaseProvider):
        def __init__(self, config):
            super().__init__(config)
            self.initialized = False
            self.closed = False
            self.api_key = config.get("api_key")
        
        async def _initialize_client(self):
            self.initialized = True
        
        async def close(self):
            self.closed = True
        
        async def generate(self, prompt: str, model: str, **kwargs):
            return f"Response to: {prompt}"
    
    return MockProvider


@pytest.fixture
def mock_import():
    """Mock importlib.import_module to avoid actual imports."""
    with patch('importlib.import_module') as mock_import:
        mock_module = MagicMock()
        mock_module.TestProvider = TestProvider
        mock_import.return_value = mock_module
        yield mock_import


@pytest.mark.asyncio
async def test_create_provider_success(provider_factory):
    """Test creating a provider successfully."""
    # Create a test provider class
    class TestProvider(BaseProvider):
        def __init__(self, config):
            super().__init__(config)
            self.initialized = False
            
        async def _initialize_client(self):
            self.initialized = True
            
        async def close(self):
            pass
            
        async def generate(self, prompt: str, model: str, **kwargs):
            return ""
    
    provider = await provider_factory.create_provider(
        "test",
        provider_class=TestProvider,
        config_overrides={"test_key": "test_value"}
    )
    
    assert isinstance(provider, TestProvider)
    assert provider.config["test_key"] == "test_value"
    assert provider.initialized is True


@pytest.mark.asyncio
async def test_create_unsupported_provider(provider_factory):
    """Test creating an unsupported provider raises an error."""
    with pytest.raises(ProviderNotSupported):
        await provider_factory.create_provider("unsupported")


@pytest.mark.asyncio
async def test_get_or_create_provider(provider_factory, mock_provider_class):
    """Test getting or creating a provider."""
    # First call should create a new provider
    provider1 = await provider_factory.get_or_create_provider(
        "test",
        provider_class=mock_provider_class
    )
    
    # Second call should return the same instance
    provider2 = await provider_factory.get_or_create_provider(
        "test",
        provider_class=mock_provider_class
    )
    
    assert provider1 is provider2


@pytest.mark.asyncio
async def test_close_provider(provider_factory, mock_provider_class):
    """Test closing a provider."""
    # Create a provider
    provider = await provider_factory.create_provider(
        "test",
        provider_class=mock_provider_class
    )
    
    # Close the provider directly
    await provider.close()
    
    # Verify it's marked as closed
    assert provider.closed is True


@pytest.mark.asyncio
async def test_context_manager(provider_factory, mock_provider_class):
    """Test using the factory as a context manager."""
    # Create a provider using get_or_create_provider to ensure it's tracked by the factory
    provider = await provider_factory.get_or_create_provider(
        "test",
        provider_class=mock_provider_class
    )
    
    # Verify the provider was initialized
    assert provider.initialized is True
    
    # Use the factory as a context manager
    async with provider_factory as factory:
        # Create another provider within the context
        context_provider = await factory.get_or_create_provider(
            "test2",
            provider_class=mock_provider_class
        )
        # Verify the provider was initialized
        assert context_provider.initialized is True
    
    # After exiting the context, the factory's close() method should have been called
    # and all tracked providers should be closed
    assert provider.closed is True
    assert context_provider.closed is True


@pytest.mark.asyncio
async def test_global_functions():
    """Test the global factory functions."""
    # Test get_factory
    factory1 = get_factory()
    assert factory1 is not None
    
    # Test set_factory
    new_factory = ProviderFactory()
    set_factory(new_factory)
    assert get_factory() is new_factory
    
    # Test close_factory
    await close_factory()
    # After closing, getting a new factory should return a new instance
    new_factory = get_factory()
    assert new_factory is not None
    assert new_factory is not factory1


def test_get_factory_with_config():
    """Test that get_factory creates a factory with the provided config."""
    # Ensure we start with no factory
    close_factory()
    
    # Create a config dictionary with a custom provider config
    # Note: Top-level config keys other than 'providers' are not merged by get_factory
    config = {
        "providers": {
            "test": {
                "provider": "test",
                "model": "test-model",
                "test_key": "test_value"
            }
        }
    }
    
    # Get a factory with the config
    factory = get_factory(config)
    
    # Verify the factory was created with the correct config
    # Get the config as a dictionary
    config_dict = factory._config.model_dump()
    
    # Check that the provider config was set correctly
    assert "test" in config_dict["providers"]
    assert config_dict["providers"]["test"]["test_key"] == "test_value"
    
    # Test updating config on existing factory
    # Note: get_factory doesn't merge top-level config keys, only provider configs
    new_config = {
        "providers": {
            "test2": {
                "provider": "test2",
                "model": "test-model-2"
            }
        }
    }
    updated_factory = get_factory(new_config)
    
    # The factory should be a new instance with the updated config
    assert updated_factory is not factory
    
    # Get the updated config as a dictionary
    updated_config_dict = updated_factory._config.model_dump()
    
    # The new provider config should be present
    assert "test2" in updated_config_dict["providers"]
    # The original provider config should not be present as we created a new instance with new config
    assert "test" not in updated_config_dict["providers"]
    
    # The config should have the default values from DEFAULT_FACTORY_CONFIG
    assert "defaults" in updated_config_dict
    assert isinstance(updated_config_dict["defaults"], dict)
    assert "timeout" in updated_config_dict["defaults"]
    assert "max_retries" in updated_config_dict["defaults"]
    assert "temperature" in updated_config_dict["defaults"]
    
    # The environment should be set to production by default
    assert updated_config_dict.get("environment") == "production"
    
    # Clean up
    close_factory()

@pytest.mark.asyncio
async def test_get_provider_with_class_reference():
    """Test getting a provider with a direct class reference."""
    # Create a test provider class with all required methods
    class TestProvider(BaseProvider):
        def __init__(self, config):
            super().__init__(config)
            self.initialized = False
            
        async def _initialize_client(self):
            self.initialized = True
            
        async def close(self):
            pass
            
        async def generate(self, prompt: str, model: str, **kwargs):
            return ""
    
    # Create a factory with the test provider class directly
    factory = ProviderFactory()
    
    # Test getting the provider with a direct class reference
    provider = await factory.create_provider(
        "test", 
        provider_class=TestProvider,
        config_overrides={"test_key": "test_value"}
    )
    
    assert provider is not None
    assert isinstance(provider, TestProvider)
    assert provider.initialized is True
    assert provider.config["test_key"] == "test_value"


def test_get_provider_with_invalid_provider():
    """Test _get_provider with an invalid provider."""
    with pytest.raises(ProviderNotSupported):
        _get_provider("nonexistent_provider", {})


def test__get_provider_with_direct_class():
    """Test _get_provider with a direct class reference."""
    class TestProvider(BaseProvider):
        pass
    
    # Create a test PROVIDER_CLASSES with a direct class reference
    test_providers = {"test": TestProvider}
    
    # Patch the PROVIDER_CLASSES
    with patch('myjobspyai.analysis.factory.PROVIDER_CLASSES', test_providers):
        provider_class = _get_provider("test", {})
        assert provider_class is TestProvider


def test__get_provider_with_invalid_subclass():
    """Test _get_provider with a class that's not a BaseProvider subclass."""
    class NotAProvider:
        pass
    
    # Create a test PROVIDER_CLASSES with an invalid provider class
    test_providers = {"test": NotAProvider}
    
    # Patch the PROVIDER_CLASSES
    with patch('myjobspyai.analysis.factory.PROVIDER_CLASSES', test_providers):
        with pytest.raises(ValueError, match="is not a subclass of BaseProvider"):
            _get_provider("test", {})


def test__get_provider_with_dynamic_import():
    """Test _get_provider with dynamic import of a provider."""
    # Create a mock module with a test provider class
    mock_module = MagicMock()
    
    class MockProvider(BaseProvider):
        pass
    
    # Set up the mock module to return our test provider class
    mock_module.MockproviderClient = MockProvider
    
    # Patch the import_module function to return our mock module
    with patch('importlib.import_module', return_value=mock_module) as mock_import:
        # Test with a provider that's not in PROVIDER_CLASSES
        provider_class = _get_provider("mockprovider", {})
        assert provider_class is MockProvider
        
        # Verify the import was attempted with the correct path
        mock_import.assert_called_once_with('.providers.mockprovider', package='myjobspyai.analysis')


def test__get_provider_with_invalid_dynamic_import():
    """Test _get_provider with a non-existent dynamic import."""
    # Patch the import_module function to raise ImportError
    with patch('importlib.import_module', side_effect=ImportError("Module not found")):
        with pytest.raises(ProviderNotSupported, match="Provider 'nonexistent' is not supported"):
            _get_provider("nonexistent", {})


@pytest.mark.asyncio
async def test_provider_factory_context_manager():
    """Test using the ProviderFactory as a context manager."""
    # Create a mock provider class
    class MockProvider(BaseProvider):
        def __init__(self, config):
            super().__init__(config)
            self.initialized = False
            self.closed = False
        
        async def _initialize_client(self):
            self.initialized = True
        
        async def close(self):
            self.closed = True
            
        async def generate(self, prompt: str, model: str, **kwargs) -> str:
            return f"Response to: {prompt}"
    
    # Create a new factory for this test
    factory = ProviderFactory()
    
    # Test that the context manager properly closes providers
    async with factory as f:
        # Create provider within the context
        provider = await f.create_provider("test", provider_class=MockProvider)
        
        # Verify the provider was initialized
        assert provider.initialized is True
        assert provider.closed is False
        
        # Manually add the provider to the factory's _providers dict
        # to ensure it gets closed when the context exits
        f._providers["test"] = provider
    
    # Verify the provider was closed when exiting the context
    assert provider.closed is True
    
    # Test that the factory can be used again after closing
    async with factory as f:
        provider2 = await f.create_provider("test2", provider_class=MockProvider)
        assert provider2.initialized is True
        assert provider2.closed is False
        
        # Manually add the second provider to the factory's _providers dict
        f._providers["test2"] = provider2
    
    # Verify the second provider was also closed
    assert provider2.closed is True


@pytest.mark.asyncio
async def test_provider_config_merging(provider_factory):
    """Test that provider configs are merged correctly with Pydantic v2."""
    # Test that provider config is merged with defaults
    config = provider_factory.get_provider_config("test")
    assert config["model"] == "test-model"
    assert config["timeout"] == 30.0  # From defaults
    
    # Test that overrides take precedence
    overridden = provider_factory.get_provider_config(
        "test", 
        {"model": "overridden", "temperature": 0.5}
    )
    assert overridden["model"] == "overridden"
    assert overridden["temperature"] == 0.5


@pytest.mark.asyncio
async def test_initialization_error(provider_factory):
    """Test error handling during provider initialization."""
    class FailingProvider(BaseProvider):
        def __init__(self, config):
            super().__init__(config)
            self.initialized = False
            
        async def _initialize_client(self):
            raise RuntimeError("Initialization failed")
            
        async def close(self):
            pass
            
        async def generate(self, prompt: str, model: str, **kwargs):
            return ""
    
    with pytest.raises(RuntimeError, match="Initialization failed"):
        await provider_factory.create_provider(
            "failing",
            provider_class=FailingProvider,
            config_overrides={"test_key": "test_value"}
        )


@pytest.mark.asyncio
@pytest.fixture
def provider_factory():
    """Fixture that provides a configured ProviderFactory for testing."""
    # Create a factory with test configuration
    config = {
        "providers": {
            "test": {
                "provider": "test",
                "model": "test-model",
                "timeout": 30.0,
                "max_retries": 3,
                "temperature": 0.7
            },
            "openai": {
                "provider": "openai",
                "model": "gpt-4",
                "timeout": 30.0,
                "max_retries": 3,
                "temperature": 0.7
            }
        }
    }
    return ProviderFactory(config)


async def test_environment_variables(monkeypatch, provider_factory):
    """Test that environment variables are loaded correctly with Pydantic v2."""
    # Set environment variables
    monkeypatch.setenv("OPENAI_API_KEY", "test-env-key")
    
    # Get config for openai provider
    config = provider_factory.get_provider_config("openai")
    
    # Should load API key from environment
    assert config["api_key"] == "test-env-key"
    
    # Should still use other config from defaults
    assert config["model"] == "gpt-4"  # From OpenAIConfig defaults
    
    # Test environment variable interpolation in config
    monkeypatch.setenv("CUSTOM_MODEL", "gpt-4-turbo")
    # Update the factory config to use the environment variable
    updated_config = provider_factory.config.model_dump()
    updated_config["providers"]["openai"]["model"] = "${CUSTOM_MODEL}"
    provider_factory._config = provider_factory.config.__class__.model_validate(updated_config)
    
    # Get the config again to test interpolation
    config = provider_factory.get_provider_config("openai")
    assert config["model"] == "gpt-4-turbo"  # From environment variable


@pytest.mark.asyncio
async def test_provider_config_with_env_vars(monkeypatch):
    """Test that provider config correctly loads environment variables with Pydantic v2."""
    # Set up environment variables
    monkeypatch.setenv("TEST_API_KEY", "test-api-key")
    
    # Create a test config that uses environment variables
    config = {
        "api_key": "${TEST_API_KEY}",
        "model": "test-model"
    }
    
    # Create a factory with the test config
    factory = ProviderFactory({
        "providers": {
            "test": config
        },
        "environment": "test"
    })
    
    # Get the provider config - should resolve the environment variable
    provider_config = factory.get_provider_config("test")
    assert provider_config["api_key"] == "test-api-key"
    assert provider_config["model"] == "test-model"
    
    # Test that the config is validated by Pydantic
    assert "timeout" in provider_config  # From defaults


@pytest.mark.asyncio
async def test_provider_config_with_precedence():
    """Test that provider configs respect precedence rules with Pydantic v2."""
    # Create a factory with multiple levels of config
    factory = ProviderFactory({
        "defaults": {
            "timeout": 10.0,
            "max_retries": 2,
            "temperature": 0.7
        },
        "providers": {
            "test": {
                "timeout": 20.0,
                "model": "test-model"
            }
        },
        "environment": "test"
    })
    
    # Test that provider config overrides defaults
    config = factory.get_provider_config("test")
    assert config["timeout"] == 20.0  # From provider config
    assert config["max_retries"] == 2  # From defaults
    assert config["temperature"] == 0.7  # From defaults
    
    # Test that overrides take highest precedence
    overrides = {"timeout": 30.0, "temperature": 1.0}
    overridden = factory.get_provider_config("test", overrides)
    assert overridden["timeout"] == 30.0  # From overrides
    assert overridden["temperature"] == 1.0  # From overrides
    
    # Test that the config is properly validated
    with pytest.raises(ValueError):
        # Temperature must be <= 2.0
        factory.get_provider_config("test", {"temperature": 3.0})
