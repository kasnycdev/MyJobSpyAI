"""Tests for the provider factory."""
import asyncio
import pytest
from typing import Dict, Any, TypeVar
from unittest.mock import MagicMock, patch, AsyncMock

from myjobspyai.analysis.factory import ProviderFactory, ProviderNotSupported
from myjobspyai.analysis.providers.base import BaseProvider

# Type variable for provider classes
T = TypeVar('T', bound=BaseProvider)

class TestProvider(BaseProvider):
    """Base class for all provider implementations."""
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize with provider configuration."""
        super().__init__(config)
        self._initialize_client = AsyncMock()

# Mock provider classes for testing
class MockOpenaiClient(BaseProvider):
    """Mock OpenAI client for testing."""
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._initialize_client = AsyncMock()
        self._initialize_client.return_value = None
        self.generate = AsyncMock(return_value="Mocked response")
        
    async def _initialize_client(self):
        """Mock initialize client method."""
        return None
        
    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Mock generate method."""
        return "Mocked response"
        
    async def close(self) -> None:
        """Mock close method."""
        pass

class MockOllamaClient(BaseProvider):
    """Mock Ollama client for testing."""
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._initialize_client = AsyncMock()
        self._initialize_client.return_value = None
        self.generate = AsyncMock(return_value="Mocked response")
        
    async def _initialize_client(self):
        """Mock initialize client method."""
        return None
        
    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Mock generate method."""
        return "Mocked response"
        
    async def close(self) -> None:
        """Mock close method."""
        pass

class MockGeminiClient(BaseProvider):
    """Mock Gemini client for testing."""
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self._initialize_client = AsyncMock()
        self._initialize_client.return_value = None
        self.generate = AsyncMock(return_value="Mocked response")
        
    async def _initialize_client(self):
        """Mock initialize client method."""
        return None
        
    async def generate(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """Mock generate method."""
        return "Mocked response"
        
    async def close(self) -> None:
        """Mock close method."""
        pass

# Test fixtures
@pytest.fixture
def provider_config() -> Dict[str, Any]:
    """Return a basic provider configuration."""
    return {
        "providers": {
            "openai": {
                "class": "myjobspyai.analysis.providers.openai.OpenaiProvider",
                "api_key": "test-api-key",
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1000,
                "timeout": 30.0
            },
            "ollama": {
                "class": "myjobspyai.analysis.providers.ollama.OllamaProvider",
                "model": "llama2",
                "temperature": 0.7,
                "max_tokens": 1000,
                "timeout": 30.0
            },
            "gemini": {
                "class": "myjobspyai.analysis.providers.gemini.GeminiProvider",
                "api_key": "test-gemini-key",
                "model": "gemini-pro",
                "temperature": 0.7,
                "max_tokens": 1000,
                "timeout": 30.0
            }
        }
    }

@pytest.fixture
def mock_imports(monkeypatch):
    """Fixture to mock imports for testing."""
    # Store the original PROVIDER_CLASSES
    from myjobspyai.analysis import factory
    original_providers = factory.PROVIDER_CLASSES.copy()
    
    # Update with our mock providers
    factory.PROVIDER_CLASSES.update({
        'openai': MockOpenaiClient,
        'ollama': MockOllamaClient,
        'gemini': MockGeminiClient,
        'test': MockOpenaiClient  # For testing unknown providers
    })
    
    # Mock import_module to return our mock module
    def mock_import(module_name, package=None):
        mock_module = MagicMock()
        if module_name == 'myjobspyai.analysis.providers.openai':
            mock_module.OpenAIClient = MockOpenaiClient
        elif module_name == 'myjobspyai.analysis.providers.ollama':
            mock_module.OllamaClient = MockOllamaClient
        elif module_name == 'myjobspyai.analysis.providers.gemini':
            mock_module.GeminiClient = MockGeminiClient
        elif module_name.startswith('.'):
            # Handle relative imports
            full_name = f"{package}{module_name}" if package else module_name[1:]
            if 'openai' in full_name:
                mock_module.OpenAIClient = MockOpenaiClient
            elif 'ollama' in full_name:
                mock_module.OllamaClient = MockOllamaClient
            elif 'gemini' in full_name:
                mock_module.GeminiClient = MockGeminiClient
        return mock_module
    
    with patch('importlib.import_module', side_effect=mock_import):
        yield
    
    # Restore original providers
    factory.PROVIDER_CLASSES = original_providers

@pytest.fixture
def mock_factory(provider_config: Dict[str, Any], mock_imports) -> ProviderFactory:
    """Return a mock factory instance."""
    # Update provider config with test values
    provider_config['providers']['openai']['api_key'] = 'test-key'
    provider_config['providers']['ollama']['api_key'] = 'test-ollama-key'
    provider_config['providers']['gemini']['api_key'] = 'test-gemini-key'
    
    return ProviderFactory(provider_config)

# Tests
@pytest.mark.asyncio
class TestProviderFactory:
    """Test cases for ProviderFactory."""

    async def test_create_provider(self, provider_config: Dict[str, Any]):
        """Test creating a provider instance."""
        factory = ProviderFactory(provider_config)
        provider = await factory.get_or_create_provider('openai')
        assert provider is not None

    @patch('myjobspyai.analysis.factory.PROVIDER_CLASSES', new_callable=dict)
    async def test_create_unknown_provider(self, mock_provider_classes, provider_config: Dict[str, Any]):
        """Test creating an unknown provider raises an error."""
        # Configure the mock to not have our test provider
        mock_provider_classes.clear()
        mock_provider_classes.update({
            'openai': 'myjobspyai.analysis.providers.openai.OpenAIClient',
            'ollama': 'myjobspyai.analysis.providers.ollama.OllamaClient',
            'gemini': 'myjobspyai.analysis.providers.gemini.GeminiClient'
        })
        
        # Create factory and test
        factory = ProviderFactory(provider_config)
        with pytest.raises(ProviderNotSupported, match="Provider 'nonexistent' is not supported"):
            await factory.get_or_create_provider('nonexistent')

    async def test_config_overrides(self, provider_config: Dict[str, Any]):
        """Test that config overrides are applied correctly."""
        factory = ProviderFactory(provider_config)
        overrides = {"temperature": 0.9}
        provider = await factory.get_or_create_provider('openai', overrides)
        assert provider.config["temperature"] == 0.9

    async def test_concurrent_initialization(self, provider_config: Dict[str, Any]):
        """Test that concurrent initialization is handled correctly."""
        factory = ProviderFactory(provider_config)
        
        async def get_provider():
            return await factory.get_or_create_provider('openai')
        
        providers = await asyncio.gather(
            get_provider(),
            get_provider(),
            get_provider()
        )
        assert len(providers) == 3
        assert all(p is providers[0] for p in providers)

    @pytest.mark.asyncio
    async def test_multiple_providers(self, provider_config: Dict[str, Any]):
        """Test creating multiple different provider types."""
        factory = ProviderFactory(provider_config)
        
        openai = await factory.get_or_create_provider('openai')
        ollama = await factory.get_or_create_provider('ollama')
        gemini = await factory.get_or_create_provider('gemini')
        
        assert openai is not None
        assert ollama is not None
        assert gemini is not None

    
@pytest.mark.asyncio
async def test_get_or_create_provider(mock_factory: ProviderFactory):
    """Test that get_or_create_provider returns the same instance for the same provider."""
    provider1 = await mock_factory.get_or_create_provider('openai')
    provider2 = await mock_factory.get_or_create_provider('openai')
    assert provider1 is provider2

@pytest.mark.asyncio
async def test_create_provider_with_overrides(
    provider_config: Dict[str, Any]
):
    """Test creating a provider with config overrides."""
    factory = ProviderFactory(provider_config)
    overrides = {"temperature": 0.9}
    provider = await factory.get_or_create_provider('openai', overrides)
    assert provider.config["temperature"] == 0.9

@pytest.mark.asyncio
async def test_create_openai_provider(
    provider_config: Dict[str, Any]
):
    """Test creating an OpenAI provider."""
    factory = ProviderFactory(provider_config)
    provider = await factory.get_or_create_provider('openai')
    assert provider is not None
    assert isinstance(provider, BaseProvider)

@pytest.mark.asyncio
async def test_concurrent_access(
    provider_config: Dict[str, Any]
):
    """Test concurrent access to the factory."""
    factory = ProviderFactory(provider_config)
    
    async def get_provider():
        return await factory.get_or_create_provider('openai')
    
    providers = await asyncio.gather(
        get_provider(),
        get_provider(),
        get_provider()
    )
    assert len(providers) == 3
    assert all(p is providers[0] for p in providers)

@pytest.mark.asyncio
async def test_environment_variable_loading():
    """Test that environment variables are loaded correctly."""
    # Set up test environment
    import os
    os.environ['OPENAI_API_KEY'] = 'test-key'
    
    # Create a minimal config that will use environment variables
    config = {
        'providers': {
            'openai': {
                'model': 'gpt-4',
                # api_key should be loaded from environment
            }
        }
    }
    
    factory = ProviderFactory(config)
    
    # Get the provider config to check if environment variable was loaded
    provider_config = factory.get_provider_config('openai')
    assert provider_config.get("api_key") == 'test-key'
    assert provider_config.get("timeout") == 30.0
