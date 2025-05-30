"""Additional tests for the provider factory."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from myjobspyai.analysis.factory import (
    ProviderFactory,
    ProviderNotConfigured,
    ProviderNotSupported,
)
from myjobspyai.analysis.providers import BaseProvider
from myjobspyai.exceptions import AuthenticationError


@pytest.mark.asyncio
async def test_create_provider_with_invalid_name():
    """Test creating a provider with an invalid name raises ProviderNotSupported."""
    # Arrange
    config = {
        'providers': {
            'openai': {
                'api_key': 'test-key',
                'model': 'gpt-4',
            }
        }
    }
    
    # Act & Assert
    factory = ProviderFactory(config)
    with pytest.raises(ProviderNotSupported):
        factory.create_provider('invalid-provider')


@pytest.mark.asyncio
async def test_create_provider_with_missing_config():
    """Test creating a provider with missing configuration raises ProviderNotConfigured."""
    # Arrange - config has no providers section
    config = {}
    
    # Act & Assert
    factory = ProviderFactory(config)
    with pytest.raises(ProviderNotConfigured):
        factory.create_provider('openai')
    
    # Test with providers section but missing the requested provider
    config = {
        'providers': {
            'gemini': {
                'api_key': 'test-key',
                'model': 'gemini-pro',
            }
        }
    }
    
    factory = ProviderFactory(config)
    with pytest.raises(ProviderNotConfigured):
        factory.create_provider('openai')


@pytest.mark.asyncio
async def test_provider_error_handling():
    """Test error handling when provider initialization fails."""
    # Arrange
    config = {
        'providers': {
            'openai': {
                'api_key': 'test-key',
                'model': 'gpt-4',
            }
        }
    }
    
    # Mock the OpenAIClient to raise an exception
    with patch('myjobspyai.analysis.providers.OpenAIClient') as mock_client:
        # Set up the mock to raise an exception
        mock_client.side_effect = AuthenticationError("Invalid API key")
        
        factory = ProviderFactory(config)
        
        # Act & Assert
        with pytest.raises(AuthenticationError):
            factory.create_provider('openai')


@pytest.mark.asyncio
async def test_default_provider_handling():
    """Test getting the default provider."""
    # Arrange
    config = {
        'providers': {
            'openai': {
                'api_key': 'test-key',
                'model': 'gpt-4',
            },
            'gemini': {
                'api_key': 'gemini-key',
                'model': 'gemini-pro',
            }
        },
        'default_provider': 'gemini'
    }
    
    # Act
    with patch('myjobspyai.analysis.providers.OpenAIClient') as mock_openai, \
         patch('myjobspyai.analysis.providers.GeminiClient') as mock_gemini:
        
        factory = ProviderFactory(config)
        
        # Get default provider
        provider = factory.get_or_create_provider()
        
        # Assert
        assert provider is not None
        mock_gemini.assert_called_once()  # Should use the default provider (gemini)
        mock_openai.assert_not_called()
        
        # Test with no default set (should use first provider)
        del config['default_provider']
        factory = ProviderFactory(config)
        provider = factory.get_or_create_provider()
        assert provider is not None
        mock_openai.assert_called_once()  # Should use the first provider (openai)



@pytest.mark.asyncio
async def test_provider_caching():
    """Test that providers are cached and reused."""
    # Arrange
    config = {
        'providers': {
            'openai': {
                'api_key': 'test-key',
                'model': 'gpt-4',
            }
        }
    }
    
    # Act
    with patch('myjobspyai.analysis.providers.OpenAIClient') as mock_client:
        factory = ProviderFactory(config)
        
        # First call creates the provider
        provider1 = factory.get_or_create_provider('openai')
        
        # Second call should return the same instance
        provider2 = factory.get_or_create_provider('openai')
        
        # Assert
        assert provider1 is provider2
        mock_client.assert_called_once()  # Should only be created once


@pytest.mark.asyncio
async def test_provider_cleanup_on_error():
    """Test that providers are properly cleaned up if an error occurs during creation."""
    # Arrange
    config = {
        'providers': {
            'openai': {
                'api_key': 'test-key',
                'model': 'gpt-4',
            }
        }
    }
    
    # Create a mock provider that raises an error during initialization
    mock_provider = MagicMock(spec=BaseProvider)
    mock_provider.close = AsyncMock()
    
    # Mock the OpenAIClient to return our mock provider
    with patch('myjobspyai.analysis.providers.OpenAIClient', return_value=mock_provider) as mock_client:
        # Make the generate method raise an error
        mock_provider.generate.side_effect = RuntimeError("Test error")
        
        factory = ProviderFactory(config)
        
        # Try to use the provider - this should fail
        provider = factory.get_or_create_provider('openai')
        with pytest.raises(RuntimeError):
            await provider.generate("Test prompt")
        
        # The provider should still be in the cache
        assert 'openai' in factory._providers
        
        # Clean up the factory - should close the provider
        await factory.close()
        
        # The provider should have been closed
        mock_provider.close.assert_awaited_once()
        
        # The cache should be empty
        assert not factory._providers


@pytest.mark.asyncio
async def test_provider_with_custom_config():
    """Test creating a provider with custom configuration."""
    # Arrange
    config = {
        'providers': {
            'openai': {
                'api_key': 'test-key',
                'model': 'gpt-4',
                'temperature': 0.8,
                'max_tokens': 1000,
                'custom_param': 'custom-value',
            }
        }
    }
    
    # Act
    with patch('myjobspyai.analysis.providers.OpenAIClient') as mock_client:
        factory = ProviderFactory(config)
        provider = factory.create_provider('openai')
        
        # Assert
        assert provider is not None
        mock_client.assert_called_once_with({
            'api_key': 'test-key',
            'model': 'gpt-4',
            'temperature': 0.8,
            'max_tokens': 1000,
            'custom_param': 'custom-value',
        })


@pytest.mark.asyncio
async def test_provider_with_environment_overrides():
    """Test that environment variables override config values."""
    # Arrange
    import os
    os.environ['OPENAI_API_KEY'] = 'env-test-key'
    os.environ['OPENAI_MODEL'] = 'env-gpt-4'
    
    config = {
        'providers': {
            'openai': {
                'api_key': 'config-test-key',
                'model': 'config-gpt-4',
            }
        }
    }
    
    # Act
    with patch('myjobspyai.analysis.providers.OpenAIClient') as mock_client:
        factory = ProviderFactory(config)
        provider = factory.create_provider('openai')
        
        # Assert
        assert provider is not None
        mock_client.assert_called_once()
        
        # Should use environment variables over config values
        assert mock_client.call_args[0][0]['api_key'] == 'env-test-key'
        assert mock_client.call_args[0][0]['model'] == 'env-gpt-4'
    
    # Cleanup
    del os.environ['OPENAI_API_KEY']
    del os.environ['OPENAI_MODEL']
