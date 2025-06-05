"""
Tests for the BaseProvider class.

This module contains tests to verify that the BaseProvider
class works as expected and provides the required interface.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

# Import the modules to test
from myjobspyai.analysis.providers.base import BaseProvider

# Test configuration
TEST_CONFIG = {"param1": "value1", "param2": 42, "nested": {"key": "value"}}


class TestBaseProvider:
    """Test cases for the BaseProvider class."""
    
    class ConcreteProvider(BaseProvider):
        """Concrete implementation of BaseProvider for testing."""

        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            self.custom_init = True

        async def generate(self, prompt, **kwargs):
            """Generate a response to the prompt."""
            return f"Concrete response to: {prompt}"

    def test_initialization(self):
        """Test that the base provider initializes correctly."""
        provider = self.ConcreteProvider(
            config=TEST_CONFIG, name="test_provider", provider_type="test_type"
        )

        assert provider.provider_name == "test_provider"
        assert provider.provider_type == "test_type"
        assert provider.config == TEST_CONFIG
        assert hasattr(provider, 'generate')  # Should have the abstract method implemented

    def test_get_config_value(self):
        """Test retrieving values from the config."""
        provider = self.ConcreteProvider(
            config=TEST_CONFIG, name="test_provider", provider_type="test_type"
        )

        # Test getting a top-level value
        assert provider.get_config_value("param1") == "value1"
        assert provider.get_config_value("param2") == 42

        # Test getting a nested value
        assert provider.get_config_value("nested.key") == "value"

        # Test default value when key doesn't exist
        assert provider.get_config_value("nonexistent", "default") == "default"

        # Test that KeyError is raised when key doesn't exist and no default is provided
        with pytest.raises(KeyError):
            provider.get_config_value("nonexistent")

    @pytest.mark.asyncio
    async def test_generate_not_implemented(self):
        """Test that a provider without generate method raises TypeError."""
        # Create a new class that doesn't implement generate
        class UnimplementedProvider(BaseProvider):
            pass
            
        with pytest.raises(TypeError) as exc_info:
            UnimplementedProvider(config=TEST_CONFIG, name="test_provider")
            
        error_msg = str(exc_info.value)
        expected_msg = "Can't instantiate abstract class UnimplementedProvider with abstract method generate"
        assert error_msg == expected_msg

    def test_str_representation(self):
        """Test the string representation of the provider."""
        provider = self.ConcreteProvider(
            config=TEST_CONFIG, name="test_provider", provider_type="test_type"
        )
        
        # Get the string representation
        str_rep = str(provider)
        
        # Should contain the full class path
        assert "TestBaseProvider.ConcreteProvider" in str_rep
        # Should contain the memory address
        assert hex(id(provider)) in str_rep
        # Should be in the standard Python object representation format
        assert str_rep.startswith("<") and str_rep.endswith(">")

    def test_repr_representation(self):
        """Test the representation of the provider."""
        provider = self.ConcreteProvider(
            config=TEST_CONFIG, name="test_provider", provider_type="test_type"
        )
        
        # Get the repr representation
        repr_rep = repr(provider)
        
        # Should be the same as str() for the default implementation
        assert repr_rep == str(provider)
        # Should contain the full class path
        assert "TestBaseProvider.ConcreteProvider" in repr_rep
        # Should contain the memory address
        assert hex(id(provider)) in repr_rep
        # Should be in the standard Python object representation format
        assert repr_rep.startswith("<") and repr_rep.endswith(">")


class TestConcreteProvider:
    """Test a concrete implementation of BaseProvider."""

    class ConcreteProvider(BaseProvider):
        """Concrete implementation of BaseProvider for testing."""

        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            self.custom_init = True

        async def generate(self, prompt, **kwargs):
            """Generate a response to the prompt."""
            return f"Concrete response to: {prompt}"

    def test_concrete_provider_initialization(self):
        """Test that a concrete provider can be initialized."""
        provider = self.ConcreteProvider(
            config=TEST_CONFIG, name="concrete_provider", provider_type="concrete"
        )

        assert provider.provider_name == "concrete_provider"
        assert provider.provider_type == "concrete"
        assert provider.config == TEST_CONFIG
        assert provider.custom_init is True

    @pytest.mark.asyncio
    async def test_concrete_provider_generate(self):
        """Test that the concrete provider's generate method works."""
        provider = self.ConcreteProvider(
            config=TEST_CONFIG, name="concrete_provider", provider_type="concrete"
        )

        response = await provider.generate("Test prompt")
        assert response == "Concrete response to: Test prompt"


class TestProviderWithHooks:
    """Test a provider with pre/post generation hooks."""

    class HookedProvider(BaseProvider):
        """Provider with pre/post generation hooks for testing."""

        def __init__(self, config, **kwargs):
            super().__init__(config, **kwargs)
            self.pre_called = False
            self.post_called = False

        async def pre_generate(self, prompt, **kwargs):
            """Pre-generation hook."""
            self.pre_called = True
            return prompt.upper(), kwargs

        async def post_generate(self, response, **kwargs):
            """Post-generation hook."""
            self.post_called = True
            return f"PREFIX: {response}"

        async def generate(self, prompt, **kwargs):
            """Generate a response to the prompt."""
            return f"Response to: {prompt}"

    @pytest.mark.asyncio
    async def test_pre_generate_hook(self):
        """Test that the pre_generate hook is called."""
        class PreHookTestProvider(self.HookedProvider):
            async def generate(self, prompt, **kwargs):
                # Call pre_generate manually for testing
                prompt, kwargs = await self.pre_generate(prompt, **kwargs)
                return await super().generate(prompt, **kwargs)
                
        provider = PreHookTestProvider(
            config=TEST_CONFIG, name="hooked_provider", provider_type="hooked"
        )

        # Call the generate method which will call pre_generate
        prompt = "test prompt"
        response = await provider.generate(prompt, extra="value")

        # The pre_generate hook should have been called
        assert provider.pre_called is True
        # The response should reflect the uppercase prompt from pre_generate
        assert response == f"Response to: {prompt.upper()}"

    @pytest.mark.asyncio
    async def test_post_generate_hook(self):
        """Test that the post_generate hook is called."""
        # Create a test class that overrides generate to call post_generate
        class PostHookTestProvider(self.HookedProvider):
            async def generate(self, prompt, **kwargs):
                response = await super().generate(prompt, **kwargs)
                return await self.post_generate(response, **kwargs)
                
        provider = PostHookTestProvider(
            config=TEST_CONFIG, name="hooked_provider", provider_type="hooked"
        )

        # Call the generate method which will call post_generate
        response = await provider.generate("test prompt", extra="value")

        # The post_generate hook should have been called
        assert provider.post_called is True
        # The response should have the PREFIX added by post_generate
        assert response == "PREFIX: Response to: test prompt"


if __name__ == "__main__":
    # Run the tests
    import sys

    sys.exit(pytest.main([__file__] + sys.argv[1:]))
