"""Configuration models for LLM providers using Pydantic v2."""
from __future__ import annotations

from pydantic import BaseModel, Field, HttpUrl, field_validator, ConfigDict
from typing import Dict, Optional, Any, Type, TypeVar
import os

T = TypeVar('T', bound='ProviderConfig')

class ProviderConfig(BaseModel):
    """Base configuration for LLM providers."""
    model_config = ConfigDict(
        extra="ignore",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
    )
    
    provider: str = Field(
        default="ollama",
        description="Provider name (e.g., 'openai', 'ollama', 'gemini')"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the provider"
    )
    model: str = Field(
        default="gpt-4",
        description="Model name to use"
    )
    base_url: Optional[HttpUrl] = Field(
        default=None,
        description="Base URL for self-hosted instances"
    )
    timeout: float = Field(
        default=30.0,
        gt=0,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retry attempts"
    )
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    
    @field_validator('api_key', mode='before')
    @classmethod
    def validate_api_key(cls, v: Optional[str], info: Any) -> Optional[str]:
        """Validate API key based on provider requirements."""
        if info.data.get('provider') != 'ollama' and not v:
            raise ValueError(f"API key is required for {info.data.get('provider')} provider")
        return v

    @classmethod
    def create(cls: Type[T], provider: str, **data: Any) -> T:
        """Create a provider config with proper type based on provider name."""
        provider_map: Dict[str, Type[ProviderConfig]] = {
            'openai': OpenAIConfig,
            'ollama': OllamaConfig,
            'gemini': GeminiConfig,
        }
        provider_class = provider_map.get(provider.lower(), ProviderConfig)
        return provider_class(provider=provider, **data)

class OpenAIConfig(ProviderConfig):
    """OpenAI provider configuration."""
    model: str = "gpt-4"
    organization: Optional[str] = None
    project: Optional[str] = None

class OllamaConfig(ProviderConfig):
    """Ollama provider configuration."""
    model: str = Field(
        default=os.getenv('OLLAMA_MODEL', 'llama3'),
        description="Model name to use with Ollama"
    )
    base_url: HttpUrl = Field(
        default=os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434'),  # type: ignore
        description="Base URL for Ollama server"
    )
    timeout: float = Field(
        default=float(os.getenv('OLLAMA_TIMEOUT', '300.0')),
        description="Request timeout in seconds"
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key (not typically needed for local Ollama)"
    )

class GeminiConfig(ProviderConfig):
    """Gemini provider configuration."""
    model: str = "gemini-pro"

class ProviderFactoryConfig(BaseModel):
    """Configuration for the ProviderFactory."""
    model_config = ConfigDict(
        extra="ignore",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
    )
    
    defaults: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default configuration for all providers"
    )
    providers: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Provider-specific configurations"
    )
    environment: str = Field(
        default="production",
        description="Environment name (e.g., development, production)"
    )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProviderFactoryConfig':
        """Create config from dictionary with environment variable resolution."""
        return cls.model_validate(data)
    
    def get_provider_config(
        self, 
        provider_name: str, 
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get configuration for a specific provider with overrides."""
        # Start with defaults
        config = self.defaults.copy()
        
        # Apply provider-specific config if exists
        if provider_name in self.providers:
            config.update(self.providers[provider_name])
            
        # Apply any runtime overrides
        if overrides:
            config.update(overrides)
            
        return config

# Default configuration for the factory
DEFAULT_FACTORY_CONFIG = ProviderFactoryConfig(
    defaults={
        'timeout': 30.0,
        'max_retries': 3,
        'temperature': 0.7,
        'provider': 'ollama',  # Set default provider
    },
    providers={
        'openai': OpenAIConfig().model_dump(exclude_none=True),
        'ollama': OllamaConfig().model_dump(exclude_none=True),
        'gemini': GeminiConfig().model_dump(exclude_none=True),
    }
)
