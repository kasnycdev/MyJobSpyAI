"""Factory for creating analysis components."""
from typing import Dict, Any, Optional, Union
import yaml
import logging
from pathlib import Path

from myjobspyai.analysis.components.langchain_integration import (
    ResumeAnalyzer,
    CandidateMatcher
)
from myjobspyai.analysis.components.analyzers.resume_analyzer import ResumeAnalyzer as BaseResumeAnalyzer
from myjobspyai.analysis.components.analyzers.langchain_resume_analyzer import LangChainResumeAnalyzer
from myjobspyai.analysis.providers.base import BaseProvider

# Import config only for type checking
from myjobspyai.config import get_config as _get_config

logger = logging.getLogger(__name__)

class ComponentFactory:
    """Factory for creating analysis components."""
    
    _instance = None
    _config = None
    _components = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ComponentFactory, cls).__new__(cls)
            cls._load_config()
        return cls._instance
    
    @classmethod
    def _load_config(cls):
        """Load configuration from file."""
        try:
            config_path = Path(__file__).parent.parent.parent / 'config' / 'langchain_config.yaml'
            with open(config_path, 'r', encoding='utf-8') as f:
                cls._config = yaml.safe_load(f).get('langchain', {})
            logger.info("Loaded LangChain configuration")
        except Exception as e:
            logger.error("Failed to load LangChain config: %s", e)
            cls._config = {}
    
    def get_analyzer(self, provider: Optional[BaseProvider] = None, **kwargs) -> Union[ResumeAnalyzer, BaseResumeAnalyzer]:
        """Get a ResumeAnalyzer instance.
        
        Args:
            provider: Optional provider instance (for backward compatibility)
            **kwargs: Override configuration values
            
        Returns:
            Configured ResumeAnalyzer instance (LangChain or legacy)
        """
        cache_key = 'analyzer'
        use_langchain = kwargs.pop('use_langchain', True)
        
        if cache_key not in self._components:
            model_config = self._config.get('model', {})
            model_config.update(kwargs.get('model', {}))
            
            if use_langchain:
                # Use the new LangChain-based analyzer
                self._components[cache_key] = ResumeAnalyzer(
                    model_name=model_config.get('name', 'llama3'),
                    temperature=model_config.get('temperature', 0.1)
                )
                logger.info("Created new LangChain ResumeAnalyzer instance")
            else:
                # Fall back to legacy analyzer if needed
                if provider is None:
                    raise ValueError("Provider is required for legacy ResumeAnalyzer")
                self._components[cache_key] = LangChainResumeAnalyzer(
                    provider=provider,
                    model=model_config.get('name', 'llama3'),
                    config=model_config
                )
                logger.info("Created new legacy ResumeAnalyzer instance")
            
        return self._components[cache_key]
    
    def get_matcher(self, **kwargs) -> CandidateMatcher:
        """Get a CandidateMatcher instance.
        
        Args:
            **kwargs: Override configuration values
            
        Returns:
            Configured CandidateMatcher instance
        """
        cache_key = 'matcher'
        if cache_key not in self._components:
            model_config = self._config.get('model', {})
            model_config.update(kwargs.get('model', {}))
            
            self._components[cache_key] = CandidateMatcher(
                model_name=model_config.get('name', 'llama3'),
                temperature=model_config.get('temperature', 0.1)
            )
            logger.info("Created new CandidateMatcher instance")
            
        return self._components[cache_key]
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current configuration.
        
        Returns:
            Current configuration dictionary
        """
        return dict(self._config)  # Return a copy to prevent modification
    
    def reset(self):
        """Reset the factory (for testing)."""
        self._components = {}
        self._load_config()

# Global factory instance
factory = ComponentFactory()

def get_analyzer(**kwargs) -> ResumeAnalyzer:
    """Get a ResumeAnalyzer instance."""
    return factory.get_analyzer(**kwargs)

def get_matcher(**kwargs) -> CandidateMatcher:
    """Get a CandidateMatcher instance."""
    return factory.get_matcher(**kwargs)

def get_component_config() -> Dict[str, Any]:
    """Get the current component configuration."""
    return factory.get_config()
