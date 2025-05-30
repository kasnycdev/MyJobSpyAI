"""
Configuration validation and loading utilities.
"""
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from pydantic import BaseModel, validator, Field

logger = logging.getLogger(__name__)

class TextProcessingConfig(BaseModel):
    """Configuration for text processing."""
    chunk_size: int = Field(1500, ge=100, le=10000, description="Size of text chunks in characters")
    chunk_overlap: int = Field(300, ge=0, le=2000, description="Overlap between chunks in characters")
    separators: List[str] = Field(
        default_factory=lambda: ["\n\n", "\n", ". ", " "],
        description="List of separators used for splitting text"
    )
    max_retries: int = Field(3, ge=0, description="Maximum number of retries for operations")
    retry_delay: float = Field(1.0, ge=0.1, le=10.0, description="Initial delay between retries in seconds")

class LLMConfig(BaseModel):
    """Configuration for LLM providers."""
    provider: str = Field(..., description="LLM provider (openai, ollama, gemini)")
    model: str = Field(..., description="Model name or ID")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(2000, ge=1, le=8000, description="Maximum number of tokens to generate")
    timeout: int = Field(120, ge=10, description="Request timeout in seconds")

class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = Field("INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    file: Optional[str] = Field("logs/app.log", description="Path to log file")
    max_size: int = Field(10, ge=1, description="Maximum log file size in MB")
    backup_count: int = Field(5, ge=0, description="Number of backup logs to keep")

class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""
    type: str = Field("milvus", description="Vector store type (milvus, faiss, chroma)")
    collection: str = Field(..., description="Collection/Index name")
    embedding_dim: int = Field(384, description="Dimensionality of embeddings")

class AppConfig(BaseModel):
    """Main application configuration."""
    text_processing: TextProcessingConfig = Field(
        default_factory=TextProcessingConfig,
        description="Text processing configuration"
    )
    llm: LLMConfig = Field(..., description="LLM configuration")
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig,
        description="Logging configuration"
    )
    vectorstore: VectorStoreConfig = Field(
        default_factory=lambda: VectorStoreConfig(collection="job_descriptions"),
        description="Vector store configuration"
    )
    
    class Config:
        extra = "ignore"  # Ignore extra fields in config
    
    @classmethod
    def from_file(cls, config_path: Optional[Union[str, Path]] = None) -> 'AppConfig':
        """
        Load configuration from file with environment variable overrides.
        
        Args:
            config_path: Path to config file
            
        Returns:
            AppConfig: Loaded configuration
        """
        config_dict = {}
        
        # Default config path
        if config_path is None:
            config_path = os.getenv('MYJOBS_CONFIG')
            if config_path is None:
                config_path = Path('config.yaml')
                if not config_path.exists():
                    logger.warning(f"Config file not found at {config_path}, using defaults")
                    return cls()
        
        # Load from file
        config_path = Path(config_path)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f) or {}
                    config_dict.update(file_config)
            except Exception as e:
                logger.error(f"Failed to load config from {config_path}: {e}")
                raise
        
        # Apply environment variable overrides
        env_config = {
            'text_processing': {
                'chunk_size': os.getenv('TEXT_CHUNK_SIZE'),
                'chunk_overlap': os.getenv('TEXT_CHUNK_OVERLAP'),
                'max_retries': os.getenv('MAX_RETRIES'),
                'retry_delay': os.getenv('RETRY_DELAY')
            },
            'llm': {
                'provider': os.getenv('LLM_PROVIDER'),
                'model': os.getenv('LLM_MODEL'),
                'temperature': os.getenv('LLM_TEMPERATURE'),
                'max_tokens': os.getenv('LLM_MAX_TOKENS'),
                'timeout': os.getenv('LLM_TIMEOUT')
            },
            'logging': {
                'level': os.getenv('LOG_LEVEL'),
                'file': os.getenv('LOG_FILE'),
                'max_size': os.getenv('LOG_MAX_SIZE'),
                'backup_count': os.getenv('LOG_BACKUP_COUNT')
            },
            'vectorstore': {
                'type': os.getenv('VECTORSTORE_TYPE'),
                'collection': os.getenv('VECTORSTORE_COLLECTION'),
                'embedding_dim': os.getenv('VECTORSTORE_EMBEDDING_DIM')
            }
        }
        
        # Remove None values and empty dicts
        env_config = {
            k1: {
                k2: v2 for k2, v2 in v1.items() 
                if v2 is not None and not (isinstance(v2, dict) and not v2)
            } 
            for k1, v1 in env_config.items() 
            if v1 and any(v is not None for v in v1.values())
        }
        
        # Deep merge configs
        def deep_merge(base: dict, update: dict) -> dict:
            for k, v in update.items():
                if isinstance(v, dict) and k in base and isinstance(base[k], dict):
                    base[k] = deep_merge(base[k], v)
                else:
                    base[k] = v
            return base
        
        config_dict = deep_merge(config_dict, env_config)
        
        return cls(**config_dict)

def get_config(config_path: Optional[Union[str, Path]] = None) -> AppConfig:
    """
    Get the application configuration.
    
    Args:
        config_path: Optional path to config file
        
    Returns:
        AppConfig: The application configuration
    """
    return AppConfig.from_file(config_path)
