"""Factory for creating vector store instances."""

import logging
from typing import Any, Dict, Optional, Type

from langchain_core.embeddings import Embeddings

from myjobspyai.rag.vector_store.base import BaseVectorStore
from myjobspyai.rag.vector_store.config import VectorStoreConfig

logger = logging.getLogger(__name__)


class VectorStoreFactory:
    """Factory for creating vector store instances.

    This factory provides a unified interface for creating different types of
    vector stores based on configuration.
    """

    # Registry of available vector store implementations
    _registry: Dict[str, Type[BaseVectorStore]] = {}

    @classmethod
    def register(cls, name: str) -> callable:
        """Register a vector store implementation.

        Args:
            name: Name of the vector store type

        Returns:
            Decorator function
        """

        def decorator(vector_store_cls: Type[BaseVectorStore]) -> Type[BaseVectorStore]:
            """Register the vector store class.

            Args:
                vector_store_cls: Vector store class to register

            Returns:
                The registered vector store class
            """
            if name in cls._registry:
                logger.warning(
                    "Vector store type '%s' is already registered. "
                    "Overwriting with %s",
                    name,
                    vector_store_cls.__name__,
                )
            cls._registry[name] = vector_store_cls
            return vector_store_cls

        return decorator

    @classmethod
    def create(
        cls, config: VectorStoreConfig, embeddings: Embeddings, **kwargs: Any
    ) -> BaseVectorStore:
        """Create a vector store instance.

        Args:
            config: Vector store configuration
            embeddings: Embeddings model to use
            **kwargs: Additional arguments to pass to the vector store constructor

        Returns:
            A vector store instance

        Raises:
            ValueError: If the vector store type is not supported
        """
        store_type = config.type.lower()
        if store_type not in cls._registry:
            raise ValueError(
                f"Unsupported vector store type: {store_type}. "
                f"Supported types: {', '.join(cls._registry.keys())}"
            )

        try:
            return cls._registry[store_type](config, embeddings, **kwargs)
        except Exception as e:
            logger.error(
                "Failed to create vector store of type '%s': %s", store_type, e
            )
            raise

    @classmethod
    def get_available_types(cls) -> Dict[str, Type[BaseVectorStore]]:
        """Get all registered vector store types.

        Returns:
            Dictionary mapping vector store type names to their classes
        """
        return dict(cls._registry)


# Register built-in vector store implementations
try:
    from myjobspyai.rag.vector_store.milvus_store import (
        MilvusVectorStore,
        MilvusVectorStoreConfig,
    )

    VectorStoreFactory.register("milvus")(MilvusVectorStore)
except ImportError:
    logger.debug("Milvus vector store not available. Install pymilvus to enable it.")

# Add other vector store implementations here as they are implemented
