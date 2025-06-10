"""Utility functions for working with vector stores."""

from typing import Any, Dict, Type, Union

from langchain_core.embeddings import Embeddings

from myjobspyai.rag.vector_store.base import BaseVectorStore
from myjobspyai.rag.vector_store.config import VectorStoreConfig
from myjobspyai.rag.vector_store.factory import VectorStoreFactory


def get_vector_store(
    config: Union[Dict[str, Any], VectorStoreConfig],
    embeddings: Embeddings,
    **kwargs: Any,
) -> BaseVectorStore:
    """Get a vector store instance based on configuration.

    Args:
        config: Vector store configuration as a dictionary or VectorStoreConfig instance
        embeddings: Embeddings model to use
        **kwargs: Additional arguments to pass to the vector store constructor

    Returns:
        A vector store instance

    Example:
        ```python
        from myjobspyai.rag.vector_store import get_vector_store
        from langchain.embeddings import OpenAIEmbeddings

        config = {
            "type": "milvus",
            "collection_name": "my_collection",
            "milvus_host": "localhost",
            "milvus_port": 19530,
        }
        embeddings = OpenAIEmbeddings()
        vector_store = get_vector_store(config, embeddings)
        ```
    """
    if not isinstance(config, VectorStoreConfig):
        config = VectorStoreConfig(**config)

    return VectorStoreFactory.create(config, embeddings, **kwargs)


def register_vector_store(name: str, vector_store_cls: Type[BaseVectorStore]) -> None:
    """Register a custom vector store implementation.

    Args:
        name: Name of the vector store type
        vector_store_cls: Vector store class to register

    Example:
        ```python
        from myapp.vector_stores import MyCustomVectorStore
        from myapp.vector_store import register_vector_store

        register_vector_store("my_custom_store", MyCustomVectorStore)
        ```
    """
    VectorStoreFactory.register(name)(vector_store_cls)


def list_available_vector_stores() -> Dict[str, Type[BaseVectorStore]]:
    """List all available vector store types.

    Returns:
        Dictionary mapping vector store type names to their classes
    """
    return VectorStoreFactory.get_available_types()
