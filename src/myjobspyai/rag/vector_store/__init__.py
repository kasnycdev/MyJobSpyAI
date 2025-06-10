"""Vector store implementations for MyJobSpyAI.

This package provides various vector store implementations for the RAG pipeline,
including support for Milvus, FAISS, and Chroma.

Basic usage:
    ```python
    from myjobspyai.rag.vector_store import get_vector_store
    from langchain.embeddings import OpenAIEmbeddings

    # Create a vector store with default configuration
    config = {
        "type": "milvus",  # or 'faiss', 'chroma', etc.
        "collection_name": "my_collection",
    }
    embeddings = OpenAIEmbeddings()
    vector_store = get_vector_store(config, embeddings)

    # Add documents
    documents = [{"page_content": "Hello, world!", "metadata": {"source": "example"}}]
    vector_store.add_documents(documents)

    # Search for similar documents
    results = vector_store.similarity_search("Hello")
    ```
"""

# Import core components first to avoid circular imports
from myjobspyai.rag.vector_store.base import BaseVectorStore  # noqa: F401
from myjobspyai.rag.vector_store.config import VectorStoreConfig  # noqa: F401
from myjobspyai.rag.vector_store.factory import VectorStoreFactory  # noqa: F401
from myjobspyai.rag.vector_store.utils import (  # noqa: F401
    get_vector_store,
    list_available_vector_stores,
    register_vector_store,
)

# Import vector store implementations
try:
    from myjobspyai.rag.vector_store.milvus_store import (  # noqa: F401
        MilvusVectorStore,
        MilvusVectorStoreConfig,
    )

    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

__all__ = [
    # Base classes
    "BaseVectorStore",
    "VectorStoreConfig",
    # Factory and utils
    "VectorStoreFactory",
    "get_vector_store",
    "list_available_vector_stores",
    "register_vector_store",
    # Availability flags
    "MILVUS_AVAILABLE",
]

# Add Milvus-specific exports if available
if MILVUS_AVAILABLE:
    __all__.extend(["MilvusVectorStore", "MilvusVectorStoreConfig"])

# Initialize the factory with built-in vector stores
# This is done in factory.py to avoid circular imports
