"""Base vector store implementation for MyJobSpyAI.

This module defines the abstract base class for all vector store implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings

from myjobspyai.rag.vector_store.config import VectorStoreConfig


class BaseVectorStore(ABC):
    """Abstract base class for vector store implementations."""

    def __init__(self, config: VectorStoreConfig, embeddings: Embeddings):
        """Initialize the vector store.

        Args:
            config: Configuration for the vector store
            embeddings: Embeddings model to use
        """
        self.config = config
        self.embeddings = embeddings

    @abstractmethod
    def load(self) -> None:
        """Load the vector store from disk."""
        pass

    @abstractmethod
    def save(self) -> None:
        """Save the vector store to disk."""
        pass

    @abstractmethod
    def add_documents(
        self, documents: List[Dict[str, Any]], **kwargs: Any
    ) -> List[str]:
        """Add documents to the vector store.

        Args:
            documents: List of documents to add
            **kwargs: Additional arguments to pass to the vector store

        Returns:
            List of document IDs
        """
        pass

    @abstractmethod
    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Search for similar documents.

        Args:
            query: Query text
            k: Number of results to return
            **kwargs: Additional arguments to pass to the similarity search

        Returns:
            List of documents with content and metadata
        """
        pass

    @abstractmethod
    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[tuple[Dict[str, Any], float]]:
        """Search for similar documents with scores.

        Args:
            query: Query text
            k: Number of results to return
            **kwargs: Additional arguments to pass to the similarity search

        Returns:
            List of (document, score) tuples
        """
        pass

    @abstractmethod
    def as_retriever(self, **kwargs: Any) -> Any:
        """Return a retriever for the vector store.

        Args:
            **kwargs: Additional arguments to pass to the retriever

        Returns:
            A retriever object
        """
        pass
