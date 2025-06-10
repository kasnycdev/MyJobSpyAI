"""RAG (Retrieval-Augmented Generation) pipeline implementation."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document as LCDocument
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field, field_validator

from .embeddings import EmbeddingConfig, get_embedding_model
from .loader import DocumentLoaderConfig, load_documents
from .splitter import TextSplitterConfig, split_documents
from .vector_store import VectorStoreConfig, get_vector_store

logger = logging.getLogger(__name__)


class RAGConfig(BaseModel):
    """Configuration for the RAG pipeline."""

    # Document loading configuration
    document_loader: DocumentLoaderConfig = Field(
        default_factory=DocumentLoaderConfig,
        description="Configuration for document loading",
    )

    # Text splitting configuration
    text_splitter: TextSplitterConfig = Field(
        default_factory=TextSplitterConfig,
        description="Configuration for text splitting",
    )

    # Embedding configuration
    embedding: EmbeddingConfig = Field(
        default_factory=EmbeddingConfig,
        description="Configuration for embedding model",
    )

    # Vector store configuration
    vector_store: VectorStoreConfig = Field(
        default_factory=VectorStoreConfig,
        description="Configuration for vector store",
    )

    # Retrieval configuration
    retrieval: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional retrieval parameters",
    )

    # Generation configuration
    generation: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional generation parameters",
    )

    class Config:
        extra = "forbid"
        frozen = True


class RAGPipeline:
    """RAG (Retrieval-Augmented Generation) pipeline."""

    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize the RAG pipeline.

        Args:
            config: Configuration for the RAG pipeline. If None, default values will be used.
        """
        self.config = config or RAGConfig()
        self.embedding_model = None
        self.vector_store = None
        self.retriever = None

        # Initialize components
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize the RAG pipeline components."""
        # Initialize embedding model
        self.embedding_model = get_embedding_model(self.config.embedding)

        # Initialize vector store
        self.vector_store = get_vector_store(
            self.config.vector_store, self.embedding_model
        )
        self.vector_store.load()

        # Initialize retriever
        self.retriever = self.vector_store.as_retriever(
            search_type=self.config.vector_store.distance_metric,
            search_kwargs={
                "k": self.config.retrieval.get("top_k", 4),
                **self.config.retrieval.get("search_kwargs", {}),
            },
        )

    def add_documents(
        self,
        file_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """Add documents to the RAG pipeline.

        Args:
            file_path: Path to the document file
            metadata: Additional metadata to include with the document
            **kwargs: Additional arguments to pass to the document loader

        Returns:
            List of document IDs
        """
        # Create document loader config
        loader_config = self.config.document_loader.model_copy()
        loader_config.file_path = str(file_path)
        if metadata:
            loader_config.metadata = {
                **(loader_config.metadata or {}),
                **metadata,
            }

        # Load documents
        documents = load_documents(loader_config)

        # Convert to list of dicts for splitting
        doc_dicts = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in documents
        ]

        # Split documents
        split_docs = split_documents(doc_dicts, self.config.text_splitter)

        # Add to vector store
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")

        return self.vector_store.add_documents(split_docs, **kwargs)

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query.

        Args:
            query: Query text
            top_k: Number of results to return. If None, use the value from config.
            **kwargs: Additional arguments to pass to the retriever

        Returns:
            List of relevant documents with metadata and scores
        """
        if self.retriever is None:
            raise ValueError("Retriever not initialized")

        # Update top_k if provided
        search_kwargs = self.retriever.search_kwargs.copy()
        if top_k is not None:
            search_kwargs["k"] = top_k
        search_kwargs.update(kwargs.get("search_kwargs", {}))

        # Perform retrieval
        results = self.retriever.get_relevant_documents(
            query,
            **{"search_kwargs": search_kwargs, **kwargs},
        )

        # Convert to list of dicts
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": doc.metadata.get("score", 0.0),
            }
            for doc in results
        ]

    def generate(
        self,
        query: str,
        context: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Generate a response using the RAG pipeline.

        Args:
            query: Query text
            context: Optional list of context documents. If None, retrieval will be performed.
            **kwargs: Additional arguments to pass to the generator

        Returns:
            Dictionary containing the generated response and context
        """
        # Retrieve context if not provided
        if context is None:
            context = self.retrieve(query, **kwargs)

        # Prepare the prompt with context
        context_text = "\n\n".join([doc["content"] for doc in context])

        # TODO: Implement actual generation with an LLM
        # For now, just return the context
        return {
            "query": query,
            "context": context,
            "answer": "[Generated response would appear here]",
        }

    def save(self) -> None:
        """Save the vector store to disk."""
        if self.vector_store is not None:
            self.vector_store.save()

    def close(self) -> None:
        """Clean up resources."""
        self.save()

    def __enter__(self):
        """Support for context manager protocol."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context manager."""
        self.close()
