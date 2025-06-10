"""Vector store functionality for RAG pipeline."""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from pydantic import BaseModel, Field, field_validator

# Try to import Milvus, but make it optional
try:
    from langchain_community.vectorstores import Milvus as LangChainMilvus
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
    )

    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

logger = logging.getLogger(__name__)


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""

    type: str = Field(
        default="faiss",
        description="Type of vector store to use (faiss, chroma, milvus, etc.)",
    )
    persist_directory: Optional[Union[str, Path]] = Field(
        None,
        description="Directory to persist the vector store. If None, the store will be in-memory only.",
    )
    collection_name: Optional[str] = Field(
        None,
        description="Name of the collection (used by some vector stores like Chroma)",
    )
    embedding_model: str = Field(
        "sentence-transformers/all-mpnet-base-v2",
        description="Name or path of the embedding model to use",
    )
    embedding_dimension: Optional[int] = Field(
        None,
        description="Dimension of the embeddings (required for some vector stores)",
    )
    distance_metric: str = Field(
        "cosine",
        description="Distance metric to use for similarity search (cosine, l2, ip, etc.)",
    )

    # Milvus-specific configuration
    milvus_host: str = Field(
        "localhost",
        description="Milvus server host",
    )
    milvus_port: int = Field(
        19530,
        description="Milvus server port",
    )
    milvus_user: Optional[str] = Field(
        None,
        description="Milvus username (if authentication is enabled)",
    )
    milvus_password: Optional[str] = Field(
        None,
        description="Milvus password (if authentication is enabled)",
    )
    milvus_secure: bool = Field(
        False,
        description="Whether to use secure connection to Milvus",
    )
    milvus_collection_description: str = Field(
        "MyJobSpyAI RAG Collection",
        description="Description of the Milvus collection",
    )
    milvus_auto_id: bool = Field(
        True,
        description="Whether to auto-generate document IDs in Milvus",
    )
    milvus_index_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for Milvus index creation",
    )
    milvus_search_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for Milvus search",
    )
    milvus_drop_old: bool = Field(
        False,
        description="Whether to drop existing collection with the same name",
    )
    normalize_embeddings: bool = Field(
        True,
        description="Whether to normalize embeddings before storing them",
    )
    index_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for the vector store index",
    )
    search_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for similarity search",
    )

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        """Validate the vector store type.

        Args:
            v: Vector store type to validate

        Returns:
            Validated vector store type

        Raises:
            ValueError: If the vector store type is not supported
        """
        supported_types = ["faiss", "chroma", "annoy", "hnswlib", "milvus", "qdrant"]
        if v not in supported_types:
            raise ValueError(
                f"Unsupported vector store type: {v}. "
                f"Supported types: {', '.join(supported_types)}"
            )
        return v

    class Config:
        extra = "forbid"
        frozen = True


class BaseVectorStore(ABC):
    """Base class for vector stores."""

    def __init__(self, config: VectorStoreConfig, embeddings: Embeddings):
        """Initialize the vector store.

        Args:
            config: Configuration for the vector store
            embeddings: Embeddings model to use
        """
        self.config = config
        self.embeddings = embeddings
        self.vector_store: Optional[VectorStore] = None

    @abstractmethod
    def load(self) -> None:
        """Load the vector store from disk if it exists, otherwise initialize a new one."""
        pass

    @abstractmethod
    def save(self) -> None:
        """Save the vector store to disk if a persist directory is configured."""
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
            List of documents with similarity scores
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

    def as_retriever(self, **kwargs: Any) -> Any:
        """Return a retriever for the vector store.

        Args:
            **kwargs: Additional arguments to pass to the retriever

        Returns:
            A retriever object
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call load() first.")
        return self.vector_store.as_retriever(**kwargs)


class FAISSVectorStore(BaseVectorStore):
    """FAISS vector store implementation."""

    def __init__(self, config: VectorStoreConfig, embeddings: Embeddings):
        """Initialize the FAISS vector store.

        Args:
            config: Configuration for the vector store
            embeddings: Embeddings model to use
        """
        super().__init__(config, embeddings)
        self.index_path = (
            Path(config.persist_directory) / "faiss_index"
            if config.persist_directory
            else None
        )

    def load(self) -> None:
        """Load the FAISS index from disk if it exists, otherwise initialize a new one."""
        try:
            from langchain_community.vectorstores import FAISS

            if self.index_path and self.index_path.exists():
                logger.info(f"Loading FAISS index from {self.index_path}")
                self.vector_store = FAISS.load_local(
                    str(self.index_path.parent),
                    self.embeddings,
                    index_name=self.index_path.name,
                    allow_dangerous_deserialization=True,  # Required for FAISS
                )
            else:
                logger.info("Initializing new FAISS index")
                self.vector_store = FAISS.from_texts(
                    [""], self.embeddings
                )  # Initialize with empty document
        except ImportError as e:
            raise ImportError(
                "FAISS not installed. Please install with: pip install faiss-cpu"
            ) from e
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            raise

    def save(self) -> None:
        """Save the FAISS index to disk if a persist directory is configured."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call load() first.")

        if self.index_path:
            os.makedirs(self.index_path.parent, exist_ok=True)
            self.vector_store.save_local(
                str(self.index_path.parent),
                index_name=self.index_path.name,
            )
            logger.info(f"Saved FAISS index to {self.index_path}")

    def add_documents(
        self, documents: List[Dict[str, Any]], **kwargs: Any
    ) -> List[str]:
        """Add documents to the FAISS index.

        Args:
            documents: List of documents to add
            **kwargs: Additional arguments to pass to FAISS

        Returns:
            List of document IDs
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call load() first.")

        try:
            from langchain_core.documents import Document as LCDocument

            # Convert documents to LangChain format
            lc_docs = [
                LCDocument(
                    page_content=doc["content"],
                    metadata=doc.get("metadata", {}),
                )
                for doc in documents
            ]

            # Add documents to the index
            doc_ids = self.vector_store.add_documents(lc_docs, **kwargs)
            logger.info(f"Added {len(doc_ids)} documents to FAISS index")

            # Save the updated index
            self.save()

            return doc_ids
        except Exception as e:
            logger.error(f"Error adding documents to FAISS: {e}")
            raise

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
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call load() first.")

        try:
            results = self.vector_store.similarity_search(query, k=k, **kwargs)
            return [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in results
            ]
        except Exception as e:
            logger.error(f"Error in FAISS similarity search: {e}")
            raise

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
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call load() first.")

        try:
            results_with_scores = self.vector_store.similarity_search_with_score(
                query, k=k, **kwargs
            )
            return [
                ({"content": doc.page_content, "metadata": doc.metadata}, score)
                for doc, score in results_with_scores
            ]
        except Exception as e:
            logger.error(f"Error in FAISS similarity search with score: {e}")
            raise


class ChromaVectorStore(BaseVectorStore):
    """Chroma vector store implementation."""

    def __init__(self, config: VectorStoreConfig, embeddings: Embeddings):
        """Initialize the Chroma vector store.

        Args:
            config: Configuration for the vector store
            embeddings: Embeddings model to use
        """
        super().__init__(config, embeddings)
        self.collection_name = config.collection_name or "langchain"
        self.persist_directory = (
            str(config.persist_directory) if config.persist_directory else None
        )

    def load(self) -> None:
        """Load the Chroma collection."""
        try:
            import chromadb
            from chromadb.config import Settings
            from langchain_community.vectorstores import Chroma

            client_settings = Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=self.persist_directory,
                anonymized_telemetry=False,
            )

            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                client_settings=client_settings,
                persist_directory=self.persist_directory,
            )

            logger.info(
                f"Initialized Chroma vector store with collection: {self.collection_name}"
            )
        except ImportError as e:
            raise ImportError(
                "Chroma not installed. Please install with: pip install chromadb"
            ) from e
        except Exception as e:
            logger.error(f"Error initializing Chroma vector store: {e}")
            raise

    def save(self) -> None:
        """Persist the Chroma collection to disk if a persist directory is configured."""
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call load() first.")

        if self.persist_directory:
            self.vector_store.persist()
            logger.info(f"Persisted Chroma collection to {self.persist_directory}")

    def add_documents(
        self, documents: List[Dict[str, Any]], **kwargs: Any
    ) -> List[str]:
        """Add documents to the Chroma collection.

        Args:
            documents: List of documents to add
            **kwargs: Additional arguments to pass to Chroma

        Returns:
            List of document IDs
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call load() first.")

        try:
            from langchain_core.documents import Document as LCDocument

            # Convert documents to LangChain format
            lc_docs = [
                LCDocument(
                    page_content=doc["content"],
                    metadata=doc.get("metadata", {}),
                )
                for doc in documents
            ]

            # Add documents to the collection
            doc_ids = self.vector_store.add_documents(lc_docs, **kwargs)
            logger.info(f"Added {len(doc_ids)} documents to Chroma collection")

            # Persist the updated collection
            self.save()

            return doc_ids
        except Exception as e:
            logger.error(f"Error adding documents to Chroma: {e}")
            raise

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
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call load() first.")

        try:
            results = self.vector_store.similarity_search(query, k=k, **kwargs)
            return [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in results
            ]
        except Exception as e:
            logger.error(f"Error in Chroma similarity search: {e}")
            raise

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
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call load() first.")

        try:
            results_with_scores = self.vector_store.similarity_search_with_score(
                query, k=k, **kwargs
            )
            return [
                ({"content": doc.page_content, "metadata": doc.metadata}, score)
                for doc, score in results_with_scores
            ]
        except Exception as e:
            logger.error(f"Error in Chroma similarity search with score: {e}")
            raise

    def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
        """Return a retriever for the vector store.

        Args:
            **kwargs: Additional arguments to pass to the retriever

        Returns:
            A retriever object
        """
        from langchain_core.vectorstores import VectorStoreRetriever

        search_kwargs = {
            "k": kwargs.pop("k", 4),
            **self.search_params,
            **kwargs.pop("search_kwargs", {}),
        }

        return VectorStoreRetriever(
            vectorstore=self,
            search_type="similarity",
            search_kwargs=search_kwargs,
            **kwargs,
        )


class VectorStoreFactory:
    """Factory for creating vector stores."""

    VECTOR_STORE_CLASSES = {
        "faiss": FAISSVectorStore,
        "chroma": ChromaVectorStore,
        "milvus": MilvusVectorStore if MILVUS_AVAILABLE else None,
        # Add other vector store implementations here
    }

    @classmethod
    def get_vector_store(
        cls, config: VectorStoreConfig, embeddings: Embeddings
    ) -> "BaseVectorStore":
        """Get a vector store instance based on the configuration.

        Args:
            config: Configuration for the vector store
            embeddings: Embeddings model to use

        Returns:
            A vector store instance

        Raises:
            ValueError: If the vector store type is not supported or dependencies are missing
        """
        store_type = config.type.lower()
        store_class = cls.VECTOR_STORE_CLASSES.get(store_type)

        if store_class is None:
            raise ValueError(
                f"Unsupported vector store type: {store_type}. "
                f"Supported types: {', '.join(cls.VECTOR_STORE_CLASSES.keys())}"
            )

        if store_class is None and store_type == 'milvus':
            raise ImportError(
                "Milvus dependencies not found. Please install them with: "
                "pip install pymilvus langchain-community"
            )

        return store_class(config, embeddings)


# For backward compatibility
get_vector_store = VectorStoreFactory.get_vector_store
