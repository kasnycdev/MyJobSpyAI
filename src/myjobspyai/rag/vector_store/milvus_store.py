"""Milvus vector store implementation for MyJobSpyAI.

This module provides a Milvus-based vector store implementation for the RAG pipeline.
It uses the pymilvus client to interact with a Milvus server for efficient
similarity search and document retrieval.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from pydantic import Field

from myjobspyai.rag.vector_store.base import BaseVectorStore
from myjobspyai.rag.vector_store.config import VectorStoreConfig

logger = logging.getLogger(__name__)

# Try to import Milvus, but make it optional
try:
    from pymilvus import (
        Collection,
        CollectionSchema,
        DataType,
        FieldSchema,
        connections,
        utility,
    )
    from pymilvus.exceptions import MilvusException

    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False


class MilvusVectorStoreConfig(VectorStoreConfig):
    """Configuration for Milvus vector store."""

    type: str = Field(
        default="milvus",
        description="Type of vector store (must be 'milvus')",
    )
    milvus_host: str = Field(
        default="localhost",
        description="Milvus server host",
    )
    milvus_port: int = Field(
        default=19530,
        description="Milvus server port",
    )
    milvus_user: Optional[str] = Field(
        default=None,
        description="Milvus username (if authentication is enabled)",
    )
    milvus_password: Optional[str] = Field(
        default=None,
        description="Milvus password (if authentication is enabled)",
    )
    milvus_secure: bool = Field(
        default=False,
        description="Whether to use secure connection to Milvus",
    )
    milvus_collection_description: str = Field(
        default="MyJobSpyAI RAG Collection",
        description="Description of the Milvus collection",
    )
    milvus_auto_id: bool = Field(
        default=True,
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
        default=False,
        description="Whether to drop existing collection with the same name",
    )


class MilvusVectorStore(BaseVectorStore):
    """Milvus vector store implementation."""

    def __init__(self, config: MilvusVectorStoreConfig, embeddings: Embeddings):
        """Initialize the Milvus vector store.

        Args:
            config: Configuration for the vector store
            embeddings: Embeddings model to use

        Raises:
            ImportError: If required dependencies are not installed
        """
        if not MILVUS_AVAILABLE:
            raise ImportError(
                "pymilvus is not installed. Please install it with: "
                "pip install pymilvus langchain-community"
            )

        super().__init__(config, embeddings)
        self.connection_args = {
            "host": config.milvus_host,
            "port": str(config.milvus_port),
            "secure": config.milvus_secure,
        }

        if config.milvus_user and config.milvus_password:
            self.connection_args.update(
                {
                    "user": config.milvus_user,
                    "password": config.milvus_password,
                }
            )

        self.collection_name = config.collection_name or "myjobspyai_rag"
        self.embedding_dimension = config.embedding_dimension or 768
        self.auto_id = config.milvus_auto_id
        self.collection_description = config.milvus_collection_description
        self.index_params = config.milvus_index_params or {
            "metric_type": config.distance_metric.upper(),
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        self.search_params = config.milvus_search_params or {
            "metric_type": config.distance_metric.upper(),
            "params": {"nprobe": 10},
        }
        self.drop_old = config.milvus_drop_old

        # Initialize Milvus connection and collection
        self._init_connection()
        self.collection = self._get_or_create_collection()

    def _init_connection(self) -> None:
        """Initialize connection to Milvus server."""
        try:
            connections.connect(alias="default", **self.connection_args)
            logger.info(
                "Connected to Milvus server at %s:%s",
                self.connection_args["host"],
                self.connection_args["port"],
            )
        except MilvusException as e:
            logger.error("Failed to connect to Milvus: %s", e)
            raise

    def _get_or_create_collection(self) -> Collection:
        """Get existing collection or create a new one if it doesn't exist.

        Returns:
            A Milvus Collection instance

        Raises:
            RuntimeError: If collection creation fails
        """
        try:
            if utility.has_collection(self.collection_name):
                if self.drop_old:
                    logger.warning(
                        "Dropping existing collection: %s", self.collection_name
                    )
                    utility.drop_collection(self.collection_name)
                else:
                    logger.info("Using existing collection: %s", self.collection_name)
                    return Collection(self.collection_name)

            # Define the schema for the collection
            fields = [
                FieldSchema(
                    name="id",
                    dtype=DataType.INT64 if self.auto_id else DataType.VARCHAR,
                    is_primary=True,
                    auto_id=self.auto_id,
                ),
                FieldSchema(
                    name="text",
                    dtype=DataType.VARCHAR,
                    max_length=65535,  # Maximum length for VARCHAR in Milvus
                ),
                FieldSchema(
                    name="embedding",
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.embedding_dimension,
                ),
                FieldSchema(
                    name="metadata",
                    dtype=DataType.JSON,
                ),
            ]

            schema = CollectionSchema(
                fields=fields,
                description=self.collection_description,
            )

            logger.info("Creating new collection: %s", self.collection_name)
            collection = Collection(
                name=self.collection_name,
                schema=schema,
                using="default",
                shards_num=2,  # Default number of shards
            )

            # Create an index for the vector field
            collection.create_index(
                field_name="embedding",
                index_params=self.index_params,
            )

            return collection

        except MilvusException as e:
            logger.error("Failed to create Milvus collection: %s", e)
            raise RuntimeError(f"Failed to create Milvus collection: {e}")

    def load(self) -> None:
        """Load the Milvus collection.

        Raises:
            RuntimeError: If collection loading fails
        """
        try:
            self.collection = Collection(self.collection_name)
            self.collection.load()
            logger.info("Loaded Milvus collection: %s", self.collection_name)
        except MilvusException as e:
            logger.error("Failed to load Milvus collection: %s", e)
            raise RuntimeError(f"Failed to load Milvus collection: {e}")

    def save(self) -> None:
        """Save the Milvus collection.

        Note: Milvus collections are persisted automatically, so this is a no-op.
        """
        logger.debug("Milvus collection is automatically persisted")

    def add_documents(
        self, documents: List[Dict[str, Any]], **kwargs: Any
    ) -> List[str]:
        """Add documents to the Milvus collection.

        Args:
            documents: List of documents to add
            **kwargs: Additional arguments to pass to Milvus

        Returns:
            List of document IDs

        Raises:
            RuntimeError: If document addition fails
        """
        if not documents:
            return []

        try:
            # Generate embeddings for the documents
            texts = [doc["page_content"] for doc in documents]
            embeddings = self.embeddings.embed_documents(texts)

            # Prepare data for insertion
            data = []
            ids = []

            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                doc_id = doc.get("id", str(i))
                data.append(
                    {
                        "id": None if self.auto_id else doc_id,
                        "text": doc["page_content"],
                        "embedding": embedding,
                        "metadata": doc.get("metadata", {}),
                    }
                )
                ids.append(doc_id)

            # Insert data into Milvus
            insert_result = self.collection.insert(data)

            # If auto_id is enabled, update the IDs with the ones generated by Milvus
            if self.auto_id and insert_result.primary_keys:
                ids = [str(pid) for pid in insert_result.primary_keys]

            # Flush to make sure the data is searchable immediately
            self.collection.flush()

            logger.info("Added %d documents to Milvus collection", len(ids))
            return ids

        except MilvusException as e:
            logger.error("Failed to add documents to Milvus: %s", e)
            raise RuntimeError(f"Failed to add documents to Milvus: {e}")

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

        Raises:
            RuntimeError: If search fails
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embeddings.embed_query(query)

            # Prepare search parameters
            search_params = {
                "data": [query_embedding],
                "anns_field": "embedding",
                "param": self.search_params,
                "limit": k,
                "output_fields": ["text", "metadata"],
            }

            # Execute search
            results = self.collection.search(**search_params)

            # Format results
            documents = []
            for hits in results:
                for hit in hits:
                    doc = {
                        "page_content": hit.entity.get("text", ""),
                        "metadata": hit.entity.get("metadata", {}),
                        "score": hit.distance,
                    }
                    documents.append(doc)

            return documents

        except MilvusException as e:
            logger.error("Failed to perform similarity search in Milvus: %s", e)
            raise RuntimeError(f"Failed to perform similarity search in Milvus: {e}")

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar documents with scores.

        Args:
            query: Query text
            k: Number of results to return
            **kwargs: Additional arguments to pass to the similarity search

        Returns:
            List of (document, score) tuples

        Raises:
            RuntimeError: If search fails
        """
        try:
            # Generate embedding for the query
            query_embedding = self.embeddings.embed_query(query)

            # Prepare search parameters
            search_params = {
                "data": [query_embedding],
                "anns_field": "embedding",
                "param": self.search_params,
                "limit": k,
                "output_fields": ["text", "metadata"],
            }

            # Execute search
            results = self.collection.search(**search_params)

            # Format results
            scored_docs = []
            for hits in results:
                for hit in hits:
                    doc = {
                        "page_content": hit.entity.get("text", ""),
                        "metadata": hit.entity.get("metadata", {}),
                    }
                    scored_docs.append((doc, hit.score))

            return scored_docs

        except MilvusException as e:
            logger.error(
                "Failed to perform similarity search with scores in Milvus: %s", e
            )
            raise RuntimeError(
                f"Failed to perform similarity search with scores in Milvus: {e}"
            )

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

    def __del__(self):
        """Clean up Milvus connection when the object is destroyed."""
        try:
            if hasattr(self, 'collection'):
                self.collection.release()
            connections.disconnect("default")
        except Exception as e:  # pylint: disable=broad-except
            logger.debug("Error cleaning up Milvus connection: %s", e)
