"""
Milvus vector store implementation for document storage and retrieval.

This module provides integration with Milvus vector database for efficient
similarity search and document retrieval.
"""
from contextlib import asynccontextmanager
import datetime
import uuid
from typing import Any, Dict, List, Optional, AsyncIterator

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
    MilvusException
)

from myjobspyai.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Default collection schema for document storage
DEFAULT_COLLECTION_NAME = "rag_documents"
DEFAULT_PRIMARY_FIELD = "id"
DEFAULT_VECTOR_FIELD = "embedding"
DEFAULT_TEXT_FIELD = "text"
DEFAULT_METADATA_FIELD = "metadata"
DEFAULT_INDEX_PARAMS = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 16384}
}

class MilvusVectorStore(VectorStore):
    """Milvus vector store implementation for RAG pipeline."""

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding: Optional[Embeddings] = None,
        connection_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize with Milvus connection."""
        self.collection_name = collection_name
        self.embedding = embedding
        self.connection_args = connection_args or {}
        self.index_params = kwargs.get("index_params", DEFAULT_INDEX_PARAMS)
        self.consistency_level = kwargs.get("consistency_level", "Bounded")
        self.timeout = kwargs.get("timeout", 30)
        self._collection = None
        self._ensure_connection()

    def _ensure_connection(self) -> None:
        """Ensure connection to Milvus server."""
        try:
            connections.connect("default", **self.connection_args)
            self._initialize_collection()
        except MilvusException as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise

    def _initialize_collection(self) -> None:
        """Initialize the Milvus collection with proper schema."""
        if utility.has_collection(self.collection_name):
            self._collection = Collection(self.collection_name)
            return

        # Define collection schema
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),  # Default dimension
            FieldSchema(name="metadata", dtype=DataType.JSON),
            FieldSchema(name="created_at", dtype=DataType.INT64),
            FieldSchema(name="updated_at", dtype=DataType.INT64),
            FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=64),
        ]

        schema = CollectionSchema(fields, description="RAG Document Store")
        self._collection = Collection(self.collection_name, schema, consistency_level=self.consistency_level)
        
        # Create index on vector field
        self._collection.create_index(
            field_name="embedding",
            index_params=self.index_params
        )
        logger.info(f"Created new collection {self.collection_name} with vector index")

    async def add_documents(
        self,
        documents: List[Document],
        **kwargs
    ) -> List[str]:
        """Add documents to the vector store."""
        if not documents:
            return []

        # Generate embeddings if not provided
        texts = [doc.page_content for doc in documents]
        embeddings = await self.embedding.aembed_documents(texts)
        
        # Prepare data for insertion
        now = int(datetime.now().timestamp())
        entities = []
        doc_ids = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = str(uuid.uuid4())
            doc_ids.append(doc_id)
            
            entities.append({
                "id": doc_id,
                "text": doc.page_content,
                "embedding": embedding,
                "metadata": doc.metadata,
                "created_at": now,
                "updated_at": now,
                "document_id": doc.metadata.get("document_id", ""),
            })
        
        # Insert into Milvus
        try:
            self._collection.insert(entities)
            self._collection.flush()
            logger.info(f"Inserted {len(doc_ids)} documents into {self.collection_name}")
            return doc_ids
        except MilvusException as e:
            logger.error(f"Failed to insert documents: {e}")
            raise

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs
    ) -> List[Document]:
        """Return documents most similar to query."""
        # Generate query embedding
        query_embedding = await self.embedding.aembed_query(query)
        
        # Prepare search parameters
        search_params = {
            "metric_type": self.index_params["metric_type"],
            "params": {"nprobe": 16}  # Adjust based on your data
        }
        
        # Execute search
        results = self._collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["text", "metadata"]
        )
        
        # Convert results to Document objects
        docs = []
        for hits in results:
            for hit in hits:
                doc = Document(
                    page_content=hit.entity.get("text", ""),
                    metadata=hit.entity.get("metadata", {})
                )
                docs.append(doc)
        
        return docs

    async def adelete(
        self,
        ids: Optional[List[str]] = None,
        **kwargs
    ) -> bool:
        """Delete documents by IDs."""
        if not ids:
            return False
            
        try:
            expr = f"id in {ids}"
            self._collection.delete(expr)
            self._collection.flush()
            logger.info(f"Deleted {len(ids)} documents")
            return True
        except MilvusException as e:
            logger.error(f"Failed to delete documents: {e}")
            return False

    async def aupdate_document(
        self,
        document_id: str,
        document: Document,
        **kwargs
    ) -> bool:
        """Update a document in the vector store."""
        # Generate new embedding if content changed
        new_embedding = await self.embedding.aembed_documents([document.page_content])
        
        # Prepare update data
        now = int(datetime.now().timestamp())
        
        try:
            self._collection.upsert([{
                "id": document_id,
                "text": document.page_content,
                "embedding": new_embedding[0],
                "metadata": document.metadata,
                "updated_at": now,
            }])
            self._collection.flush()
            logger.info(f"Updated document {document_id}")
            return True
        except MilvusException as e:
            logger.error(f"Failed to update document {document_id}: {e}")
            return False

    @classmethod
    def from_documents(
        cls,
        documents: List[Document],
        embedding: Embeddings,
        **kwargs
    ) -> "MilvusVectorStore":
        """Create a vector store from documents."""
        store = cls(embedding=embedding, **kwargs)
        store.add_documents(documents)
        return store

@asynccontextmanager
async def pipeline_context(config: Dict[str, Any]) -> AsyncIterator[MilvusVectorStore]:
    """Context manager for Milvus vector store operations."""
    connection_args = {
        "uri": config.get("MILVUS_URI", "http://localhost:19530"),
        "token": config.get("MILVUS_TOKEN", ""),
        "timeout": config.get("MILVUS_TIMEOUT", 30),
    }
    
    store = MilvusVectorStore(
        collection_name=config.get("MILVUS_COLLECTION", "rag_documents"),
        connection_args=connection_args,
    )
    
    try:
        yield store
    finally:
        # Clean up resources if needed
        pass