"""
RAG (Retrieval-Augmented Generation) module for MyJobSpyAI.

This module provides functionality for document loading, text splitting, vector storage,
and retrieval to support RAG applications.
"""

from typing import Any

from .config import RAGConfig
from .embeddings import (
    BaseEmbeddings,
    EmbeddingConfig,
    OpenAIEmbeddings,
    SentenceTransformerEmbeddings,
    get_embedding_model,
)

# Document Loaders
from .loader import (
    DocumentLoaderConfig,
    DocxLoader,
    PDFLoader,
    TextLoader,
    UnstructuredFileLoader,
    load_documents,
)

# Pipeline
from .pipeline import RAGPipeline

# Text Splitting
from .splitter import TextSplitterConfig, TextSplitterType, split_documents, split_text

# Utils
from .utils import (
    batch_iterable,
    calculate_md5,
    convert_from_langchain_documents,
    convert_to_langchain_documents,
    ensure_directory_exists,
    get_file_metadata,
    merge_metadata,
    validate_documents,
    validate_embedding,
)

# Vector Stores
from .vector_store import (
    BaseVectorStore,
    ChromaVectorStore,
    FAISSVectorStore,
    VectorStoreConfig,
    get_vector_store,
)

__all__ = [
    # Config
    "RAGConfig",
    # Embeddings
    "EmbeddingConfig",
    "get_embedding_model",
    "BaseEmbeddings",
    "SentenceTransformerEmbeddings",
    "OpenAIEmbeddings",
    # Document Loaders
    "DocumentLoaderConfig",
    "load_documents",
    "PDFLoader",
    "TextLoader",
    "DocxLoader",
    "UnstructuredFileLoader",
    # Pipeline
    "RAGPipeline",
    # Text Splitting
    "TextSplitterConfig",
    "TextSplitterType",
    "split_text",
    "split_documents",
    # Utils
    "calculate_md5",
    "get_file_metadata",
    "validate_documents",
    "convert_to_langchain_documents",
    "convert_from_langchain_documents",
    "merge_metadata",
    "validate_embedding",
    "batch_iterable",
    "ensure_directory_exists",
    # Vector Stores
    "VectorStoreConfig",
    "get_vector_store",
    "BaseVectorStore",
    "FAISSVectorStore",
    "ChromaVectorStore",
]
