"""Utility functions for the RAG pipeline."""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langchain_core.documents import Document as LCDocument

logger = logging.getLogger(__name__)


def calculate_md5(file_path: Union[str, Path], chunk_size: int = 8192) -> str:
    """Calculate the MD5 hash of a file.

    Args:
        file_path: Path to the file
        chunk_size: Size of chunks to read from the file

    Returns:
        MD5 hash of the file
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        # Read and update hash in chunks
        for chunk in iter(lambda: f.read(chunk_size), b""):
            md5_hash.update(chunk)

    return md5_hash.hexdigest()


def get_file_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Get metadata about a file.

    Args:
        file_path: Path to the file

    Returns:
        Dictionary containing file metadata
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    return {
        "file_name": file_path.name,
        "file_path": str(file_path.absolute()),
        "file_size": file_path.stat().st_size,
        "file_extension": file_path.suffix.lower(),
        "last_modified": file_path.stat().st_mtime,
        "md5_hash": calculate_md5(file_path),
    }


def validate_documents(documents: List[Dict[str, Any]]) -> None:
    """Validate a list of document dictionaries.

    Args:
        documents: List of document dictionaries

    Raises:
        ValueError: If any document is invalid
    """
    if not isinstance(documents, list):
        raise ValueError(f"Expected a list of documents, got {type(documents)}")

    for i, doc in enumerate(documents):
        if not isinstance(doc, dict):
            raise ValueError(f"Document at index {i} is not a dictionary")

        if "content" not in doc:
            raise ValueError(f"Document at index {i} is missing 'content' field")

        if not isinstance(doc["content"], str):
            raise ValueError(
                f"Document at index {i} has invalid content type: "
                f"{type(doc['content']).__name__}, expected str"
            )

        if "metadata" in doc and not isinstance(doc["metadata"], dict):
            raise ValueError(
                f"Document at index {i} has invalid metadata type: "
                f"{type(doc['metadata']).__name__}, expected dict or None"
            )


def convert_to_langchain_documents(documents: List[Dict[str, Any]]) -> List[LCDocument]:
    """Convert a list of document dictionaries to LangChain Documents.

    Args:
        documents: List of document dictionaries with 'content' and 'metadata' keys

    Returns:
        List of LangChain Document objects
    """
    validate_documents(documents)

    return [
        LCDocument(
            page_content=doc["content"],
            metadata=doc.get("metadata", {}),
        )
        for doc in documents
    ]


def convert_from_langchain_documents(
    documents: List[LCDocument],
) -> List[Dict[str, Any]]:
    """Convert a list of LangChain Documents to document dictionaries.

    Args:
        documents: List of LangChain Document objects

    Returns:
        List of document dictionaries with 'content' and 'metadata' keys
    """
    if not isinstance(documents, list):
        raise ValueError(f"Expected a list of documents, got {type(documents)}")

    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in documents
    ]


def merge_metadata(
    base_metadata: Optional[Dict[str, Any]],
    new_metadata: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Merge two metadata dictionaries.

    Args:
        base_metadata: Base metadata dictionary
        new_metadata: New metadata to merge into base

    Returns:
        Merged metadata dictionary
    """
    if base_metadata is None:
        return new_metadata or {}
    if new_metadata is None:
        return base_metadata or {}

    # Create a copy to avoid mutating the input
    result = base_metadata.copy()

    # Merge the new metadata, with new values taking precedence
    for key, value in new_metadata.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = merge_metadata(result[key], value)
        else:
            result[key] = value

    return result


def validate_embedding(embedding: List[float]) -> None:
    """Validate an embedding vector.

    Args:
        embedding: Embedding vector to validate

    Raises:
        ValueError: If the embedding is invalid
    """
    if not isinstance(embedding, list):
        raise ValueError(f"Expected embedding to be a list, got {type(embedding)}")

    if not all(isinstance(x, (int, float)) for x in embedding):
        raise ValueError("All elements in the embedding must be numbers")

    if not embedding:
        raise ValueError("Embedding cannot be empty")


def batch_iterable(iterable, batch_size: int):
    """Yield batches from an iterable.

    Args:
        iterable: Input iterable
        batch_size: Size of each batch

    Yields:
        Batches of items from the iterable
    """
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def ensure_directory_exists(directory: Union[str, Path]) -> Path:
    """Ensure that a directory exists, creating it if necessary.

    Args:
        directory: Directory path

    Returns:
        Path to the directory
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
