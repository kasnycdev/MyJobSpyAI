"""Configuration classes for vector stores."""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, Field, field_validator


class VectorStoreConfig(BaseModel):
    """Configuration for vector store."""

    type: str = Field(
        default="faiss",
        description="Type of vector store to use (faiss, chroma, milvus, etc.)",
    )
    persist_directory: Optional[Union[str, Path]] = Field(
        None,
        description="Directory to persist the vector store",
    )
    collection_name: Optional[str] = Field(
        None,
        description="Name of the collection/index to use",
    )
    embedding_dimension: Optional[int] = Field(
        None,
        description="Dimension of the embeddings (optional, will be inferred from model if not provided)",
    )
    distance_metric: str = Field(
        "cosine",
        description="Distance metric to use for similarity search (cosine, l2, ip, etc.)",
    )
    normalize_embeddings: bool = Field(
        True,
        description="Whether to normalize embeddings before storing them",
    )

    @field_validator("distance_metric")
    @classmethod
    def validate_distance_metric(cls, v: str) -> str:
        """Validate the distance metric.

        Args:
            v: Distance metric to validate

        Returns:
            Validated distance metric

        Raises:
            ValueError: If the distance metric is not supported
        """
        valid_metrics = {"cosine", "l2", "ip"}  # Inner product, L2, cosine
        if v.lower() not in valid_metrics:
            raise ValueError(
                f"Unsupported distance metric: {v}. "
                f"Supported metrics: {', '.join(valid_metrics)}"
            )
        return v.lower()

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields for specific vector store implementations
