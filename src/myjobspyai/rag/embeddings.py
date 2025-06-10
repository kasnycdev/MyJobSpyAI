"""Embedding models for RAG pipeline."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type, Union

from langchain_core.embeddings import Embeddings as LangChainEmbeddings
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class EmbeddingConfig(BaseModel):
    """Configuration for embedding models."""

    model_name: str = Field(
        "sentence-transformers/all-mpnet-base-v2",
        description="Name or path of the pre-trained model to use",
    )
    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments to pass to the model",
    )
    encode_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments to pass to the model's encode method",
    )
    cache_folder: Optional[str] = Field(
        None,
        description="Path to store the model files. If None, the default cache folder is used.",
    )
    device: Optional[str] = Field(
        None,
        description="Device to run the model on (e.g., 'cuda', 'cpu', 'mps')",
    )
    normalize_embeddings: bool = Field(
        True,
        description="Whether to normalize the embeddings to unit length",
    )
    batch_size: int = Field(
        32,
        description="Batch size for embedding multiple texts",
        gt=0,
    )
    show_progress_bar: bool = Field(
        True,
        description="Whether to show a progress bar during embedding",
    )

    class Config:
        extra = "forbid"
        frozen = True


class BaseEmbeddings(ABC, LangChainEmbeddings):
    """Base class for embedding models."""

    def __init__(self, config: EmbeddingConfig):
        """Initialize the embedding model.

        Args:
            config: Configuration for the embedding model
        """
        super().__init__()
        self.config = config
        self.model = self._load_model()

    @abstractmethod
    def _load_model(self) -> Any:
        """Load the embedding model.

        Returns:
            The loaded embedding model
        """
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings, one for each text
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: Text to embed

        Returns:
            Embedding for the text
        """
        pass

    def embed_documents_batch(
        self, texts: List[str], batch_size: Optional[int] = None, **kwargs: Any
    ) -> List[List[float]]:
        """Embed a list of texts in batches.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for embedding. If None, use the value from config.
            **kwargs: Additional keyword arguments to pass to the embedding model

        Returns:
            List of embeddings, one for each text
        """
        batch_size = batch_size or self.config.batch_size
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self.embed_documents(batch, **kwargs)
            embeddings.extend(batch_embeddings)

        return embeddings


class SentenceTransformerEmbeddings(BaseEmbeddings):
    """Sentence Transformers embedding model."""

    def _load_model(self) -> Any:
        """Load the Sentence Transformers model.

        Returns:
            The loaded Sentence Transformers model

        Raises:
            ImportError: If sentence-transformers is not installed
        """
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                self.config.model_name,
                cache_folder=self.config.cache_folder,
                device=self.config.device,
                **self.config.model_kwargs,
            )
            return model
        except ImportError as e:
            raise ImportError(
                "sentence-transformers not installed. Please install with: "
                "pip install sentence-transformers"
            ) from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings, one for each text
        """
        return self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=self.config.show_progress_bar,
            normalize_embeddings=self.config.normalize_embeddings,
            **self.config.encode_kwargs,
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: Text to embed

        Returns:
            Embedding for the text
        """
        return self.embed_documents([text])[0]


class OpenAIEmbeddings(BaseEmbeddings):
    """OpenAI embedding model."""

    def _load_model(self) -> Any:
        """Load the OpenAI embeddings model.

        Returns:
            The loaded OpenAI embeddings model

        Raises:
            ImportError: If openai is not installed
        """
        try:
            from langchain_openai import OpenAIEmbeddings as LangChainOpenAIEmbeddings

            # Remove model_kwargs that are not supported by LangChain's OpenAIEmbeddings
            model_kwargs = self.config.model_kwargs.copy()
            model_kwargs.pop("model", None)  # model is set separately

            model = LangChainOpenAIEmbeddings(
                model=self.config.model_name,
                **model_kwargs,
            )
            return model
        except ImportError as e:
            raise ImportError(
                "openai not installed. Please install with: pip install openai"
            ) from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings, one for each text
        """
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text.

        Args:
            text: Text to embed

        Returns:
            Embedding for the text
        """
        return self.model.embed_query(text)


class EmbeddingFactory:
    """Factory for creating embedding models."""

    # Map of model prefixes to their implementation classes
    EMBEDDING_CLASSES = {
        "sentence-transformers": SentenceTransformerEmbeddings,
        "text-embedding": OpenAIEmbeddings,  # For OpenAI text-embedding-* models
        "text-embedding-ada": OpenAIEmbeddings,  # For backward compatibility
    }

    @classmethod
    def get_embedding_model(cls, config: EmbeddingConfig) -> BaseEmbeddings:
        """Get an embedding model based on the configuration.

        Args:
            config: Configuration for the embedding model

        Returns:
            An embedding model instance

        Raises:
            ValueError: If the model type is not supported
        """
        # Check for OpenAI models
        if config.model_name.startswith("text-embedding"):
            return OpenAIEmbeddings(config)

        # Check for Sentence Transformers models
        for prefix, embedding_class in cls.EMBEDDING_CLASSES.items():
            if config.model_name.startswith(prefix):
                return embedding_class(config)

        # Default to Sentence Transformers if no specific match found
        try:
            return SentenceTransformerEmbeddings(config)
        except ImportError as e:
            raise ValueError(
                f"Could not load embedding model: {config.model_name}. "
                "Make sure the required dependencies are installed."
            ) from e


# For backward compatibility
get_embedding_model = EmbeddingFactory.get_embedding_model
