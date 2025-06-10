"""Document loading functionality for RAG pipeline."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Type, Union

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_core.documents import Document
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class DocumentLoaderConfig(BaseModel):
    """Configuration for document loading."""

    file_path: Union[str, Path] = Field(..., description="Path to the document file")
    file_type: Optional[str] = Field(
        None, description="Explicit file type (inferred from extension if not provided)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata to include with the document"
    )

    class Config:
        extra = "forbid"
        frozen = True


class BaseDocumentLoader(ABC):
    """Base class for document loaders."""

    SUPPORTED_EXTENSIONS: List[str] = []

    @classmethod
    def is_supported(cls, file_path: Union[str, Path]) -> bool:
        """Check if the file type is supported by this loader.

        Args:
            file_path: Path to the file to check

        Returns:
            bool: True if the file type is supported, False otherwise
        """
        file_path = Path(file_path)
        return file_path.suffix.lower() in cls.SUPPORTED_EXTENSIONS

    @abstractmethod
    def load(self) -> List[Document]:
        """Load documents from the configured source.

        Returns:
            List[Document]: List of loaded documents
        """
        pass


class PDFDocumentLoader(BaseDocumentLoader):
    """Loader for PDF documents."""

    SUPPORTED_EXTENSIONS = [".pdf"]

    def __init__(self, config: DocumentLoaderConfig):
        """Initialize the PDF document loader.

        Args:
            config: Configuration for the loader
        """
        self.config = config
        self.loader = PyPDFLoader(str(config.file_path))

    def load(self) -> List[Document]:
        """Load PDF document.

        Returns:
            List[Document]: List of document pages
        """
        try:
            docs = self.loader.load()
            if self.config.metadata:
                for doc in docs:
                    doc.metadata.update(self.config.metadata)
            return docs
        except Exception as e:
            logger.error(f"Error loading PDF document: {e}")
            raise


class TextDocumentLoader(BaseDocumentLoader):
    """Loader for plain text documents."""

    SUPPORTED_EXTENSIONS = [".txt", ".md", ".rst", ".json", ".yaml", ".yml"]

    def __init__(self, config: DocumentLoaderConfig):
        """Initialize the text document loader.

        Args:
            config: Configuration for the loader
        """
        self.config = config
        self.loader = TextLoader(str(config.file_path), autodetect_encoding=True)

    def load(self) -> List[Document]:
        """Load text document.

        Returns:
            List[Document]: List containing a single document
        """
        try:
            docs = self.loader.load()
            if self.config.metadata:
                for doc in docs:
                    doc.metadata.update(self.config.metadata)
            return docs
        except Exception as e:
            logger.error(f"Error loading text document: {e}")
            raise


class WordDocumentLoader(BaseDocumentLoader):
    """Loader for Microsoft Word documents."""

    SUPPORTED_EXTENSIONS = [".docx", ".doc"]

    def __init__(self, config: DocumentLoaderConfig):
        """Initialize the Word document loader.

        Args:
            config: Configuration for the loader
        """
        self.config = config
        self.loader = UnstructuredWordDocumentLoader(str(config.file_path))

    def load(self) -> List[Document]:
        """Load Word document.

        Returns:
            List[Document]: List of document elements
        """
        try:
            docs = self.loader.load()
            if self.config.metadata:
                for doc in docs:
                    doc.metadata.update(self.config.metadata)
            return docs
        except Exception as e:
            logger.error(f"Error loading Word document: {e}")
            raise


class UnstructuredDocumentLoader(BaseDocumentLoader):
    """Loader for unstructured documents using UnstructuredIO."""

    SUPPORTED_EXTENSIONS = [".html", ".htm", ".eml", ".msg", ".epub", ".odt", ".rtf"]

    def __init__(self, config: DocumentLoaderConfig):
        """Initialize the unstructured document loader.

        Args:
            config: Configuration for the loader
        """
        self.config = config
        self.loader = UnstructuredFileLoader(str(config.file_path))

    def load(self) -> List[Document]:
        """Load unstructured document.

        Returns:
            List[Document]: List of document elements
        """
        try:
            docs = self.loader.load()
            if self.config.metadata:
                for doc in docs:
                    doc.metadata.update(self.config.metadata)
            return docs
        except Exception as e:
            logger.error(f"Error loading unstructured document: {e}")
            raise


class DocumentLoaderFactory:
    """Factory for creating document loaders based on file type."""

    # Map of loader classes to their supported file extensions
    LOADER_CLASSES: List[Type[BaseDocumentLoader]] = [
        PDFDocumentLoader,
        WordDocumentLoader,
        TextDocumentLoader,
        UnstructuredDocumentLoader,
    ]

    @classmethod
    def get_loader(cls, config: DocumentLoaderConfig) -> BaseDocumentLoader:
        """Get the appropriate document loader for the given file.

        Args:
            config: Configuration for the loader

        Returns:
            BaseDocumentLoader: An instance of the appropriate document loader

        Raises:
            ValueError: If no suitable loader is found for the file type
        """
        file_path = Path(config.file_path)

        # If file_type is provided explicitly, use it
        if config.file_type:
            for loader_class in cls.LOADER_CLASSES:
                if f".{config.file_type.lower()}" in loader_class.SUPPORTED_EXTENSIONS:
                    return loader_class(config)

        # Otherwise, try to determine from file extension
        for loader_class in cls.LOADER_CLASSES:
            if file_path.suffix.lower() in loader_class.SUPPORTED_EXTENSIONS:
                return loader_class(config)

        raise ValueError(
            f"Unsupported file type: {file_path.suffix}. "
            f"Supported types: {', '.join(cls.get_supported_extensions())}"
        )

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get a list of all supported file extensions.

        Returns:
            List[str]: List of supported file extensions
        """
        extensions = set()
        for loader_class in cls.LOADER_CLASSES:
            extensions.update(loader_class.SUPPORTED_EXTENSIONS)
        return sorted(extensions)


def load_documents(config: DocumentLoaderConfig) -> List[Document]:
    """Load documents from a file.

    Args:
        config: Configuration for document loading

    Returns:
        List[Document]: List of loaded documents
    """
    loader = DocumentLoaderFactory.get_loader(config)
    return loader.load()
