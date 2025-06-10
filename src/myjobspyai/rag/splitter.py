"""Text splitting functionality for RAG pipeline."""

from typing import Any, Dict, List, Optional

from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
)
from pydantic import BaseModel, Field, field_validator


class TextSplitterConfig(BaseModel):
    """Configuration for text splitting."""

    chunk_size: int = Field(
        default=1000,
        description="Maximum size of chunks to create (in characters or tokens)",
        gt=0,
    )
    chunk_overlap: int = Field(
        default=200,
        description="Number of characters/tokens to overlap between chunks",
        ge=0,
    )
    separator: str = Field(
        default="\n\n",
        description="Separator to use when splitting text",
    )
    is_separator_regex: bool = Field(
        default=False,
        description="Whether the separator is a regular expression",
    )
    keep_separator: bool = Field(
        default=True,
        description="Whether to keep the separator in the chunks",
    )
    length_function: str = Field(
        default="len",
        description="Function to use for calculating length of text (len for characters, 'token_counter' for tokens)",
    )
    splitter_type: str = Field(
        default="recursive",
        description="Type of splitter to use (recursive, character, token)",
    )
    separators: Optional[List[str]] = Field(
        default=None,
        description="List of separators to use for recursive splitting (only for recursive splitter)",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="Model name for token-based splitting (only for token splitter)",
    )
    encoding_name: Optional[str] = Field(
        default=None,
        description="Encoding name for token-based splitting (only for token splitter)",
    )

    @field_validator("splitter_type")
    @classmethod
    def validate_splitter_type(cls, v: str) -> str:
        """Validate the splitter type.

        Args:
            v: Splitter type to validate

        Returns:
            Validated splitter type

        Raises:
            ValueError: If the splitter type is not supported
        """
        if v not in ["recursive", "character", "token"]:
            raise ValueError(
                "splitter_type must be one of: recursive, character, token"
            )
        return v

    @field_validator("length_function")
    @classmethod
    def validate_length_function(cls, v: str, values: Dict[str, Any]) -> str:
        """Validate the length function.

        Args:
            v: Length function to validate
            values: Other field values

        Returns:
            Validated length function

        Raises:
            ValueError: If the length function is not supported
        """
        if v not in ["len", "token_counter"]:
            raise ValueError("length_function must be one of: len, token_counter")

        if v == "token_counter" and values.get("splitter_type") != "token":
            raise ValueError("token_counter can only be used with token splitter_type")

        return v

    class Config:
        extra = "forbid"
        frozen = True


def get_text_splitter(config: TextSplitterConfig):
    """Get a text splitter based on the configuration.

    Args:
        config: Configuration for the text splitter

    Returns:
        A text splitter instance

    Raises:
        ValueError: If the configuration is invalid
    """
    splitter_type = config.splitter_type

    # Common arguments for all splitters
    common_args = {
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "keep_separator": config.keep_separator,
    }

    if splitter_type == "recursive":
        return RecursiveCharacterTextSplitter(
            **common_args,
            separators=config.separators or ["\n\n", "\n", ". ", " ", ""],
            is_separator_regex=config.is_separator_regex,
        )
    elif splitter_type == "character":
        return CharacterTextSplitter(
            **common_args,
            separator=config.separator,
            is_separator_regex=config.is_separator_regex,
        )
    elif splitter_type == "token":
        if not config.model_name and not config.encoding_name:
            raise ValueError(
                "Either model_name or encoding_name must be provided for token splitter"
            )
        return TokenTextSplitter(
            **common_args,
            model_name=config.model_name,
            encoding_name=config.encoding_name,
        )
    else:
        raise ValueError(f"Unknown splitter type: {splitter_type}")


def split_documents(
    documents: List[Dict[str, Any]], config: Optional[TextSplitterConfig] = None
) -> List[Dict[str, Any]]:
    """Split documents into chunks based on the configuration.

    Args:
        documents: List of documents to split
        config: Configuration for text splitting. If None, default values will be used.

    Returns:
        List of split documents with metadata
    """
    if not documents:
        return []

    config = config or TextSplitterConfig()
    text_splitter = get_text_splitter(config)

    # Convert documents to the format expected by LangChain
    docs = [
        {
            "page_content": doc.get("content", ""),
            "metadata": doc.get("metadata", {}),
        }
        for doc in documents
    ]

    # Split documents
    split_docs = text_splitter.split_documents(docs)

    # Convert back to the original format
    return [
        {
            "content": doc.page_content,
            "metadata": doc.metadata,
        }
        for doc in split_docs
    ]
