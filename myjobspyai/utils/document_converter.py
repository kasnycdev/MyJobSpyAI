"""Document conversion utilities using DocLing.

This module provides functionality to convert between various document formats
using the DocLing library. It supports a wide range of input and output formats
including PDF, DOCX, HTML, Markdown, and more.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional, Union

# Initialize logger at module level
logger = logging.getLogger(__name__)

# Define fallback types when docling is not available
class _FallbackConversionStatus:
    SUCCESS = "success"
    FAILURE = "failure"

class _FallbackConversionResult:
    def __init__(self, content: bytes = b"", status: str = _FallbackConversionStatus.SUCCESS):
        self.content = content
        self.status = status

# Try to import DocLing components
try:
    from docling.document_converter import DocumentConverter
    from docling.backend import InputFormat
    from docling.models import ConversionResult, ConversionStatus
    DOCLING_AVAILABLE = True
except ImportError:
    logger.error("Failed to import DocLing components. Document conversion will not be available.")
    DOCLING_AVAILABLE = False
    
    # Define fallback types
    class DocumentConverter:
        pass
        
    class InputFormat:
        PDF = "pdf"
        DOCX = "docx"
        
    class ConversionStatus:
        SUCCESS = _FallbackConversionStatus.SUCCESS
        FAILURE = _FallbackConversionStatus.FAILURE
        
    class ConversionResult(_FallbackConversionResult):
        pass

class DocumentConversionError(Exception):
    """Exception raised for errors in document conversion."""
    pass

class DocumentConverterWrapper:
    """A wrapper around DocLing's DocumentConverter for easier use.
    
    This class provides a simplified interface to the DocLing document converter
    with support for common document conversion tasks.
    """
    def __init__(self, allowed_formats: Optional[List[str]] = None):
        """Initialize the document converter.
        
        Args:
            allowed_formats: List of allowed input formats. If None, all formats are allowed.
            
        Raises:
            DocumentConversionError: If docling is not available or initialization fails
        """
        if not DOCLING_AVAILABLE:
            raise DocumentConversionError(
                "DocLing is not available. Please install it with: pip install docling"
            )
            
        self.allowed_formats = [InputFormat(fmt.upper()) for fmt in (allowed_formats or [])]
        
        try:
            # Initialize the converter with default options
            self.converter = DocumentConverter(
                allowed_formats=self.allowed_formats if self.allowed_formats else None
            )
            
            # Configure PDF processing pipeline if available
            self._configure_pdf_pipeline()
            
        except Exception as e:
            raise DocumentConversionError(
                f"Failed to initialize document converter: {str(e)}"
            ) from e
    
    def _configure_pdf_pipeline(self) -> None:
        """Configure the PDF processing pipeline if available."""
        if not hasattr(self, 'converter') or not hasattr(self, 'allowed_formats'):
            return
            
        if InputFormat.PDF not in self.allowed_formats:
            return
            
        try:
            from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
            self.converter.format_to_options[InputFormat.PDF] = StandardPdfPipeline(
                extract_tables=True,
                extract_images=True,
                extract_metadata=True,
                extract_bookmarks=True,
            )
        except ImportError:
            logger.warning(
                "StandardPdfPipeline not found. PDF processing will be limited. "
                "Install with: pip install docling[pdf]"
            )
        except Exception as e:
            logger.warning(
                "Failed to configure PDF pipeline: %s",
                str(e),
                exc_info=True
            )
    
    async def convert_file(
        self,
        input_path: Union[str, Path],
        output_format: str = 'text',
        **kwargs: Any
    ) -> ConversionResult:
        """Convert a file to the specified format.
        
        Args:
            input_path: Path to the input file
            output_format: Desired output format (e.g., 'text', 'html', 'markdown')
            **kwargs: Additional arguments to pass to the converter
            
        Returns:
            A ConversionResult object containing the conversion results
            
        Raises:
            DocumentConversionError: If the file cannot be converted
        """
        if not DOCLING_AVAILABLE:
            raise DocumentConversionError(
                "DocLing is not available. Please install it with: pip install docling"
            )
            
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise DocumentConversionError(f"Input file not found: {input_path}")
        
        if self.allowed_formats:
            input_format = input_path.suffix.lower()
            allowed_format_values = [fmt.value for fmt in self.allowed_formats]
            if input_format not in allowed_format_values:
                raise DocumentConversionError(
                    f"Input format '{input_format}' is not in the list of allowed formats: "
                    f"{allowed_format_values}"
                )
        
        try:
            result = await self.converter.convert(
                str(input_path.absolute()),
                output_format=output_format,
                **kwargs
            )
            return result
        except Exception as e:
            logger.error("Error during document conversion: %s", str(e), exc_info=True)
            raise DocumentConversionError(f"Document conversion failed: {str(e)}") from e
    
    def _get_format_from_extension(self, extension: str) -> InputFormat:
        """Get the input format from a file extension."""
        extension = extension.lower().lstrip('.')
        format_map = {
            'pdf': InputFormat.PDF,
            'docx': InputFormat.DOCX,
            'doc': InputFormat.DOCX,
            'html': InputFormat.HTML,
            'htm': InputFormat.HTML,
            'md': InputFormat.MD,
            'markdown': InputFormat.MD,
            'pptx': InputFormat.PPTX,
            'ppt': InputFormat.PPTX,
            'xlsx': InputFormat.XLSX,
            'xls': InputFormat.XLSX,
            'csv': InputFormat.CSV,
            'adoc': InputFormat.ASCIIDOC,
            'asciidoc': InputFormat.ASCIIDOC,
            'json': InputFormat.JSON_DOCLING,
            'xml': InputFormat.XML_JATS,  # Default XML format
            'jats': InputFormat.XML_JATS,
            'uspto': InputFormat.XML_USPTO,
        }
        
        if extension not in format_map:
            raise ValueError(f"Unsupported file extension: {extension}")
            
        return format_map[extension]


# Example usage
async def example_usage():
    """Example usage of the DocumentConverterWrapper."""
    converter = DocumentConverterWrapper()
    
    # Convert PDF to Markdown
    result = await converter.convert_file(
        "input.pdf",
        "output.md",
        output_format="md"
    )
    print(f"Conversion successful: {result.status == ConversionStatus.SUCCESS}")
    
    # Convert DOCX to HTML
    result = await converter.convert_file(
        "input.docx",
        "output.html",
        output_format="html"
    )
    print(f"Conversion successful: {result.status == ConversionStatus.SUCCESS}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
