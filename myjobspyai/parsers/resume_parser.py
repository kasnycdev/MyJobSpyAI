import os
import logging
import re
import html  # For HTML escaping in error messages
from pathlib import Path
from opentelemetry import trace
from docx import Document
from pypdf import PdfReader
import textract  # For additional file format support
from bs4 import BeautifulSoup  # For HTML content extraction

# Get a logger for this module
logger = logging.getLogger(__name__)

# Get a logger for this module
logger = logging.getLogger(__name__)

# Import tracer from logging_utils
try:
    from myjobspyai.utils.logging_utils import tracer as global_tracer
    if global_tracer is None: # Check if OTEL was disabled in logging_utils
        from opentelemetry import trace
        tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
        logger.warning("OpenTelemetry not configured in logging_utils, using NoOpTracer for resume_parser.")
    else:
        tracer = global_tracer
except ImportError:
    from opentelemetry import trace
    tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
    logger.error("Could not import global_tracer from myjobspyai.utils.logging_utils. Using NoOpTracer for resume_parser.", exc_info=True)


# init(autoreset=True) # Colorama init, if needed, should be in main.py
# console = Console() # Replaced by logger

# The logging.basicConfig call here is removed. 
# It's assumed that main.py (or the entry point) calls setup_logging from logging_utils.py,
# which configures the root logger and all its handlers (including RichHandler for console).

def _preprocess_text(text: str) -> str:
    """
    Preprocesses extracted text to clean and normalize it.

    Args:
        text: The raw extracted text.

    Returns:
        The cleaned and normalized text.
    """
    if not text or not isinstance(text, str):
        return ""
        
    # Remove non-printable characters except newlines and tabs
    text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
    
    # Normalize whitespace and line breaks
    text = ' '.join(text.split())  # Replace all whitespace with single space
    
    # Replace multiple newlines with a single one
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Remove common resume artifacts
    text = re.sub(r'\b(?:page|resume|cv)\s*\d*\s*[\-–—]?\s*\d*\b', '', text, flags=re.IGNORECASE)
    
    return text.strip()


def _validate_extracted_text(text: str, min_length: int = 50) -> bool:
    """
    Validates that the extracted text meets quality criteria.
    
    Args:
        text: The extracted text to validate
        min_length: Minimum required length of meaningful text
        
    Returns:
        bool: True if text is valid, False otherwise
    """
    if not text or not isinstance(text, str):
        return False
        
    # Check minimum length
    if len(text.strip()) < min_length:
        return False
        
    # Check for common error patterns
    error_patterns = [
        r'error',
        r'unable to extract',
        r'not available',
        r'protected document',
        r'password protected'
    ]
    
    for pattern in error_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return False
            
    # Check for reasonable word count
    words = text.split()
    if len(words) < 10:  # Arbitrary threshold
        return False
        
    return True


def _parse_other_formats(file_path: str, file_extension: str) -> str:
    """
    Parse other supported file formats using textract.
    
    Args:
        file_path: Path to the file to parse
        file_extension: File extension including dot (e.g., '.txt')
        
    Returns:
        Extracted and preprocessed text
    """
    try:
        # Use textract for various formats
        text = textract.process(file_path).decode('utf-8')
        
        # Special handling for HTML/HTM
        if file_extension in ['.html', '.htm']:
            soup = BeautifulSoup(text, 'html.parser')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
        
        return _preprocess_text(text)
    except Exception as e:
        logger.error(f"Error parsing {file_extension} file {file_path}: {str(e)}")
        return ""


@tracer.start_as_current_span("parse_docx")
def _parse_docx(file_path: str) -> str:
    """Parses text content from a DOCX file."""
    trace.get_current_span().set_attribute("file.path", file_path)
    try:
        doc = Document(file_path)
        full_text = [para.text for para in doc.paragraphs]
        raw_text = '\n'.join(full_text)
        return _preprocess_text(raw_text) # Correctly indented
    except Exception as e:
        logger.error(f"Error parsing DOCX file {html.escape(file_path)}: {html.escape(str(e))}", exc_info=True)
        trace.get_current_span().record_exception(e)
        trace.get_current_span().set_status(trace.Status(trace.StatusCode.ERROR, "DOCX parsing failed"))
        return ""


@tracer.start_as_current_span("parse_pdf")
def _parse_pdf(file_path: str) -> str:
    """Parses text content from a PDF file using pypdf."""
    trace.get_current_span().set_attribute("file.path", file_path)
    text = ""
    try:
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            # Iterate through pages and extract text
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                if (page_text := page.extract_text()):
                    text += page_text + "\n"
        return _preprocess_text(text)
    except Exception as e:
        logger.error(f"Error parsing PDF file {file_path}: {e}", exc_info=True)
        trace.get_current_span().record_exception(e)
        trace.get_current_span().set_status(trace.Status(trace.StatusCode.ERROR, "PDF parsing failed"))
        return ""


@tracer.start_as_current_span("parse_resume")
def parse_resume(file_path: str) -> str:
    """
    Parses text from a resume file (DOCX or PDF).

    Args:
        file_path: Path to the resume file.

    Returns:
        The extracted text content as a string, or empty string on error.
    """
    # Only use tracing if it's available
    current_span = None
    try:
        current_span = trace.get_current_span()
        if current_span and hasattr(current_span, 'set_attribute'):
            current_span.set_attribute("file.path", str(file_path))
    except Exception as e:
        logger.debug(f"Could not set trace attributes: {e}")

    if not os.path.exists(file_path):
        logger.error(f"Resume file not found: {file_path}")
        if current_span and hasattr(current_span, 'set_attribute'):
            current_span.set_attribute("parse_error", "file_not_found")
            try:
                from opentelemetry.trace.status import Status, StatusCode
                current_span.set_status(Status(StatusCode.ERROR, "File not found"))
            except Exception as e:
                logger.debug(f"Could not set trace status: {e}")
        return ""

    file_extension = Path(file_path).suffix.lower()
    if current_span and hasattr(current_span, 'set_attribute'):
        current_span.set_attribute("file.extension", file_extension)

    try:
        if file_extension == ".docx":
            logger.info(f"Parsing DOCX resume: {file_path}")
            text = _parse_docx(file_path)
        elif file_extension == ".pdf":
            logger.info(f"Parsing PDF resume: {file_path}")
            text = _parse_pdf(file_path)
        elif file_extension in ['.txt', '.rtf', '.odt', '.html', '.htm']:
            logger.info(f"Parsing {file_extension.upper()} resume: {file_path}")
            text = _parse_other_formats(file_path, file_extension)
        else:
            logger.error(f"Unsupported resume file format: {file_extension}")
            current_span.set_attribute("parse_error", "unsupported_format")
            current_span.set_status(trace.Status(trace.StatusCode.ERROR, "Unsupported file format"))
            return ""
        
        # Validate extracted text
        if not _validate_extracted_text(text):
            logger.warning(f"Extracted text from {file_path} failed validation")
            current_span.set_status(trace.Status(trace.StatusCode.WARN, "Text validation failed"))
        
        return text
    except Exception as e:
        logger.error(f"Error parsing {file_path}: {str(e)}", exc_info=True)
        current_span.record_exception(e)
        current_span.set_status(trace.Status(trace.StatusCode.ERROR, f"Parsing failed: {str(e)}"))
        return ""
