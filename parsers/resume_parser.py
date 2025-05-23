import os
import logging # Keep for standard logging
import html
from docx import Document
from pypdf import PdfReader # Changed from PyPDF2 to pypdf
# from colorama import Fore, Style, init # Likely no longer needed
# from rich.console import Console # Replaced by logger
# from rich.logging import RichHandler # Handled by setup_logging in main

# Get a logger for this module
logger = logging.getLogger(__name__)

# Import tracer from logging_utils
try:
    from logging_utils import tracer as global_tracer
    if global_tracer is None: # Check if OTEL was disabled in logging_utils
        from opentelemetry import trace
        tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
        logger.warning("OpenTelemetry not configured in logging_utils, using NoOpTracer for resume_parser.")
    else:
        tracer = global_tracer
except ImportError:
    from opentelemetry import trace
    tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
    logger.error("Could not import global_tracer from logging_utils. Using NoOpTracer for resume_parser.", exc_info=True)


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
    # Remove extra whitespace and normalize line breaks
    text = text.strip()
    text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
    return text


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
    current_span = trace.get_current_span()
    current_span.set_attribute("file.path", file_path)

    if not os.path.exists(file_path):
        logger.error(f"Resume file not found: {file_path}")
        current_span.set_attribute("parse_error", "file_not_found")
        current_span.set_status(trace.Status(trace.StatusCode.ERROR, "File not found"))
        return ""

    _, file_extension = os.path.splitext(file_path.lower())
    current_span.set_attribute("file.extension", file_extension)

    if file_extension == ".docx":
        logger.info(f"Parsing DOCX resume: {file_path}")
        return _parse_docx(file_path) # Already traced
    elif file_extension == ".pdf":
        logger.info(f"Parsing PDF resume: {file_path}")
        return _parse_pdf(file_path) # Already traced
    else:
        logger.error(f"Unsupported resume file format: {file_extension}. Please use .docx or .pdf.")
        current_span.set_attribute("parse_error", "unsupported_format")
        current_span.set_status(trace.Status(trace.StatusCode.ERROR, "Unsupported file format"))
        return ""
