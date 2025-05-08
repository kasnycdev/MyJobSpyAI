import os
import logging
import html
from docx import Document
from pypdf import PdfReader # Changed from PyPDF2 to pypdf
from colorama import Fore, Style, init  # Import colorama
from rich.console import Console
from rich.logging import RichHandler

# Initialize colorama
init(autoreset=True)

# Initialize rich console
console = Console()

# Update logging configuration to use RichHandler
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
)


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


def _parse_docx(file_path: str) -> str:
    """Parses text content from a DOCX file."""
    try:
        doc = Document(file_path)
        full_text = [para.text for para in doc.paragraphs]
        raw_text = '\n'.join(full_text)
        return _preprocess_text(raw_text)
    except Exception as e:
        console.log(f"[red]Error parsing DOCX file {html.escape(file_path)}: {html.escape(str(e))}[/red]")
        return ""


def _parse_pdf(file_path: str) -> str:
    """Parses text content from a PDF file using pypdf."""
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
        console.log(f"[red]Error parsing PDF file {file_path}: {e}[/red]")
        return ""


def parse_resume(file_path: str) -> str:
    """
    Parses text from a resume file (DOCX or PDF).

    Args:
        file_path: Path to the resume file.

    Returns:
        The extracted text content as a string, or empty string on error.
    """
    if not os.path.exists(file_path):
        console.log(f"[red]Resume file not found: {file_path}[/red]")
        return ""

    _, file_extension = os.path.splitext(file_path.lower())

    if file_extension == ".docx":
        console.log(f"Parsing DOCX resume: {file_path}")
        return _parse_docx(file_path)
    elif file_extension == ".pdf":
        console.log(f"Parsing PDF resume: {file_path}")
        return _parse_pdf(file_path)
    else:
        console.log(f"[red]Unsupported resume file format: {file_extension}. Please use .docx or .pdf.[/red]")
        return ""
