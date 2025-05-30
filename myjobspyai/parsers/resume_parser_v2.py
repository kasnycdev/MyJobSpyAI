"""Resume parser with DocLing integration for improved document processing.

This module provides functionality to parse text from various resume formats
using DocLing for enhanced document conversion and text extraction.
"""
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from opentelemetry import trace
from pydantic import BaseModel, Field, HttpUrl, validator

# Import our document converter
from myjobspyai.utils.document_converter import DocumentConverterWrapper

# Configure logging
logger = logging.getLogger(__name__)

# Import tracer from logging_utils
try:
    from myjobspyai.utils.logging_utils import tracer as global_tracer
    if global_tracer is None:
        tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
        logger.warning("OpenTelemetry not configured in logging_utils, using NoOpTracer for resume_parser_v2.")
    else:
        tracer = global_tracer
except ImportError:
    tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
    logger.error(
        "Could not import global_tracer from myjobspyai.utils.logging_utils. "
        "Using NoOpTracer for resume_parser_v2.",
        exc_info=True
    )

# Supported file extensions and their corresponding mime types
SUPPORTED_EXTENSIONS = {
    '.pdf': 'application/pdf',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.doc': 'application/msword',
    '.rtf': 'application/rtf',
    '.odt': 'application/vnd.oasis.opendocument.text',
    '.txt': 'text/plain',
    '.html': 'text/html',
    '.htm': 'text/html',
}

# Pydantic models for resume data structure
class ContactInfo(BaseModel):
    """Contact information model."""
    name: Optional[str] = Field(None, description="Full name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    location: Optional[str] = Field(None, description="Location (city, state, country)")
    linkedin: Optional[HttpUrl] = Field(None, description="LinkedIn profile URL")
    portfolio: Optional[HttpUrl] = Field(None, description="Portfolio or personal website URL")
    github: Optional[HttpUrl] = Field(None, description="GitHub profile URL")

class ExperienceItem(BaseModel):
    """Work experience item model."""
    company: str = Field(..., description="Company name")
    title: str = Field(..., description="Job title")
    location: Optional[str] = Field(None, description="Job location")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD or 'Present')")
    is_current: bool = Field(False, description="Whether this is the current position")
    description: Optional[str] = Field(None, description="Job description and achievements")
    
    @validator('start_date', 'end_date', pre=True)
    def parse_date(cls, v):
        if not v:
            return None
        # Handle 'Present' or similar values for end_date
        if isinstance(v, str) and v.lower() in ['present', 'current', 'now']:
            return datetime.now().strftime('%Y-%m-%d')
        return v

class EducationItem(BaseModel):
    """Education item model."""
    institution: str = Field(..., description="Name of the educational institution")
    degree: str = Field(..., description="Degree or qualification obtained")
    field_of_study: Optional[str] = Field(None, description="Field of study")
    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    gpa: Optional[float] = Field(None, description="GPA (out of 4.0 or 5.0)")

class ResumeData(BaseModel):
    """Structured resume data model."""
    # Top-level fields
    name: Optional[str] = Field(None, description="Full name of the candidate")
    contact_info: Dict[str, str] = Field(
        default_factory=dict,
        description="Contact information including email, phone, etc."
    )
    summary: Optional[str] = Field(None, description="Professional summary or objective")
    
    # Experience section
    experience: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Work experience in reverse chronological order"
    )
    
    # Education section
    education: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Education history in reverse chronological order"
    )
    
    # Skills and competencies
    skills: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of skills with categories and proficiency levels"
    )
    
    # Projects and achievements
    projects: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Notable projects with descriptions and technologies used"
    )
    
    # Certifications and licenses
    certifications: List[str] = Field(
        default_factory=list,
        description="List of professional certifications"
    )
    
    # Languages
    languages: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Languages spoken with proficiency levels"
    )
    
    # Additional sections
    honors: List[str] = Field(
        default_factory=list,
        description="Honors and awards received"
    )
    
    publications: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Published works or papers"
    )
    
    def to_json(self, **kwargs) -> str:
        """Convert the model to a JSON string."""
        return self.model_dump_json(indent=2, **kwargs)
    
    @classmethod
    def from_text(cls, text: str) -> 'ResumeData':
        """Create a ResumeData instance from unstructured text."""
        # This is a placeholder for the actual parsing logic
        # In a real implementation, this would use NLP or pattern matching
        # to extract structured data from the text
        return cls()
    
    # Add dict method for backward compatibility
    def dict(self, **kwargs) -> Dict[str, Any]:
        """Convert the model to a dictionary."""
        return self.model_dump(**kwargs)


# DocumentConverterWrapper is imported from myjobspyai.utils.document_converter


# Supported file extensions and their MIME types
SUPPORTED_EXTENSIONS = {
    '.pdf': 'application/pdf',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.doc': 'application/msword',
    '.rtf': 'application/rtf',
    '.odt': 'application/vnd.oasis.opendocument.text',
    '.txt': 'text/plain',
    '.html': 'text/html',
    '.htm': 'text/html',
}


class ResumeParser:
    """Parser for extracting and structuring resume data using DocLing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the resume parser.
        
        Args:
            config: Optional configuration dictionary for the document converter
        """
        self.config = config or {}
        self.converter = DocumentConverterWrapper(
            allowed_formats=list(SUPPORTED_EXTENSIONS.keys())
        )
    
    async def parse_resume(
        self, 
        file_path: Union[str, Path],
        output_format: str = 'json',
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """Parse text from a resume file and return structured data.
        
        Args:
            file_path: Path to the resume file
            output_format: Desired output format ('json' or 'text')
            **kwargs: Additional arguments to pass to the document converter
            
        Returns:
            The extracted resume data as a dictionary (for 'json' format) or 
            plain text (for 'text' format)
        """
        try:
            file_path = Path(file_path)
            
            # Validate file exists and is a file
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return {} if output_format == 'json' else ""
                
            if not file_path.is_file():
                logger.error(f"Not a file: {file_path}")
                return {} if output_format == 'json' else ""
                
            # Check file extension
            ext = file_path.suffix.lower()
            if ext not in SUPPORTED_EXTENSIONS:
                logger.error(f"Unsupported file format: {ext}")
                return {} if output_format == 'json' else ""
            
            # Convert the document to text first
            text = await self._extract_text(file_path)
            if not text:
                return {} if output_format == 'json' else ""
            
            # If text format is requested, return the cleaned text
            if output_format == 'text':
                return text
                
            # Otherwise, parse the text into structured data
            resume_data = self._parse_resume_text(text)
            
            # Convert to dictionary if JSON output is requested
            if output_format == 'json':
                return resume_data.model_dump()
                
            return resume_data
            
        except Exception as e:
            logger.error(f"Error parsing resume {file_path}: {str(e)}")
            return {} if output_format == 'json' else ""
    
    async def _extract_text(self, file_path: Union[str, Path]) -> str:
        """Extract text from a resume file."""
        try:
            # Convert the document to text
            result = await self.converter.convert_file(
                file_path,
                output_format='txt'
            )
            
            # Extract text from the conversion result
            text = self._extract_text_from_result(result)
            
            # Clean and preprocess the text
            return self._preprocess_text(text)
            
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {str(e)}")
            return ""
    
    def _parse_resume_text(self, text: str) -> ResumeData:
        """Parse resume text into structured data.
        
        Args:
            text: The extracted resume text
            
        Returns:
            A ResumeData instance with the parsed resume data
        """
        # This is a simplified implementation that creates a basic ResumeData instance
        # In a real implementation, you would use NLP or pattern matching to extract
        # the structured data from the text
        
        # Create a basic ResumeData instance
        resume_data = ResumeData(
            contact_info=ContactInfo(
                name="John Doe",  # Would be extracted from text
                email="john.doe@example.com",  # Would be extracted from text
            ),
            summary=text[:500] + "..." if len(text) > 500 else text,  # First 500 chars as summary
            skills=[{"name": "Python", "level": "Advanced"}],  # Would be extracted
            experience=[],  # Would be extracted
            education=[],  # Would be extracted
            certifications=[],  # Would be extracted
            languages=[],  # Would be extracted
            projects=[]  # Would be extracted
        )
        
        return resume_data
    
    def _extract_text_from_result(self, result: Any) -> str:
        """Extract text from a document conversion result.
        
        Args:
            result: The result from the document converter
            
        Returns:
            The extracted text as a string
        """
        if result is None:
            return ""
            
        try:
            # Handle different result types
            if hasattr(result, 'text'):
                return str(result.text)
                
            if hasattr(result, 'assembled') and hasattr(result.assembled, 'content'):
                if isinstance(result.assembled.content, bytes):
                    return result.assembled.content.decode('utf-8', errors='replace')
                return str(result.assembled.content)
            
            # Fallback to document text if available
            if hasattr(result, 'document') and hasattr(result.document, 'text'):
                return str(result.document.text)
                
            # Fallback to pages if available
            if hasattr(result, 'pages') and result.pages:
                return '\n'.join(page.text for page in result.pages if hasattr(page, 'text'))
                
            # Try to convert to string as last resort
            return str(result)
            
        except Exception as e:
            logger.error(f"Error extracting text from result: {str(e)}")
            return ""
    
    def _preprocess_text(self, text: str) -> str:
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
    
    def _validate_extracted_text(self, text: str, min_length: int = 50) -> bool:
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
            r'password protected',
            r'this document is corrupted',
            r'failed to extract',
            r'decryption failed',
        ]
        
        for pattern in error_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return False
                
        # Check for reasonable word count
        words = text.split()
        if len(words) < 10:  # Arbitrary threshold
            return False
            
        return True


# For backward compatibility
async def parse_resume(file_path: Union[str, Path], **kwargs) -> str:
    """Parse text from a resume file (backward compatibility wrapper).
    
    Args:
        file_path: Path to the resume file
        **kwargs: Additional arguments to pass to the parser
        
    Returns:
        The extracted text content as a string, or empty string on error
    """
    parser = ResumeParser()
    return await parser.parse_resume(file_path, **kwargs)
