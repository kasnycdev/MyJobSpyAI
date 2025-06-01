"""
Resume analysis module for extracting and processing resume information.
"""
import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Type, cast

from pydantic import BaseModel, Field, HttpUrl, ValidationError

from myjobspyai.analysis.components.analyzers.base import BaseAnalyzer, AnalysisError

logger = logging.getLogger(__name__)

# Pydantic models for type-safe data validation
class ContactInfo(BaseModel):
    """Contact information model."""
    name: str = Field(..., description="Full name")
    email: Optional[str] = Field(None, description="Email address")
    phone: Optional[str] = Field(None, description="Phone number")
    location: Optional[str] = Field(None, description="Location (city, state, country)")
    linkedin: Optional[HttpUrl] = Field(None, description="LinkedIn profile URL")
    portfolio: Optional[HttpUrl] = Field(None, description="Portfolio or personal website URL")

class ExperienceItem(BaseModel):
    """Work experience item model."""
    company: str = Field(..., description="Company name")
    title: str = Field(..., description="Job title")
    location: Optional[str] = Field(None, description="Job location")
    start_date: Optional[date] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[date] = Field(None, description="End date (YYYY-MM-DD), None if current")
    is_current: bool = Field(False, description="Whether this is the current position")
    description: Optional[str] = Field(None, description="Job description and achievements")

class EducationItem(BaseModel):
    """Education item model."""
    institution: str = Field(..., description="Name of the educational institution")
    degree: str = Field(..., description="Degree or certification obtained")
    field_of_study: Optional[str] = Field(None, description="Field of study")
    start_date: Optional[date] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[date] = Field(None, description="End date (YYYY-MM-DD)")
    gpa: Optional[float] = Field(None, description="GPA if available")

class ResumeData(BaseModel):
    """Complete resume data model."""
    contact_info: ContactInfo = Field(..., description="Contact information")
    summary: Optional[str] = Field(None, description="Professional summary")
    skills: List[str] = Field(default_factory=list, description="List of skills")
    experience: List[ExperienceItem] = Field(
        default_factory=list, 
        description="Work experience in reverse chronological order"
    )
    education: List[EducationItem] = Field(
        default_factory=list,
        description="Education history in reverse chronological order"
    )
    certifications: List[str] = Field(
        default_factory=list,
        description="List of certifications"
    )
    languages: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Languages spoken with proficiency levels"
    )
    projects: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Notable projects"
    )

class ResumeAnalyzer(BaseAnalyzer[ResumeData, Any]):
    """Analyzer for extracting structured data from resumes."""
    
    # Define the response model for this analyzer
    RESPONSE_MODEL = ResumeData
    
    # Required fields that must be present in the response
    REQUIRED_FIELDS = ["contact_info", "skills", "experience", "education"]
    
    # Default model name
    DEFAULT_MODEL = "llama3"  # Default model to use if not specified in config
    
    # Path to the prompt template
    PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "resume_extraction.prompt"
    
    def __init__(self, provider: Any, model: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the resume analyzer with a provider and model."""
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        # Initialize the base class
        super().__init__(provider=provider, model=model, config=config)
        
        # Load the prompt template
        self.prompt_template = self._load_prompt_template()
        
        # Store the provider instance
        self._provider_instance = provider
    
    def _load_prompt_template(self) -> str:
        """Load the prompt template from file."""
        # First try to load the new CO-STAR template
        template_path = Path(__file__).parent / "templates" / "resume_analysis_prompt_v2.txt"
        
        if not template_path.exists():
            # Fall back to the legacy template if the new one doesn't exist
            template_path = Path(__file__).parent / "templates" / "resume_analysis_prompt.txt"
            self.logger.warning(
                "CO-STAR prompt template not found. Falling back to legacy template. "
                "This may result in lower quality analysis results."
            )
            
            if not template_path.exists():
                raise AnalysisError(
                    f"Could not find any prompt template at {template_path}. "
                    "Please ensure the template file exists."
                )
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template = f.read()
                
            # The template now uses double curly braces for JSON examples, so we don't need to escape them
            # Just make sure the actual template variable is properly formatted
            if '{resume_text}' not in template:
                raise AnalysisError(
                    "Invalid template: Missing required placeholder {resume_text}. "
                    "Please check the template file."
                )
                
            return template
            
        except Exception as e:
            raise AnalysisError(f"Error loading prompt template: {str(e)}") from e

    def _clean_response(self, response: str) -> Dict[str, Any]:
        """Clean and parse the LLM response."""
        if not response:
            raise AnalysisError("Empty response from LLM")
            
        # Clean the response
        cleaned = response.strip()
        
        # Debug logging
        self.logger.debug("=== ORIGINAL RESPONSE ===")
        self.logger.debug(response)
        self.logger.debug("========================")
        
        # Try to extract JSON from markdown code blocks
        code_blocks = re.findall(r'```(?:json\n)?(.*?)```', cleaned, re.DOTALL)
        if code_blocks:
            cleaned = code_blocks[0].strip()
            self.logger.debug("=== EXTRACTED FROM MARKDOWN CODE BLOCK ===")
            self.logger.debug(cleaned)
            self.logger.debug("========================================")
        
        # Remove any thinking/planning text
        thinking_patterns = [
            r'(?i)^\s*(?:thinking|planning|analysis|reasoning):?\s*[\n\r]+',
            r'(?i)^\s*\*\*thinking\*\*:?\s*[\n\r]+',
            r'(?i)^\s*\[thinking\]\s*[\n\r]+',
            r'(?i)^\s*thought process:?\s*[\n\r]+',
            r'(?i)^\s*here(?:\'"s)? (?:is|are) (?:the )?(?:extracted|parsed|analysis|result|response|data)(?: in json)?:?\s*[\n\r]+',
        ]
        
        for pattern in thinking_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)
        
        # Clean up any remaining markdown formatting
        cleaned = re.sub(r'^\s*[#*\-]\s*', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Normalize multiple newlines
        
        # Define parsing strategies
        strategies = [
            # Strategy 1: Direct JSON parse
            ('direct_json', lambda: json.loads(cleaned)),
            
            # Strategy 2: Try with common JSON fixes
            ('fixed_json', lambda: json.loads(self._fix_common_json_issues(cleaned)) 
                      if hasattr(self, '_fix_common_json_issues') 
                      else None),
                      
            # Strategy 3: Try to extract JSON object or array
            ('extract_json', lambda: (
                json.loads(json_match.group(0)) 
                if (json_match := re.search(r'[\[\{](?:[^\[\]\{\}]|"(?:\\.|[^"\\])*"|\d+|"[^"]*"|true|false|null|\s)*[\]\}]', cleaned, re.DOTALL))
                else None
            )),
            
            # Strategy 4: Try with more aggressive fixes if available
            ('aggressive_fix', lambda: (
                json.loads(self._fix_common_json_issues(cleaned, aggressive=True))
                if (hasattr(self, '_fix_common_json_issues') and 
                    callable(getattr(self, '_fix_common_json_issues')))
                else None
            )),
            
            # Strategy 5: Try to parse as JSON with trailing comma fix
            ('trailing_comma_fix', lambda: json.loads(re.sub(r',\s*([}\]])', r'\1', cleaned)))
        ]
        
        last_error = None
        
        for name, parse_func in strategies:
            self.logger.debug(f"Trying strategy: {name}")
            try:
                result = parse_func()
                if result is not None:  # Skip strategies that returned None
                    self.logger.debug(f"Successfully parsed response using strategy: {name}")
                    return result
            except (json.JSONDecodeError, AttributeError, IndexError, KeyError) as e:
                last_error = e
                self.logger.debug(f"Strategy '{name}' failed: {str(e)}")
                continue
            except Exception as e:
                last_error = e
                self.logger.debug(f"Unexpected error in strategy '{name}': {str(e)}", exc_info=True)
                continue
        
        # If we get here, all strategies failed
        error_msg = (
            "Could not extract valid JSON from LLM response.\n"
            f"Last error: {str(last_error)}\n"
            f"Cleaned response preview: {cleaned[:1000]}\n"
            f"Original response preview: {response[:1000]}"
        )
        
        # Try to include more context in the error message
        if "```" in response:
            error_msg += "\n\nNote: The response contains markdown code blocks. "
            error_msg += "Please ensure the LLM is configured to return raw JSON without markdown formatting."
        
        self.logger.error(error_msg)
        raise AnalysisError(error_msg)
    
    def _fix_common_json_issues(self, json_str: str, aggressive: bool = False) -> str:
        """Fix common JSON formatting issues in the LLM response."""
        if not json_str or not isinstance(json_str, str):
            raise ValueError("Input must be a non-empty string")
            
        # Make a copy to avoid modifying the original
        fixed = json_str.strip()
        
        # Remove any control characters that might break JSON parsing
        if aggressive:
            # Remove control characters except newlines and tabs
            fixed = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', fixed)
        
        # Common JSON fixes
        try:
            # Try direct parse first
            json.loads(fixed)
            return fixed
        except json.JSONDecodeError:
            pass
            
        # Fix common issues
        try:
            # Fix 1: Replace single quotes with double quotes for JSON keys
            fixed = re.sub(r"(?<![\\\w])'(.*?)'(?![\\\w]):", r'"\1":', fixed)
            
            # Fix 2: Fix escaped quotes within strings
            fixed = re.sub(r'(?<!\\)\"', '"', fixed)
            
            # Fix 3: Fix missing quotes around property names
            fixed = re.sub(r'([\{\,]\s*)([a-zA-Z0-9_]+)(\s*:)' , r'\1"\2"\3', fixed)
            
            # Fix 4: Fix trailing commas
            fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
            
            # Try parsing again
            json.loads(fixed)
            return fixed
            
        except json.JSONDecodeError as e:
            self.logger.debug(f"Could not fix JSON: {str(e)}")
            raise ValueError(f"Could not fix JSON: {str(e)}") from e
