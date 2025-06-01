"""
Resume analysis module for extracting and processing resume information.
"""
import asyncio
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar

from pydantic import BaseModel, Field, HttpUrl, ValidationError

from myjobspyai.analysis.components.analyzers.base import BaseAnalyzer, AnalysisError
from myjobspyai.analysis.providers.base import BaseProvider

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

# Create a type variable for the response model
R = TypeVar('R', bound=BaseModel)

class ResumeAnalyzer(BaseAnalyzer[ResumeData, BaseProvider]):
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
        """Initialize the resume analyzer with a provider and model.
        
        Args:
            provider: The LLM provider to use for analysis
            model: The name of the model to use
            config: Optional configuration dictionary
        """
        # Initialize logger first
        self.logger = logging.getLogger(__name__)
        
        # Initialize the base class
        super().__init__(provider=provider, model=model, config=config)
        
        # Load the prompt template
        self.prompt_template = self._load_prompt_template()
        
        # Store the provider instance
        self._provider_instance = provider
        
    def _load_prompt_template(self) -> str:
        """Load the prompt template from file or return a default template if file loading fails.
        
        Returns:
            str: The prompt template string
        """
        default_template = """Extract the following information from the resume and return ONLY a valid JSON object.

The JSON object should have the following structure:
{
    "contact_info": {
        "name": "Full name",
        "email": "Email address",
        "phone": "Phone number",
        "location": "Location (city, state, country)",
        "linkedin": "LinkedIn profile URL",
        "portfolio": "Portfolio or personal website URL"
    },
    "summary": "Professional summary",
    "skills": ["Skill 1", "Skill 2", "..."],
    "experience": [
        {
            "company": "Company name",
            "title": "Job title",
            "location": "Job location",
            "start_date": "Start date (YYYY-MM-DD)",
            "end_date": "End date (YYYY-MM-DD or null if current)",
            "is_current": true,
            "description": "Job description and achievements"
        }
    ],
    "education": [
        {
            "institution": "Name of institution",
            "degree": "Degree or certification",
            "field_of_study": "Field of study",
            "start_date": "Start date (YYYY-MM-DD)",
            "end_date": "End date (YYYY-MM-DD)",
            "gpa": 4.0
        }
    ],
    "certifications": ["Certification 1", "Certification 2", "..."],
    "languages": [
        {
            "language": "Language name",
            "proficiency": "Proficiency level"
        }
    ],
    "projects": [
        {
            "name": "Project name",
            "description": "Project description",
            "technologies": ["Tech 1", "Tech 2", "..."],
            "url": "Project URL",
            "start_date": "Start date (YYYY-MM-DD)",
            "end_date": "End date (YYYY-MM-DD)",
            "contributions": ["Contribution 1", "Contribution 2", "..."]
        }
    ]
}

Resume Text:
{resume_text}

IMPORTANT: Return ONLY the JSON object, without any other text or markdown formatting."""
        
        try:
            if self.PROMPT_TEMPLATE_PATH.exists():
                with open(self.PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
                    return f.read()
            return default_template
        except Exception as e:
            self.logger.warning(f"Failed to load prompt template from {self.PROMPT_TEMPLATE_PATH}: {e}")
            return default_template

    def _validate_response(self, response: 'ResumeData') -> None:
        """Validate that the response contains all required fields.
        
        Args:
            response: The response to validate
            
        Raises:
            AnalysisError: If any required fields are missing
        """
        missing_fields = []
        for field in self.REQUIRED_FIELDS:
            if not getattr(response, field, None):
                missing_fields.append(field)
        
        if missing_fields:
            raise AnalysisError(f"Missing required fields in response: {', '.join(missing_fields)}")
            
    async def analyze(self, resume_text: str) -> Optional[ResumeData]:
        """Analyze resume text and extract structured data.
        
        This is a high-level method that processes the resume text and returns
        structured data. It handles the entire analysis pipeline including:
        - Input validation
        - Prompt generation
        - LLM interaction
        - Response cleaning and validation
        - Error handling
        
        Args:
            resume_text: The raw text content of the resume to analyze
            
        Returns:
            ResumeData object containing structured resume information, or None if analysis fails
            
        Raises:
            ValueError: If the input is invalid
            AnalysisError: If there's an error during analysis
            
        Example:
            >>> analyzer = ResumeAnalyzer(provider=provider, model="llama3")
            >>> resume_data = await analyzer.analyze("John Doe\\nSoftware Engineer...")
            >>> print(resume_data.contact_info.name)
            'John Doe'
        """
        if not resume_text or not isinstance(resume_text, str):
            raise ValueError("resume_text must be a non-empty string")
            
        try:
            # Generate the prompt using the template
            prompt = self.prompt_template.format(resume_text=resume_text)
            
            # Generate the response from the LLM
            response = await self.generate(
                prompt=prompt,
                task_name="resume_analysis"
            )
            
            if not response:
                raise AnalysisError("Empty response from LLM")
                
            # Clean the response
            cleaned_response = self._clean_response(response)
            
            # Parse the JSON response into a dictionary
            try:
                response_dict = json.loads(cleaned_response)
            except json.JSONDecodeError as je:
                raise AnalysisError(f"Failed to parse LLM response: {str(je)}") from je
                
            # Convert the dictionary to a ResumeData object
            try:
                resume_data = ResumeData(**response_dict)
            except ValidationError as ve:
                raise AnalysisError(f"Invalid data in response: {str(ve)}") from ve
                
            # Validate that all required fields are present
            self._validate_response(resume_data)
            
            return resume_data
            
        except Exception as e:
            error_msg = f"Error analyzing resume: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            if not isinstance(e, (ValueError, AnalysisError)):
                raise AnalysisError(error_msg) from e
            raise

    def _clean_response(self, response: str) -> str:
        """Clean and prepare the LLM response for JSON parsing.
        
        This method processes the raw LLM response to extract valid JSON content.
        It handles various response formats including markdown, code blocks, and
        thinking/planning tags. The method is designed to be robust against
        different LLM output formats and includes multiple fallback strategies
        to extract valid JSON.
        
        Args:
            response: The raw response string from the LLM
            
        Returns:
            str: The cleaned JSON string ready for parsing
            
        Raises:
            ValueError: If the response is empty, doesn't contain valid JSON,
                      or cannot be cleaned into a valid format
            
        Example:
            >>> cleaned = analyzer._clean_response("Here's your JSON: {\"name\": \"John\"}")
            >>> print(cleaned)
            {"name": "John"}
        """
        if not response:
            raise ValueError("Empty response from LLM")
            
        # Convert to string if not already and strip whitespace
        text = str(response).strip()
        
        # Log the raw response for debugging (truncated)
        self.logger.debug("Raw response before cleaning: %s...", text[:500])
        
        # Common patterns to remove or clean
        patterns_to_remove = [
            r'^[^{]*',                     # Everything before the first {
            r'[^}]*$',                     # Everything after the last }
            r'```(?:json)?',               # Markdown code blocks
            r'<think>.*?</think>',         # Thinking/planning tags (non-greedy match)
            r'^[\s\S]*?(?={)',             # Non-greedy match everything before first {
            r'}[\s\S]*$',                 # Everything after last }
            r'^[^\n]*\n',                  # First line if it doesn't start with {
            r'\n[^\n]*$',                   # Last line if it doesn't end with }
            r'^[\s\S]*?({[\s\S]*})[\s\S]*$'  # Extract JSON from any text
        ]
        
        # Apply patterns
        import re
        for pattern in patterns_to_remove:
            text = re.sub(pattern, '\\1' if '\\1' in pattern else '', 
                         text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove any remaining non-printable characters except newlines
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        
        # Final cleanup of any remaining whitespace
        text = text.strip()
        
        # If we still don't have valid JSON, try to extract it
        if not (text.startswith('{') and text.endswith('}')):
            # Try to find JSON object in the text
            match = re.search(r'({[^{}]*})', text, re.DOTALL)
            if match:
                text = match.group(1)
        
        # Basic validation
        if not text or not (text.startswith('{') and text.endswith('}')):
            raise ValueError(f"Response does not contain valid JSON. Content: {text[:200]}...")
            
        # Try to validate JSON structure
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError as json_err:
            # If we get here, the JSON is still not valid
            # Try to find the largest valid JSON object
            stack = []
            start = text.find('{')
            if start == -1:
                raise ValueError("No JSON object found in response") from json_err
                
            max_len = 0
            best_json = None
            
            for i in range(start, len(text)):
                if text[i] == '{':
                    stack.append(i)
                elif text[i] == '}':
                    if stack:
                        start_idx = stack.pop()
                        if not stack:  # Found matching pair
                            potential_json = text[start_idx:i+1]
                            try:
                                json.loads(potential_json)
                                if len(potential_json) > max_len:
                                    max_len = len(potential_json)
                                    best_json = potential_json
                            except json.JSONDecodeError:
                                continue
            
            if best_json:
                self.logger.debug("Extracted valid JSON after recovery")
                return best_json
                
            # If we couldn't find valid JSON, try one last time with the original text
            try:
                json.loads(text)
                return text
            except json.JSONDecodeError as final_err:
                self.logger.error("Failed to parse LLM response as JSON: %s", str(final_err))
                raise ValueError(f"Failed to parse LLM response as JSON: {str(final_err)}") from final_err
