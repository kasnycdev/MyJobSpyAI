"""
Resume analysis module for extracting and processing resume information.
"""
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar

from pydantic import BaseModel, Field, HttpUrl

from .base_analyzer import BaseAnalyzer, AnalysisError

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

class ResumeAnalyzer(BaseAnalyzer[ResumeData]):
    """Analyzer for extracting structured data from resumes."""
    
    # Define the response model for this analyzer
    RESPONSE_MODEL = ResumeData
    
    # Required fields that must be present in the response
    REQUIRED_FIELDS = ["contact_info", "skills", "experience", "education"]
    
    # Default model name
    DEFAULT_MODEL = "llama3"  # Default model to use if not specified in config
    
    # Path to the prompt template
    PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "resume_extraction.prompt"
    
    def _load_prompt_template(self) -> str:
        """Load the prompt template from file.
        
        Returns:
            str: The prompt template content
            
        Raises:
            AnalysisError: If the prompt template file is not found or cannot be read
        """
        try:
            if not self.PROMPT_TEMPLATE_PATH.exists():
                raise FileNotFoundError(f"Prompt template not found at {self.PROMPT_TEMPLATE_PATH}")
                
            with open(self.PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
                return f.read()
                
        except Exception as e:
            error_msg = f"Failed to load prompt template: {str(e)}"
            self.logger.error(error_msg)
            raise AnalysisError(error_msg) from e
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the resume analyzer.
        
        Args:
            config: Optional configuration dictionary. Can include:
                - model: Name of the model to use
                - provider: Name of the LLM provider
                - Any other provider-specific configuration
        """
        # Initialize with default config
        config = config or {}
        
        # Make a copy of the config to avoid modifying the input
        config = dict(config)
        
        # Set model name from config or use default
        model_name = config.get('model')
        if not model_name:
            # If no model in config, use the default but don't modify the config
            self.model_name = self.DEFAULT_MODEL
        else:
            # Use the configured model name
            self.model_name = model_name
            # Ensure the model is set in the config for the base class
            config['model'] = model_name
        
        logger.info(f"Initializing ResumeAnalyzer with model: {self.model_name}")
        
        # Initialize the base analyzer with the config
        super().__init__(config)
        
        # Load the prompt template
        self._load_prompt_template()
        
        logger.debug(f"Initialized ResumeAnalyzer with model: {self.model_name}")
    
    def _load_prompt_template(self) -> str:
        """Load the prompt template from file.
        
        Returns:
            str: The loaded prompt template or default template if loading fails
            
        Raises:
            AnalysisError: If template file is not found and no default is available
        """
        try:
            with open(self.PROMPT_TEMPLATE_PATH, 'r', encoding='utf-8') as f:
                self.prompt_template = f.read()
                return self.prompt_template
        except Exception as e:
            self.logger.error(f"Failed to load prompt template: {e}")
            self.prompt_template = self._get_default_prompt()
            return self.prompt_template
    
    def _get_default_prompt(self) -> str:
        """Return a default prompt template if file loading fails."""
        return """Extract the following information from the resume and return ONLY a valid JSON object.

Required fields:
- contact_info: Object with name, email, phone, location
- summary: String with professional summary
- skills: List of technical/professional skills
- experience: List of job experiences with company, title, dates, and description
- education: List of education entries with degree, institution, and dates
- certifications: List of certifications
- languages: List of languages spoken with proficiency
- projects: List of projects with name, description, and technologies

Example response format:
```json
{
  "contact_info": {
    "name": "John Doe",
    "email": "john@example.com",
    "phone": "+1-555-555-5555",
    "location": "New York, NY"
  },
  "summary": "Experienced software engineer...",
  "skills": ["Python", "JavaScript", "Docker"],
  "experience": [
    {
      "company": "Tech Corp",
      "title": "Senior Software Engineer",
      "dates": "2020-01 - Present",
      "description": "Led development of..."
    }
  ],
  "education": [
    {
      "degree": "BS in Computer Science",
      "institution": "MIT",
      "dates": "2016-2020"
    }
  ],
  "certifications": ["AWS Certified Developer"],
  "languages": [{"name": "English", "proficiency": "Native"}],
  "projects": [
    {
      "name": "Project X",
      "description": "Description of project...",
      "technologies": ["Python", "FastAPI", "React"]
    }
  ]
}
```

Resume:
{resume_text}

IMPORTANT: Return ONLY the JSON object, without any other text or markdown formatting."""
    
    async def analyze(self, resume_text: str, max_retries: int = 3) -> ResumeData:
        """Analyze resume text and extract structured data.
        
        Args:
            resume_text: Raw text content of the resume
            max_retries: Maximum number of retry attempts for failed parsing
            
        Returns:
            ResumeData object containing structured resume information
            
        Raises:
            AnalysisError: If there's an error during analysis after all retries
        """
        # Ensure prompt template is loaded
        if not hasattr(self, 'prompt_template') or not self.prompt_template:
            self._load_prompt_template()
            
        if not hasattr(self, 'prompt_template') or not self.prompt_template:
            raise AnalysisError("Failed to load or generate prompt template")
            
        # Try different prompt variations if needed
        prompts = [
            self.prompt_template + "\n\nIMPORTANT: Respond ONLY with a valid JSON object that matches the schema above. Do not include any other text, explanations, or markdown formatting.",
            self.prompt_template + "\n\nCRITICAL: Your response MUST be a valid JSON object that strictly follows the schema. Do not include any markdown formatting, backticks, or additional text outside the JSON.",
            self.prompt_template + "\n\nWARNING: Your response must be a valid JSON object. The entire response must be parseable as JSON with no additional text before or after the JSON object."
        ]
        
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Use a different prompt variation for each retry
                prompt_variation = prompts[min(attempt, len(prompts) - 1)]
                formatted_prompt = prompt_variation.format(resume_text=resume_text)
                
                # Get the LLM response using the base class's _call_llm method
                self.logger.debug("Sending prompt to LLM...")
                response = await self._call_llm(
                    prompt=formatted_prompt,
                    temperature=0.1,  # Use lower temperature for more deterministic output
                    max_tokens=2000,  # Sufficient for resume data
                    stop=None
                )
                
                # Log the raw response for debugging
                self.logger.debug(f"Raw LLM response: {response[:500]}..." if len(response) > 500 else f"Raw LLM response: {response}")
                
                # Clean up the response
                cleaned_response = response.strip()
                
                # Helper function to find JSON in text
                def extract_json(text):
                    # Try to find JSON object
                    try:
                        # Look for the first { and last }
                        start = text.find('{')
                        end = text.rfind('}')
                        if start >= 0 and end > start:
                            json_str = text[start:end+1]
                            # Try to parse it to validate
                            import json
                            json.loads(json_str)  # Will raise ValueError if not valid JSON
                            return json_str
                    except (ValueError, json.JSONDecodeError):
                        pass
                    return None
                
                # Try different extraction methods
                json_str = None
                
                # Method 1: Look for ```json ... ```
                if '```json' in cleaned_response:
                    parts = cleaned_response.split('```json')
                    if len(parts) > 1:
                        json_part = '```json' + parts[1]
                        json_str = extract_json(json_part)
                
                # Method 2: Look for ``` ... ```
                if not json_str and '```' in cleaned_response:
                    parts = cleaned_response.split('```')
                    # Get the first code block that contains a {
                    for i in range(1, len(parts), 2):
                        if i < len(parts) and '{' in parts[i]:
                            json_str = extract_json(parts[i])
                            if json_str:
                                break
                
                # Method 3: Look for raw JSON
                if not json_str:
                    json_str = extract_json(cleaned_response)
                
                if not json_str:
                    raise AnalysisError("Could not extract valid JSON from LLM response")
                
                cleaned_response = json_str
                self.logger.debug(f"Extracted JSON: {cleaned_response}")
                
                # Parse and validate the response
                return await self._extract_structured_data(
                    cleaned_response,
                    response_model=self.RESPONSE_MODEL,
                    required_fields=self.REQUIRED_FIELDS
                )
                
            except Exception as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    self.logger.info(f"Retrying analysis (attempt {attempt + 2}/{max_retries})...")
                continue
        
        # If we get here, all retries failed
        error_msg = f"Failed to analyze resume after {max_retries} attempts"
        if last_error:
            error_msg += f": {str(last_error)}"
        self.logger.error(error_msg)
        raise AnalysisError(error_msg) from last_error
