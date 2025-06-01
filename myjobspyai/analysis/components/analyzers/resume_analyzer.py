"""
Resume analyzer for extracting structured data from resumes using LLMs.

This module provides the ResumeAnalyzer class which handles the extraction of
structured data from resume text using language models. It includes robust
error handling, retry logic, and response validation.
"""

import json
import logging
import re
import asyncio
from typing import Dict, Any, Optional, Type, TypeVar, List, Tuple
from pathlib import Path
from enum import Enum

from pydantic import BaseModel, ValidationError

from myjobspyai.analysis.components.base_analyzer import BaseAnalyzer
from myjobspyai.analysis.exceptions import AnalysisError
from myjobspyai.analysis.utils.logging_utils import log_execution_time

# Import the Pydantic models
from myjobspyai.analysis.components.analyzers.schemas.models import Resume

# Type variable for Pydantic models
T = TypeVar('T', bound=BaseModel)

class AnalysisType(str, Enum):
    """Supported analysis types."""
    RESUME = "resume"
    JOB = "job"
    SUITABILITY = "suitability"

logger = logging.getLogger(__name__)

class ResumeAnalyzer(BaseAnalyzer):
    """Analyzes resumes and extracts structured data using LLMs.
    
    This analyzer processes resume text and extracts structured information
    such as contact details, skills, experience, and education.
    """
    
    # Default model to use if none specified
    DEFAULT_MODEL = 'llama3'
    
    # Required fields for resume validation
    REQUIRED_FIELDS = [
        'contact_info',
        'summary',
        'skills',
        'experience',
        'education',
        'certifications',
        'languages',
        'projects'
    ]
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the ResumeAnalyzer with configuration.
        
        Args:
            config: Configuration dictionary with the following keys:
                - model: The model to use for analysis (default: 'llama3')
                - provider: The provider to use for analysis
                - templates_dir: Base directory containing prompt templates
                - schemas_dir: Base directory containing schemas
        """
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        self.model = config.get('model', self.DEFAULT_MODEL)
        
        # Initialize cache for templates and schemas
        self._templates_cache: Dict[str, str] = {}
        self._schemas_cache: Dict[str, Type[BaseModel]] = {}
        
        # Initialize provider
        self.provider = self._initialize_provider()
        
        # Store template and schema directories from config
        self._templates_dir = Path(config.get('templates_dir', Path(__file__).parent / 'templates'))
        self._schemas_dir = Path(config.get('schemas_dir', Path(__file__).parent / 'schemas'))

    def _load_template(self, analysis_type: AnalysisType) -> str:
        """Load the appropriate template for the given analysis type.
        
        Args:
            analysis_type: Type of analysis to perform
            
        Returns:
            The template content as a string
            
        Raises:
            AnalysisError: If template cannot be loaded
        """
        if analysis_type.value in self._templates_cache:
            return self._templates_cache[analysis_type.value]
            
        template_path = Path(__file__).parent / f'templates/{analysis_type.value}'
        template_file = template_path / f"{analysis_type.value}_extraction.prompt"
        
        if not template_file.exists():
            # Fall back to root templates directory
            template_file = template_path.parent / f"{analysis_type.value}_extraction.prompt"
            
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                template = f.read()
                self._templates_cache[analysis_type.value] = template
                self.logger.debug(f"Loaded template: {template_file}")
                return template
        except Exception as e:
            error_msg = f"Error loading template {template_file}: {str(e)}"
            self.logger.error(error_msg)
            raise AnalysisError(error_msg) from e

    def _get_schema_model(self, analysis_type: AnalysisType) -> Type[BaseModel]:
        """Get the Pydantic model for the given analysis type.
        
        Args:
            analysis_type: Type of analysis to perform
            
        Returns:
            A Pydantic model class for the given analysis type
            
        Raises:
            AnalysisError: If the model cannot be found or is invalid
        """
        if analysis_type == AnalysisType.RESUME:
            return Resume
        elif analysis_type == AnalysisType.JOB:
            # Import job model here to avoid circular imports
            from myjobspyai.analysis.components.analyzers.schemas.job_models import JobPosting
            return JobPosting
        elif analysis_type == AnalysisType.SUITABILITY:
            # Import suitability model here to avoid circular imports
            from myjobspyai.analysis.components.analyzers.schemas.suitability_models import SuitabilityAnalysis
            return SuitabilityAnalysis
        else:
            raise AnalysisError(f"Unsupported analysis type: {analysis_type}")

    @log_execution_time
    async def analyze(
        self, 
        input_text: str,
        analysis_type: AnalysisType = AnalysisType.RESUME,
        generation_params: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        **kwargs
    ) -> BaseModel:
        """Analyze input text and extract structured data based on the specified type.
        
        Args:
            input_text: Raw text content to analyze
            analysis_type: Type of analysis to perform (resume, job, or suitability)
            generation_params: Additional parameters for the LLM generation
            max_retries: Maximum number of retry attempts
            **kwargs: Additional keyword arguments
            
        Returns:
            A Pydantic model instance containing the extracted data
            
        Raises:
            AnalysisError: If analysis fails after max_retries
        """
        if not input_text or not isinstance(input_text, str):
            raise ValueError("input_text must be a non-empty string")
            
        # Get the appropriate model and template
        model = self._get_schema_model(analysis_type)
        template = self._load_template(analysis_type)
        
        # Prepare the prompt with the input text
        prompt = template.replace("*** Paste the entire content here ***", input_text)
        
        # Set up generation parameters
        params = {
            'temperature': 0.2,  # Lower temperature for more deterministic output
            'max_tokens': 4000,  # Increased token limit for detailed responses
            'top_p': 0.95,
            'frequency_penalty': 0.2,
            'presence_penalty': 0.1,
            'format': 'json',  # Force JSON output
            'schema': model.model_json_schema()  # Pass the JSON schema
        }
        
        # Update with any provided generation parameters
        if generation_params:
            params.update(generation_params)
        
        # Make the API call with retries
        last_error = None
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Analyzing {analysis_type.value} (attempt {attempt + 1}/{max_retries})")
                
                # Call the provider with the prompt and parameters
                response = await self.provider.generate(
                    prompt=prompt,
                    **params
                )
                
                # Parse and validate the response using the Pydantic model
                result = None
                if isinstance(response, str):
                    result = model.model_validate_json(response)
                elif hasattr(response, 'message') and hasattr(response.message, 'content'):
                    result = model.model_validate_json(response.message.content)
                elif hasattr(response, 'content'):
                    result = model.model_validate_json(response.content)
                else:
                    # Try to convert the response to a string and parse it
                    try:
                        result = model.model_validate_json(json.dumps(response))
                    except (TypeError, json.JSONDecodeError) as e:
                        self.logger.error(f"Failed to parse response: {str(e)}")
                        raise AnalysisError(f"Failed to parse response: {str(e)}") from e
                
                # Ensure we have a result
                if result is None:
                    raise AnalysisError("Failed to parse response: No valid data found")
                
                return result
                
            except json.JSONDecodeError as e:
                error_msg = f"Failed to decode JSON response: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                raise AnalysisError(error_msg) from e
                
            except ValidationError as e:
                error_msg = f"Response validation failed: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                raise AnalysisError(error_msg) from e
                
            except Exception as e:
                error_msg = f"Unexpected error during analysis: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                raise AnalysisError(error_msg) from e
                
        # If we get here, all retries failed
        error_msg = f"Failed to analyze {analysis_type.value} after {max_retries} attempts"
        if last_error:
            error_msg += f": {str(last_error)}"
        self.logger.error(error_msg)
        raise AnalysisError(error_msg) from last_error
    
    # Backward compatibility
    async def analyze_resume(self, resume_text: str, **kwargs) -> Dict[str, Any]:
        """Analyze a resume and extract structured data (legacy method)."""
        result = await self.analyze(resume_text, AnalysisType.RESUME, **kwargs)
        return result.model_dump() if hasattr(result, 'model_dump') else result.dict()

    def _process_response(self, response: str) -> Dict[str, Any]:
        """Process the raw response from the LLM into structured data.
        
        This method handles the entire pipeline of extracting, validating, and 
        normalizing the JSON response from the LLM.
        
        Args:
            response: The raw response text from the LLM
            
        Returns:
            Dict containing the processed resume data
            
        Raises:
            AnalysisError: If the response cannot be processed or validated
        """
        try:
            # Extract JSON from the response text
            json_str = self._extract_json_from_text(response)
            
            # Fix common JSON issues
            json_str = self._fix_common_json_issues(json_str)
            
            # Parse the JSON
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON: {json_str}")
                raise AnalysisError(f"Invalid JSON in response: {str(e)}") from e
            
            # Validate required fields
            self._validate_resume_data(data)
            
            # Log a summary of the extracted data for debugging
            summary = {}
            for field in self.REQUIRED_FIELDS:
                if field in data:
                    if isinstance(data[field], (list, dict)):
                        summary[field] = f"{len(data[field])} items"
                    else:
                        summary[field] = str(data[field])[:50] + ("..." if len(str(data[field])) > 50 else "")
            
            self.logger.info("Extracted resume data: %s", summary)
            return data
            
        except AnalysisError:
            # Re-raise AnalysisError directly
            raise
            
        except Exception as e:
            error_msg = f"Unexpected error processing response: {str(e)}"
            self.logger.error("%s\nResponse (first 500 chars): %s", 
                           error_msg, response[:500], exc_info=True)
            raise AnalysisError(error_msg) from e
    
    def _extract_json_from_text(self, text: str) -> str:
        """Extract JSON from the response text.
        
        This method handles various formats of JSON in the response text,
        including handling markdown code blocks, malformed JSON, and other common response formats.
        
        Args:
            text: Raw text response that may contain JSON
            
        Returns:
            Extracted and cleaned JSON string
            
        Raises:
            AnalysisError: If no valid JSON can be extracted after all attempts
        """
        if not text or not text.strip():
            raise AnalysisError("Empty or invalid input text provided")

        self.logger.debug("Attempting to extract JSON from text (length: %d)", len(text))

        # Common patterns to extract JSON from text, in order of specificity
        patterns = [
            # 1. Pattern for markdown code blocks with json language specifier
            (r'```(?:json\n)?([\s\S]*?)```', "Markdown code block with JSON"),
            # 2. Pattern for markdown code blocks without language specifier
            (r'```([\s\S]*?)```', "Markdown code block"),
            # 3. Pattern for JSON object with potential leading/trailing text
            (r'[\s\S]*?({[\s\S]*})[\s\S]*', "JSON object with surrounding text"),
            # 4. Pattern for JSON array with potential leading/trailing text
            (r'[\s\S]*(\[[\s\S]*\])[\s\S]*', "JSON array with surrounding text"),
            # 5. Last resort: try to extract any content that looks like JSON
            (r'([\s\S]+)', "Any content")
        ]

        for pattern_idx, (pattern, pattern_desc) in enumerate(patterns, 1):
            self.logger.debug("Attempt %d: Trying pattern '%s' (%s)", 
                           pattern_idx, pattern_desc, pattern)

            try:
                match = re.search(pattern, text, re.DOTALL)
                if not match:
                    self.logger.debug("Pattern %d did not match", pattern_idx)
                    continue

                # Get the first non-None group (should be the JSON content)
                json_str = next((g for g in match.groups() if g is not None), '').strip()
                if not json_str:
                    self.logger.debug("Pattern %d matched but no content captured", pattern_idx)
                    continue

                self.logger.debug("Pattern %d matched. Extracted content (first 200 chars): %s", 
                               pattern_idx, json_str[:200])

                # Try to fix common JSON issues
                try:
                    fixed_json = self._fix_common_json_issues(json_str)

                    # Validate the fixed JSON by attempting to parse it
                    parsed = json.loads(fixed_json)
                    self.logger.debug("Successfully parsed JSON after fixing (type: %s)", 
                                     type(parsed).__name__)

                    # If we got this far, return the fixed JSON
                    return fixed_json

                except (json.JSONDecodeError, TypeError) as e:
                    self.logger.debug("Failed to fix/validate JSON (attempt %d): %s", 
                                     pattern_idx, str(e))
                    continue

            except Exception:  # noqa: B902 - We're logging the error and continuing
                self.logger.debug("Error applying pattern %d", pattern_idx, exc_info=True)
                continue

        # If we get here, no valid JSON was found after all attempts
        error_msg = ("Could not extract valid JSON from response after multiple attempts. "
                    "Response start (200 chars): {}").format(text[:200])
        self.logger.error(error_msg)
        raise AnalysisError(error_msg)

    def _fix_common_json_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues in the LLM response.

        Args:
            json_str: The JSON string to fix

        Returns:
            The fixed JSON string
        """
        if not json_str or not json_str.strip():
            return json_str

        fixed = json_str.strip()

        # Fix 1: Remove any text before the first { or [
        fixed = re.sub(r'^[^{\[]*([{\[])', r'\1', fixed, flags=re.DOTALL)

        # Fix 2: Remove any text after the last } or ]
        fixed = re.sub(r'([}\]])[^}\]]*$', r'\1', fixed, flags=re.DOTALL)

        # Fix 3: Replace single quotes with double quotes for JSON keys
        fixed = re.sub(r'(?<!\\)\'(.*?)\'(?!\\):', r'"\1":', fixed)

        # Fix 4: Handle unescaped quotes within strings
        fixed = re.sub(r'(?<!\\)"(.*?)(?<!\\)"', 
                     lambda m: '"' + m.group(1).replace('"', '\\"').replace('\\', '\\\\') + '"', 
                     fixed)

        # Fix 5: Remove trailing commas before closing brackets/braces
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)

        # Fix 6: Add missing commas between array/object elements
        fixed = re.sub(r'([}"\w])\s*(?={\s*"|"|\[)', r'\1,', fixed)

        # Fix 7: Fix unescaped control characters
        fixed = fixed.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')

        # Fix 8: Handle non-ASCII characters
        try:
            fixed = fixed.encode('utf-8', 'ignore').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass

        # Fix 9: Handle boolean values (true/false) that might be in quotes
        fixed = re.sub(r'"(true|false)"', r'\1', fixed, flags=re.IGNORECASE)

        # Fix 10: Handle null values that might be in quotes or have inconsistent casing
        fixed = re.sub(r'"(null|NULL|Null)"', r'null', fixed)

        # Fix 11: Remove any remaining control characters that might break JSON parsing
        fixed = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', fixed)

        # Fix 12: Ensure proper escaping of forward slashes
        fixed = fixed.replace('/', '\\/')

        # Fix 13: Fix missing or extra quotes around property names
        fixed = re.sub(r'(?<![\[\{, ])(\w+)(?=\s*:)', r'"\1"', fixed)

        # Fix 14: Handle missing colons between keys and values
        fixed = re.sub(r'"(\w+)"\s*([^\s{\[\]},:])', r'"\1": \2', fixed)

        # Fix 15: Ensure commas between array elements
        fixed = re.sub(r'([}"\w])(\s*[{\[\"])', r'\1,\2', fixed)

        
        # Fix 16: Fix any remaining unescaped quotes in the middle of strings
        fixed = re.sub(r'(?<!\\)"(.*?)(?<!\\)"', 
                     lambda m: '"' + m.group(1).replace('"', '\\"').replace('\\', '\\\\') + '"', 
                     fixed)
        
        # Fix 17: Remove any remaining control characters that might break JSON parsing
        fixed = re.sub(r'[\x00-\x1F\x7F-\x9F]', ' ', fixed)
        
        # Fix 18: Ensure proper escaping of backslashes
        fixed = fixed.replace('\\', '\\\\')
        
        # Fix 19: Handle any remaining unescaped quotes
        fixed = fixed.replace('"', '\\"')
        
        # Fix 20: If the string doesn't start with { or [, try to find and extract a JSON object/array
        if not re.match(r'^\s*[{\[]', fixed):
            match = re.search(r'[{\[](?:[^{}\[\]]|\{[^{}]*\}|\[[^\[\]]*\])*[}\]]', fixed)
            if match:
                fixed = match.group(0)
        
        return fixed
    
    def _fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding issues in the response text.
        
        Args:
            text: The text to fix
            
        Returns:
            The fixed text
        """
        # Replace common encoding issues
        replacements = {
            '\u201c': '"',  # Left double quote
            '\u201d': '"',  # Right double quote
            '\u2018': "'",   # Left single quote
            '\u2019': "'",   # Right single quote
            '\u2013': '-',   # En dash
            '\u2014': '--',  # Em dash
            '\u2026': '...'  # Ellipsis
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
            
        return text