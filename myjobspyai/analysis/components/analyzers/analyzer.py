"""Resume and job analysis using LLM providers with semantic chunking.

This module provides analyzers for processing resumes and job descriptions
using configured LLM providers with support for semantic chunking of large texts.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.documents import Document  # noqa: F401
from .base import BaseAnalyzer
from ...factory import get_factory
from ...components.models.models import ResumeData
from ....rag.text_processor import TextProcessor
from ....config import settings

class DateTimeEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles date and datetime objects."""
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_CHUNK_SIZE = 3000
DEFAULT_CHUNK_OVERLAP = 200
RESUME_EXTRACTION_PROMPT_PATH = os.path.join(
    os.path.dirname(__file__), 'prompts', 'resume_extraction.prompt'
)

def load_prompt_template(file_path: str) -> str:
    """Load a prompt template from a file.
    
    Args:
        file_path: Path to the prompt template file
        
    Returns:
        The content of the prompt template
        
    Raises:
        FileNotFoundError: If the prompt template file is not found
        IOError: If there's an error reading the file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Prompt template not found at {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading prompt template: {str(e)}")
        raise

class JobAnalyzer(BaseAnalyzer):
    """Analyzes job descriptions and matches them against resumes.
    
    This analyzer extracts structured information from job descriptions
    and evaluates candidate suitability based on resume data.
    """
    
    def __init__(
        self,
        provider: Optional[Any] = None,  # Can be a provider instance or string
        model: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the job analyzer.
        
        Args:
            provider: Either a pre-initialized provider instance or a string with the provider name.
                     If None, uses the default from settings.
            model: The model to use. If None, uses the default from settings.
            config: Additional configuration overrides.
        """
        # Initialize with provider and model from settings if not provided
        if provider is None:
            provider = settings.llm.provider
        if model is None:
            model = settings.llm.model
            
        super().__init__(provider=provider, model=model, config=config)
        
        # If we got a pre-initialized provider, ensure it's properly set up
        if self._provider_instance is not None:
            logger.info(f"Using pre-initialized provider: {self._provider_instance.__class__.__name__}")
            if model and hasattr(self._provider_instance, 'model'):
                self._provider_instance.model = model
        
        # Load prompt templates from config or fall back to defaults
        self.job_extraction_prompt = self._load_prompt_from_config('job_extraction')
        self.suitability_analysis_prompt = self._load_prompt_from_config('suitability_analysis')
        
        # Configure text processing
        self.chunk_size = getattr(settings.analysis, 'chunk_size', DEFAULT_CHUNK_SIZE)
        self.chunk_overlap = getattr(settings.analysis, 'chunk_overlap', DEFAULT_CHUNK_OVERLAP)
    
    def _load_prompt_from_config(self, prompt_type: str) -> str:
        """Load a prompt template from the configuration or fall back to default.
        
        Args:
            prompt_type: Type of prompt to load (e.g., 'job_extraction', 'suitability_analysis')
            
        Returns:
            str: The loaded prompt template
            
        Raises:
            FileNotFoundError: If the prompt template file is not found
            IOError: If there's an error reading the file
        """
        try:
            # First try to get the path from config
            config_path = getattr(settings.analysis.prompts, prompt_type, None)
            
            if config_path:
                # If we have a config path, use it
                prompt_path = Path(config_path)
                if not prompt_path.is_absolute():
                    # If path is relative, make it relative to the project root
                    project_root = Path(__file__).parent.parent.parent.parent.parent
                    prompt_path = (project_root / config_path).resolve()
                
                logger.debug(f"Loading {prompt_type} prompt from config: {prompt_path}")
            else:
                # Fall back to default location
                prompt_path = Path(__file__).parent / 'templates' / f"{prompt_type}.prompt"
                logger.debug(f"Using default prompt path: {prompt_path}")
            
            # Ensure the file exists
            if not prompt_path.exists():
                raise FileNotFoundError(
                    f"Prompt template not found at: {prompt_path}. "
                    f"Please ensure the file exists or update the configuration."
                )
                
            # Load the prompt content
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
                
        except Exception as e:
            logger.error(f"Error loading {prompt_type} prompt: {str(e)}")
            raise
            
    def log_llm_call_summary(self) -> None:
        """Log a summary of LLM calls for debugging and monitoring."""
        if hasattr(self, '_llm_call_stats') and self._llm_call_stats:
            total_calls = sum(self._llm_call_stats.values())
            logger.info(f"LLM Call Summary - Total calls: {total_calls}")
            for call_type, count in self._llm_call_stats.items():
                logger.info(f"  - {call_type}: {count} calls")
        else:
            logger.debug("No LLM call statistics available")
    
    async def extract_job_details_async(self, job_description: Any, job_title: str = "") -> Dict[str, Any]:
        """Extract structured details from a job description.
        
        Args:
            job_description: The job description text to analyze (can be str, float, or other types)
            job_title: Optional job title for logging purposes
            
        Returns:
            Dictionary containing structured job details
        """
        try:
            # Ensure job_description is a string and handle various input types
            if job_description is None:
                job_description_str = ""
            elif isinstance(job_description, str):
                job_description_str = job_description
            elif isinstance(job_description, (int, float, bool)):
                job_description_str = str(job_description)
            elif hasattr(job_description, 'model_dump'):  # Pydantic v2
                job_description_str = str(job_description.model_dump())
            elif hasattr(job_description, 'dict'):  # Pydantic v1
                job_description_str = str(job_description.dict())
            else:
                # Try string conversion as last resort
                try:
                    job_description_str = str(job_description)
                except Exception as e:
                    logger.warning(f"Failed to convert job description to string: {e}")
                    job_description_str = ""
            
            # Log the type of job description for debugging
            if job_description is not None:
                logger.debug(f"Processing job description of type: {type(job_description).__name__}")
            
            # Format the prompt with the job description
            try:
                prompt = self.job_extraction_prompt.replace(
                    '{{ job_description }}', 
                    job_description_str
                )
            except (AttributeError, TypeError) as e:
                logger.error(f"Error formatting prompt: {e}")
                raise ValueError("Failed to format job description into prompt") from e
                # For other types, try to convert to string, but be more graceful
                try:
                    job_description_str = str(job_description)
                except Exception as e:
                    logger.warning(f"Failed to convert job description (type: {type(job_description).__name__}) to string: {str(e)}")
                    job_description_str = ""
            
            # Log the type of job description for debugging
            if job_description is not None:
                logger.debug(f"Processing job description of type: {type(job_description).__name__}")
            
            # Format the prompt with the job description
            try:
                prompt = self.job_extraction_prompt.replace(
                    '{{ job_description }}', 
                    job_description_str
                )
            except (AttributeError, TypeError) as e:
                logger.error(f"Error formatting prompt: {e}")
                raise ValueError("Failed to format job description into prompt") from e
            
            # Call the LLM to extract job details
            response = await self.generate(
                prompt=prompt,
                task_name="extract_job_details",
                temperature=0.2,  # Lower temperature for more consistent output
                max_tokens=2000,
            )
            
            # Parse the response (assuming it's a JSON string)
            try:
                if not response or not response.strip():
                    logger.warning("Empty response from LLM")
                    return {}
                    
                # Clean the response if it contains markdown code blocks
                response_text = response.strip()
                if '```json' in response_text:
                    # Extract JSON from markdown code block
                    json_str = response_text.split('```json')[1].split('```')[0].strip()
                elif '```' in response_text:
                    # Handle case where language isn't specified
                    json_str = response_text.split('```')[1].split('```')[0].strip()
                else:
                    json_str = response_text
                    
                result = json.loads(json_str)
                # Ensure we have at least the required fields
                result.setdefault('title', job_title)
                result.setdefault('skills', [])
                result.setdefault('requirements', [])
                result.setdefault('responsibilities', [])
                result.setdefault('extracted_details', {})
                return result
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse job details as JSON: {response}")
                return {
                    "title": job_title,
                    "skills": [],
                    "requirements": [],
                    "responsibilities": [],
                    "extracted_details": {},
                    "error": "Failed to parse job details"
                }
                
        except Exception as e:
            logger.error(f"Error extracting job details: {str(e)}", exc_info=True)
            return {
                "title": job_title,
                "skills": [],
                "requirements": [],
                "responsibilities": [],
                "extracted_details": {},
                "error": str(e)
            }
    
    async def analyze_resume_suitability(
        self, 
        resume_data: ResumeData, 
        job_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze how well a resume matches a job description.
        
        Args:
            resume_data: The structured resume data
            job_dict: The job dictionary containing job details
            
        Returns:
            Dictionary containing match analysis results
        """
        # Initialize response variable at the beginning to ensure it's always defined
        response = None
        
        try:
            # Log input data for debugging
            logger.debug("Starting resume suitability analysis")
            
            # Convert resume_data to dict if it's a Pydantic model
            try:
                # Handle both Pydantic v1 and v2
                if hasattr(resume_data, 'model_dump'):  # Pydantic v2
                    resume_dict = resume_data.model_dump()
                elif hasattr(resume_data, 'dict'):  # Pydantic v1
                    resume_dict = resume_data.dict()
                else:
                    resume_dict = dict(resume_data) if hasattr(resume_data, '__dict__') else {}
                
                resume_json = json.dumps(
                    resume_dict, 
                    indent=2, 
                    cls=DateTimeEncoder
                )
                job_json = json.dumps(
                    job_dict, 
                    indent=2, 
                    cls=DateTimeEncoder
                )
            except Exception as e:
                error_msg = f"Error serializing data for analysis: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg) from e
            
            # Format the prompt with the resume and job data
            try:
                prompt = self.suitability_analysis_prompt\
                    .replace('{{ resume_data_json }}', resume_json)\
                    .replace('{{ job_data_json }}', job_json)
            except Exception as e:
                error_msg = f"Error formatting prompt template: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg) from e
            
            logger.debug("Sending prompt to LLM for analysis")
            # Call the LLM to analyze suitability
            try:
                response = await self.generate(
                    prompt=prompt,
                    task_name="analyze_resume_suitability",
                    temperature=0.3,  # Slightly higher temperature for analysis
                    max_tokens=2500,
                )
                logger.debug(f"Received response from LLM: {response[:200]}...")  # Log first 200 chars
            except Exception as e:
                error_msg = f"Error generating analysis from LLM: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg) from e
            
            # Parse the response (trying multiple formats)
            result = None
            analysis = {}
            
            # Clean up the response
            response = response.strip()
            
            # Try to extract JSON from markdown code blocks
            if '```json' in response and '```' in response:
                try:
                    json_str = response.split('```json')[1].split('```')[0].strip()
                    result = json.loads(json_str)
                    logger.debug("Successfully extracted JSON from markdown code block")
                except (IndexError, json.JSONDecodeError) as e:
                    logger.debug(f"Failed to extract JSON from markdown code block: {str(e)}, trying full response")
                    pass
            
            # If no result yet, try to parse the entire response as JSON
            if result is None:
                try:
                    result = json.loads(response)
                    logger.debug("Successfully parsed response as raw JSON")
                except json.JSONDecodeError:
                    logger.debug("Response is not valid JSON, treating as plain text")
                    # If we can't parse as JSON, create a default analysis with the text
                    result = {
                        'analysis': {
                            'justification': response,
                            'suitability_score': 0.5,  # Default neutral score
                            'pros': ['Analysis available in raw_analysis'],
                            'cons': ['Could not parse detailed analysis'],
                            'summary': 'Analysis available in raw_analysis'
                        }
                    }
            
            # Log successful parsing
            logger.debug("Processing analysis result")
            
            # Ensure we have at least the required fields
            analysis = result.get('analysis', result)  # Handle both nested and flat structures
            
            # Safely extract all fields with defaults
            try:
                match_score = float(analysis.get('suitability_score', 0.0)) if analysis else 0.0
            except (TypeError, ValueError):
                match_score = 0.5  # Default to neutral score if can't parse
                
            matching_skills = analysis.get('pros', [])
            if not isinstance(matching_skills, list):
                matching_skills = [str(matching_skills)] if matching_skills else []
                
            missing_skills = analysis.get('cons', [])
            if not isinstance(missing_skills, list):
                missing_skills = [str(missing_skills)] if missing_skills else []
            
            # Get the analysis text, defaulting to the raw response if not found
            analysis_text = analysis.get('justification', '')
            if not analysis_text and isinstance(analysis, str):
                analysis_text = analysis
            
            result = {
                "match_score": match_score,
                "matching_skills": matching_skills,
                "missing_skills": missing_skills,
                "analysis": analysis_text,
                "skill_match_summary": analysis.get('skill_match_summary', analysis.get('summary', '')),
                "experience_match_summary": analysis.get('experience_match_summary', ''),
                "education_match_summary": analysis.get('education_match_summary', ''),
                "missing_keywords": analysis.get('missing_keywords', []),
                "raw_analysis": analysis,
                "raw_response": response  # Include the raw response for debugging
            }
            
            logger.debug(f"Analysis completed with score: {match_score}")
            return result
        
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            logger.error(f"Failed to parse suitability analysis as JSON. Response: {response[:500]}...", exc_info=True)
            return {
                "match_score": 0.0,
                "matching_skills": [],
                "missing_skills": ["Failed to parse analysis"],
                "analysis": f"Error: Could not parse analysis results. Raw response: {response[:200]}...",
                "error": f"Failed to parse analysis results: {str(e)}",
                "raw_response": response[:500]  # Include part of raw response for debugging
            }
                
        except Exception as e:
            error_msg = f"Error analyzing resume suitability: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                "match_score": 0.0,
                "matching_skills": [],
                "missing_skills": [f"Error: {str(e)[:200]}"],  # Truncate long errors
                "analysis": f"Error during analysis: {str(e)[:500]}",
                "error": str(e)[:500],
                "error_type": type(e).__name__
            }


class ResumeAnalyzer(BaseAnalyzer):
    """Analyzes resumes using configured LLM provider with semantic chunking.
    
    This analyzer processes resume text, chunks it for efficient processing,
    and extracts structured information using the configured LLM provider.
    """
    
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the resume analyzer.
        
        Args:
            provider: The LLM provider to use. If None, uses the default from settings.
            model: The model to use. If None, uses the default from settings.
            config: Additional configuration overrides.
        """
        # Initialize with provider and model from settings if not provided
        if provider is None:
            provider = settings.llm.provider
        if model is None:
            model = settings.llm.model
            
        super().__init__(provider=provider, model=model, config=config)
        
        # Configure text processing from settings
        analysis_config = settings.analysis
        self.chunk_size = analysis_config.chunk_size if hasattr(analysis_config, 'chunk_size') else DEFAULT_CHUNK_SIZE
        self.chunk_overlap = analysis_config.chunk_overlap if hasattr(analysis_config, 'chunk_overlap') else DEFAULT_CHUNK_OVERLAP
        
        # Initialize text processor with configured chunking
        self.text_processor = TextProcessor(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            use_semantic_chunking=getattr(analysis_config, 'semantic_chunking', True),
            max_parallel_chunks=getattr(analysis_config, 'max_parallel_chunks', 5)
        )
        
        # Load extraction prompt
        self.extraction_prompt = load_prompt_template(RESUME_EXTRACTION_PROMPT_PATH)
    
    async def _initialize_client(self) -> None:
        """Initialize the LLM client using the provider factory.
        
        This method is called automatically when the client is first accessed.
        It sets up the appropriate client based on the configured provider.
        """
        try:
            # Get the factory instance
            factory = get_factory()
            
            # Create provider configuration
            provider_config = {
                'provider': self.provider,
                'model': self.model,
                **self.config  # Allow config overrides
            }
            
            # Get or create the provider
            self._client = await factory.get_or_create_provider(
                provider_name=self.provider,
                config_overrides=provider_config
            )
            
            # Verify the client was created
            if self._client is None:
                raise RuntimeError(f"Failed to initialize {self.provider} provider")
                
            logger.info(
                "Initialized %s provider with model: %s", 
                self.provider, 
                self.model
            )
            
        except Exception as e:
            logger.error(
                "Failed to initialize %s provider: %s", 
                self.provider, 
                str(e),
                exc_info=True
            )
            raise
    
    async def generate(
        self,
        prompt: str,
        task_name: str = "generate",
        **kwargs
    ) -> str:
        """Generate text from the LLM with retry logic.
        
        This method uses the configured provider to generate text based on the given prompt.
        It includes retry logic and metrics collection.
        
        Args:
            prompt: The prompt to send to the LLM
            task_name: Name of the task for metrics
            **kwargs: Additional arguments to pass to the LLM
            
        Returns:
            The generated text
            
        Raises:
            RuntimeError: If the LLM call fails after retries
        """
        if not hasattr(self, '_provider'):
            await self._initialize_clients()
            
        try:
            # Set default parameters if not provided
            if 'temperature' not in kwargs:
                kwargs['temperature'] = 0.7
            if 'max_tokens' not in kwargs:
                kwargs['max_tokens'] = 2000
                
            start_time = asyncio.get_event_loop().time()
            
            # Make the API call with retries
            max_retries = getattr(settings.llm, 'max_retries', 3)
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    # Use the provider's generate method
                    response = await self._provider.generate(
                        prompt=prompt,
                        model=self.model_name,
                        **kwargs
                    )
                    
                    # Log successful generation
                    duration = asyncio.get_event_loop().time() - start_time
                    logger.debug(
                        "Generated response in %.2fs (attempt %d/%d)",
                        duration, attempt + 1, max_retries
                    )
                    
                    return response
                    
                except Exception as e:
                    last_error = e
                    wait_time = min(2 ** attempt, 10)  # Exponential backoff, max 10s
                    logger.warning(
                        "Attempt %d/%d failed: %s. Retrying in %.1fs...",
                        attempt + 1, max_retries, str(e), wait_time
                    )
                    await asyncio.sleep(wait_time)
            
            # If we get here, all retries failed
            raise RuntimeError(
                f"Failed to generate text after {max_retries} attempts: {str(last_error)}"
            ) from last_error
            
        except Exception as e:
            logger.error("Error in generate: %s", str(e), exc_info=True)
            raise
    
    async def _initialize_clients(self) -> None:
        """Initialize the LLM provider using the ProviderFactory."""
        logger.info(f"Initializing {self.provider} provider for ResumeAnalyzer")
        
        try:
            # Get the provider factory with current configuration
            factory = get_factory()
            
            # Get the provider instance
            self._provider = await factory.get_or_create_provider(
                self.provider,
                model=self.model_name,
                config=getattr(settings.llm, self.provider, {})
            )
            
            logger.info(f"Successfully initialized {self.provider} provider for ResumeAnalyzer")
            return self._provider
            
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider} provider: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to initialize {self.provider} provider: {str(e)}") from e
            
    async def _process_chunk(self, chunk: str) -> Dict[str, Any]:
        """Process a single chunk of resume text.
        
        This method processes a chunk of resume text using the configured LLM provider
        to extract structured information. It handles prompt formatting, response
        generation, and response parsing.
        
        Args:
            chunk: A chunk of resume text to process
            
        Returns:
            Dictionary containing extracted resume data from this chunk
        """
        if not chunk or not isinstance(chunk, str):
            logger.warning(f"Invalid chunk data: {chunk}")
            return {}
            
        try:
            # Ensure the chunk is properly escaped for JSON
            try:
                chunk = json.dumps(chunk)[1:-1]  # Remove the surrounding quotes
            except (TypeError, ValueError) as e:
                logger.warning(f"Error escaping chunk data: {e}")
                chunk = str(chunk).replace('"', '\\"').replace('\n', ' ').strip()
            
            # Format the prompt with the current chunk
            try:
                prompt = self.extraction_prompt.replace('{{ resume_chunk }}', chunk)
            except Exception as e:
                logger.error(f"Error formatting prompt: {e}")
                return {}
            
            # Generate the response using the base class method
            try:
                response = await self.generate(
                    prompt=prompt,
                    task_name="resume_chunk_extraction",
                    temperature=0.1  # Lower temperature for more consistent extraction
                )
            except Exception as e:
                logger.error(f"Error generating response from LLM: {e}")
                return {}
            
            # Parse the JSON response
            try:
                if not response:
                    logger.warning("Received empty response from LLM")
                    return {}
                
                # Clean up the response before parsing
                response = response.strip()
                if response.startswith('```json'):
                    response = response[response.find('{'):response.rfind('}')+1]
                elif '```' in response:
                    response = response[response.find('```')+3:response.rfind('```')].strip()
                
                # Handle case where response might be wrapped in markdown code block without json specifier
                if response.startswith('{') and response.endswith('}'):
                    response = response.strip()
                
                if not response.strip():
                    logger.warning("Empty response after cleaning")
                    return {}
                    
                try:
                    result = json.loads(response)
                    if not isinstance(result, dict):
                        logger.warning("Unexpected response format from LLM (not a dictionary): %s", response)
                        return {}
                    
                    # Ensure all values in contact_info are strings or None
                    if 'contact_info' in result and isinstance(result['contact_info'], dict):
                        result['contact_info'] = {
                            k: str(v) if v is not None else None 
                            for k, v in result['contact_info'].items()
                        }
                        
                    return result
                    
                except json.JSONDecodeError as e:
                    # Try to extract JSON if it's embedded in text
                    try:
                        # Look for JSON-like content in the response
                        import re
                        json_match = re.search(r'\{.*\}', response, re.DOTALL)
                        if json_match:
                            result = json.loads(json_match.group(0))
                            if isinstance(result, dict):
                                return result
                    except Exception:
                        pass
                        
                    logger.error("Failed to parse LLM response as JSON: %s", response)
                    logger.exception(e)
                    return {}
                
            except Exception as e:
                logger.error("Unexpected error parsing LLM response: %s", str(e))
                logger.exception(e)
                return {}
            
        except Exception as e:
            logger.error("Error processing resume chunk: %s", str(e))
            logger.exception(e)
            return {}

    async def extract_resume_data_async(self, resume_text: str) -> Optional[ResumeData]:
        """Extract structured data from resume text using LLM with semantic chunking.
        
        This method processes the resume text by splitting it into semantic chunks,
        processing each chunk in parallel, and then combining the results into a
        single structured ResumeData object.
        
        Args:
            resume_text: Raw text content of the resume
            
        Returns:
            ResumeData object containing structured resume information, or None if processing fails
            
        Raises:
            RuntimeError: If there's an error during processing that can't be recovered from
        """
        if not resume_text or not isinstance(resume_text, str):
            logger.error("Invalid resume text provided")
            return None
            
        try:
            # Clean and normalize the resume text
            resume_text = resume_text.strip()
            if not resume_text:
                logger.error("Empty resume text provided")
                return None
                
            # Split the resume into semantic chunks
            try:
                documents = self.text_processor.split_text(resume_text)
                logger.info("Split resume into %d chunks for processing", len(documents))
                
                if not documents:
                    logger.warning("No valid chunks generated from resume text")
                    return None
                    
            except Exception as e:
                logger.error("Error splitting resume text into chunks: %s", str(e))
                # Fall back to simple chunking if semantic chunking fails
                chunk_size = self.chunk_size or DEFAULT_CHUNK_SIZE
                documents = [resume_text[i:i+chunk_size] for i in range(0, len(resume_text), chunk_size)]
                logger.info("Fallback: Split resume into %d fixed-size chunks", len(documents))
            
            # Process each chunk in parallel with error handling
            tasks = []
            for doc in documents:
                try:
                    if hasattr(doc, 'page_content'):
                        content = doc.page_content
                    else:
                        content = str(doc)
                    if content.strip():
                        tasks.append(self._process_chunk(content))
                except Exception as e:
                    logger.warning("Error preparing chunk for processing: %s", str(e))
            
            if not tasks:
                logger.error("No valid chunks to process")
                return None
                
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results from all chunks
            combined_result = {
                'name': '',
                'contact_info': {},
                'summary': '',
                'experience': [],
                'education': [],
                'skills': [],
                'projects': [],
                'certifications': [],
                'languages': [],
                'honors': [],
                'publications': []
            }
            
            for i, result in enumerate(chunk_results):
                if isinstance(result, Exception):
                    logger.error("Error processing chunk %d: %s", i, str(result))
                    continue
                    
                if not isinstance(result, dict):
                    logger.warning("Unexpected chunk result type: %s", type(result).__name__)
                    continue
                
                # Merge the results (simple strategy - for production, you might want more sophisticated merging)
                for key, value in result.items():
                    if key not in combined_result:
                        combined_result[key] = value
                    elif isinstance(value, list) and key in combined_result and isinstance(combined_result[key], list):
                        # For lists, extend with non-duplicate items
                        existing_items = {json.dumps(item, sort_keys=True) for item in combined_result[key]}
                        for item in value:
                            try:
                                item_str = json.dumps(item, sort_keys=True) if isinstance(item, (dict, list)) else str(item)
                                if item_str not in existing_items:
                                    combined_result[key].append(item)
                                    existing_items.add(item_str)
                            except (TypeError, ValueError) as e:
                                logger.warning(f"Error processing item in {key}: {e}")
                                continue
                    elif isinstance(value, dict) and key in combined_result and isinstance(combined_result[key], dict):
                        # For dictionaries, update with new keys or merge values
                        for k, v in value.items():
                            if k not in combined_result[key]:
                                combined_result[key][k] = v
                            elif isinstance(v, str) and k in combined_result[key] and isinstance(combined_result[key][k], str):
                                # For strings, concatenate with a space if they're different
                                if v.strip() and v not in combined_result[key][k]:
                                    combined_result[key][k] = f"{combined_result[key][k]} {v}".strip()
                    elif isinstance(value, str) and key in combined_result and isinstance(combined_result[key], str):
                        # For strings, concatenate with a space if they're different
                        if value.strip() and value not in combined_result[key]:
                            combined_result[key] = f"{combined_result[key]} {value}".strip()
                    else:
                        # For other types, prefer non-empty values
                        if value:
                            combined_result[key] = value
                            
            # Create a ResumeData object from the combined results
            try:
                # Ensure required fields are present
                if not combined_result.get('name') and 'contact_info' in combined_result:
                    contact = combined_result['contact_info']
                    if isinstance(contact, dict) and 'name' in contact:
                        combined_result['name'] = contact.get('name', '')
                
                # Ensure all list fields are lists
                list_fields = ['experience', 'education', 'skills', 'projects', 'certifications', 'languages', 'honors', 'publications']
                for field in list_fields:
                    if field not in combined_result or not isinstance(combined_result[field], list):
                        combined_result[field] = []
                
                # Ensure contact_info is a dictionary
                if 'contact_info' not in combined_result or not isinstance(combined_result['contact_info'], dict):
                    combined_result['contact_info'] = {}
                
                # Prepare contact_info ensuring it's a dictionary with string values
                contact_info = combined_result.get('contact_info', {})
                if isinstance(contact_info, dict):
                    # Ensure all values in contact_info are strings or None
                    contact_info = {
                        k: str(v) if v is not None else None 
                        for k, v in contact_info.items()
                    }
                else:
                    contact_info = {}
                
                # Create the ResumeData object with properly formatted data
                resume_data = ResumeData(
                    contact_info=contact_info,
                    summary=combined_result.get('summary', ''),
                    skills=combined_result.get('skills', []),
                    experience=combined_result.get('experience', []),
                    education=combined_result.get('education', []),
                    certifications=combined_result.get('certifications', []),
                    languages=combined_result.get('languages', []),
                    projects=combined_result.get('projects', []),
                    raw_text_hash=None  # This will be set by the model if needed
                )
                
                logger.info("Successfully extracted resume data")
                return resume_data
                
            except Exception as e:
                logger.error(f"Error creating ResumeData object: {str(e)}", exc_info=True)
                # Try to create a minimal valid ResumeData object as fallback
                try:
                    return ResumeData(
                        contact_info={
                            k: str(v) if v is not None else None 
                            for k, v in combined_result.get('contact_info', {}).items()
                        },
                        summary=combined_result.get('summary', ''),
                        skills=combined_result.get('skills', []),
                        experience=combined_result.get('experience', []),
                        education=combined_result.get('education', []),
                        certifications=combined_result.get('certifications', []),
                        languages=combined_result.get('languages', []),
                        projects=combined_result.get('projects', [])
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback ResumeData creation failed: {str(fallback_error)}", exc_info=True)
                    return None
                    
        except Exception as e:
            logger.error(f"Error in extract_resume_data_async: {str(e)}", exc_info=True)
            raise RuntimeError(f"Failed to extract resume data: {str(e)}") from e
