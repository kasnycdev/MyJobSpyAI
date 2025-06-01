"""Job description and resume suitability analysis using LLM providers.

This module provides analyzers for processing job descriptions and evaluating
resume suitability with support for structured output and semantic analysis.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, cast

from pydantic import BaseModel, ValidationError

from myjobspyai.analysis.components.analyzers.base import BaseAnalyzer
from myjobspyai.analysis.components.analyzers.models import ResumeData
from myjobspyai.analysis.components.analyzers.exceptions import AnalysisError
from myjobspyai.analysis.providers.instructor_ollama import InstructorOllamaClient
from myjobspyai.config import settings

logger = logging.getLogger(__name__)

# Type variable for generic model types
T = TypeVar('T', bound=BaseModel)

class JobAnalyzer(BaseAnalyzer):
    """Analyzes job descriptions and matches them against resumes.
    
    This analyzer extracts structured information from job descriptions
    and evaluates candidate suitability based on resume data.
    """
    
    # Default paths to prompt templates and schemas
    DEFAULT_TEMPLATE_DIR = Path(__file__).parent / "templates" / "job"
    DEFAULT_SCHEMA_DIR = Path(__file__).parent / "schemas"
    
    # Template and schema filenames
    EXTRACTION_INSTRUCTIONS = "extraction_instructions.prompt"
    SUITABILITY_INSTRUCTIONS = "suitability_instructions.prompt"
    JOB_SCHEMA = "job_schema.json"
    SUITABILITY_SCHEMA = "suitability_schema.json"
    
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
        super().__init__(provider=provider, model=model, config=config)
        
        # Load prompt templates and schemas
        self.extraction_instructions = self._load_prompt_template(self.EXTRACTION_INSTRUCTIONS)
        self.suitability_instructions = self._load_prompt_template(self.SUITABILITY_INSTRUCTIONS)
        self.job_schema = self._load_schema(self.JOB_SCHEMA)
        self.suitability_schema = self._load_schema(self.SUITABILITY_SCHEMA)
    
    def _load_prompt_template(self, template_name: str) -> str:
        """Load a prompt template from file.
        
        Args:
            template_name: Name of the template file to load
            
        Returns:
            str: The loaded prompt template
            
        Raises:
            FileNotFoundError: If the template file is not found
            IOError: If there's an error reading the file
        """
        try:
            # First try to get the path from config
            config_path = self.config.get('analysis', {}).get('prompts', {}).get(template_name)
            
            if config_path:
                # If we have a config path, use it
                template_path = Path(config_path)
                if not template_path.is_absolute():
                    # If path is relative, make it relative to the project root
                    project_root = Path(__file__).parent.parent.parent.parent.parent
                    template_path = (project_root / config_path).resolve()
            else:
                # Fall back to default location
                template_path = self.DEFAULT_TEMPLATE_DIR / template_path
            
            logger.debug(f"Loading prompt template from: {template_path}")
            
            # Try to load the specified template file
            if not template_path.exists():
                raise FileNotFoundError(
                    f"Prompt template not found at: {template_path}. "
                    f"Please ensure the file exists or update the configuration."
                )
                
            with open(template_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
                
        except Exception as e:
            logger.error(f"Error loading prompt template {template_name}: {str(e)}", exc_info=True)
            raise
    
    def _load_schema(self, schema_name: str) -> Dict[str, Any]:
        """Load a JSON schema from file.
        
        Args:
            schema_name: Name of the schema file to load
            
        Returns:
            dict: The loaded schema
            
        Raises:
            FileNotFoundError: If the schema file is not found
            json.JSONDecodeError: If the schema file contains invalid JSON
        """
        try:
            # First try to get the path from config
            config_path = self.config.get('analysis', {}).get('schemas', {}).get(schema_name)
            
            if config_path:
                # If we have a config path, use it
                schema_path = Path(config_path)
                if not schema_path.is_absolute():
                    # If path is relative, make it relative to the project root
                    project_root = Path(__file__).parent.parent.parent.parent.parent
                    schema_path = (project_root / config_path).resolve()
            else:
                # Fall back to default location
                schema_path = self.DEFAULT_SCHEMA_DIR / schema_name
            
            logger.debug(f"Loading schema from: {schema_path}")
            
            # Try to load the specified schema file
            if not schema_path.exists():
                raise FileNotFoundError(
                    f"Schema file not found at: {schema_path}. "
                    f"Please ensure the file exists or update the configuration."
                )
                
            with open(schema_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse schema file {schema_path}: {str(e)}"
            logger.error(error_msg)
            raise
        except Exception as e:
            logger.error(f"Error loading schema {schema_name}: {str(e)}", exc_info=True)
            raise
    
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
            
            # Format the prompt with the job description
            prompt = self.extraction_instructions.format(
                job_description=job_description_str
            )
            
            # Check if we're using InstructorOllamaClient
            use_instructor = isinstance(self._provider_instance, InstructorOllamaClient)
            
            # Call the LLM to extract job details
            if use_instructor:
                # For Instructor, we can directly use the schema
                response = await self.generate(
                    prompt=prompt,
                    task_name="extract_job_details",
                    response_model=dict,  # We'll validate against schema separately
                    schema=self.job_schema,
                    temperature=0.2,
                    max_tokens=2000,
                )
                result = response
            else:
                # For other providers, include schema in the prompt
                schema_instructions = (
                    "\n\nEXPECTED OUTPUT SCHEMA (for reference only, follow these field names and types):\n"
                    f"{json.dumps(self.job_schema, indent=2)}"
                )
                
                full_prompt = (
                    f"{prompt}\n"
                    f"{schema_instructions}\n"
                    "IMPORTANT: Return ONLY a valid JSON object that matches the schema above. "
                    "Do not include any explanatory text outside the JSON."
                )
                
                response = await self.generate(
                    prompt=full_prompt,
                    task_name="extract_job_details",
                    temperature=0.2,
                    max_tokens=2000,
                )
                
                # Parse the response
                try:
                    if not response or not response.strip():
                        raise ValueError("Empty response from LLM")
                        
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
                except (json.JSONDecodeError, IndexError) as e:
                    logger.error(f"Failed to parse job details: {str(e)}\nResponse: {response}")
                    raise ValueError(f"Failed to parse job details: {str(e)}") from e
            
            # Validate the result against the schema
            try:
                # Basic validation - in a real implementation, you might want to use jsonschema
                if not isinstance(result, dict):
                    raise ValueError("Expected a JSON object in the response")
                    
                # Ensure required fields are present
                required_fields = ["job_title_extracted", "key_responsibilities", "required_skills"]
                for field in required_fields:
                    if field not in result:
                        raise ValueError(f"Missing required field: {field}")
                        
                return result
                
            except Exception as e:
                logger.error(f"Validation error in job details: {str(e)}\nData: {result}")
                raise ValueError(f"Invalid job details format: {str(e)}") from e
                
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
        try:
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
                    indent=2
                )
                job_json = json.dumps(
                    job_dict, 
                    indent=2
                )
            except Exception as e:
                error_msg = f"Error serializing data for analysis: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg) from e
            
            # Format the prompt with the resume and job data
            prompt = self.suitability_instructions.format(
                resume_data=resume_json,
                job_data=job_json
            )
            
            # Check if we're using InstructorOllamaClient
            use_instructor = isinstance(self._provider_instance, InstructorOllamaClient)
            
            # Call the LLM to analyze suitability
            if use_instructor:
                # For Instructor, we can directly use the schema
                response = await self.generate(
                    prompt=prompt,
                    task_name="analyze_resume_suitability",
                    response_model=dict,  # We'll validate against schema separately
                    schema=self.suitability_schema,
                    temperature=0.3,
                    max_tokens=2500,
                )
                result = response
            else:
                # For other providers, include schema in the prompt
                schema_instructions = (
                    "\n\nEXPECTED OUTPUT SCHEMA (for reference only, follow these field names and types):\n"
                    f"{json.dumps(self.suitability_schema, indent=2)}"
                )
                
                full_prompt = (
                    f"{prompt}\n"
                    f"{schema_instructions}\n"
                    "IMPORTANT: Return ONLY a valid JSON object that matches the schema above. "
                    "Do not include any explanatory text outside the JSON."
                )
                
                response = await self.generate(
                    prompt=full_prompt,
                    task_name="analyze_resume_suitability",
                    temperature=0.3,
                    max_tokens=2500,
                )
                
                # Parse the response
                try:
                    if not response or not response.strip():
                        raise ValueError("Empty response from LLM")
                        
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
                except (json.JSONDecodeError, IndexError) as e:
                    logger.error(f"Failed to parse suitability analysis: {str(e)}\nResponse: {response}")
                    raise ValueError(f"Failed to parse suitability analysis: {str(e)}") from e
            
            # Validate the result against the schema
            try:
                # Basic validation - in a real implementation, you might want to use jsonschema
                if not isinstance(result, dict):
                    raise ValueError("Expected a JSON object in the response")
                    
                # Ensure required fields are present
                if 'analysis' not in result:
                    raise ValueError("Missing required field: analysis")
                    
                analysis = result['analysis']
                if not isinstance(analysis, dict):
                    raise ValueError("Analysis field should be an object")
                    
                if 'suitability_score' not in analysis or 'justification' not in analysis:
                    raise ValueError("Analysis is missing required fields: suitability_score and/or justification")
                    
                return result
                
            except Exception as e:
                logger.error(f"Validation error in suitability analysis: {str(e)}\nData: {result}")
                raise ValueError(f"Invalid suitability analysis format: {str(e)}") from e
                
        except Exception as e:
            logger.error(f"Error analyzing resume suitability: {str(e)}", exc_info=True)
            return {
                "analysis": {
                    "suitability_score": 0,
                    "justification": f"Error during analysis: {str(e)}",
                    "pros": [],
                    "cons": [],
                    "error": str(e)
                }
            }
