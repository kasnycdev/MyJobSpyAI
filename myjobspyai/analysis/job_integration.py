"""
Job Integration Module

This module provides integration between the job parser and the main application.
It handles loading job data, parsing it, and making it available for analysis.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from opentelemetry import trace

from myjobspyai.parsers.job_parser import load_job_mandates, parse_job_data
from myjobspyai.utils.logging_utils import setup_logging

# Set up logging
logger = logging.getLogger(__name__)
setup_logging()

class JobIntegration:
    """
    Handles the integration of job data into the application.
    """
    
    def __init__(self, job_data_path: Optional[str] = None):
        """
        Initialize the JobIntegration.
        
        Args:
            job_data_path: Optional path to the job data JSON file.
        """
        self.job_data_path = Path(job_data_path) if job_data_path else None
        self._jobs: List[Dict[str, Any]] = []
        self._current_job_index = 0
    
    def load_jobs(self, job_data_path: Optional[str] = None) -> bool:
        """
        Load jobs from a JSON file.
        
        Args:
            job_data_path: Path to the job data JSON file. If None, uses the path
                         provided during initialization.
                          
        Returns:
            bool: True if jobs were loaded successfully, False otherwise.
        """
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("load_jobs"):
            if job_data_path:
                self.job_data_path = Path(job_data_path)
            
            if not self.job_data_path or not self.job_data_path.exists():
                logger.error(f"Job data file not found: {self.job_data_path}")
                return False
            
            try:
                # Load and parse job data
                raw_jobs = load_job_mandates(str(self.job_data_path))
                
                # Parse each job to ensure it matches our schema
                self._jobs = []
                for job in raw_jobs:
                    try:
                        parsed_job = parse_job_data(job, validate='strict')
                        self._jobs.append(parsed_job)
                    except ValueError as e:
                        logger.warning(f"Skipping invalid job data: {e}")
                
                if not self._jobs:
                    logger.error("No valid jobs found in the provided data")
                    return False
                
                logger.info(f"Successfully loaded {len(self._jobs)} jobs from {self.job_data_path}")
                return True
                
            except Exception as e:
                logger.error(f"Error loading job data: {e}", exc_info=True)
                return False
    
    def get_jobs(self) -> List[Dict[str, Any]]:
        """
        Get all loaded jobs.
        
        Returns:
            List of job dictionaries.
        """
        return self._jobs
    
    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific job by ID.
        
        Args:
            job_id: The ID of the job to retrieve.
            
        Returns:
            The job dictionary if found, None otherwise.
        """
        for job in self._jobs:
            if job.get('job_id') == job_id:
                return job
        return None
    
    def get_jobs_by_title(self, title: str) -> List[Dict[str, Any]]:
        """
        Get all jobs with a specific title.
        
        Args:
            title: The job title to search for.
            
        Returns:
            List of matching job dictionaries.
        """
        return [job for job in self._jobs 
                if title.lower() in job.get('job_title_extracted', '').lower()]
    
    def get_jobs_by_company(self, company: str) -> List[Dict[str, Any]]:
        """
        Get all jobs from a specific company.
        
        Args:
            company: The company name to search for.
            
        Returns:
            List of matching job dictionaries.
        """
        return [job for job in self._jobs 
                if company.lower() in job.get('company_name', '').lower()]
    
    def get_next_job(self) -> Optional[Dict[str, Any]]:
        """
        Get the next job in the sequence.
        
        Returns:
            The next job dictionary, or None if at the end of the list.
        """
        if not self._jobs or self._current_job_index >= len(self._jobs):
            return None
        
        job = self._jobs[self._current_job_index]
        self._current_job_index += 1
        return job
    
    def reset_iterator(self) -> None:
        """Reset the job iterator to the beginning."""
        self._current_job_index = 0
    
    def __iter__(self):
        """Make the class iterable over jobs."""
        return iter(self._jobs)
    
    def __len__(self) -> int:
        """Return the number of loaded jobs."""
        return len(self._jobs)
