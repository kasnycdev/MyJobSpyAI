"""JobSpy scraper implementation for MyJobSpy AI."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ..models.job import Job, JobSource, JobType
from .base import BaseJobScraper

logger = logging.getLogger(__name__)

class JobSpyScraper(BaseJobScraper):
    """Scraper that uses JobSpy to search for jobs across multiple platforms."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the JobSpy scraper.

        Args:
            config: Configuration dictionary for the scraper
        """
        super().__init__("jobspy", config or {})

        # Initialize browser if needed (JobSpy will handle this)
        self._browser_initialized = False

    async def _init_browser(self):
        """Initialize browser if needed."""
        if not self._browser_initialized:
            try:
                from jobspy import init_browser
                init_browser()
                self._browser_initialized = True
            except Exception as e:
                logger.error(f"Failed to initialize browser: {e}")
                raise

    async def search_jobs(
        self,
        query: str,
        location: str = "",
        max_results: int = 15,
        **kwargs
    ) -> List[Job]:
        """Search for jobs using JobSpy.

        Args:
            query: Job search query (e.g., 'software engineer')
            location: Location for the job search (e.g., 'New York, NY')
            max_results: Maximum number of results to return
            **kwargs: Additional search parameters

        Returns:
            List of Job objects
        """
        try:
            from jobspy import scrape_jobs

            # Map our parameters to JobSpy parameters
            jobspy_params = {
                'search_term': query,
                'location': location,
                'results_wanted': max_results,
                'site_name': kwargs.get('site_name'),
                'distance': kwargs.get('distance', 50),
                'job_type': kwargs.get('job_type'),
                'is_remote': kwargs.get('is_remote', False),
                'easy_apply': kwargs.get('easy_apply', False),
                'offset': kwargs.get('offset', 0),
                'hours_old': kwargs.get('hours_old'),
                'country_indeed': kwargs.get('country_indeed'),
                'description_format': kwargs.get('description_format', 'markdown'),
                'linkedin_fetch_description': kwargs.get('linkedin_fetch_description', False),
                'linkedin_company_ids': kwargs.get('linkedin_company_ids'),
                'proxies': kwargs.get('proxies'),
                'ca_cert': kwargs.get('ca_cert'),
                'verbose': kwargs.get('verbose', 2)
            }

            # Remove None values
            jobspy_params = {k: v for k, v in jobspy_params.items() if v is not None}

            logger.info(f"Searching for jobs with params: {jobspy_params}")

            # Scrape jobs
            jobs_df = scrape_jobs(**jobspy_params)

            if jobs_df.empty:
                logger.info("No jobs found matching the criteria")
                return []

            # Convert to our Job model
            jobs = []
            for _, row in jobs_df.iterrows():
                try:
                    # Map job type - handle None, float, or string values
                    job_type_str = str(row.get('job_type', 'full_time')).lower().strip()
                    job_type = self._map_job_type(job_type_str) if job_type_str else JobType.FULL_TIME

                    # Map source
                    source_url = row.get('job_url', '')
                    source = self._get_job_source(source_url)

                    # Get row as dict, handling None case
                    row_dict = row.to_dict() if hasattr(row, 'to_dict') and callable(row.to_dict) else {}

                    # Prepare job data with proper handling of None/NaN values
                    posted_date = row.get('date_posted')
                    if pd.isna(posted_date):
                        posted_date = None

                    job_data = {
                        'id': str(row.get('job_id', '')),
                        'title': str(row.get('title', '')),
                        'company': str(row.get('company', '')),
                        'location': str(row.get('location', '')),
                        'description': str(row.get('description', 'No description available')),
                        'job_type': job_type,
                        'url': source_url,
                        'posted_date': posted_date,
                        'salary': str(row.get('salary', '')) if pd.notna(row.get('salary')) else None,
                        'remote': bool(row.get('is_remote', False)),
                        'source': source,
                        'metadata': {
                            'site': str(row.get('site', '')),
                            'emails': list(row.get('emails', []) if pd.notna(row.get('emails')) else []),
                            'skills': list(row.get('skills', []) if pd.notna(row.get('skills')) else []),
                            'source_url': source_url,
                            **{k: v for k, v in (row_dict or {}).items()
                               if pd.notna(v) and v is not None and not k.startswith('_')}
                        }
                    }

                    # Handle None/NaN values for required fields
                    if pd.isna(job_data['description']) or not job_data['description'].strip():
                        job_data['description'] = 'No description available'

                    # Create job object
                    job = Job(**job_data)
                    jobs.append(job)
                except Exception as e:
                    logger.error(f"Error processing job {row.get('job_id')}: {e}")

            return jobs

        except Exception as e:
            logger.error(f"Error searching for jobs: {e}", exc_info=True)
            raise

    def _map_job_type(self, job_type_str: str) -> JobType:
        """Map JobSpy job type to our JobType enum."""
        if not job_type_str or not isinstance(job_type_str, str):
            return JobType.FULL_TIME

        # Convert to string and clean up
        job_type_str = str(job_type_str).lower().strip()

        # Map common variations to standard types
        type_mapping = {
            'full_time': JobType.FULL_TIME,
            'full-time': JobType.FULL_TIME,
            'full time': JobType.FULL_TIME,
            'ft': JobType.FULL_TIME,

            'part_time': JobType.PART_TIME,
            'part-time': JobType.PART_TIME,
            'part time': JobType.PART_TIME,
            'pt': JobType.PART_TIME,

            'contract': JobType.CONTRACT,
            'contractor': JobType.CONTRACT,
            'contract to hire': JobType.CONTRACT,
            'contract-to-hire': JobType.CONTRACT,
            'c2h': JobType.CONTRACT,

            'temporary': JobType.TEMPORARY,
            'temp': JobType.TEMPORARY,
            'temp to perm': JobType.TEMPORARY,
            'temporary to permanent': JobType.TEMPORARY,

            'internship': JobType.INTERNSHIP,
            'intern': JobType.INTERNSHIP,
            'co-op': JobType.INTERNSHIP,
            'coop': JobType.INTERNSHIP,

            'volunteer': JobType.VOLUNTEER,
            'voluntary': JobType.VOLUNTEER,
            'volunteer position': JobType.VOLUNTEER,

            'permanent': JobType.FULL_TIME,
            'employee': JobType.FULL_TIME,
            'staff': JobType.FULL_TIME,
        }

        # Try exact match first
        if job_type_str in type_mapping:
            return type_mapping[job_type_str]

        # Try partial matching
        for key, job_type in type_mapping.items():
            if key in job_type_str:
                return job_type

        # Default to FULL_TIME for professional jobs, OTHER otherwise
        return JobType.FULL_TIME if any(term in job_type_str for term in
                                     ['full', 'part', 'contract', 'temp', 'intern', 'volunt']) else JobType.OTHER

    def _get_job_source(self, url: str) -> JobSource:
        """Determine job source from URL."""
        if 'linkedin.com' in url:
            return JobSource.LINKEDIN
        elif 'indeed.com' in url:
            return JobSource.INDEED
        elif 'glassdoor.com' in url:
            return JobSource.GLASSDOOR
        elif 'ziprecruiter.com' in url:
            return JobSource.ZIPRECRUITER
        else:
            return JobSource.OTHER

    async def close(self):
        """Clean up resources."""
        # JobSpy handles its own cleanup
        self._browser_initialized = False
        logger.info("JobSpy scraper closed")
