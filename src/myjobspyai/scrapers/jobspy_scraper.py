"""JobSpy scraper implementation for MyJobSpy AI."""

import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from jobspy import scrape_jobs

from ..models.job import Job, JobSource, JobType, JobEnums
from .base import BaseJobScraper
from ..models.job_listing import JobListing

logger = logging.getLogger(__name__)


class JobSpyScraper(BaseJobScraper):
    """Scraper that uses JobSpy to search for jobs across multiple platforms."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the JobSpy scraper.

        Args:
            config: Configuration dictionary for the scraper
        """
        super().__init__("jobspy", config or {})
        self._browser_initialized = False

    def _map_job_type(self, job_type_str: str) -> JobType:
        """Map raw job type string to JobType enum.

        Args:
            job_type_str: Raw job type string from job listing

        Returns:
            JobType enum value
        """
        return BaseJobScraper.map_job_type(job_type_str)

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
        max_retries: int = 3,
        retry_delay: int = 5,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Search for jobs using JobSpy with retry logic and error handling.

        Args:
            query: Job search query (e.g., 'software engineer')
            location: Location for the job search (e.g., 'New York, NY')
            max_results: Maximum number of results to return
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            **kwargs: Additional search parameters

        Returns:
            List of Job objects

        Raises:
            RuntimeError: If job search fails after all retry attempts
        """
        # Get site-specific timeouts from config
        site_timeouts = self.config.get('timeouts', {})
        default_timeout = site_timeouts.get('default', 30)

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                # Initialize search parameters
                search_params = {
                    'search_term': query,
                    'location': location,
                    'country_indeed': kwargs.get(
                        'country_indeed', self.config.get('country_indeed', 'usa')
                    ),
                    'results_wanted': max_results,
                    'site_name': kwargs.get(
                        'site_name',
                        self.config.get(
                            'site_name', ['linkedin', 'glassdoor', 'google']
                        ),
                    ),
                    'distance': kwargs.get('distance', self.config.get('distance', 50)),
                    'job_type': kwargs.get('job_type', self.config.get('job_type')),
                    'is_remote': kwargs.get(
                        'is_remote', self.config.get('is_remote', False)
                    ),
                    'easy_apply': kwargs.get(
                        'easy_apply', self.config.get('easy_apply', False)
                    ),
                    'offset': kwargs.get('offset', 0),
                    'hours_old': kwargs.get('hours_old', self.config.get('hours_old')),
                    'description_format': kwargs.get(
                        'description_format',
                        self.config.get('description_format', 'markdown'),
                    ),
                    'enforce_annual_salary': kwargs.get('enforce_annual_salary', False),
                    'min_salary': kwargs.get('min_salary'),
                }

                # Add site-specific timeouts
                for site in search_params.get('site_name', []):
                    if site in site_timeouts:
                        search_params[f'{site}_timeout'] = site_timeouts[site]

                # Execute job search
                logger.info(f"Searching for jobs with params: {search_params}")

                # Run the synchronous scrape_jobs in a thread
                loop = asyncio.get_running_loop()
                jobs_df = await loop.run_in_executor(
                    None,
                    lambda p=search_params: scrape_jobs(
                        **{k: v for k, v in p.items() if v is not None}
                    ),
                )

                if jobs_df is None or jobs_df.empty:
                    logger.warning(f"No jobs found in attempt {attempt + 1}")
                    if attempt == max_retries:
                        return []
                    continue

                # Process job results with all filters applied
                processed_jobs = self._process_job_results(
                    jobs_df=jobs_df,
                    is_remote_search=kwargs.get(
                        'is_remote', self.config.get('is_remote', False)
                    ),
                    job_type=kwargs.get('job_type', self.config.get('job_type')),
                    min_salary=kwargs.get('min_salary', self.config.get('min_salary')),
                )

                # Log filtering results
                if kwargs.get('is_remote', self.config.get('is_remote', False)):
                    remote_count = sum(1 for job in processed_jobs if job.remote)
                    logger.info(
                        f"Found {remote_count} remote jobs out of {len(processed_jobs)} total jobs"
                    )

                return processed_jobs

            except Exception as e:
                last_error = e
                if attempt == max_retries:
                    break

                logger.warning(
                    f"Attempt {attempt + 1} failed: {str(e)}",
                    exc_info=attempt == max_retries,
                )
                await asyncio.sleep(retry_delay)
                continue

        # If we got here, all retries failed
        error_msg = f"Failed to search for jobs after {max_retries + 1} attempts"
        if last_error:
            error_msg += f": {str(last_error)}"
        logger.error(error_msg)
        raise RuntimeError(error_msg) from last_error

    def _process_job_results(
        self,
        jobs_df: pd.DataFrame,
        is_remote_search: bool = False,
        job_type: Optional[str] = None,
        min_salary: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Process job results from DataFrame to Job objects.

        Args:
            jobs_df: DataFrame containing job listings
            is_remote_search: Whether to filter for remote jobs only
            job_type: Optional job type to filter by
            min_salary: Optional minimum salary to filter by

        Returns:
            List of processed job dictionaries
        """
        jobs = []
        if jobs_df is None or jobs_df.empty:
            logger.warning("No jobs found in the provided DataFrame")
            return []

        logger.info(f"Processing {len(jobs_df)} job listings")

        for _, row in jobs_df.iterrows():
            try:
                # Map job type - handle None, float, or string values
                job_type_str = str(row.get('job_type', 'full_time')).lower().strip()
                job_type = self._map_job_type(job_type_str)

                # Generate a unique ID based on job details
                job_id = f"{row.get('site', 'jobspy')}_{row.get('job_title', '')}_{row.get('company_name', '')}_{row.get('posted_date', datetime.now()).strftime('%Y%m%d_%H%M%S')}"
                job_id = job_id.lower().replace(' ', '_').replace(',', '').replace('.', '')[:100]  # Clean and truncate

                # Clean job data
                title = str(row.get('job_title', ''))
                company = str(row.get('company_name', ''))
                location = str(row.get('location', ''))
                description = str(row.get('description', ''))
                if not description.strip():
                    description = f"Job posting for {title} at {company} in {location}"

                # Create Job object
                job = Job(
                    id=job_id,
                    title=title,
                    company=company,
                    location=location,
                    description=description,
                    url=row.get('job_url', ''),
                    posted_date=row.get('posted_date', datetime.now()),
                    source=JobSource(row.get('site', 'jobspy')),
                    remote=row.get('is_remote', False),
                    job_type=job_type,
                    salary=row.get('salary', None),
                    experience_level=row.get('experience_level', None),
                    skills=row.get('skills', []),
                    requirements=row.get('requirements', []),
                    benefits=row.get('benefits', []),
                    additional_info=row.get('additional_info', {}),
                )

                # Apply filters
                if is_remote_search and not job.remote:
                    continue

                if job_type and job_type != job.job_type:
                    continue

                if min_salary and job.salary and job.salary < min_salary:
                    continue

                jobs.append(job)

            except Exception as e:
                logger.error(f"Error processing job row: {str(e)}", exc_info=True)
                continue

        logger.info(f"Processed {len(jobs)} jobs after filtering")
        return jobs

    def _map_job_type(self, job_type_str: str) -> JobType:
        """Map raw job type string to JobType enum.

        Args:
            job_type_str: Raw job type string from job listing

        Returns:
            JobType enum value
        """
        # Convert to lowercase and remove spaces
        job_type_str = job_type_str.lower().strip()

        # Map common job type strings to JobType enum
        job_type_map = {
            'full-time': JobType.FULL_TIME,
            'full time': JobType.FULL_TIME,
            'part-time': JobType.PART_TIME,
            'part time': JobType.PART_TIME,
            'contract': JobType.CONTRACT,
            'contractor': JobType.CONTRACT,
            'consultant': JobType.CONTRACT,
            'internship': JobType.INTERNSHIP,
            'intern': JobType.INTERNSHIP,
            'temporary': JobType.TEMPORARY,
            'temp': JobType.TEMPORARY,
            'remote': JobType.REMOTE,
            'telecommute': JobType.REMOTE,
            'work from home': JobType.REMOTE,
            'hybrid': JobType.HYBRID,
            'flexible': JobType.HYBRID,
        }

        # Try to find a match in the mapping
        for key in job_type_map:
            if key in job_type_str:
                return job_type_map[key]

        # If no match found, default to FULL_TIME
        logger.warning(f"Unknown job type: {job_type_str}, defaulting to FULL_TIME")
        return JobType.FULL_TIME
