"""JobSpy scraper implementation for MyJobSpy AI."""

import asyncio
import logging
import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from jobspy import scrape_jobs

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
        default_timeout = site_timeouts.get('default', 30)  # Default to 30 seconds

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

                # Return Job objects directly - let the caller handle serialization if needed
                return processed_jobs

            except Exception as e:
                last_error = e
                if attempt == max_retries:
                    break

                logger.warning(
                    f"Attempt {attempt + 1} failed: {str(e)}",
                    exc_info=attempt
                    == max_retries,  # Only log full traceback on last attempt
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

                # Get source from URL if not provided
                source = row.get('source', '')
                if not source and 'job_url' in row:
                    source = self._get_job_source(row['job_url'])

                # Parse posted date
                posted_date = None
                if pd.notna(row.get('posted_date')):
                    try:
                        # Try to parse the timestamp if it's in epoch format
                        if isinstance(row['posted_date'], (int, float)):
                            posted_date = datetime.fromtimestamp(
                                row['posted_date']
                            ).isoformat()
                        else:
                            # Try to parse string date
                            posted_date = datetime.strptime(
                                str(row['posted_date']), '%Y-%m-%d'
                            ).isoformat()
                    except (ValueError, TypeError) as e:
                        logger.warning(
                            f"Could not parse date: {row.get('posted_date')}"
                        )
                        posted_date = None

                # Handle salary information
                salary = None
                if pd.notna(row.get('salary')) and isinstance(row['salary'], str):
                    try:
                        salary = float(row['salary'].replace('$', '').replace(',', ''))
                    except (ValueError, AttributeError):
                        pass

                # Enhanced remote job detection
                is_remote = False

                # Check explicit remote flag first
                if row.get('is_remote') in (True, 'True', 'true', '1', 1):
                    is_remote = True

                # Check location-based indicators
                location = str(row.get('location', '')).lower()
                if any(
                    term in location
                    for term in [
                        'remote',
                        'work from home',
                        'wfh',
                        'virtual',
                        'telecommute',
                        'anywhere',
                    ]
                ):
                    is_remote = True

                # Check title and description if still not marked as remote
                if not is_remote:
                    title = str(row.get('title', '')).lower()
                    description = str(row.get('description', '')).lower()

                    remote_indicators = [
                        'remote',
                        'work from home',
                        'work-from-home',
                        'work at home',
                        'wfh',
                        'virtual',
                        'telecommute',
                        'telework',
                        'telecommuting',
                        'distributed team',
                        'work anywhere',
                        'location independent',
                        'location-independent',
                        'work from anywhere',
                        'digital nomad',
                    ]

                    # Check if any remote indicators are in title or description
                    title_remote = any(term in title for term in remote_indicators)
                    desc_remote = any(term in description for term in remote_indicators)

                    # If we're specifically searching for remote jobs, be more aggressive
                    if is_remote_search and (title_remote or desc_remote):
                        is_remote = True
                    # If not specifically searching remote, require stronger signals
                    elif title_remote and desc_remote:
                        is_remote = True

                # Extract company name with fallbacks
                company_name = (
                    str(
                        row.get('company_name')
                        or row.get('company', '')
                        or row.get('company_name_original', '')
                        or ''
                    ).strip()
                    or 'Company Not Specified'
                )

                # Ensure we have a valid job ID
                job_id = str(
                    row.get('job_id')
                    or f"job_{int(datetime.now().timestamp())}_{hash(str(row))}"
                )

                # Ensure we have a valid description
                description = str(
                    row.get('description') or 'No description available'
                ).strip()
                if not description:
                    description = 'No description available'

                # Ensure we have a valid title
                title = str(row.get('title') or 'No Title').strip()
                if not title:
                    title = 'No Title'

                # Helper function to safely get list fields
                def get_safe_list(value, default=None):
                    if value is None:
                        return default or []
                    if isinstance(value, (list, tuple, set)):
                        return list(value)
                    return [value] if value is not None else []

                # Prepare metadata with safe list handling
                metadata = {
                    'job_id': job_id,
                    'company_id': str(row.get('company_id', '')),
                    'job_url': str(row.get('job_url', '')),
                    'company_url': str(row.get('company_url', '')),
                    'location': str(row.get('location', '')),
                    'source': source,
                    'company': company_name,
                    'benefits': get_safe_list(row.get('benefits')),
                    'emails': get_safe_list(row.get('emails')),
                    'logo_photo_url': str(row.get('logo_photo_url', '')),
                    'banner_photo_url': str(row.get('banner_photo_url', '')),
                    'is_remote': bool(is_remote),
                }

                # Only include original_data if it's not too large
                try:
                    row_dict = row.to_dict()
                    if len(str(row_dict)) < 10000:  # Only include if not too large
                        metadata['original_data'] = row_dict
                except Exception as e:
                    logger.debug(f"Could not include original_data in metadata: {e}")

                # Create job data dictionary with all required fields
                job_data = {
                    'id': job_id,
                    'title': title,
                    'company': company_name,
                    'location': str(
                        row.get('location', 'Remote' if is_remote else 'Not specified')
                    ).strip(),
                    'description': description,
                    'url': str(row.get('job_url', '')),
                    'job_type': job_type,
                    'remote': bool(is_remote),
                    'salary': str(salary) if salary is not None else None,
                    'metadata': metadata,
                    'source': JobSource.JOBSPY,
                    'posted_date': posted_date or datetime.now(),
                }

                # Create job object
                job = Job(**job_data)
                jobs.append(job)

            except Exception as e:
                job_id = row.get('job_id', 'unknown')
                logger.error(f"Error processing job {job_id}: {e}", exc_info=True)

        return jobs

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
        return (
            JobType.FULL_TIME
            if any(
                term in job_type_str
                for term in ['full', 'part', 'contract', 'temp', 'intern', 'volunt']
            )
            else JobType.OTHER
        )

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
