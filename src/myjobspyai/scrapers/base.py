"""Base scraper class for job search implementations."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar

from ..models.job import JobType
from ..models.job_listing import JobListing

T = TypeVar('T')

logger = logging.getLogger(__name__)


class BaseJobScraper(ABC):
    """Abstract base class for job scrapers."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the scraper.

        Args:
            name: Name of the scraper (e.g., 'linkedin', 'indeed', 'jobspy')
            config: Configuration dictionary for the scraper
        """
        self.name = name
        self.config = config or {}
        self._initialized = False
        self._session = None

    async def _init_session(self):
        """Initialize the HTTP session if not already done."""
        if self._initialized:
            return

        from myjobspyai.utils.http_factory import get_http_client

        self._session = await get_http_client(
            timeout=self.config.get("timeout", 30),
            max_retries=self.config.get("max_retries", 3),
            headers=self.config.get("headers", {})
        )
        self._initialized = True

    @abstractmethod
    async def search_jobs(
        self,
        query: str,
        location: str = "",
        max_results: int = 15,
        **kwargs
    ) -> List[JobListing]:
        """Search for jobs using the scraper.

        Args:
            query: Job search query (e.g., 'software engineer')
            location: Location for the job search (e.g., 'New York, NY')
            max_results: Maximum number of results to return
            **kwargs: Additional search parameters specific to the scraper

        Returns:
            List of JobListing objects
        """
        pass

    async def get_job_details(self, job_url: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific job.

        Args:
            job_url: URL of the job listing

        Returns:
            Dictionary containing detailed job information, or None if not found
        """
        self.logger.warning(
            f"get_job_details not implemented for {self.name}, returning None"
        )
        return None

    async def close(self):
        """Clean up resources used by the scraper."""
        if self._session:
            await self._session.close()
            self._session = None
        self._initialized = False
        logger.debug(f"{self.name} scraper closed")

    def __str__(self) -> str:
        """Return a string representation of the scraper."""
        return f"{self.__class__.__name__}(name={self.name})"

    @staticmethod
    def map_job_type(job_type_str: str) -> JobType:
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
