"""Interfaces for job scrapers.

This module defines the standard interfaces that all job scrapers must implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from myjobspyai.models.job import JobType
from myjobspyai.models.job_listing import JobListing


class JobSearchError(Exception):
    """Base exception for job search related errors."""

    pass


class ScraperError(JobSearchError):
    """Base exception for scraper related errors."""

    pass


class AuthenticationError(ScraperError):
    """Raised when authentication with the job board fails."""

    pass


class RateLimitExceeded(ScraperError):
    """Raised when the rate limit for the job board is exceeded."""

    pass


class SearchError(ScraperError):
    """Raised when a search operation fails."""

    pass


class JobNotFound(ScraperError):
    """Raised when a specific job cannot be found."""

    pass


@dataclass
class SearchParams:
    """Parameters for job search.

    Attributes:
        query: The search query string (e.g., "software engineer")
        location: The location for the search (e.g., "New York, NY")
        job_type: Type of job (full-time, part-time, contract, etc.)
        max_results: Maximum number of results to return
        page: Page number for pagination (1-based)
        radius: Search radius in miles/kilometers
        salary_min: Minimum salary
        salary_max: Maximum salary
        posted_after: Only return jobs posted after this date
        remote: Whether to only show remote jobs
        easy_apply: Whether to only show jobs with easy apply
        custom_params: Custom parameters specific to the job board
    """

    query: str
    location: str = ""
    job_type: Optional[JobType] = None
    max_results: int = 50
    page: int = 1
    radius: Optional[int] = None
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    posted_after: Optional[datetime] = None
    remote: bool = False
    easy_apply: bool = False
    custom_params: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class IScraper(Protocol):
    """Protocol defining the interface for all job scrapers.

    This protocol defines the standard interface that all job scrapers must implement.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the scraper.

        Returns:
            str: The name of the scraper (e.g., 'indeed', 'linkedin')
        """
        ...

    @abstractmethod
    async def search_jobs(self, params: SearchParams) -> List[JobListing]:
        """Search for jobs based on the given parameters.

        Args:
            params: Search parameters

        Returns:
            List[JobListing]: List of job listings matching the search criteria

        Raises:
            SearchError: If the search fails
            AuthenticationError: If authentication fails
            RateLimitExceeded: If rate limit is exceeded
        """
        ...

    @abstractmethod
    async def get_job_details(self, job_id: str) -> Optional[JobListing]:
        """Get detailed information about a specific job.

        Args:
            job_id: The unique identifier for the job

        Returns:
            Optional[JobListing]: The job listing with detailed information,
                              or None if not found

        Raises:
            JobNotFound: If the job cannot be found
            ScraperError: If there is an error fetching the job details
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close any resources used by the scraper.

        This should be called when the scraper is no longer needed to clean up
        any resources such as HTTP sessions.
        """
        ...
