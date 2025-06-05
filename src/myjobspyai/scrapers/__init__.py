"""
Scrapers package for MyJobSpy AI.

This package contains modules for scraping job listings from various job sites.
"""

from typing import Dict, Type, Optional, List, Any, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Import base classes and types
from .base import BaseJobScraper
from ..models.job import Job, JobType as JobTypeEnum, JobSource

# Import scrapers
from .jobspy_scraper import JobSpyScraper

# Define a type variable for the job model
JobModel = TypeVar('JobModel')

# Export public API
__all__ = [
    'BaseJobScraper',
    'JobSpyScraper',
    'Job',
    'JobTypeEnum as JobType',
    'JobSource',
]


class JobType(Enum):
    """Enum for job types."""

    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    TEMPORARY = "temporary"
    INTERNSHIP = "internship"
    VOLUNTEER = "volunteer"
    OTHER = "other"


@dataclass
class JobListing:
    """Data class representing a job listing."""

    title: str
    company: str
    location: str
    description: str
    job_type: JobType
    url: str
    posted_date: Optional[str] = None
    salary: Optional[str] = None
    remote: bool = False
    source: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the job listing to a dictionary."""
        return {
            "title": self.title,
            "company": self.company,
            "location": self.location,
            "description": self.description,
            "job_type": self.job_type.value,
            "url": self.url,
            "posted_date": self.posted_date,
            "salary": self.salary,
            "remote": self.remote,
            "source": self.source,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "JobListing":
        """Create a JobListing from a dictionary."""
        return cls(
            title=data.get("title", ""),
            company=data.get("company", ""),
            location=data.get("location", ""),
            description=data.get("description", ""),
            job_type=JobType(data.get("job_type", "other")),
            url=data.get("url", ""),
            posted_date=data.get("posted_date"),
            salary=data.get("salary"),
            remote=data.get("remote", False),
            source=data.get("source"),
            metadata=data.get("metadata", {}),
        )


class BaseJobScraper:
    """Base class for job scrapers."""

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Initialize the scraper.

        Args:
            name: Name of the scraper (e.g., 'linkedin', 'indeed')
            config: Configuration dictionary for the scraper
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    async def search_jobs(
        self, query: str, location: str, **kwargs
    ) -> List[JobListing]:
        """Search for jobs matching the given query and location.

        Args:
            query: Job search query (e.g., 'software engineer')
            location: Location for the job search (e.g., 'New York, NY')
            **kwargs: Additional search parameters

        Returns:
            List of JobListing objects

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement search_jobs")

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
        pass
