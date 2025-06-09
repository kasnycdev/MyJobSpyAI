"""
Scrapers package for MyJobSpy AI.

This package contains modules for scraping job listings from various job sites.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Generic, List, Optional, Type, TypeVar

# Import scrapers
from .base import BaseJobScraper
from .jobspy_scraper import JobSpyScraper
from .linkedin import LinkedInScraper
from .indeed import IndeedScraper

# Import base classes and types


# Define a type variable for the job model


# Export public API
__all__ = [
    'BaseJobScraper',
    'JobSpyScraper',
    'LinkedInScraper',
    'IndeedScraper',
    'JobListing',
]


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
