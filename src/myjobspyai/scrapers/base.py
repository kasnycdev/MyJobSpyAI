"""Base scraper class for job search implementations."""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..models.job import Job, JobSource, JobType

logger = logging.getLogger(__name__)

class BaseJobScraper(ABC):
    """Abstract base class for job scrapers."""

    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize the scraper.

        Args:
            name: Name of the scraper (e.g., 'linkedin', 'indeed', 'jobspy')
            config: Configuration dictionary for the scraper
        """
        self.name = name
        self.config = config
        self._initialized = False

    @abstractmethod
    async def search_jobs(
        self,
        query: str,
        location: str = "",
        max_results: int = 15,
        **kwargs
    ) -> List[Job]:
        """Search for jobs using the scraper.

        Args:
            query: Job search query (e.g., 'software engineer')
            location: Location for the job search (e.g., 'New York, NY')
            max_results: Maximum number of results to return
            **kwargs: Additional search parameters specific to the scraper

        Returns:
            List of Job objects
        """
        pass

    async def close(self):
        """Clean up resources used by the scraper.

        Subclasses should override this if they need to perform cleanup.
        """
        self._initialized = False
        logger.debug(f"{self.name} scraper closed")

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"
