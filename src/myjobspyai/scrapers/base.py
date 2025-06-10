"""Base scraper class for job search implementations.

This module provides a base class for all job scrapers with common functionality
like rate limiting, retries, and error handling.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from myjobspyai.models.job import JobListing, JobType
from myjobspyai.scrapers.interfaces import IScraper, SearchParams

logger = logging.getLogger(__name__)


class RateLimitConfig(BaseModel):
    """Configuration for rate limiting."""

    max_requests: int = Field(
        default=10,
        description="Maximum number of requests allowed in the time window",
        gt=0,
    )
    time_window: float = Field(
        default=1.0,
        description="Time window in seconds for rate limiting",
        gt=0.0,
    )
    delay: float = Field(
        default=0.5,
        description="Minimum delay between requests in seconds",
        ge=0.0,
    )


class RetryConfig(BaseModel):
    """Configuration for retry behavior."""

    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts",
    )
    initial_delay: float = Field(
        default=1.0,
        description="Initial delay between retries in seconds",
    )
    max_delay: float = Field(
        default=30.0,
        description="Maximum delay between retries in seconds",
    )
    backoff_factor: float = Field(
        default=2.0,
        description="Exponential backoff factor for retries",
    )
    retry_on: list[type[Exception]] = Field(
        default_factory=lambda: [
            ConnectionError,
            TimeoutError,
            asyncio.TimeoutError,
        ],
        description="Exceptions that should trigger a retry",
    )


class BaseJobScraper(IScraper, ABC):
    """Abstract base class for job scrapers with rate limiting and retry support.

    This class provides a foundation for implementing job scrapers with built-in
    rate limiting, retry logic, and error handling. It implements the IScraper
    interface and provides common functionality that can be reused by concrete
    scraper implementations.
    """

    def __init__(
        self,
        name: str,
        config: dict[str, Any] | None = None,
        rate_limit_config: RateLimitConfig | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        """Initialize the scraper with configuration.

        Args:
            name: Name of the scraper (e.g., 'linkedin', 'indeed', 'jobspy')
            config: Configuration dictionary for the scraper
            rate_limit_config: Configuration for rate limiting. If None, uses defaults.
            retry_config: Configuration for retry behavior. If None, uses defaults.

        Raises:
            ValueError: If name is empty or None
        """
        if not name:
            raise ValueError("Scraper name cannot be empty")

        self._name = name
        self.config = config or {}
        self._initialized = False
        self._session = None

        # Initialize rate limiting
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self._request_timestamps: list[float] = []
        self._last_request_time = 0.0

        # Initialize retry configuration
        self.retry_config = retry_config or RetryConfig()

        # Initialize logger with scraper name
        self.logger = logging.getLogger(f"{__name__}.{self._name}")

    @property
    def name(self) -> str:
        """Get the name of the scraper.

        Returns:
            str: The name of the scraper.
        """
        return self._name

    async def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting by sleeping if necessary."""
        now = time.time()

        # Remove timestamps older than the time window
        self._request_timestamps = [
            ts
            for ts in self._request_timestamps
            if now - ts < self.rate_limit_config.time_window
        ]

        # Check if we've exceeded the rate limit
        if len(self._request_timestamps) >= self.rate_limit_config.max_requests:
            sleep_time = self.rate_limit_config.time_window - (
                now - self._request_timestamps[0]
            )
            if sleep_time > 0:
                self.logger.debug(
                    "Rate limit reached. Sleeping for %.2f seconds",
                    sleep_time,
                )
                await asyncio.sleep(sleep_time)

        # Enforce minimum delay between requests
        if self.rate_limit_config.delay > 0 and self._last_request_time > 0:
            time_since_last = now - self._last_request_time
            if time_since_last < self.rate_limit_config.delay:
                sleep_time = self.rate_limit_config.delay - time_since_last
                await asyncio.sleep(sleep_time)

        self._request_timestamps.append(time.time())
        self._last_request_time = time.time()

    async def _with_retry(
        self,
        func: Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Execute a function with retry logic.

        Args:
            func: The async function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            The result of the function call

        Raises:
            Exception: If all retry attempts fail, the last exception is raised
        """
        last_exception = None
        delay = self.retry_config.initial_delay

        for attempt in range(self.retry_config.max_retries + 1):
            try:
                await self._enforce_rate_limit()
                return await func(*args, **kwargs)
            except tuple(self.retry_config.retry_on) as e:
                last_exception = e
                if attempt < self.retry_config.max_retries:
                    # Calculate next delay with exponential backoff
                    delay = min(
                        delay * self.retry_config.backoff_factor,
                        self.retry_config.max_delay,
                    )

                    self.logger.warning(
                        "Attempt %d/%d failed: %s. Retrying in %.1f seconds...",
                        attempt + 1,
                        self.retry_config.max_retries,
                        str(e),
                        delay,
                    )

                    await asyncio.sleep(delay)
                else:
                    # If we get here, all retries failed
                    raise last_exception from e

    async def _init_session(self) -> None:
        """Initialize the HTTP session if not already done.

        This method sets up the HTTP client with appropriate timeouts, retries,
        and headers. It should be called before making any HTTP requests.
        """
        if self._initialized and self._session is not None:
            return

        try:
            from myjobspyai.utils.http_factory import get_http_client

            self.logger.debug("Initializing HTTP session for %s", self.name)

            # Default headers with a common user agent
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/91.0.4472.124 Safari/537.36"
                ),
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;"
                    "q=0.9,image/webp,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0",
                **self.config.get("headers", {}),
            }

            # Get HTTP client with configured settings
            self._session = await get_http_client(
                base_url=self.config.get("base_url"),
                timeout=self.config.get("timeout", 30),
                max_retries=self.config.get("max_retries", 3),
                headers=headers,
                follow_redirects=self.config.get("follow_redirects", True),
                verify_ssl=self.config.get("verify_ssl", True),
            )

            self._initialized = True
            self.logger.debug("Successfully initialized HTTP session")

        except Exception as e:
            self.logger.error("Failed to initialize HTTP session: %s", str(e))
            self._initialized = False
            raise

    @abstractmethod
    async def search_jobs(self, params: SearchParams) -> list[JobListing]:
        """Search for jobs based on the given parameters.

        This method must be implemented by subclasses to provide specific
        job search functionality for different job boards.

        Args:
            params: Search parameters including query, location, filters, etc.

        Returns:
            List of JobListing objects matching the search criteria

        Raises:
            ScraperError: If there is an error during the search
            AuthenticationError: If authentication with the job board fails
            RateLimitExceeded: If the rate limit for the job board is exceeded
            SearchError: If the search parameters are invalid or the search fails
        """
        raise NotImplementedError("Subclasses must implement search_jobs")

    async def get_job_details(self, job_id: str) -> JobListing | None:
        """Get detailed information about a specific job.

        This method can be overridden by subclasses to provide detailed
        job information. The base implementation returns None.

        Args:
            job_id: Unique identifier for the job

        Returns:
            JobListing with detailed information, or None if not found

        Raises:
            JobNotFound: If the job with the given ID cannot be found
            ScraperError: If there is an error fetching the job details
            AuthenticationError: If authentication with the job board fails
            RateLimitExceeded: If the rate limit for the job board is exceeded
        """
        return None

    async def close(self) -> None:
        """Clean up resources.

        This method should be called when the scraper is no longer needed
        to properly close any open connections and release resources.
        """
        if self._session:
            try:
                self.logger.debug("Closing HTTP session for %s", self.name)
                await self._session.close()
                self._session = None
                self._initialized = False
                self.logger.debug("Successfully closed HTTP session")
            except Exception as e:
                self.logger.error("Error closing HTTP session: %s", str(e))
                raise

    async def __aenter__(self):
        """Async context manager entry."""
        await self._init_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def _get_default_headers(self) -> dict[str, str]:
        """Get default HTTP headers for requests.

        Returns:
            Dictionary of default headers
        """
        return {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
            "Sec-Fetch-User": "?1",
            "Cache-Control": "max-age=0",
        }

    def __str__(self) -> str:
        """Return a string representation of the scraper."""
        return f"{self.__class__.__name__}(name={self.name})"

    def _map_job_type(self, job_type_str: str) -> JobType | None:
        """Map a job type string to a JobType enum value.

        Args:
            job_type_str: The job type string to map.

        Returns:
            The mapped JobType, or None if no match found.
        """
        if not job_type_str:
            return None

        job_type_str = job_type_str.lower()
        type_map = {
            "full-time": JobType.FULL_TIME,
            "full time": JobType.FULL_TIME,
            "fulltime": JobType.FULL_TIME,
            "ft": JobType.FULL_TIME,
            "part-time": JobType.PART_TIME,
            "part time": JobType.PART_TIME,
            "parttime": JobType.PART_TIME,
            "pt": JobType.PART_TIME,
            "contract": JobType.CONTRACT,
            "contractor": JobType.CONTRACT,
            "temporary": JobType.TEMPORARY,
            "temp": JobType.TEMPORARY,
            "internship": JobType.INTERNSHIP,
            "intern": JobType.INTERNSHIP,
            "volunteer": JobType.VOLUNTEER,
            "freelance": JobType.FREELANCE,
            "permanent": JobType.PERMANENT,
        }

        job_type = type_map.get(job_type_str)
        if job_type is None:
            self.logger.warning("Could not map job type: %s", job_type_str)
        return job_type
