"""Indeed job scraper for MyJobSpy AI."""

from __future__ import annotations

import asyncio
import logging
import time
from http import HTTPStatus
from typing import Any, Optional

import aiohttp
from httpx import HTTPStatusError

from myjobspyai.models.job import JobType
from myjobspyai.models.job_listing import JobListing
from myjobspyai.scrapers.base import BaseJobScraper, RateLimitConfig, RetryConfig
from myjobspyai.scrapers.interfaces import (
    AuthenticationError,
    JobNotFound,
    RateLimitExceeded,
    ScraperError,
    SearchError,
    SearchParams,
)

logger = logging.getLogger(__name__)


class IndeedScraper(BaseJobScraper):
    """Scraper for Indeed job listings.

    This class implements the IScraper interface to provide job search
    functionality for Indeed job listings.
    """

    # Constants
    MAX_RESULTS_LIMIT = 100  # Maximum number of results allowed per search

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        rate_limit_config: RateLimitConfig | None = None,
        retry_config: RetryConfig | None = None,
    ) -> None:
        """Initialize the Indeed scraper.

        Args:
            config: Configuration dictionary for the scraper
            rate_limit_config: Configuration for rate limiting. If None, uses defaults.
            retry_config: Configuration for retry behavior. If None, uses defaults.

        Raises:
            ValueError: If configuration values are invalid
        """
        super().__init__("indeed", config, rate_limit_config, retry_config)
        self.base_url = "https://www.indeed.com"

        # Validate configuration
        if config and not isinstance(config, dict):
            raise ValueError("config must be a dictionary or None")

        # Set default configuration
        self._config = config or {}
        self._config.setdefault("max_retries", 3)
        self._config.setdefault("timeout", 30.0)
        self._config.setdefault("follow_redirects", True)
        self._config.setdefault("verify_ssl", True)

        # Initialize circuit breaker state
        self._circuit_open = False
        self._circuit_last_failure = 0.0
        self._circuit_reset_timeout = 60.0  # 1 minute circuit breaker timeout

    def _validate_search_params(self, params: SearchParams) -> None:
        """Validate search parameters.

        Args:
            params: Search parameters to validate

        Raises:
            ValueError: If any parameter is invalid
        """
        if not params.query or not isinstance(params.query, str):
            raise ValueError("Search query must be a non-empty string")

        if params.max_results < 1 or params.max_results > self.MAX_RESULTS_LIMIT:
            raise ValueError(
                f"max_results must be between 1 and {self.MAX_RESULTS_LIMIT}"
            )

        if params.page < 1:
            raise ValueError("page must be greater than 0")

        if params.radius is not None and params.radius < 0:
            raise ValueError("radius cannot be negative")

        if params.salary_min is not None and params.salary_min < 0:
            raise ValueError("salary_min cannot be negative")

        if params.salary_max is not None and params.salary_max < 0:
            raise ValueError("salary_max cannot be negative")

        if (
            params.salary_min is not None
            and params.salary_max is not None
            and params.salary_min > params.salary_max
        ):
            raise ValueError("salary_min cannot be greater than salary_max")

    def _validate_job_id(self, job_id: str) -> None:
        """Validate job ID.

        Args:
            job_id: Job ID to validate

        Raises:
            ValueError: If job ID is invalid
            JobNotFound: If job ID is empty or None
        """
        if not job_id:
            raise JobNotFound("Job ID cannot be empty")

        if not isinstance(job_id, str):
            raise ValueError("Job ID must be a string")

        if len(job_id) > 100:  # Arbitrary limit to prevent abuse
            raise ValueError("Job ID is too long")

    def _check_circuit_breaker(self) -> None:
        """Check if the circuit breaker is open and should be closed.

        Raises:
            ScraperError: If the circuit breaker is open
        """
        if not self._circuit_open:
            return

        now = time.time()
        if now - self._circuit_last_failure >= self._circuit_reset_timeout:
            self.logger.warning("Resetting circuit breaker after timeout")
            self._circuit_open = False
        else:
            raise ScraperError(
                "Service temporarily unavailable due to repeated failures. "
                "Please try again later."
            )

    def _trip_circuit_breaker(self) -> None:
        """Trip the circuit breaker to prevent further requests."""
        self._circuit_open = True
        self._circuit_last_failure = time.time()
        self.logger.error(
            "Circuit breaker tripped due to repeated failures. "
            "No more requests will be made for %d seconds.",
            self._circuit_reset_timeout,
        )

    async def _init_session(self) -> None:
        """Initialize the HTTP session if not already done.

        This method sets up the HTTP client with appropriate headers and
        settings for making requests to Indeed.

        Raises:
            ScraperError: If there is an error initializing the session
        """
        if self._session is not None:
            return

        try:
            from myjobspyai.utils.http_factory import get_http_client

            self._session = await get_http_client(
                base_url=self.base_url,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/91.0.4472.124 Safari/537.36"
                    ),
                    "Accept": (
                        "text/html,application/xhtml+xml,"
                        "application/xml;q=0.9,*/*;q=0.8"
                    ),
                    "Accept-Language": "en-US,en;q=0.5",
                },
                follow_redirects=True,
                timeout=30.0,
            )
            self._initialized = True
            self.logger.debug("Initialized HTTP session for Indeed scraper")
        except Exception as e:
            self.logger.error("Failed to initialize HTTP session: %s", e)
            raise ScraperError(f"Failed to initialize HTTP session: {e}") from e

    async def search_jobs(self, params: SearchParams) -> list[JobListing]:
        """Search for jobs on Indeed based on the given parameters.

        Args:
            params: Search parameters including query, location, and filters

        Returns:
            List of JobListing objects matching the search criteria

        Raises:
            SearchError: If the search fails
            AuthenticationError: If authentication with Indeed fails
            RateLimitExceeded: If the rate limit is exceeded
            ScraperError: For other scraper-related errors
        """
        try:
            await self._init_session()
            await self._enforce_rate_limit()

            # Log the search parameters
            log_params = {
                "query": params.query,
                "location": params.location,
                "job_type": params.job_type,
                "max_results": params.max_results,
                "remote": params.remote,
            }
            self.logger.info("Searching Indeed jobs with params: %s", log_params)

            # Build search URL parameters
            search_params = {
                'q': params.query,
                'l': params.location,
                'limit': min(params.max_results, 50),  # Max 50 per page
                'fromage': 7,  # Last 7 days
                'sort': 'date',  # Most recent first
                'start': 0,
            }

            if params.job_type:
                search_params['jt'] = self._map_job_type_to_indeed(params.job_type)
            if params.remote:
                search_params['remotejob'] = '1'

            # Make the search request
            response = await self._make_request(
                '/jobs',
                params=search_params,
                error_message="Failed to search Indeed jobs",
            )
            # Parse the response
            return await self._parse_search_results(response, search_params)

        except RateLimitExceeded:
            self.logger.warning("Rate limit exceeded for Indeed search")
            raise
        except AuthenticationError as e:
            self.logger.error("Authentication error in Indeed search: %s", e)
            raise
        except Exception as e:
            self.logger.error("Error searching Indeed jobs: %s", e, exc_info=True)
            raise SearchError(f"Failed to search Indeed jobs: {e}") from e

    async def get_job_details(self, job_id: str) -> JobListing | None:
        """Get detailed information for a specific job.

        Args:
            job_id: The unique identifier for the job

        Returns:
            JobListing with detailed information, or None if not found

        Raises:
            JobNotFound: If the job with the given ID cannot be found
            ScraperError: If there is an error fetching the job details
            AuthenticationError: If authentication with Indeed fails
            RateLimitExceeded: If the rate limit is exceeded
        """
        try:
            await self._init_session()
            await self._enforce_rate_limit()

            self.logger.debug("Fetching job details for job ID: %s", job_id)

            # Make the job details request
            response = await self._make_request(
                f'/viewjob?jk={job_id}',
                error_message=f"Failed to fetch job details for ID {job_id}",
            )

            # Parse the job details
            return await self._parse_job_details(response, job_id)

        except RateLimitExceeded:
            self.logger.warning("Rate limit exceeded while fetching job details")
            raise
        except JobNotFound:
            self.logger.warning("Job not found: %s", job_id)
            raise
        except Exception as e:
            error_msg = f"Failed to fetch job details for ID {job_id}"
            self.logger.error(
                "%s: %s",
                error_msg,
                str(e),
                extra={"job_id": job_id},
                exc_info=True,
            )
            raise ScraperError(error_msg) from e

    def _handle_rate_limit(self, response_headers: Optional[dict]) -> None:
        """Handle rate limiting response.

        Args:
            response_headers: Response headers containing rate limit info

        Raises:
            RateLimitExceeded: Always raises with rate limit information
        """
        retry_after = response_headers.get('Retry-After') if response_headers else None
        raise RateLimitExceeded(
            f"Rate limit exceeded. Retry after: {retry_after or 'unknown'}",
            retry_after=retry_after,
        )

    def _handle_http_error(
        self,
        method: str,
        url: str,
        status_code: int,
        error: str,
        params: dict[str, Any] | None,
        response_headers: Optional[dict] = None,
    ) -> None:
        """Handle HTTP error responses.

        Args:
            method: HTTP method used
            url: Request URL
            status_code: HTTP status code
            error: Error message
            params: Request parameters
            response_headers: Response headers (optional)

        Raises:
            RateLimitExceeded: If rate limit is hit
            AuthenticationError: For authentication failures
            JobNotFound: For 404 errors
            ScraperError: For other HTTP errors
        """
        self._log_request_failure(method, url, status_code, error, params)

        if status_code == HTTPStatus.TOO_MANY_REQUESTS:
            self._handle_rate_limit(response_headers)

        if status_code in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}:
            raise AuthenticationError(
                f"Authentication failed with status {status_code}"
            )

        if status_code == HTTPStatus.NOT_FOUND:
            raise JobNotFound(f"Resource not found: {url}")

        raise ScraperError(f"HTTP error {status_code}: {error}")

    async def _execute_http_request(
        self, method: str, url: str, params: dict[str, Any], **kwargs: Any
    ) -> str:
        """Execute the HTTP request and handle successful response.

        Args:
            method: HTTP method
            url: Request URL
            params: Query parameters
            **kwargs: Additional request arguments

        Returns:
            Response text

        Raises:
            RateLimitExceeded: If rate limit is hit
            AuthenticationError: For authentication failures
            HTTPStatusError: For other HTTP errors
        """
        response = await self._with_retry(
            self._execute_request, method=method, url=url, params=params, **kwargs
        )

        self._log_request_success(
            method=method, url=url, status_code=response.status_code, params=params
        )

        # Handle rate limiting
        if response.status_code == HTTPStatus.TOO_MANY_REQUESTS:
            self._handle_rate_limit(response.headers)

        # Handle authentication errors
        if response.status_code in {HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN}:
            raise AuthenticationError(
                f"Authentication failed with status {response.status_code}"
            )

        response.raise_for_status()
        self._circuit_open = False
        return response.text

    async def _make_request(
        self,
        endpoint: str,
        method: str = 'GET',
        params: dict[str, Any] | None = None,
        error_message: str = 'Request failed',
        **kwargs: Any,
    ) -> str:
        """Make an HTTP request to the Indeed API with enhanced error handling.

        Args:
            endpoint: API endpoint to call
            method: HTTP method (GET, POST, etc.)
            params: Query parameters
            error_message: Custom error message for exceptions
            **kwargs: Additional arguments to pass to the request

        Returns:
            Response text

        Raises:
            ScraperError: If the request fails
            RateLimitExceeded: If rate limit is hit
            AuthenticationError: If authentication fails
            ConnectionError: If there is a network issue
            TimeoutError: If the request times out
        """
        self._check_circuit_breaker()
        url = f"{self.base_url}{endpoint}"
        params = params or {}
        self._log_request_start(method, url, params)

        try:
            return await self._execute_http_request(method, url, params, **kwargs)

        except HTTPStatusError as e:
            status_code = e.response.status_code if e.response else 'unknown'
            self._handle_http_error(
                method=method,
                url=url,
                status_code=status_code,
                error=str(e),
                params=params,
                response_headers=e.response.headers if e.response else None,
            )
            raise  # This line is actually unreachable due to _handle_http_error raising

        except asyncio.TimeoutError as e:
            self._log_request_timeout(method, url, params)
            raise TimeoutError(f"Request to {url} timed out") from e

        except (ConnectionError, aiohttp.ClientConnectionError) as e:
            self._log_request_connection_error(method, url, str(e), params)
            raise ConnectionError(f"Connection error: {e}") from e

        except Exception as e:
            self._log_request_error(method, url, str(e), params)
            self._trip_circuit_breaker()
            raise ScraperError(f"{error_message}: {e}") from e

    async def _execute_request(self, method: str, url: str, **kwargs: Any) -> Any:
        """Execute an HTTP request with the current session.

        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional arguments to pass to the request

        Returns:
            The response object

        Raises:
            Exception: If the request fails
        """
        if not self._session or not self._initialized:
            await self._init_session()

        return await self._session.request(method=method, url=url, **kwargs)

    def _log_request_success(
        self,
        method: str,
        url: str,
        status_code: int,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Log a successful HTTP request.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: The URL that was requested
            status_code: HTTP status code of the response
            params: Optional query parameters that were used
        """
        self.logger.debug(
            "%s %s completed with status %d",
            method,
            url,
            status_code,
            extra={
                "method": method,
                "url": url,
                "status_code": status_code,
                "params": params,
            },
        )

    def _log_request_start(
        self, method: str, url: str, params: dict[str, Any] | None = None
    ) -> None:
        """Log the start of an HTTP request.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: The URL being requested
            params: Optional query parameters
        """
        self.logger.debug(
            "Starting %s request to %s",
            method,
            url,
            extra={
                "method": method,
                "url": url,
                "params": params,
            },
        )

    def _log_request_failure(
        self,
        method: str,
        url: str,
        status_code: int | str,
        error: str,
        params: dict[str, Any] | None = None,
    ) -> None:
        """Log a failed HTTP request.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: The URL that was requested
            status_code: HTTP status code of the response
            error: Error message
            params: Optional query parameters that were used
        """
        self.logger.error(
            "%s %s failed with status %s: %s",
            method,
            url,
            status_code,
            error,
            extra={
                "method": method,
                "url": url,
                "status_code": status_code,
                "error": error,
                "params": params,
            },
        )

    def _log_request_timeout(
        self, method: str, url: str, params: dict[str, Any] | None = None
    ) -> None:
        """Log a request timeout.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: The URL that timed out
            params: Optional query parameters that were used
        """
        self.logger.warning(
            "Request to %s %s timed out",
            method,
            url,
            extra={
                "method": method,
                "url": url,
                "params": params,
            },
        )

    def _log_request_connection_error(
        self, method: str, url: str, error: str, params: dict[str, Any] | None = None
    ) -> None:
        """Log a connection error.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: The URL that failed to connect
            error: Error message
            params: Optional query parameters that were used
        """
        self.logger.error(
            "Connection error for %s %s: %s",
            method,
            url,
            error,
            extra={
                "method": method,
                "url": url,
                "error": error,
                "params": params,
            },
        )

    def _log_request_error(
        self, method: str, url: str, error: str, params: dict[str, Any] | None = None
    ) -> None:
        """Log an unexpected error during a request.

        Args:
            method: HTTP method (GET, POST, etc.)
            url: The URL that caused the error
            error: Error message
            params: Optional query parameters that were used
        """
        self.logger.error(
            "Unexpected error for %s %s: %s",
            method,
            url,
            error,
            extra={
                "method": method,
                "url": url,
                "error": error,
                "params": params,
            },
            exc_info=True,
        )

    async def _parse_search_results(
        self,
        html: str,
        params: dict[str, Any] | None = None,  # Unused but kept for interface
    ) -> list[JobListing]:
        """Parse HTML search results from Indeed.

        Args:
            html: Raw HTML response
            params: Original search parameters (unused, kept for interface)

        Returns:
            List of JobListing objects

        Raises:
            SearchError: If parsing fails
        """
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, 'html.parser')

            # Find all job listings in the search results
            job_cards = soup.select('div.jobsearch-SerpJobCard')
            jobs = []

            for card in job_cards:
                try:
                    # Extract job details from the card
                    title_elem = card.select_one('h2.jobTitle')
                    company_elem = card.select_one('span.company')
                    location_elem = card.select_one('div.recJobLoc')
                    summary_elem = card.select_one('div.summary')

                    # Extract job ID from data-jk attribute
                    job_id = card.get('data-jk', '')
                    if not job_id:
                        continue

                    # Create JobListing object with remote flag
                    remote_flag = bool(
                        location_elem and 'remote' in location_elem.text.lower()
                    )

                    job = JobListing(
                        title=(title_elem.text.strip() if title_elem else 'No title'),
                        company=(
                            company_elem.text.strip() if company_elem else 'Unknown'
                        ),
                        location=(
                            location_elem.text.strip() if location_elem else 'Remote'
                        ),
                        description=(
                            summary_elem.text.strip()
                            if summary_elem
                            else 'No description'
                        ),
                        job_type=JobType.OTHER,  # Updated in get_job_details
                        url=f"{self.base_url}/viewjob?jk={job_id}",
                        salary=None,  # Populated by get_job_details
                        source='indeed',
                        remote=remote_flag,
                        metadata={"job_id": job_id},
                    )
                    jobs.append(job)

                except Exception as e:
                    self.logger.warning(
                        "Failed to parse job card: %s",
                        e,
                        exc_info=True,
                    )
                    continue

            return jobs

        except Exception as e:
            self.logger.error(
                "Failed to parse search results: %s",
                e,
                exc_info=True,
            )
            raise SearchError(f"Failed to parse search results: {e}") from e

    async def _parse_job_details(self, html: str, job_id: str) -> JobListing | None:
        """Parse detailed job information from Indeed job page.

        Args:
            html: Raw HTML response
            job_id: The job ID

        Returns:
            JobListing with detailed information, or None if parsing fails

        Raises:
            JobNotFound: If the job cannot be found
            ScraperError: If parsing fails
        """
        try:
            import re

            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, 'html.parser')

            # Check if job not found
            if "We couldn't find this job" in html:
                raise JobNotFound(f"Job with ID {job_id} not found on Indeed")

            # Extract job details
            title_elem = soup.select_one('h1.jobsearch-JobInfoHeader-title')
            company_elem = soup.select_one('div[data-company-name]')
            location_elem = soup.select_one(
                'div[data-testid="jobsearch-JobInfoHeader-companyLocation"]'
            )
            salary_elem = soup.select_one('div#salaryInfoAndJobType')
            description_elem = soup.select_one('div#jobDescriptionText')

            # Extract job type from the job title or description
            job_type = 'fulltime'  # Default
            if description_elem:
                description_text = description_elem.get_text().lower()
                if 'part' in description_text:
                    job_type = 'parttime'
                elif 'contract' in description_text:
                    job_type = 'contract'

            # Extract salary information
            salary = None
            if salary_elem:
                salary = salary_elem.get_text(strip=True)
                # Clean up salary string
                salary = re.sub(r'\s+', ' ', salary).strip()

            # Create JobListing object with remote flag
            remote_flag = bool(location_elem and 'remote' in location_elem.text.lower())
            return JobListing(
                title=title_elem.text.strip() if title_elem else 'No title',
                company=(
                    company_elem.text.strip() if company_elem else 'Unknown company'
                ),
                location=(
                    location_elem.text.strip()
                    if location_elem
                    else 'Location not specified'
                ),
                description=(
                    description_elem.get_text().strip()
                    if description_elem
                    else 'No description'
                ),
                job_type=self._map_job_type(job_type),
                url=f"{self.base_url}/viewjob?jk={job_id}",
                salary=salary,
                source='indeed',
                posted_date=None,  # Not available on detail page
                remote=remote_flag,
                metadata={"job_id": job_id},
            )

        except JobNotFound:
            raise
        except Exception as e:
            self.logger.error(
                "Failed to parse job details: %s",
                e,
                exc_info=True,
            )
            raise ScraperError(f"Failed to parse job details: {e}") from e

    def _map_job_type(self, job_type: str) -> JobType:
        """Map job type string to JobType enum.

        Args:
            job_type: Job type string from Indeed

        Returns:
            JobType enum value
        """
        job_type = job_type.lower()
        if 'part' in job_type:
            return JobType.PART_TIME
        elif 'contract' in job_type:
            return JobType.CONTRACT
        elif 'intern' in job_type:
            return JobType.INTERNSHIP
        elif 'temporary' in job_type or 'temp' in job_type:
            return JobType.TEMPORARY
        return JobType.FULL_TIME

    def _map_job_type_to_indeed(self, job_type: str) -> str:
        """Map our job type to Indeed's job type parameter.

        Args:
            job_type: Our internal job type

        Returns:
            Indeed's job type parameter value
        """
        mapping = {
            'fulltime': 'fulltime',
            'parttime': 'parttime',
            'contract': 'contract',
            'internship': 'internship',
            'temporary': 'temporary',
        }
        return mapping.get(job_type.lower(), '')

    async def close(self) -> None:
        """Clean up resources used by the scraper.

        This method should be called when the scraper is no longer needed
        to properly close any open connections and release resources.
        """
        if self._session:
            self.logger.debug("Closing HTTP session for Indeed scraper")
            await self._session.close()
            self._session = None
            self._initialized = False
