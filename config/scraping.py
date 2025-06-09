"""Scraping configuration for MyJobSpyAI."""
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryCallState
)
from . import settings

logger = logging.getLogger('scraping')

@dataclass
class ScraperConfig:
    """Configuration for a web scraper."""
    name: str
    base_url: str
    timeout: int
    retry_attempts: int
    retry_delay: int
    request_delay: float = 1.0
    headers: Optional[Dict[str, str]] = None

    def __post_init__(self):
        """Set default headers if not provided."""
        if self.headers is None:
            self.headers = {
                "User-Agent": settings.settings.scraping.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Cache-Control": "max-age=0",
            }

class Scraper:
    """Base class for web scrapers with retry logic."""

    def __init__(self, config: ScraperConfig):
        """Initialize the scraper with configuration."""
        self.config = config
        self.logger = logging.getLogger(f"scraping.{config.name}")

        # Configure retry logic
        self.retry = retry(
            stop=stop_after_attempt(config.retry_attempts),
            wait=wait_exponential(multiplier=1, min=config.retry_delay, max=60),
            retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError)),
            before_sleep=self._log_retry_attempt,
            reraise=True
        )

    def _log_retry_attempt(self, retry_state: RetryCallState) -> None:
        """Log retry attempts."""
        attempt = retry_state.attempt_number
        max_attempts = self.config.retry_attempts
        delay = retry_state.idle_for
        exception = retry_state.outcome.exception() if retry_state.outcome else None

        self.logger.warning(
            f"Retry {attempt}/{max_attempts} after {delay:.2f}s: {str(exception) or 'Unknown error'}",
            extra={
                "attempt": attempt,
                "max_attempts": max_attempts,
                "delay_seconds": delay,
                "exception": str(exception) if exception else None,
                "exception_type": exception.__class__.__name__ if exception else None
            }
        )

    async def fetch(self, url: str, **kwargs) -> str:
        """Fetch a URL with retry logic."""
        return await self.retry(self._fetch_impl)(url, **kwargs)

    async def _fetch_impl(self, url: str, **kwargs) -> str:
        """Implementation of the fetch method."""
        # Add delay between requests
        if hasattr(self, '_last_request_time'):
            elapsed = time.time() - self._last_request_time
            if elapsed < self.config.request_delay:
                await asyncio.sleep(self.config.request_delay - elapsed)

        self._last_request_time = time.time()

        # Prepare request
        headers = self.config.headers.copy()
        headers.update(kwargs.pop('headers', {}))

        timeout = kwargs.pop('timeout', self.config.timeout)

        # Make the request
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.get(
                    url,
                    headers=headers,
                    follow_redirects=True,
                    **kwargs
                )
                response.raise_for_status()
                return response.text

            except httpx.HTTPStatusError as e:
                self.logger.error(
                    f"HTTP error {e.response.status_code} for {url}",
                    extra={"status_code": e.response.status_code, "url": url}
                )
                raise
            except httpx.RequestError as e:
                self.logger.error(
                    f"Request error for {url}: {str(e)}",
                    exc_info=True
                )
                raise

# Factory function to create scrapers
def create_scraper(name: str, **kwargs) -> Scraper:
    """Create a new scraper instance with the given configuration."""
    # Get default config from settings
    default_config = {
        "timeout": settings.settings.scraping.timeouts.get(name, settings.settings.scraping.timeouts["default"]),
        "retry_attempts": settings.settings.scraping.retry_attempts,
        "retry_delay": settings.settings.scraping.retry_delay,
    }

    # Override with provided kwargs
    config = {**default_config, **kwargs, "name": name}

    return Scraper(ScraperConfig(**config))

# Pre-configured scrapers for common job sites
def create_naukri_scraper() -> Scraper:
    """Create a scraper for Naukri.com."""
    return create_scraper(
        "naukri",
        base_url="https://www.naukri.com",
        headers={
            "Referer": "https://www.naukri.com/",
            "DNT": "1",
        }
    )

def create_indeed_scraper() -> Scraper:
    """Create a scraper for Indeed.com."""
    return create_scraper(
        "indeed",
        base_url="https://www.indeed.com",
        headers={
            "Referer": "https://www.indeed.com/",
            "DNT": "1",
        }
    )

def create_linkedin_scraper() -> Scraper:
    """Create a scraper for LinkedIn."""
    return create_scraper(
        "linkedin",
        base_url="https://www.linkedin.com",
        headers={
            "Referer": "https://www.linkedin.com/",
            "Accept": "application/vnd.linkedin.normalized+json+2.1",
            "X-Restli-Protocol-Version": "2.0.0",
            "X-Li-Lang": "en_US",
        },
        request_delay=2.0  # Be more gentle with LinkedIn
    )
