"""
Factory for creating HTTP clients with consistent configuration.
"""

import asyncio
from typing import Optional, Dict, Any

from aiohttp import ClientSession
from aiohttp.client import ClientTimeout

from .http_client import HTTPClient
from ..config import config as app_config


class HTTPClientFactory:
    """Factory for creating HTTP clients with consistent configuration."""

    _instance = None
    _session: Optional[ClientSession] = None
    _lock = asyncio.Lock()

    def __new__(cls):
        """Ensure singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    async def get_client(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> HTTPClient:
        """Get an HTTP client with consistent configuration.

        Args:
            base_url: Base URL for the client
            timeout: Request timeout in seconds
            headers: Request headers
            **kwargs: Additional client configuration

        Returns:
            Configured HTTP client instance
        """
        async with self._lock:
            if self._session is None:
                self._session = ClientSession(
                    timeout=ClientTimeout(
                        total=timeout or app_config.http.timeout,
                        connect=app_config.http.connect_timeout,
                    ),
                    headers=headers or {},
                )

        return HTTPClient(
            session=self._session,
            base_url=base_url,
            timeout=timeout or app_config.http.timeout,
            headers=headers or {},
            **kwargs,
        )

    async def close(self) -> None:
        """Close the shared session."""
        async with self._lock:
            if self._session and not self._session.closed:
                await self._session.close()
                self._session = None

# Singleton instance
http_client_factory = HTTPClientFactory()

async def get_http_client(
    base_url: Optional[str] = None,
    timeout: Optional[float] = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs: Any,
) -> HTTPClient:
    """Convenience function to get an HTTP client."""
    return await http_client_factory.get_client(
        base_url=base_url,
        timeout=timeout,
        headers=headers,
        **kwargs,
    )
