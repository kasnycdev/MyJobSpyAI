"""HTTP client utilities for MyJobSpy AI."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
from aiohttp import ClientError, ClientResponse, ClientSession, ClientTimeout

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = ClientTimeout(total=30, connect=10)


class HTTPClientError(Exception):
    """Base exception for HTTP client errors."""

    pass


class HTTPRequestError(HTTPClientError):
    """Exception raised for HTTP request errors."""

    def __init__(
        self,
        message: str,
        status: Optional[int] = None,
        url: Optional[str] = None,
        method: Optional[str] = None,
        response_text: Optional[str] = None,
    ) -> None:
        self.status = status
        self.url = url
        self.method = method
        self.response_text = response_text
        super().__init__(message)


class HTTPClient:
    """Asynchronous HTTP client with retry logic and response caching."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
        headers: Optional[Dict[str, str]] = None,
        raise_for_status: bool = True,
        session: Optional[ClientSession] = None,
    ) -> None:
        """Initialize the HTTP client.

        Args:
            base_url: Base URL for all requests.
            timeout: Default timeout in seconds.
            max_retries: Maximum number of retries for failed requests.
            retry_delay: Initial delay between retries in seconds.
            backoff_factor: Multiplier for the retry delay.
            headers: Default headers to include in all requests.
            raise_for_status: Whether to raise an exception for non-2xx responses.
            session: Optional aiohttp ClientSession to use.
        """
        self.base_url = base_url.rstrip("/") if base_url else None
        self.timeout = (
            ClientTimeout(total=timeout) if timeout is not None else DEFAULT_TIMEOUT
        )
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        self.headers = headers or {}
        self.raise_for_status = raise_for_status
        self._session = session
        self._session_owner = session is None

    async def __aenter__(self) -> "HTTPClient":
        if self._session is None:
            self._session = ClientSession(
                headers=self.headers,
                timeout=self.timeout,
            )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._session_owner and self._session and not self._session.closed:
            await self._session.close()

    async def request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Union[Dict[str, Any], bytes, str]] = None,
        json_data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        raise_for_status: Optional[bool] = None,
        **kwargs: Any,
    ) -> Tuple[ClientResponse, Union[Dict[str, Any], str, bytes]]:
        """Make an HTTP request with retry logic.

        Args:
            method: HTTP method (GET, POST, etc.).
            url: URL to request (can be relative if base_url is set).
            params: Query parameters.
            data: Request body as dict, bytes, or str.
            json_data: JSON-serializable data to send in the request body.
            headers: Additional headers for this request.
            timeout: Timeout in seconds for this request.
            raise_for_status: Whether to raise an exception for non-2xx responses.
            **kwargs: Additional arguments to pass to aiohttp.ClientSession.request.

        Returns:
            A tuple of (response, parsed_response_data).

        Raises:
            HTTPRequestError: If the request fails after all retries.
        """
        if self._session is None:
            raise RuntimeError(
                "HTTPClient is not initialized. Use async with HTTPClient()"
            )

        if raise_for_status is None:
            raise_for_status = self.raise_for_status

        # Build the full URL
        if self.base_url and not (
            url.startswith("http://") or url.startswith("https://")
        ):
            url = f"{self.base_url}/{url.lstrip('/')}"

        # Merge headers
        request_headers = self.headers.copy()
        if headers:
            request_headers.update(headers)

        # Set up retry logic
        retry_count = 0
        last_error = None

        while retry_count <= self.max_retries:
            try:
                # Calculate timeout for this attempt
                attempt_timeout = ClientTimeout(
                    total=timeout or self.timeout.total,
                    connect=timeout or self.timeout.connect,
                )

                # Make the request
                async with self._session.request(
                    method=method,
                    url=url,
                    params=params,
                    data=data,
                    json=json_data,
                    headers=request_headers,
                    timeout=attempt_timeout,
                    **kwargs,
                ) as response:
                    # Parse response
                    content_type = response.headers.get("Content-Type", "")

                    # Try to parse JSON if the content type suggests it
                    if "application/json" in content_type:
                        try:
                            response_data = await response.json()
                        except (json.JSONDecodeError, aiohttp.ContentTypeError):
                            response_data = await response.text()
                    elif "text/" in content_type:
                        response_data = await response.text()
                    else:
                        response_data = await response.read()

                    # Raise for status if configured
                    if raise_for_status and response.status >= 400:
                        raise HTTPRequestError(
                            f"HTTP {response.status} for {method} {url}",
                            status=response.status,
                            url=str(response.url),
                            method=method,
                            response_text=(
                                response_data
                                if isinstance(response_data, str)
                                else None
                            ),
                        )

                    return response, response_data

            except (ClientError, asyncio.TimeoutError) as e:
                last_error = e
                if retry_count >= self.max_retries:
                    break

                # Calculate delay with exponential backoff
                delay = self.retry_delay * (self.backoff_factor**retry_count)
                logger.warning(
                    "Request failed (attempt %d/%d), retrying in %.1fs: %s",
                    retry_count + 1,
                    self.max_retries + 1,
                    delay,
                    str(e),
                )

                await asyncio.sleep(delay)
                retry_count += 1

        # If we get here, all retries failed
        error_msg = f"Request failed after {self.max_retries + 1} attempts"
        if last_error is not None:
            error_msg += f": {last_error}"

        raise HTTPRequestError(
            error_msg,
            url=url,
            method=method,
        )

    # Convenience methods for common HTTP methods

    async def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Tuple[ClientResponse, Union[Dict[str, Any], str, bytes]]:
        """Send a GET request."""
        return await self.request(
            "GET",
            url,
            params=params,
            headers=headers,
            **kwargs,
        )

    async def post(
        self,
        url: str,
        data: Optional[Union[Dict[str, Any], bytes, str]] = None,
        json_data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Tuple[ClientResponse, Union[Dict[str, Any], str, bytes]]:
        """Send a POST request."""
        return await self.request(
            "POST",
            url,
            data=data,
            json_data=json_data,
            headers=headers,
            **kwargs,
        )

    async def put(
        self,
        url: str,
        data: Optional[Union[Dict[str, Any], bytes, str]] = None,
        json_data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Tuple[ClientResponse, Union[Dict[str, Any], str, bytes]]:
        """Send a PUT request."""
        return await self.request(
            "PUT",
            url,
            data=data,
            json_data=json_data,
            headers=headers,
            **kwargs,
        )

    async def delete(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Tuple[ClientResponse, Union[Dict[str, Any], str, bytes]]:
        """Send a DELETE request."""
        return await self.request(
            "DELETE",
            url,
            headers=headers,
            **kwargs,
        )

    async def patch(
        self,
        url: str,
        data: Optional[Union[Dict[str, Any], bytes, str]] = None,
        json_data: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> Tuple[ClientResponse, Union[Dict[str, Any], str, bytes]]:
        """Send a PATCH request."""
        return await self.request(
            "PATCH",
            url,
            data=data,
            json_data=json_data,
            headers=headers,
            **kwargs,
        )


# Helper functions for common HTTP operations


async def fetch_json(
    url: str,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Fetch JSON from a URL with error handling.

    Args:
        url: The URL to fetch.
        method: HTTP method to use.
        params: Query parameters.
        headers: Request headers.
        timeout: Request timeout in seconds.

    Returns:
        The parsed JSON response as a dictionary.

    Raises:
        HTTPRequestError: If the request fails or the response is not valid JSON.
    """
    async with HTTPClient(timeout=timeout) as client:
        response, data = await client.request(
            method=method,
            url=url,
            params=params,
            headers=headers or {},
        )

        if not isinstance(data, dict):
            raise HTTPRequestError(f"Expected JSON response, got {type(data).__name__}")

        return data


async def download_file(
    url: str,
    output_path: Union[str, Path],
    chunk_size: int = 8192,
    timeout: float = 300.0,
) -> Path:
    """Download a file from a URL to the specified path.

    Args:
        url: The URL of the file to download.
        output_path: Where to save the downloaded file.
        chunk_size: Size of chunks to download at a time.
        timeout: Request timeout in seconds.

    Returns:
        The path to the downloaded file.

    Raises:
        HTTPRequestError: If the download fails.
    """
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    async with HTTPClient(timeout=timeout) as client:
        async with client._session.get(url) as response:
            if response.status != 200:
                raise HTTPRequestError(
                    f"Failed to download {url}: HTTP {response.status}",
                    status=response.status,
                    url=url,
                )

            with open(output_path, "wb") as f:
                async for chunk in response.content.iter_chunked(chunk_size):
                    f.write(chunk)

    return output_path
