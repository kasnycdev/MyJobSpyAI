# Indeed Scraper

## Overview
The Indeed scraper is a robust, production-grade component of MyJobSpyAI that extracts job listings from Indeed's website. It's designed with reliability, maintainability, and performance in mind.

## Features

- **Asynchronous HTTP requests** using `aiohttp` for high performance
- **Circuit breaker pattern** to handle service unavailability gracefully
- **Rate limiting** with automatic retries and backoff
- **Comprehensive error handling** for various failure scenarios
- **Detailed logging** for monitoring and debugging
- **Type hints** throughout the codebase for better maintainability

## Architecture

### Key Components

1. **BaseScraper**
   - Core scraping functionality
   - Common utilities and error handling
   - Base configuration

2. **IndeedScraper**
   - Implements Indeed-specific scraping logic
   - Handles Indeed's HTML structure and API
   - Manages session and cookies

3. **Error Handling**
   - Custom exceptions for different error types
   - Automatic retry for transient failures
   - Circuit breaker for handling service outages

### Code Organization

```
src/myjobspyai/scrapers/
├── __init__.py
├── base.py          # Base scraper class
├── indeed.py        # Indeed scraper implementation
└── exceptions.py    # Custom exceptions
```

## Recent Improvements

### Refactoring (June 2024)

1. **Reduced Method Complexity**
   - Split large methods into smaller, focused ones
   - Improved code readability and maintainability
   - Added comprehensive docstrings

2. **Enhanced Error Handling**
   - Centralized error handling logic
   - Improved error messages and logging
   - Better handling of rate limits and timeouts

3. **Code Quality**
   - Fixed all linting issues
   - Removed trailing whitespace
   - Improved code formatting
   - Added type hints

## Usage Example

```python
from myjobspyai.scrapers import IndeedScraper

async def search_jobs():
    scraper = IndeedScraper()
    try:
        jobs = await scraper.search_jobs(
            query="software engineer",
            location="remote",
            max_results=50
        )
        return jobs
    except Exception as e:
        print(f"Error: {e}")
    finally:
        await scraper.close()
```

## Error Handling

The scraper handles various error conditions:

- **Rate Limiting**: Automatically respects `Retry-After` headers
- **Network Issues**: Implements retry logic with exponential backoff
- **Authentication**: Handles 401/403 errors appropriately
- **Timeouts**: Configurable timeouts for all requests

## Configuration

Configuration is done through the main application config file:

```yaml
scrapers:
  indeed:
    request_timeout: 30  # seconds
    max_retries: 3
    rate_limit_delay: 5  # seconds
```

## Best Practices

1. **Session Management**
   - Always close the scraper when done using `await scraper.close()`
   - Reuse scraper instances when making multiple requests

2. **Rate Limiting**
   - Be respectful of Indeed's rate limits
   - Use appropriate delays between requests
   - Handle rate limit errors gracefully

3. **Error Handling**
   - Always wrap scraping operations in try/except blocks
   - Log errors appropriately
   - Implement proper cleanup in finally blocks

## Performance Considerations

- The scraper uses connection pooling for better performance
- Asynchronous operations allow for concurrent requests
- Circuit breaker prevents cascading failures

## Future Improvements

- Add support for more job boards
- Implement job result caching
- Add more detailed metrics and monitoring
- Support for proxies and rotating user agents
