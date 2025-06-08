MMMMMMMMMMMMMMMMMMMMMMMMMM

------------------------------------------------------------------------

> (Transition content)

------------------------------------------------------------------------

------------------------------------------------------------------------

> (Transition content)

------------------------------------------------------------------------

> (Transition content)

------------------------------------------------------------------------

> (Transition content)

------------------------------------------------------------------------

> (Transition content)

------------------------------------------------------------------------

> members
>
> :
>
> undoc-members
>
> :
>
> show-inheritance
>
> :

Submodule s

------------------------------------------------------------------------

# Job Scrapers Module

RRRRRRRRRRRRRRRRRRRRR

------------------------------------------------------------------------

> (Transition content)

------------------------------------------------------------------------

------------------------------------------------------------------------

> (Transition content)

------------------------------------------------------------------------

> (Transition content)

------------------------------------------------------------------------

> (Transition content)

------------------------------------------------------------------------

> (Transition content)

------------------------------------------------------------------------

> members
>
> :
>
> undoc-members
>
> :
>
> show-inheritance
>
> :

# myjobspyai.scrapers

## Overview

The scrapers module provides functionality for scraping job listings from various sources. It includes base classes, specific scraper implementations, and a factory for creating scraper instances.

## Classes

### BaseScraper

Base class for job scrapers.

#### Initialization

```python
scraper = BaseScraper(
    config={
        "source": "indeed",
        "location": "New York",
        "search_terms": ["software engineer", "python developer"]
    }
)
```

#### Methods

##### \_validate_config

```python
# Validate the scraper configuration
is_valid = scraper._validate_config()
```

Validates the current scraper configuration against requirements.

##### scrape

```python
# Scrape job listings
job_listings = await scraper.scrape()
```

Scrapes job listings from the configured source.

##### close

```python
# Clean up resources
await scraper.close()
```

Closes the scraper's resources, including any active connections or sessions.

### Factory

Factory class for creating scrapers.

#### Methods

##### create_scraper

```python
# Create a scraper instance
scraper = Factory.create_scraper(
    "indeed",
    config={
        "location": "New York",
        "search_terms": ["software engineer"]
    }
)
```

Creates a scraper instance based on configuration.

### IndeedScraper

Scraper for Indeed job listings.

#### Methods

##### \_validate_config

```python
# Validate Indeed scraper configuration
is_valid = scraper._validate_config()
```

Validates the configuration specific to Indeed.

##### scrape

```python
# Scrape job listings from Indeed
job_listings = await scraper.scrape()
```

Scrapes job listings from Indeed.com.

### JobSpyScraper

Scraper for JobSpy job listings.

#### Methods

##### \_validate_config

```python
# Validate JobSpy scraper configuration
is_valid = scraper._validate_config()
```

Validates the configuration specific to JobSpy.

##### scrape

```python
# Scrape job listings from JobSpy
job_listings = await scraper.scrape()
```

Scrapes job listings from JobSpy.

### LinkedInScraper

Scraper for LinkedIn job listings.

#### Methods

##### \_validate_config

```python
# Validate LinkedIn scraper configuration
is_valid = scraper._validate_config()
```

Validates the configuration specific to LinkedIn.

##### scrape

```python
# Scrape job listings from LinkedIn
job_listings = await scraper.scrape()
```

Scrapes job listings from LinkedIn.

## Configuration

The scrapers module supports the following configuration options:

```python
{
    "source": "indeed",  # Source identifier (indeed, linkedin, etc.)
    "location": "New York",  # Job location
    "search_terms": ["software engineer", "python developer"],  # Search terms
    "max_pages": 5,  # Maximum pages to scrape
    "delay": 1.0,    # Delay between requests
    "proxy": None,   # Proxy configuration (optional)
    "headers": {     # Request headers
        "User-Agent": "Mozilla/5.0"
    }
}
```

## Error Handling

The scrapers module implements robust error handling:

- Invalid configuration
- Network errors
- Rate limiting
- Authentication failures
- Resource cleanup failures

All errors are wrapped in `ScraperError` with detailed context and status codes.

## Usage Example

```python
from myjobspyai.scrapers import Factory

# Create scraper
scraper = Factory.create_scraper(
    "indeed",
    config={
        "location": "New York",
        "search_terms": ["software engineer", "python developer"],
        "max_pages": 3,
        "delay": 1.5
    }
)

try:
    # Scrape job listings
    job_listings = await scraper.scrape()
    print(f"Found {len(job_listings)} job listings")
finally:
    # Clean up
    await scraper.close()
```

## Best Practices

1. Always validate scraper configurations
2. Use appropriate delays between requests
3. Handle errors gracefully
4. Clean up resources properly
5. Monitor performance
6. Use configuration validation
7. Follow security best practices
8. Implement proper error handling
9. Use appropriate headers and proxies
10. Monitor resource usage

## Source-Specific Notes

### Indeed
- Supports pagination
- Requires appropriate delays
- Handles rate limiting
- Supports multiple search terms

### LinkedIn
- Requires authentication
- Supports filtering by experience
- Handles session management
- Supports location filtering

### JobSpy
- Specialized job board
- Supports skill-based filtering
- Handles structured data
- Supports resume matching
