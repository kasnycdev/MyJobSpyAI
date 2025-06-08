# Scrapers

This directory contains modules for scraping job listings from various job sites.

## Structure

```
scrapers/
├── __init__.py           # Package initialization
├── base.py               # Base scraper class and interfaces
├── linkedin.py          # LinkedIn job scraper
├── indeed.py            # Indeed job scraper
└── glassdoor.py         # Glassdoor job scraper
```

## Adding a New Scraper

1. Create a new Python file for the job site (e.g., `monster.py`)
2. Implement a class that inherits from `BaseJobScraper`
3. Implement the required methods
4. Update the scraper factory to include your new scraper

## Example

```python
from .base import BaseJobScraper, JobListing


class ExampleScraper(BaseJobScraper):
    def __init__(self, config=None):
        super().__init__("example", config)

    async def search_jobs(self, query, location, **kwargs):
        # Implementation here
        pass
```
