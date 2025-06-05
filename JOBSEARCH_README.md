# Job Search Functionality

This document provides comprehensive documentation for the job search functionality in MyJobSpyAI, including the integration with the JobSpy library for scraping job listings from various job boards.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [CLI Usage](#cli-usage)
  - [Programmatic Usage](#programmatic-usage)
- [JobSpy Integration](#jobspy-integration)
- [Job Model](#job-model)
- [Scraper Architecture](#scraper-architecture)
- [Error Handling](#error-handling)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

The job search functionality allows users to search for job listings across multiple job boards using the JobSpy library. It provides a unified interface for searching, filtering, and processing job listings from various sources.

## Features

- Search for jobs across multiple job boards
- Filter by job type (full-time, part-time, contract, etc.)
- Filter by location and remote work availability
- Support for multiple output formats (JSON, CSV, XLSX, Markdown)
- Interactive job viewing in the terminal
- Extensible scraper architecture
- Rate limiting and error handling

## Installation

1. Ensure you have Python 3.8+ installed
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Install Playwright browsers (required for some job boards):

```bash
playwright install
```

## Configuration

Configuration can be done through environment variables or a configuration file at `~/.config/myjobspyai/config.yaml`.

Example configuration:

```yaml
jobsearch:
  default_scraper: jobspy
  default_limit: 10
  default_location: "Remote"
  default_job_type: "fulltime"
  output_dir: "./job_results"

jobspy:
  max_retries: 3
  request_timeout: 30
  headless: true
  verbose: 1
```

## Usage

### CLI Usage

```bash
# Basic search
python -m myjobspyai jobsearch --query "python developer" --location "Remote"

# With job type filter
python -m myjobspyai jobsearch --query "data scientist" --job-type fulltime --remote-only

# Limit number of results
python -m myjobspyai jobsearch --query "machine learning" --max-results 20

# Save results to file
python -m myjobspyai jobsearch --query "devops" --output-format json --output-file jobs.json

# Interactive mode
python -m myjobspyai jobsearch --query "python" --interactive
```

### Programmatic Usage

```python
from myjobspyai.scrapers.factory import create_scraper

async def search_jobs():
    # Create a scraper instance
    scraper = create_scraper('jobspy')

    # Search for jobs
    jobs = await scraper.search_jobs(
        query="python developer",
        location="Remote",
        max_results=10,
        job_type="fulltime",
        is_remote=True
    )

    # Process results
    for job in jobs:
        print(f"{job.title} at {job.company}")
        print(f"Location: {job.location}")
        print(f"Type: {job.job_type}")
        print(f"Remote: {'Yes' if job.remote else 'No'}")
        print(f"URL: {job.url}")
        print("-" * 80)

    # Clean up
    await scraper.close()
```

## JobSpy Integration

The JobSpy integration provides access to multiple job boards through a single interface. The following job boards are supported:

- LinkedIn Jobs
- Indeed
- Glassdoor
- ZipRecruiter
- Google Jobs (limited)
- Naukri
- Bayt

### JobSpy Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| query | str | Required | Job search query |
| location | str | "" | Job location |
| max_results | int | 10 | Maximum number of results to return |
| job_type | str | None | Job type (fulltime, parttime, contract, etc.) |
| is_remote | bool | False | Whether to include only remote jobs |
| distance | int | 50 | Search radius in miles |
| easy_apply | bool | False | Whether to filter for easy apply jobs |
| verbose | int | 0 | Verbosity level (0-2) |

## Job Model

The `Job` model represents a job listing with the following fields:

- `id`: Unique identifier for the job
- `title`: Job title
- `company`: Company name
- `location`: Job location
- `description`: Job description (HTML or plain text)
- `job_type`: Type of job (full-time, part-time, etc.)
- `url`: URL to the job posting
- `posted_date`: When the job was posted
- `salary`: Salary information
- `remote`: Whether the job is remote
- `source`: Source of the job listing
- `metadata`: Additional metadata (varies by source)

## Scraper Architecture

The job search functionality is built around a flexible scraper architecture:

1. **BaseJobScraper**: Abstract base class defining the scraper interface
2. **JobSpyScraper**: Implementation using the JobSpy library
3. **Scraper Factory**: Creates and manages scraper instances

### Adding a New Scraper

To add a new scraper:

1. Create a new class that inherits from `BaseJobScraper`
2. Implement the required methods
3. Register the scraper in the factory

Example:

```python
from myjobspyai.scrapers.base import BaseJobScraper

class MyCustomScraper(BaseJobScraper):
    def __init__(self, config=None):
        super().__init__(config)
        # Initialize your scraper

    async def search_jobs(self, **kwargs):
        # Implement job search logic
        pass

    async def close(self):
        # Clean up resources
        pass
```

## Error Handling

The job search functionality includes comprehensive error handling:

- Network timeouts and retries
- Rate limiting
- Invalid responses
- Missing or malformed data

## Troubleshooting

### Common Issues

1. **No results found**
   - Check your internet connection
   - Verify your search parameters
   - Try a different location or search term

2. **Rate limiting**
   - Some job boards may block frequent requests
   - Use the `--delay` parameter to slow down requests
   - Consider using proxies

3. **Playwright issues**
   - Ensure Playwright browsers are installed
   - Run `playwright install` if needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
