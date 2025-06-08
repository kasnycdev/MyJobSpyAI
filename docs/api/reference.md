# API Reference

## Overview

This section provides comprehensive documentation for the MyJobSpyAI API, including all available endpoints, request/response formats, and usage examples.

## Core API

### Job Analysis

```python
# Job analysis endpoints
from myjobspyai.analysis import JobAnalyzer

analyzer = JobAnalyzer()

# Analyze a job description
results = analyzer.analyze(
    description="Senior Python Developer with 5+ years experience",
    location="San Francisco",
    industry="Technology"
)

# Extract skills
skills = analyzer.extract_skills(description)

# Estimate salary
salary_range = analyzer.estimate_salary(description)
```

### Resume Matching

```python
# Resume matching endpoints
from myjobspyai.matching import ResumeMatcher

matcher = ResumeMatcher()

# Match resume to job description
match_score = matcher.match(
    resume_text="Experienced Python developer with 5 years experience",
    job_description="Senior Python Developer position"
)

# Get matching details
match_details = matcher.get_match_details()
```

### Job Scraping

```python
# Job scraping endpoints
from myjobspyai.scrapers import LinkedInScraper

scraper = LinkedInScraper()

# Scrape jobs
jobs = scraper.scrape(
    query="Software Engineer",
    location="San Francisco",
    pages=5
)

# Get job details
job_details = scraper.get_job_details(job_id)
```

## Data Models

### Job Model

```python
from myjobspyai.models import Job

class Job:
    title: str
    company: str
    location: str
    description: str
    salary_range: tuple[float, float]
    skills: list[str]
    requirements: list[str]
    posting_date: datetime
    source: str
```

### Resume Model

```python
from myjobspyai.models import Resume

class Resume:
    name: str
    email: str
    phone: str
    skills: list[str]
    experience: list[Experience]
    education: list[Education]
    summary: str
    
class Experience:
    company: str
    title: str
    start_date: datetime
    end_date: datetime
    description: str
    
class Education:
    institution: str
    degree: str
    field: str
    graduation_date: datetime
```

## Error Handling

```python
# Common error handling
try:
    results = analyzer.analyze(description)
except JobAnalysisError as e:
    print(f"Analysis failed: {e}")
    
try:
    jobs = scraper.scrape(query, location)
except ScrapingError as e:
    print(f"Scraping failed: {e}")
```

## Configuration

```yaml
# Configuration options
api:
  version: "1.0.0"
  debug: false
  timeout: 30
  
analysis:
  model_path: "models/job_analysis"
  language: "en"
  
scraping:
  max_retries: 3
  retry_delay: 5
  user_agent: "MyJobSpyAI/1.0"
```

## Best Practices

!!! tip "API Usage Tips"
    - Always validate input data
    - Implement proper error handling
    - Use async/await for scraping
    - Cache results when possible
    - Monitor API usage
    - Implement rate limiting
    - Use proper authentication
    - Keep dependencies up to date

## Troubleshooting

=== "Analysis"
    ```python
    # Common analysis issues
    try:
        results = analyzer.analyze(description)
    except AnalysisError as e:
        print(f"Analysis failed: {e}")
        # Check description length
        if len(description) < 100:
            print("Description too short")
    ```

=== "Scraping"
    ```python
    # Scraping rate limiting
    try:
        jobs = scraper.scrape(query, location)
    except RateLimitError:
        print("Rate limit exceeded")
        time.sleep(60)  # Wait for 1 minute
    ```

=== "Matching"
    ```python
    # Resume matching issues
    try:
        score = matcher.match(resume, job)
    except MatchingError as e:
        print(f"Matching failed: {e}")
        # Check required fields
        if not resume.skills or not job.skills:
            print("Missing required fields")
    ```

## Performance Optimization

### Caching

```python
# Cache implementation
from functools import lru_cache

@lru_cache(maxsize=128)
def analyze_job(description: str) -> dict:
    """Cache job analysis results"""
    return analyzer.analyze(description)
```

### Async Processing

```python
# Async scraping
async def scrape_jobs(query: str, location: str) -> list[Job]:
    """Scrape jobs asynchronously"""
    scraper = LinkedInScraper()
    return await scraper.scrape(query, location)
```

### Batch Processing

```python
# Batch analysis
async def analyze_batch(jobs: list[str]) -> list[dict]:
    """Analyze multiple jobs in parallel"""
    tasks = [analyzer.analyze(job) for job in jobs]
    return await asyncio.gather(*tasks)
```

## Security Considerations

!!! warning "Security Best Practices"
    - Validate all input data
    - Use HTTPS for API calls
    - Implement proper authentication
    - Rate limit API requests
    - Use secure headers
    - Keep dependencies updated
    - Regular security audits
    - Implement proper logging
