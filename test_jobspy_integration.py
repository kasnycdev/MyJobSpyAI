#!/usr/bin/env python3
"""
Test script to verify JobSpy integration with MyJobSpy AI.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = str(Path(__file__).parent.resolve())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.myjobspyai.scrapers.factory import create_scraper
from src.myjobspyai.scrapers.jobspy_scraper import JobSpyScraper

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_jobspy_scraper():
    """Test the JobSpy scraper integration."""
    print("Testing JobSpy scraper integration...")

    try:
        # Create a JobSpy scraper instance
        scraper = create_scraper('jobspy')

        # Test search parameters
        search_params = {
            'query': 'python developer',
            'location': 'Remote',
            'max_results': 5,
            'is_remote': True,
            'job_type': 'fulltime',
            'verbose': 2
        }

        print(f"Searching for jobs with params: {search_params}")

        # Search for jobs
        jobs = await scraper.search_jobs(**search_params)

        if not jobs:
            print("No jobs found. This might be expected if there are no matching jobs.")
            return False

        print(f"\nFound {len(jobs)} jobs:")
        for i, job in enumerate(jobs, 1):
            print(f"\n{i}. {job.title} at {job.company}")
            print(f"   Location: {job.location}")
            print(f"   Type: {job.job_type}")
            print(f"   Remote: {'Yes' if job.remote else 'No'}")
            print(f"   URL: {job.url}")

        return True

    except Exception as e:
        print(f"Error testing JobSpy scraper: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if 'scraper' in locals():
            await scraper.close()

if __name__ == "__main__":
    # Run the test
    success = asyncio.run(test_jobspy_scraper())
    sys.exit(0 if success else 1)
