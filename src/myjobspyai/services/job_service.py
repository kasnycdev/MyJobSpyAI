"""
Job Service for MyJobSpyAI.

This module provides services for searching, filtering, and analyzing job postings.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from myjobspyai.analysis.analyzer import JobAnalyzer
from myjobspyai.models.job_listing import JobListing
from myjobspyai.scrapers.factory import create_scraper

logger = logging.getLogger(__name__)
console = Console()


class JobService:
    """Service for job search and analysis operations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the job service.

        Args:
            config: Configuration dictionary for the service
        """
        self.config = config or {}
        self.scraper = None
        self.analyzer = JobAnalyzer()

    async def initialize_scraper(self, scraper_type: str = "jobspy") -> None:
        """Initialize the job scraper.

        Args:
            scraper_type: Type of scraper to use (default: "jobspy")
        """
        try:
            self.scraper = create_scraper(scraper_type, self.config.get("scrapers", {}).get(scraper_type, {}))
            logger.info(f"Initialized {scraper_type} scraper")
        except Exception as e:
            logger.error(f"Failed to initialize {scraper_type} scraper: {e}")
            raise

    async def search_jobs(
        self,
        query: str = "",
        location: str = "",
        max_results: int = 15,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Search for jobs using the configured scraper.

        Args:
            query: Search query string
            location: Location to search in
            max_results: Maximum number of results to return
            **kwargs: Additional search parameters

        Returns:
            List of job dictionaries
        """
        if not self.scraper:
            await self.initialize_scraper()

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task(description="Searching for jobs...", total=None)
                jobs = await self.scraper.search_jobs(
                    query=query,
                    location=location,
                    max_results=max_results,
                    **kwargs
                )

                # Convert jobs to dicts for easier handling
                jobs_list = []
                for job in jobs:
                    if hasattr(job, 'model_dump'):
                        # Pydantic v2 model
                        job_dict = job.model_dump()
                    elif hasattr(job, 'dict'):
                        # Pydantic v1 model (fallback)
                        job_dict = job.dict()
                    elif isinstance(job, dict):
                        # Already a dictionary
                        job_dict = job
                    else:
                        # For other objects, convert to dict safely
                        job_dict = {}
                        for attr in dir(job):
                            if not attr.startswith('_') and not callable(getattr(job, attr)):
                                try:
                                    value = getattr(job, attr)
                                    # Handle nested Pydantic models
                                    if hasattr(value, 'model_dump'):
                                        value = value.model_dump()
                                    elif hasattr(value, 'dict'):
                                        value = value.dict()
                                    job_dict[attr] = value
                                except Exception as e:
                                    logger.warning(f"Could not get attribute {attr} from job object: {e}")
                    jobs_list.append(job_dict)

                return jobs_list

        except Exception as e:
            logger.error(f"Error during job search: {str(e)}", exc_info=True)
            raise

    async def filter_jobs_by_salary(
        self, jobs: List[Dict[str, Any]], min_salary: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Filter jobs by minimum salary.

        Args:
            jobs: List of job dictionaries
            min_salary: Minimum salary threshold

        Returns:
            Filtered list of jobs
        """
        if min_salary is None:
            return jobs

        try:
            filtered_jobs = []
            for job in jobs:
                if not job.get('salary') or not job['salary'].get('min_amount'):
                    filtered_jobs.append(job)
                elif job['salary']['min_amount'] >= min_salary:
                    filtered_jobs.append(job)

            filtered_count = len(jobs) - len(filtered_jobs)
            if filtered_count > 0:
                logger.info(
                    f"Filtered out {filtered_count} jobs with confirmed salaries below ${min_salary:,.0f}"
                )
                logger.info(
                    f"Kept {len(filtered_jobs)} jobs "
                    f"(including {len([j for j in filtered_jobs if not j.get('salary') or not j['salary'].get('min_amount')])} with unknown salaries)"
                )

            return filtered_jobs

        except Exception as e:
            logger.error(f"Error applying min-salary filter: {str(e)}", exc_info=True)
            return jobs

    async def analyze_jobs(
        self, jobs: List[Dict[str, Any]], resume_data: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Analyze jobs with optional resume comparison.

        Args:
            jobs: List of job dictionaries
            resume_data: Optional resume data for analysis

        Returns:
            List of analyzed job dictionaries
        """
        if not resume_data:
            return jobs

        try:
            analyzed_jobs = []
            for job in jobs:
                try:
                    analysis = await self.analyzer.analyze_resume_suitability(
                        resume_data=resume_data,
                        job_data=job
                    )
                    job['_analysis'] = analysis
                    analyzed_jobs.append(job)
                except Exception as e:
                    logger.error(f"Error analyzing job: {e}", exc_info=True)
                    job['_analysis'] = {"error": str(e)}
                    analyzed_jobs.append(job)
            return analyzed_jobs
        except Exception as e:
            logger.error(f"Error during job analysis: {e}", exc_info=True)
            return jobs

    @staticmethod
    def save_jobs_to_file(
        jobs: List[Dict[str, Any]],
        output_file: Union[str, Path],
        output_format: str = 'json'
    ) -> None:
        """Save jobs to a file in the specified format.

        Args:
            jobs: List of job dictionaries
            output_file: Path to the output file
            output_format: Output format (json, csv, xlsx, markdown)
        """
        try:
            df = pd.DataFrame(jobs)

            # Flatten analysis if it exists
            if '_analysis' in df.columns:
                analysis = df.pop('_analysis')
                if 'suitability_score' in analysis:
                    df['match_score'] = analysis['suitability_score']
                if 'pros' in analysis:
                    df['strengths'] = analysis['pros'].apply(
                        lambda x: "; ".join(x[:3]) if x else ""
                    )
                if 'cons' in analysis:
                    df['areas_for_improvement'] = analysis['cons'].apply(
                        lambda x: "; ".join(x[:3]) if x else ""
                    )

            if output_format == 'csv':
                df.to_csv(output_file, index=False)
            elif output_format == 'xlsx':
                df.to_excel(output_file, index=False)
            elif output_format == 'markdown':
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(df.to_markdown(index=False))
            else:  # default to json
                df.to_json(output_file, orient='records', indent=2, force_ascii=False)

            logger.info(f"Saved {len(jobs)} jobs to {output_file}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}", exc_info=True)
            raise
