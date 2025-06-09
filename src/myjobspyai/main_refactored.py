"""
Refactored main module for MyJobSpy AI application.

This module has been refactored to use the new service layer.
"""

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from myjobspyai.config import config as app_config
from myjobspyai.services import JobService, ResumeService
from myjobspyai.utils.logging_config import setup_logging_custom

# Initialize console for Rich output
console = Console()
logger = logging.getLogger(__name__)

# Create a reusable table creation function
def _create_table(show_details: bool = False) -> Table:
    """Create a table for displaying job listings.

    Args:
        show_details: Whether to show detailed columns

    Returns:
        Table: Rich Table object
    """
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", style="cyan", no_wrap=True)
    table.add_column("Company", style="green")
    table.add_column("Location", style="yellow")
    table.add_column("Type", style="blue")
    table.add_column("Posted", style="purple")

    if show_details:
        table.add_column("Salary", style="magenta")
        table.add_column("Match Score", style="green")
        table.add_column("Strengths", style="green")
        table.add_column("Areas for Improvement", style="yellow")

    return table

def parse_args(args: List[str]) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        args: Command line arguments

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="MyJobSpy AI - Job Search and Analysis Tool")

    # Search parameters
    search_group = parser.add_argument_group("Search Parameters")
    search_group.add_argument(
        "--search",
        type=str,
        default="",
        help="Search query (job title, keywords, etc.)"
    )
    search_group.add_argument(
        "--location",
        type=str,
        default="",
        help="Location for job search"
    )
    search_group.add_argument(
        "--scraper",
        type=str,
        default="jobspy",
        choices=["jobspy", "linkedin", "indeed"],
        help="Job scraper to use"
    )
    search_group.add_argument(
        "--min-salary",
        type=float,
        help="Minimum salary threshold"
    )

    # Resume and analysis
    analysis_group = parser.add_argument_group("Resume and Analysis")
    analysis_group.add_argument(
        "--resume",
        type=str,
        help="Path to resume file for analysis"
    )
    analysis_group.add_argument(
        "--analyze",
        action="store_true",
        help="Enable job analysis against resume"
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output",
        type=str,
        help="Output file to save results"
    )
    output_group.add_argument(
        "--format",
        type=str,
        choices=["json", "csv", "xlsx", "markdown"],
        default="json",
        help="Output format"
    )
    output_group.add_argument(
        "--details",
        action="store_true",
        help="Show detailed job information"
    )

    # General options
    general_group = parser.add_argument_group("General Options")
    general_group.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    general_group.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (can be used multiple times)"
    )
    general_group.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit"
    )

    return parser.parse_args(args)

async def main_async() -> int:
    """Async main entry point for the application."""
    try:
        # Parse command line arguments
        args = parse_args(sys.argv[1:])

        # Handle version flag
        if getattr(args, 'version', False):
            console.print(f"MyJobSpy AI v{app_config.version}")
            return 0

        # Set up logging
        setup_logging_custom(debug=getattr(args, 'debug', False))

        # Initialize services
        job_service = JobService(app_config.dict())
        resume_service = ResumeService(app_config.dict())

        # Load and parse resume if provided
        resume_data = None
        if getattr(args, 'resume', None):
            console.print(f"[blue]Loading and analyzing resume: {args.resume}[/blue]")
            resume_data = await resume_service.parse_resume(args.resume)
            if not resume_data:
                console.print(
                    "[yellow]Warning: Could not parse resume. Analysis will be limited.[/yellow]"
                )

        # Search for jobs
        jobs = await job_service.search_jobs(
            query=args.search,
            location=args.location,
            max_results=15
        )

        # Apply salary filter if specified
        if getattr(args, 'min_salary', None) is not None:
            jobs = await job_service.filter_jobs_by_salary(
                jobs,
                min_salary=args.min_salary
            )

        # Analyze jobs if resume is provided and analysis is enabled
        if resume_data and getattr(args, 'analyze', False):
            jobs = await job_service.analyze_jobs(jobs, resume_data)

        # Display results
        if getattr(args, 'output', None):
            await job_service.save_jobs_to_file(
                jobs,
                args.output,
                getattr(args, 'format', 'json')
            )
            console.print(f"[green]Results saved to {args.output}[/green]")
        else:
            # Display in console
            display_jobs_table(jobs, console, show_details=getattr(args, 'details', False))

        return 0

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        console.print(f"[red]Error: {e}[/red]")
        return 1

def main() -> int:
    """Main entry point for the application."""
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("\nApplication interrupted by user")
        return 0
    except Exception as e:
        logger.exception("Fatal error in main function")
        print(f"Fatal error: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
