"""Main module for MyJobSpy AI application."""

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import aiohttp
import pandas as pd
import yaml
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from myjobspyai.analysis.analyzer import JobAnalyzer, ResumeAnalyzer
from myjobspyai.analysis.models import JobAnalysisResult, ResumeData
from myjobspyai.config import config as app_config
from myjobspyai.main_matcher import load_and_extract_resume_async

# Import application components
from myjobspyai.scrapers import JobType
from myjobspyai.scrapers.factory import create_scraper

# Import JobSpy if available
try:
    from jobspy import scrape_jobs
    JOBSPY_AVAILABLE = True
except ImportError:
    JOBSPY_AVAILABLE = False

# Set up logging before other imports that might log
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[RichHandler(rich_tracebacks=True, markup=False)]
)
logger = logging.getLogger("myjobspyai")

# Set up rich console
console = Console()


def parse_config_overrides(overrides: List[str]) -> Dict[str, Any]:
    """Parse configuration overrides from command line arguments.

    Args:
        overrides: List of key=value strings.

    Returns:
        Dictionary of configuration overrides.
    """
    result = {}
    for override in overrides:
        if "=" not in override:
            logger.warning(f"Invalid config override (missing '='): {override}")
            continue

        key, value = override.split("=", 1)
        keys = key.split(".")

        # Handle nested keys
        current = result
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Convert value to appropriate type
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass  # Keep as string

        current[keys[-1]] = value

    return result


def setup_logging_custom(debug: bool = False) -> None:
    """Configure logging for the application.

    Args:
        debug: Enable debug logging if True.
    """
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


def parse_args(args: list[str]) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        args: Command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="MyJobSpy AI - AI-powered job search and analysis tool"
    )

    # Required arguments
    required_group = parser.add_argument_group('Required Arguments')
    required_group.add_argument(
        '--search',
        type=str,
        required=True,
        help='Search term (e.g., "software engineer" or "data scientist")',
    )
    required_group.add_argument(
        '--resume',
        type=str,
        required=True,
        help='Path to resume file (PDF or DOCX)',
    )

    # Scraper selection
    scraper_group = parser.add_argument_group('Scraper Options')
    scraper_group.add_argument(
        '--scraper',
        type=str,
        default='jobspy',
        choices=['jobspy', 'linkedin', 'indeed'],
        help='Scraper to use (default: jobspy)',
    )

    # Search options
    search_group = parser.add_argument_group('Search Options')
    search_group.add_argument(
        '--site-name',
        type=str,
        nargs='+',
        choices=['linkedin', 'indeed', 'glassdoor', 'google', 'zip_recruiter', 'bayt', 'naukri'],
        help='Job sites to search (default: all supported sites)',
    )
    search_group.add_argument(
        '--search-term',
        type=str,
        help='Search term (e.g., "software engineer" or "data scientist")',
    )
    search_group.add_argument(
        '--google-search-term',
        type=str,
        help='Search term specifically for Google Jobs (overrides --search-term for Google)',
    )
    search_group.add_argument(
        '--location',
        type=str,
        default='',
        help='Location for job search (e.g., "New York, NY" or "Remote")',
    )
    search_group.add_argument(
        '--distance',
        type=int,
        default=50,
        help='Distance in miles from location (default: 50)',
    )
    search_group.add_argument(
        '--job-type',
        type=str,
        choices=['fulltime', 'parttime', 'contract', 'internship'],
        help='Type of job to search for',
    )
    search_group.add_argument(
        '--results-wanted',
        type=int,
        default=15,
        help='Number of results wanted per site (default: 15)',
    )
    search_group.add_argument(
        '--hours-old',
        type=int,
        help='Maximum age of job postings in hours',
    )
    search_group.add_argument(
        '--is-remote',
        action='store_true',
        help='Only show remote jobs',
    )
    search_group.add_argument(
        '--easy-apply',
        action='store_true',
        help='Filter for jobs that support quick apply',
    )
    search_group.add_argument(
        '--offset',
        type=int,
        default=0,
        help='Start results from this offset (default: 0)',
    )
    search_group.add_argument(
        '--description-format',
        type=str,
        choices=['markdown', 'html'],
        default='markdown',
        help='Format for job descriptions (default: markdown)',
    )
    search_group.add_argument(
        '--proxies',
        type=str,
        nargs='+',
        help='List of proxies to use (format: user:pass@host:port)',
    )
    search_group.add_argument(
        '--enforce-annual-salary',
        action='store_true',
        help='Convert wages to annual salary',
    )
    search_group.add_argument(
        '--min-salary',
        type=float,
        help='Minimum annual salary to filter jobs',
    )
    search_group.add_argument(
        '--country-indeed',
        type=str,
        help='Country for Indeed/Glassdoor searches (e.g., "USA", "Canada")',
    )

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory to save output files (default: output/)',
    )
    output_group.add_argument(
        '--output-format',
        type=str,
        choices=['json', 'csv', 'xlsx', 'markdown'],
        default='json',
        help='Output format for results (default: json)',
    )
    output_group.add_argument(
        '--no-save',
        action='store_true',
        help="Don't save results to disk (only show in console)",
    )

    # Proxy settings
    proxy_group = parser.add_argument_group('Proxy Settings')
    proxy_group.add_argument(
        '--proxy',
        type=str,
        action='append',
        help='Proxy server to use (format: user:pass@host:port or localhost)',
    )
    proxy_group.add_argument(
        '--ca-cert',
        type=str,
        help='Path to CA certificate file for SSL verification',
    )

    # LinkedIn specific options
    linkedin_group = parser.add_argument_group('LinkedIn Options')
    linkedin_group.add_argument(
        '--linkedin-fetch-description',
        action='store_true',
        help='Fetch full job descriptions from LinkedIn (increases requests)',
    )
    linkedin_group.add_argument(
        '--linkedin-company-ids',
        type=int,
        nargs='+',
        help='Filter LinkedIn jobs by company IDs',
    )

    # Runtime options
    runtime_group = parser.add_argument_group('Runtime Options')
    verbose_group = runtime_group.add_mutually_exclusive_group()
    verbose_group.add_argument(
        '-v', '--verbose',
        action='count',
        default=2,
        help='Increase verbosity (can be used multiple times, e.g., -vv for debug)',
    )
    verbose_group.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress all output except errors',
    )
    runtime_group.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (more verbose output)',
    )
    runtime_group.add_argument(
        '--version',
        action='store_true',
        help='Show version and exit',
    )
    runtime_group.add_argument(
        '--interactive',
        action='store_true',
        help='Enable interactive mode to view job details',
    )

    return parser.parse_args(args)


async def initialize_llm_provider(
    provider_config: Dict[str, Any],
) -> Optional[Any]:
    """Initialize an LLM provider from configuration.

    Args:
        provider_config: Provider configuration.

    Returns:
        Initialized LLM provider or None if initialization failed.
    """
    provider_type = provider_config.get("type", "").lower()
    config = provider_config.get("config", {})

    if not provider_type:
        logger.error("Provider type not specified in configuration")
        return None

    try:
        if provider_type == "openai":
            from langchain_openai import ChatOpenAI

            # Handle LM Studio (local) configuration
            if config.get("base_url", "").startswith(("http://", "https://")):
                return ChatOpenAI(
                    base_url=config["base_url"],
                    model=config.get("model", "local-model"),
                    api_key=config.get("api_key", "not-needed"),
                    temperature=config.get("temperature", 0.7),
                    max_tokens=config.get("max_tokens", 1000),
                    streaming=config.get("streaming", True),
                    request_timeout=config.get("timeout", 60)
                )
            # Standard OpenAI configuration
            return ChatOpenAI(
                model=config.get("model", "gpt-4-turbo-preview"),
                api_key=config.get("api_key") or os.getenv("OPENAI_API_KEY"),
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 1000),
                streaming=config.get("streaming", True),
                request_timeout=config.get("timeout", 60),
                organization=config.get("organization") or os.getenv("OPENAI_ORG_ID")
            )

        elif provider_type == "anthropic":
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=config.get("model", "claude-3-opus-20240229"),
                api_key=config.get("api_key") or os.getenv("ANTHROPIC_API_KEY"),
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 1000)
            )

        elif provider_type == "google":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=config.get("model", "gemini-pro"),
                google_api_key=config.get("api_key") or os.getenv("GOOGLE_API_KEY"),
                temperature=config.get("temperature", 0.7)
            )

        elif provider_type == "langchain":
            # For backward compatibility with LangChain provider
            return LangChainProvider(config)

        elif provider_type == "ollama":
            try:
                from langchain.callbacks.streaming_stdout import (
                    StreamingStdOutCallbackHandler,
                )
                from langchain_community.llms import Ollama
                from langchain_core.callbacks.manager import CallbackManager

                # Set up callbacks for streaming if enabled
                callbacks = []
                if config.get("streaming", False):
                    callbacks = [StreamingStdOutCallbackHandler()]

                # Initialize Ollama with the specified configuration
                return Ollama(
                    base_url=config.get("base_url", "http://localhost:11434"),
                    model=config.get("model", "llama2"),  # Default to llama2 if not specified
                    temperature=config.get("temperature", 0.7),
                    top_p=config.get("top_p", 1.0),
                    callback_manager=CallbackManager(callbacks) if callbacks else None,
                    timeout=config.get("timeout", 60),
                    num_predict=config.get("max_tokens", 1000)
                )
            except ImportError as e:
                logger.error("Failed to import Ollama dependencies. Install with: pip install langchain-community")
                return None
            except Exception as e:
                logger.error(f"Failed to initialize Ollama provider: {e}", exc_info=True)
                return None

        else:
            logger.warning(f"Unsupported LLM provider: {provider_type}")
            return None

    except ImportError as e:
        logger.error(f"Failed to import dependencies for {provider_type} provider: {e}")
        logger.info("You may need to install additional packages. For example:")
        if provider_type == "openai":
            logger.info("pip install langchain-openai")
        elif provider_type == "anthropic":
            logger.info("pip install langchain-anthropic")
        elif provider_type == "google":
            logger.info("pip install langchain-google-genai")
        return None

    except Exception as e:
        logger.error(f"Failed to initialize {provider_type} provider: {e}", exc_info=True)
        return None


def display_jobs_table(jobs: list, title: str = "Job Search Results"):
    """Display job search results in a formatted table with analysis."""
    if not jobs:
        console.print("[yellow]No jobs to display.[/yellow]")
        return

    # Check if we have analysis results
    has_analysis = bool(jobs and isinstance(jobs[0], dict) and '_analysis' in jobs[0])

    # Create a table with appropriate columns
    table = Table(title=title, show_header=True, header_style="bold magenta")

    # Add columns
    table.add_column("#", style="dim", width=4)
    table.add_column("Title")
    table.add_column("Company")
    table.add_column("Location")
    table.add_column("Type")
    table.add_column("Remote")
    table.add_column("Posted")

    # Add analysis column if we have analysis
    if has_analysis:
        table.add_column("Match Score", justify="right")

    # Add rows
    for i, job in enumerate(jobs, 1):
        # Handle both object and dict job formats
        if isinstance(job, dict):
            title = job.get('title', 'N/A')
            company = job.get('company', 'N/A')
            location = job.get('location', 'N/A')
            job_type = job.get('job_type', 'N/A')
            is_remote = "✅" if job.get('is_remote', False) else "❌"
            posted_date = job.get('posted_date', 'N/A')

            # Get analysis score if available
            match_score = ""
            if has_analysis and '_analysis' in job and job['_analysis']:
                analysis = job['_analysis']
                if 'suitability_score' in analysis and analysis['suitability_score'] is not None:
                    score = analysis['suitability_score']
                    # Color code the score
                    if score >= 75:
                        match_score = f"[green]{score}%[/green]"
                    elif score >= 50:
                        match_score = f"[yellow]{score}%[/yellow]"
                    else:
                        match_score = f"[red]{score}%[/red]"
        else:
            # Original object access
            title = getattr(job, 'title', 'N/A')
            company = getattr(job, 'company', 'N/A')
            location = getattr(job, 'location', 'N/A')
            job_type = getattr(job, 'job_type', 'N/A')
            is_remote = "✅" if getattr(job, 'is_remote', False) else "❌"
            posted_date = getattr(job, 'posted_date', 'N/A')
            match_score = ""

        # Format the date if it's a datetime object
        if hasattr(posted_date, 'strftime'):
            posted_date = posted_date.strftime('%Y-%m-%d')

        # Build row data
        row_data = [
            str(i),
            str(title)[:50] + ("..." if len(str(title)) > 50 else ""),
            str(company)[:20] + ("..." if len(str(company)) > 20 else ""),
            str(location)[:20] + ("..." if len(str(location)) > 20 else ""),
            str(job_type),
            is_remote,
            str(posted_date)
        ]

        # Add match score if available
        if has_analysis:
            row_data.append(match_score)

        table.add_row(*row_data)

    # Print the table
    console.print(table)

    # Print analysis legend if we have analysis
    if has_analysis:
        console.print("\n[bold]Match Score Legend:[/bold]")
        console.print("  [green]75-100%:[/green] Strong match")
        console.print("  [yellow]50-74%:[/yellow] Moderate match")
        console.print("  [red]0-49%:[/red] Weak match")

    console.print(f"\n[yellow]Found {len(jobs)} jobs. Use the job number to view details.[/yellow]")


def display_job_details(job):
    """Display detailed information about a job with analysis if available."""
    console = Console()

    # Handle both dict and object job formats
    is_dict = isinstance(job, dict)

    # Get job details
    title = job.get('title') if is_dict else getattr(job, 'title', 'N/A')
    company = job.get('company') if is_dict else getattr(job, 'company', 'N/A')
    location = job.get('location') if is_dict else getattr(job, 'location', 'N/A')
    job_type = job.get('job_type') if is_dict else getattr(job, 'job_type', 'N/A')
    remote = job.get('is_remote') if is_dict else getattr(job, 'is_remote', False)
    posted_date = job.get('posted_date') if is_dict else getattr(job, 'posted_date', None)
    description = job.get('description') if is_dict else getattr(job, 'description', '')
    url = job.get('url') if is_dict else getattr(job, 'url', '')

    # Get analysis if available
    analysis = job.get('_analysis') if is_dict and '_analysis' in job else None

    # Create a panel for the job details
    console.print(f"\n[bold blue]{'=' * 80}[/bold blue]")
    console.print(f"[bold]{title}[/bold]")
    console.print(f"[bold cyan]{company}[/bold]")
    console.print(f"[yellow]{location}[/yellow]")

    # Job type and remote status
    job_type_str = str(job_type or "Not specified").replace("_", " ").title()
    remote_status = "✅ Remote" if remote else "❌ On-site"
    console.print(f"{job_type_str} • {remote_status}")

    # Display match score if analysis is available
    if analysis and 'suitability_score' in analysis and analysis['suitability_score'] is not None:
        score = analysis['suitability_score']
        # Color code the score
        if score >= 75:
            score_display = f"[green]{score}% Match[/green]"
        elif score >= 50:
            score_display = f"[yellow]{score}% Match[/yellow]"
        else:
            score_display = f"[red]{score}% Match[/red]"
        console.print(f"\n[bold]Match Score:[/bold] {score_display}")

        # Display strengths if available
        if 'pros' in analysis and analysis['pros']:
            console.print("\n[bold green]Strengths:[/bold green]")
            for strength in analysis['pros'][:3]:  # Limit to top 3 strengths
                console.print(f"  • {strength}")

        # Display areas for improvement if available
        if 'cons' in analysis and analysis['cons']:
            console.print("\n[bold yellow]Areas for Improvement:[/bold yellow]")
            for con in analysis['cons'][:3]:  # Limit to top 3 areas
                console.print(f"  • {con}")

    # Posted date
    if posted_date:
        if hasattr(posted_date, 'strftime'):
            posted_date = posted_date.strftime('%Y-%m-%d')
        console.print(f"\n[dim]Posted: {posted_date}[/dim]")

    # Job description
    if description:
        console.print("\n[bold]Description:[/bold]")
        console.print(description[:1000] + ("..." if len(description) > 1000 else ""))

    # Salary information if available
    salary = job.get('salary') if is_dict else (getattr(job, 'salary', None) if hasattr(job, 'salary') else None)
    if salary:
        salary_str = []

        # Handle both dict and object salary formats
        min_amount = salary.get('min_amount') if isinstance(salary, dict) else getattr(salary, 'min_amount', None)
        max_amount = salary.get('max_amount') if isinstance(salary, dict) else getattr(salary, 'max_amount', None)
        currency = salary.get('currency') if isinstance(salary, dict) else getattr(salary, 'currency', None)
        period = salary.get('period') if isinstance(salary, dict) else getattr(salary, 'period', None)

        if min_amount is not None:
            salary_str.append(f"${min_amount:,.0f}")
        if max_amount is not None:
            if salary_str:
                salary_str.append("-")
            salary_str.append(f"${max_amount:,.0f}")
        if currency:
            salary_str.append(currency)
        if period:
            salary_str.append(f"per {period}")

        if salary_str:
            console.print("\n[bold]Salary:[/bold]", " ".join(salary_str))

    # Application URL if available
    if url:
        console.print(f"\n[bold]Apply:[/bold] {url}")

    console.print(f"[bold blue]{'=' * 80}[/bold blue]\n")


async def analyze_jobs_with_resume(jobs: List[Any], resume_data: ResumeData) -> List[Dict[str, Any]]:
    """Analyze jobs against a resume.

    Args:
        jobs: List of job objects to analyze
        resume_data: Parsed resume data

    Returns:
        List of job dicts with analysis results
    """
    if not jobs:
        return []

    job_analyzer = JobAnalyzer()
    analyzed_jobs = []

    for job in jobs:
        try:
            # Convert job to dict if it's an object
            job_dict = job.model_dump() if hasattr(job, 'model_dump') else dict(job)

            # Add analysis results
            analysis = await job_analyzer.analyze_resume_suitability(
                resume_data=resume_data,
                job_data=job_dict
            )

            # Add analysis to job data
            job_dict['_analysis'] = analysis.dict() if hasattr(analysis, 'dict') else analysis
            analyzed_jobs.append(job_dict)

        except Exception as e:
            logger.error(f"Error analyzing job {getattr(job, 'title', 'Unknown')}: {e}")
            # Add job without analysis if analysis fails
            job_dict = job.model_dump() if hasattr(job, 'model_dump') else dict(job)
            job_dict['_analysis'] = {'error': str(e)}
            analyzed_jobs.append(job_dict)

    return analyzed_jobs

async def search_jobs(args) -> int:
    """Search for jobs using the specified scraper."""
    resume_data = None
    analyzed_jobs = []  # Initialize to empty list

    # Load and analyze resume if provided
    if hasattr(args, 'resume') and args.resume:
        try:
            console.print(f"[blue]Loading and analyzing resume: {args.resume}[/blue]")
            resume_data = await load_and_extract_resume_async(args.resume)
            if not resume_data:
                console.print("[yellow]Warning: Could not parse resume. Analysis will be limited.[/yellow]")
        except Exception as e:
            logger.error(f"Error loading resume: {e}", exc_info=True)
            console.print(f"[yellow]Warning: Error loading resume: {e}. Analysis will be limited.[/yellow]")

    try:
        # Get search query from args
        search_query = getattr(args, 'search', '')
        if not search_query:
            console.print("[red]Error: Search query is required[/red]")
            return 1

        # Use JobSpy by default
        scraper_type = getattr(args, 'scraper', 'jobspy')

        # Check if JobSpy is available if that's the selected scraper
        if scraper_type == 'jobspy' and not JOBSPY_AVAILABLE:
            console.print("[yellow]Warning: JobSpy is not installed. Falling back to default scraper.[/yellow]")
            console.print("  Install with: pip install python-jobspy")
            scraper_type = 'linkedin'  # Fall back to LinkedIn scraper

        # Get scraper config from app_config
        scraper_config = {}
        if hasattr(app_config, 'scrapers') and scraper_type in app_config.scrapers:
            scraper_config = app_config.scrapers[scraper_type]
            logger.debug(f"Using scraper config from app_config: {scraper_config}")

        # Create scraper instance with config
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            progress.add_task(description=f"Initializing {scraper_type} scraper...", total=None)
            try:
                scraper = create_scraper(scraper_type, scraper_config)
            except Exception as e:
                logger.error(f"Failed to create {scraper_type} scraper: {e}", exc_info=True)
                console.print(f"[red]Error: Failed to initialize {scraper_type} scraper: {str(e)}[/red]")
                if scraper_type == 'jobspy' and 'No module named' in str(e):
                    console.print("\n[bold]JobSpy is not installed. Please install it with:[/bold]")
                    console.print("pip install python-jobspy")
                return 1

        # Prepare search parameters, starting with config values
        search_params = {}

        # Add query, location, and max_results from args, falling back to config
        search_params['query'] = getattr(args, 'search', search_query)
        search_params['location'] = getattr(args, 'location', scraper_config.get('location', ''))
        search_params['max_results'] = getattr(args, 'results_wanted', scraper_config.get('max_results', 15))

        # Add JobSpy specific parameters if using JobSpy
        if scraper_type == 'jobspy':
            # Define JobSpy parameters with defaults from config, overridden by command-line args
            jobspy_params = {
                'site_name': getattr(args, 'site_name', scraper_config.get('site_name')),
                'distance': getattr(args, 'distance', scraper_config.get('distance', 50)),
                'job_type': getattr(args, 'job_type', scraper_config.get('job_type')),
                'is_remote': getattr(args, 'is_remote', scraper_config.get('is_remote', False)),
                'easy_apply': getattr(args, 'easy_apply', scraper_config.get('easy_apply', False)),
                'offset': getattr(args, 'offset', scraper_config.get('offset', 0)),
                'hours_old': getattr(args, 'hours_old', scraper_config.get('hours_old')),
                'country_indeed': getattr(args, 'country_indeed', scraper_config.get('country_indeed')),
                'description_format': getattr(args, 'description_format',
                                           scraper_config.get('description_format', 'markdown')),
                'linkedin_fetch_description': getattr(args, 'linkedin_fetch_description',
                                                   scraper_config.get('linkedin_fetch_description', False)),
                'linkedin_company_ids': getattr(args, 'linkedin_company_ids',
                                             scraper_config.get('linkedin_company_ids')),
                'proxies': getattr(args, 'proxies', scraper_config.get('proxies')),
                'ca_cert': getattr(args, 'ca_cert', scraper_config.get('ca_cert')),
                'enforce_annual_salary': getattr(args, 'enforce_annual_salary',
                                              scraper_config.get('enforce_annual_salary', False)),
                'verbose': getattr(args, 'verbose', scraper_config.get('verbose', 2))
            }
            # Remove None values
            search_params.update({k: v for k, v in jobspy_params.items() if v is not None and v != ''})

        # Filter out None values
        search_params = {k: v for k, v in search_params.items() if v is not None}

        # Execute search with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task(description="Searching for jobs...", total=None)
            try:
                # Execute the job search
                jobs = await scraper.search_jobs(**search_params)

                # Convert jobs to dicts for easier handling
                jobs_list = [job.model_dump() if hasattr(job, 'model_dump') else dict(job) for job in jobs]

                # Apply min-salary filter if specified
                min_salary = getattr(args, 'min_salary', None)
                if min_salary is not None:
                    original_count = len(jobs_list)
                    filtered_jobs = []
                    for job in jobs_list:
                        # Keep jobs where we can't determine the salary (None or missing)
                        if not job.get('salary') or not job['salary'].get('min_amount'):
                            filtered_jobs.append(job)
                        # Keep jobs where salary meets or exceeds minimum
                        elif job['salary']['min_amount'] >= min_salary:
                            filtered_jobs.append(job)

                    filtered_count = original_count - len(filtered_jobs)
                    jobs_list = filtered_jobs
                    if filtered_count > 0:
                        logger.info(f"Filtered out {filtered_count} jobs with confirmed salaries below ${min_salary:,.0f}")
                        logger.info(f"Kept {len(jobs_list)} jobs (including {len([j for j in jobs_list if not j.get('salary') or not j['salary'].get('min_amount')])} with unknown salaries)")

                # Analyze jobs against resume if resume is provided
                if resume_data:
                    progress.update(task, description="Analyzing job matches...")
                    jobs_list = await analyze_jobs_with_resume(jobs_list, resume_data)

                    # Sort by analysis score if available
                    if jobs_list and '_analysis' in jobs_list[0] and 'suitability_score' in jobs_list[0]['_analysis']:
                        jobs_list.sort(key=lambda x: x['_analysis'].get('suitability_score', 0), reverse=True)

                progress.update(task, description=f"Found {len(jobs_list)} jobs")

                # Store the analyzed jobs for display
                analyzed_jobs = jobs_list

            except Exception as e:
                logger.error(f"Job search failed: {e}", exc_info=True)
                console.print(f"[red]Error: Job search failed - {str(e)}[/red]")
                return 1

        if not analyzed_jobs:
            console.print("[yellow]No jobs found matching your criteria.[/yellow]")
            return 0

        # Display results with analysis if available
        display_title = f"{scraper_type.upper()} Job Search Results"
        if resume_data:
            display_title += " (with Resume Analysis)"
        display_jobs_table(analyzed_jobs, display_title)

        # Initialize save_results flag
        save_results = False

        # Interactive mode
        if getattr(args, 'interactive', False):
            while True:
                try:
                    console.print("\n[bold]Enter job number to view details, 's' to save results, or 'q' to quit:[/bold]")
                    choice = input("> ").strip().lower()

                    if choice == 'q':
                        break
                    elif choice == 's':
                        # Save results
                        save_results = True
                        break
                    elif choice.isdigit():
                        job_index = int(choice) - 1
                        if 0 <= job_index < len(analyzed_jobs):
                            display_job_details(analyzed_jobs[job_index])
                        else:
                            console.print("[yellow]Invalid job number. Please try again.[/yellow]")
                    else:
                        console.print("[yellow]Invalid input. Please enter a job number, 's' to save, or 'q' to quit.[/yellow]")
                except (KeyboardInterrupt, EOFError):
                    console.print("\n[yellow]Exiting interactive mode...[/yellow]")
                    return 0
        else:
            save_results = True

        # Save results if not disabled and not in interactive mode or user chose to save
        if not getattr(args, 'no_save', False) and save_results:
            output_dir = getattr(args, 'output_dir', 'output')
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_format = getattr(args, 'output_format', 'json')
            output_file = os.path.join(output_dir, f'job_search_results_{timestamp}.{output_format}')

            try:
                # Convert jobs to DataFrame for saving
                import pandas as pd
                jobs_data = []
                for job in analyzed_jobs:
                    if hasattr(job, 'model_dump'):
                        job_dict = job.model_dump()
                    else:
                        job_dict = dict(job)  # Already a dict from our processing

                    # Flatten analysis if it exists
                    if '_analysis' in job_dict and job_dict['_analysis']:
                        analysis = job_dict.pop('_analysis', {})
                        if 'suitability_score' in analysis:
                            job_dict['match_score'] = analysis['suitability_score']
                        if 'pros' in analysis:
                            job_dict['strengths'] = "; ".join(analysis['pros'][:3]) if analysis['pros'] else ""
                        if 'cons' in analysis:
                            job_dict['areas_for_improvement'] = "; ".join(analysis['cons'][:3]) if analysis['cons'] else ""

                    jobs_data.append(job_dict)

                df = pd.DataFrame(jobs_data)

                if output_format == 'csv':
                    df.to_csv(output_file, index=False)
                elif output_format == 'xlsx':
                    df.to_excel(output_file, index=False)
                elif output_format == 'markdown':
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(df.to_markdown(index=False))
                else:  # default to json
                    df.to_json(output_file, orient='records', indent=2, force_ascii=False)

                console.print(f"\n[green]Results saved to: {output_file}[/green]")
            except Exception as e:
                logger.error(f"Failed to save results: {e}")
                console.print(f"[yellow]Warning: Failed to save results: {str(e)}[/yellow]")

        # If interactive mode, allow viewing details
        if getattr(args, 'interactive', False) and analyzed_jobs:
            while True:
                try:
                    choice = console.input("\n[bold]Enter job number to view details, or 'q' to quit: [/bold]")
                    if choice.lower() == 'q':
                        break

                    job_idx = int(choice) - 1
                    if 0 <= job_idx < len(analyzed_jobs):
                        display_job_details(analyzed_jobs[job_idx])
                    else:
                        console.print("[red]Invalid job number. Please try again.[/red]")
                except ValueError:
                    console.print("[red]Please enter a valid number or 'q' to quit.[/red]")

        return 0

    except Exception as e:
        logger.error(f"Job search failed: {e}", exc_info=True)
        console.print(f"[red]Error: {str(e)}[/red]")
        return 1
        return 0

        # Display results with analysis if available
        display_title = f"{scraper_type.upper()} Job Search Results"
        if resume_data:
            display_title += " (with Resume Analysis)"
        display_jobs_table(analyzed_jobs, display_title)

        # Initialize save_results flag
        save_results = False

        # Interactive mode
        if getattr(args, 'interactive', False):
            while True:
                try:
                    console.print("\n[bold]Enter job number to view details, 's' to save results, or 'q' to quit:[/bold]")
                    choice = input("> ").strip().lower()

                    if choice == 'q':
                        break
                    elif choice == 's':
                        # Save results
                        save_results = True
                        break
                    elif choice.isdigit():
                        job_index = int(choice) - 1
                        if 0 <= job_index < len(analyzed_jobs):
                            display_job_details(analyzed_jobs[job_index])
                        else:
                            console.print("[yellow]Invalid job number. Please try again.[/yellow]")
                    else:
                        console.print("[yellow]Invalid input. Please enter a job number, 's' to save, or 'q' to quit.[/yellow]")
                except (KeyboardInterrupt, EOFError):
                    console.print("\n[yellow]Exiting interactive mode...[/yellow]")
                    return 0
        else:
            save_results = True
    else:
        save_results = True

    # Save results if not disabled and not in interactive mode or user chose to save
    if not getattr(args, 'no_save', False) and save_results:
        output_dir = getattr(args, 'output_dir', 'output')
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_format = getattr(args, 'output_format', 'json')
        output_file = os.path.join(output_dir, f'job_search_results_{timestamp}.{output_format}')

        try:
            # Convert jobs to DataFrame for saving
            import pandas as pd
            jobs_data = []
            for job in analyzed_jobs:
                if hasattr(job, 'model_dump'):
                    job_dict = job.model_dump()
                else:
                    job_dict = dict(job)  # Already a dict from our processing

                # Flatten analysis if it exists
                if '_analysis' in job_dict and job_dict['_analysis']:
                    analysis = job_dict.pop('_analysis', {})
                    if 'suitability_score' in analysis:
                        job_dict['match_score'] = analysis['suitability_score']
                    if 'pros' in analysis:
                        job_dict['strengths'] = "; ".join(analysis['pros'][:3]) if analysis['pros'] else ""
                    if 'cons' in analysis:
                        job_dict['areas_for_improvement'] = "; ".join(analysis['cons'][:3]) if analysis['cons'] else ""

                jobs_data.append(job_dict)

            df = pd.DataFrame(jobs_data)

            if output_format == 'csv':
                df.to_csv(output_file, index=False)
            elif output_format == 'xlsx':
                df.to_excel(output_file, index=False)
            elif output_format == 'markdown':
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(df.to_markdown(index=False))
            else:  # default to json
                df.to_json(output_file, orient='records', indent=2, force_ascii=False)

            console.print(f"\n[green]Results saved to: {output_file}[/green]")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            console.print(f"[yellow]Warning: Failed to save results: {str(e)}[/yellow]")

    # If interactive mode, allow viewing details
    if getattr(args, 'interactive', False) and jobs:
        while True:
            try:
                choice = console.input("\n[bold]Enter job number to view details, or 'q' to quit: [/bold]")
                if choice.lower() == 'q':
                    break

                job_idx = int(choice) - 1
                if 0 <= job_idx < len(jobs):
                    display_job_details(jobs[job_idx])
                else:
                    console.print("[red]Invalid job number. Please try again.[/red]")
            except ValueError:
                console.print("[red]Please enter a valid number or 'q' to quit.[/red]")

    return 0


async def main_async() -> int:
    """Async main entry point for the application."""
    try:
        # Parse command line arguments
        args = parse_args(sys.argv[1:])

        # Handle version flag
        if getattr(args, 'version', False):
            console.print(f"MyJobSpy AI v{version('myjobspyai')}")
            return 0

        # Set log level based on verbosity
        verbose_level = getattr(args, 'verbose', 0)
        if verbose_level > 1:
            logger.setLevel(logging.DEBUG)
        elif verbose_level == 1:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

        # Log command line arguments (safely, without sensitive data)
        safe_args = {k: v for k, v in vars(args).items()
                    if k not in ['linkedin_password', 'proxy', 'api_key'] and not k.startswith('_')}
        logger.debug(f"Command line arguments: {safe_args}")

        logger.info("Starting MyJobSpy AI")

        # Perform job search with the provided arguments
        return await search_jobs(args)

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        return 1


def main() -> int:
    """Main entry point for the application.

    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    try:
        # Parse command line arguments
        args = parse_args(sys.argv[1:])

        # Configure logging
        debug_mode = getattr(args, 'debug', False)
        setup_logging_custom(debug_mode)

        # Set log level based on verbosity
        verbose_level = getattr(args, 'verbose', 0)
        if verbose_level > 1:
            logger.setLevel(logging.DEBUG)
        elif verbose_level == 1:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

        logger.debug("Starting MyJobSpy AI")

        # Run the async main function
        return asyncio.run(main_async())

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 0
    except Exception as e:
        logger.exception("Fatal error in main function")
        print(f"Fatal error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
