# Standard library imports
import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

# Third-party imports
import pandas as pd
from colorama import init
from rich.console import Console

# Local application imports
from myjobspyai.config import settings
from myjobspyai.filtering.filter_utils import DateEncoder
from myjobspyai.utils.logging_utils import setup_logging, tracer
from myjobspyai.analysis.main_matcher import (
    load_and_extract_resume_async,
    analyze_jobs_async,
    apply_filters_sort_and_save
)

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize colorama
init(autoreset=True)

# Initialize rich console (primarily for direct use, logging uses its own RichHandler instance)
console = Console()

logger.info("Rich console and logging initialized successfully.") 

# Import the loaded settings from myjobspyai.config
try:
    from myjobspyai.config import settings
except ImportError:
    # This case should ideally be caught by config.py itself if it fails to load
    logger.critical("CRITICAL ERROR: Failed to import settings from myjobspyai.config", exc_info=True)
    sys.exit(1)

# Use the jobspy library for scraping
try:
    from jobspy import scrape_jobs
except ImportError:
    console.print("[red]CRITICAL ERROR: 'jobspy' library not found.[/red]")
    sys.exit(1)

# Import analysis components
try:
    from myjobspyai.analysis.main_matcher import load_and_extract_resume_async, analyze_jobs_async, apply_filters_sort_and_save
except ImportError as e:
    console.print(f"[red]CRITICAL ERROR: Could not import analysis functions from myjobspyai.analysis.main_matcher: {e}[/red]")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Rich for UX needs settings for logging setup
try:
    from rich.table import Table
except ImportError:
    logger.warning("'rich' library not found. Console output will be basic.")
    class Table: 
        def __init__(self, title=None): self.title = title
        def add_column(self, *args, **kwargs): pass
        def add_row(self, *args, **kwargs): print(args) 
        def __str__(self): return f"Table: {self.title}" if self.title else "Table"

# Helper function for logging exceptions (now uses standard logger)
def log_exception(message, exception):
    logger.error(message, exc_info=True)

# --- Utility functions (basic implementations) ---
@tracer.start_as_current_span("convert_and_save_scraped")
def convert_and_save_scraped(jobs_df: Optional[pd.DataFrame], output_path: str) -> List[Dict[str, Any]]:
    """Converts DataFrame to list of dicts and saves to JSON."""
    if jobs_df is None or jobs_df.empty:
        logger.warning("No jobs DataFrame to convert or save.")
        try:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=4, cls=DateEncoder) 
            logger.info(f"Empty scraped jobs file saved to {output_path}")
        except Exception as e:
            log_exception(f"Error saving empty scraped jobs file to {output_path}: {e}", e)
        return []

    jobs_list = jobs_df.to_dict(orient='records')
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir: 
            os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(jobs_list, f, indent=4, cls=DateEncoder) 
        logger.info(f"Successfully saved {len(jobs_list)} scraped jobs to {output_path}")
    except Exception as e:
        log_exception(f"Error saving scraped jobs to {output_path}: {e}", e)
        # Still return the jobs list even if saving to disk fails
    return jobs_list

@tracer.start_as_current_span("print_summary_table")
def print_summary_table(analyzed_jobs_list: List[Dict[str, Any]], top_n: int = 10):
    """Prints a basic summary of the top N analyzed jobs."""
    if not analyzed_jobs_list:
        logger.warning("No analyzed jobs to summarize.")
        return

    logger.info(f"Top {min(top_n, len(analyzed_jobs_list))} (or fewer) jobs after filtering and sorting:")
    
    if 'Table' in globals() and hasattr(globals()['Table'], 'add_column') and isinstance(globals()['Table'], type):
        table = Table(title="Job Summary")
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Title", style="bold cyan")
        table.add_column("Company", style="green")
        table.add_column("Location", style="magenta")
        table.add_column("Score", style="bold yellow", justify="right")

        for i, job_summary in enumerate(analyzed_jobs_list[:top_n]):
            original_data = job_summary.get('original_job_data', {})
            analysis_data = job_summary.get('analysis', {})
            
            title = original_data.get('title', 'N/A')
            company = original_data.get('company', 'N/A')
            location_value = original_data.get('location') 
            city = 'N/A'
            state = None
            country = None
            location_str = 'N/A' 

            if isinstance(location_value, dict):
                city = location_value.get('city', 'N/A')
                state = location_value.get('state')
                country = location_value.get('country')
                
                location_parts = []
                if city and city != 'N/A':  
                    location_parts.append(city)
                if state:
                    location_parts.append(state)
                if (country and 
                    country.lower() != city.lower() and 
                    (not state or country.lower() != state.lower())):
                    location_parts.append(f"({country})")
                
                if location_parts:
                    location_str = ", ".join(location_parts)

            elif isinstance(location_value, str):
                location_str = location_value 
                city = location_value 


            # Safely get the score, handling both dictionary and object access
            if hasattr(analysis_data, 'get'):
                # It's a dictionary
                score = analysis_data.get('suitability_score', 'N/A')
            elif hasattr(analysis_data, 'suitability_score'):
                # It's an object with the attribute
                score = getattr(analysis_data, 'suitability_score', 'N/A')
            else:
                score = 'N/A'
                
            table.add_row(str(i + 1), title, company, location_str, str(score))
        
        # Use the Rich console instance for printing the table directly
        # This is okay as it's a direct user-facing output, not typical logging
        console.print(table)
    else: # Fallback to basic print if Rich Table is not fully available
        for i, job_summary in enumerate(analyzed_jobs_list[:top_n]):
            # Safely get the data, handling both dictionary and object access
            original_data = job_summary.get('original_job_data', {}) if hasattr(job_summary, 'get') else getattr(job_summary, 'original_job_data', {})
            analysis_data = job_summary.get('analysis', {}) if hasattr(job_summary, 'get') else getattr(job_summary, 'analysis', {})
            
            title = original_data.get('title', 'N/A') if hasattr(original_data, 'get') else getattr(original_data, 'title', 'N/A')
            company = original_data.get('company', 'N/A') if hasattr(original_data, 'get') else getattr(original_data, 'company', 'N/A')
            
            # Get the score safely
            if hasattr(analysis_data, 'get'):
                score = analysis_data.get('suitability_score', 'N/A')
            elif hasattr(analysis_data, 'suitability_score'):
                score = getattr(analysis_data, 'suitability_score', 'N/A')
            else:
                score = 'N/A'
                
            logger.info(f"{i+1}. {title} at {company} - Score: {score}")

@tracer.start_as_current_span("print_detailed_analysis")
def print_detailed_analysis(analyzed_jobs_list: List[Dict[str, Any]], top_n: int = 3):
    """Prints detailed analysis for the top N jobs."""
    if not analyzed_jobs_list:
        logger.warning("No analyzed jobs to provide detailed analysis for.")
        return

    logger.info(f"\n--- Detailed Analysis for Top {min(top_n, len(analyzed_jobs_list))} Jobs ---")
    for i, job_data_dict in enumerate(analyzed_jobs_list[:top_n]):
        console.rule(f"[bold cyan]Rank {i+1}[/bold cyan]", style="cyan")

        original_job = job_data_dict.get('original_job_data', {})
        parsed_details = job_data_dict.get('parsed_job_details') # This is already a dict or None
        analysis = job_data_dict.get('analysis') # This is already a dict or None

        console.print(f"[bold]Title:[/bold] {original_job.get('title', 'N/A')}")
        console.print(f"[bold]Company:[/bold] {original_job.get('company', 'N/A')}")
        console.print(f"[bold]Location:[/bold] {original_job.get('location', 'N/A')}")
        console.print(f"[bold]URL:[/bold] {original_job.get('job_url', original_job.get('url', 'N/A'))}")
        console.print(f"[bold]Description Snippet:[/bold]\n {original_job.get('description', 'N/A')[:500]}...")

        if parsed_details:
            console.print("\n[bold green]--- Extracted Job Details (LLM) ---[/bold green]")
            for key, value in parsed_details.items():
                if isinstance(value, list) and value:
                    console.print(f"  [bold]{key.replace('_', ' ').title()}:[/bold]")
                    for item in value:
                        if isinstance(item, dict): # For SkillDetail
                            item_str = f"    - Name: {item.get('name')}"
                            if item.get('level'):
                                item_str += f", Level: {item.get('level')}"
                            if item.get('years_experience') is not None:
                                item_str += f", Years: {item.get('years_experience')}"
                            console.print(item_str)
                        else:
                            console.print(f"    - {item}")
                elif value and not isinstance(value, list):
                     console.print(f"  [bold]{key.replace('_', ' ').title()}:[/bold] {value}")
        else:
            console.print("\n[bold green]--- Extracted Job Details (LLM) ---[/bold green]")
            console.print("  Not available or extraction failed.")

        if analysis:
            console.print("\n[bold magenta]--- Suitability Analysis (LLM) ---[/bold magenta]")
            console.print(f"  [bold]Suitability Score:[/bold] {analysis.get('suitability_score', 'N/A')}%")
            console.print(f"  [bold]Justification:[/bold]\n    {analysis.get('justification', 'N/A')}")
            if analysis.get('pros'):
                console.print("  [bold]Pros:[/bold]")
                for pro in analysis.get('pros', []):
                    console.print(f"    - {pro}")
            if analysis.get('cons'):
                console.print("  [bold]Cons:[/bold]")
                for con in analysis.get('cons', []):
                    console.print(f"    - {con}")
            console.print(
                f"  [bold]Skill Match Summary:[/bold]\n    "
                f"{analysis.get('skill_match_summary', 'N/A')}"
            )
            console.print(
                f"  [bold]Experience Match Summary:[/bold]\n    "
                f"{analysis.get('experience_match_summary', 'N/A')}"
            )
            console.print(
                f"  [bold]Education Match Summary:[/bold]\n    "
                f"{analysis.get('education_match_summary', 'N/A')}"
            )
            if analysis.get('missing_keywords'):
                console.print("  [bold]Missing Keywords:[/bold]")
                for kw in analysis.get('missing_keywords', []):
                    console.print(f"    - {kw}")
        else:
            console.print("\n[bold magenta]--- Suitability Analysis (LLM) ---[/bold magenta]")
            console.print("  Not available or analysis failed.")
        
        if i < min(top_n, len(analyzed_jobs_list)) - 1:
            console.print("\n") # Add space before next rule, unless it's the last job

    console.rule(style="cyan")


# --- scrape_jobs_with_jobspy function ---
@tracer.start_as_current_span("scrape_jobs_with_jobspy")
async def scrape_jobs_with_jobspy(
    sites: list[str], # Renamed from site_name to match caller
    search_terms: str,
    location: Optional[str],
    results_wanted: int,
    hours_old: Optional[int],
    country_indeed: str,
    proxies: Optional[list[str]] = None,
    offset: int = 0,
    # google_search_term: Optional[str] = None, # Removed
    distance: Optional[int] = None,
    is_remote: bool = False,
    job_type: Optional[str] = None,
    easy_apply: Optional[bool] = None,
    ca_cert: Optional[str] = None,
    linkedin_company_ids: Optional[list[int]] = None,
    enforce_annual_salary: bool = False,
    verbose: int = 0,
    cli_args: Optional[argparse.Namespace] = None, # Added to pass cli_args for file paths
    linkedin_fetch_description: bool = True, # Added missing parameter
) -> Optional[pd.DataFrame]:
    """Uses the jobspy library to scrape jobs, with better logging."""
    logger.info("Starting job scraping via JobSpy...")
    logger.info(
        f"Search: '{search_terms}' | Location: '{location}' | "
        f"Sites: {sites}"
    )
    logger.info(
        f"Params: Results ~{results_wanted}, Max Age={hours_old}h, " 
        f"Indeed Country='{country_indeed}', Offset={offset}, "
        f"Is Remote={is_remote}, Job Type={job_type}, Easy Apply={easy_apply}, "
        f"Distance={distance}, Enforce Annual Salary={enforce_annual_salary}"
    )
    # if google_search_term: # Removed
    #     logger.info(f"Google Search Term: '{google_search_term}'") # Removed
    if proxies:
        logger.info(f"Using {len(proxies)} proxies.")
    if linkedin_company_ids:
        logger.info(f"LinkedIn Company IDs: {linkedin_company_ids}")
    if ca_cert:
        logger.info(f"Using CA Cert: {ca_cert}")

    try:
        with tracer.start_as_current_span("jobspy.scrape_jobs_call"):
            # Run the synchronous scrape_jobs in a separate thread
            jobs_df = await asyncio.to_thread(
                scrape_jobs,
                site_name=sites, # Pass the sites list as site_name
            search_term=search_terms, # Pass search_terms as search_term
            # google_search_term=google_search_term, # Removed
            location=location,
            distance=distance,
            is_remote=is_remote,
            job_type=job_type,
            easy_apply=easy_apply,
            results_wanted=results_wanted,
            hours_old=hours_old,
            country_indeed=country_indeed,
            proxies=proxies,
            ca_cert=ca_cert,
            offset=offset,
            verbose=verbose, # Use the verbose level from args/config
            description_format="markdown", # Keep markdown as default
            linkedin_fetch_description=linkedin_fetch_description, # Use passed parameter
            linkedin_company_ids=linkedin_company_ids,
            enforce_annual_salary=enforce_annual_salary,
            )
        if jobs_df is None or jobs_df.empty:
            logger.warning("Scraping yielded no results. Pipeline cannot continue.")
            with tracer.start_as_current_span("save_empty_scrape_results"):
                try: # Correctly indented
                    # Use cli_args for file paths if available, otherwise fallback to settings
                    scraped_jobs_target_path = cli_args.scraped_jobs_file if cli_args and hasattr(cli_args, 'scraped_jobs_file') else settings.get('output', {}).get('scraped_jobs_file')
                    analysis_output_target_path = cli_args.analysis_output if cli_args and hasattr(cli_args, 'analysis_output') else settings.get('output', {}).get('analysis_output_file')

                    if scraped_jobs_target_path: # Now inside TRY
                        if scraped_jobs_dir := os.path.dirname(scraped_jobs_target_path):
                            os.makedirs(scraped_jobs_dir, exist_ok=True)
                        pd.DataFrame().to_json(scraped_jobs_target_path, orient='records', indent=4)
                        logger.info(f"Empty scraped jobs file created at {scraped_jobs_target_path}")

                    if analysis_output_target_path: # Also save an empty analysis results file, now inside TRY
                        if analysis_output_dir := os.path.dirname(analysis_output_target_path):
                            os.makedirs(analysis_output_dir, exist_ok=True)
                        with open(analysis_output_target_path, 'w', encoding='utf-8') as f:
                            json.dump([], f, indent=4)
                        logger.info(f"Empty analysis results file created at {analysis_output_target_path}")
                
                    sys.exit(0)  # Exit after creating empty file(s), now inside TRY
                except Exception as e: # Correctly indented
                        log_exception(f"Error saving empty results files: {e}", e)
                        sys.exit(1)  # Exit with error code

        # --- Steps 2-6 remain unchanged ---
        with tracer.start_as_current_span("internal_convert_and_save_scraped"):
            convert_and_save_scraped(jobs_df, cli_args.scraped_jobs_file if cli_args and hasattr(cli_args, 'scraped_jobs_file') else settings.get('output', {}).get('scraped_jobs_file')) # Use cli_args, assignment to jobs_list removed as it's unused here
        return jobs_df # Ensure jobs_df is returned

    except KeyboardInterrupt:
        print() # Keep a simple print for immediate feedback on Ctrl+C before logger might flush
        logger.warning("Pipeline interrupted by user (Ctrl+C).")
        sys.exit(130)
    except Exception as e:
        log_exception(f"Unexpected critical error: {e}", e)
        sys.exit(1)


# --- Main ASYNC Execution Function ---
@tracer.start_as_current_span("run_pipeline_async")
async def run_pipeline_async():
    parser = argparse.ArgumentParser(
        description="Run Job Scraping (JobSpy) & LLM Analysis Pipeline (using LM Studio/OpenAI-compatible API).", # Updated description
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    scrape_cfg = settings.scraping
    default_sites = scrape_cfg.default_sites
    default_results = scrape_cfg.default_results_limit
    default_days_old = scrape_cfg.default_days_old

    scrape_group = parser.add_argument_group('Scraping Options (JobSpy)')
    scrape_group.add_argument("--search", required=True, help="Job title, keywords, or company.")
    scrape_group.add_argument("--location", default=None, help="Primary location for scraping. Overridden if --filter-remote-country.")
    scrape_group.add_argument("--sites", default=",".join(default_sites), help="Comma-separated sites.")
    scrape_group.add_argument("--results", type=int, default=default_results, help="Approx total jobs per site.")
    scrape_group.add_argument("--days-old", type=int, default=default_days_old, help="Max job age in days (0=disable). JobSpy uses hours, so this will be converted.")
    scrape_group.add_argument("--country-indeed", default=scrape_cfg.default_country_indeed, help="Country for Indeed search.")
    scrape_group.add_argument("--proxies", default=scrape_cfg.proxies, help="Comma-separated proxies.") # Added default from config
    scrape_group.add_argument("--offset", type=int, default=0, help="Search results offset.")
    scrape_group.add_argument(
        "--is-remote",
        type=lambda x: str(x).lower() == 'true',
        default=scrape_cfg.is_remote, # Corrected default lookup from config
        help="Filter for remote jobs only (e.g., true, false)."
    )
    scrape_group.add_argument(
        "--job-type",
        type=str,
        default=scrape_cfg.job_type, # Corrected default lookup from config
        help="Job type to filter for (e.g., 'fulltime', 'contract', 'parttime')."
    )
    # scrape_group.add_argument("--google-search-term", default=scrape_cfg.google_search_term, help="Specific search term for Google Jobs scraper.") # Removed
    scrape_group.add_argument("--distance", type=int, default=scrape_cfg.distance, help="Distance in miles from location (for supported sites). JobSpy default is 50.") # Corrected default
    scrape_group.add_argument(
        "--easy-apply",
        type=lambda x: str(x).lower() == 'true' if x is not None else None,
        default=scrape_cfg.easy_apply,
        help="Filter for jobs with easy apply option (e.g., true, false). LinkedIn easy apply no longer works."
    )
    scrape_group.add_argument("--ca-cert", default=scrape_cfg.ca_cert, help="Path to CA certificate file for proxies.")
    scrape_group.add_argument("--linkedin-company-ids", default=",".join(map(str, scrape_cfg.linkedin_company_ids)), type=str, help="Comma-separated LinkedIn company IDs to filter by.") # Fetch from config, parse later
    scrape_group.add_argument(
        "--enforce-annual-salary",
        type=lambda x: str(x).lower() == 'true',
        default=scrape_cfg.enforce_annual_salary,
        help="Convert all salaries to annual (e.g., true, false)."
    )
    # Get output settings with defaults
    output_settings = settings.output
    default_scraped_jobs_file = output_settings.scraped_jobs_file
    
    scrape_group.add_argument(
        "--scraped-jobs-file", 
        default=default_scraped_jobs_file, 
        help="Intermediate file for scraped jobs."
    )

    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument("--resume", required=True, help="Path to the resume file.")
    analysis_group.add_argument(
        "--analysis-output", 
        default=getattr(settings, 'analysis_output_path', None), 
        help="Final analysis output JSON."
    )
    analysis_group.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG level logging (overrides config).")

    filter_group = parser.add_argument_group('Filtering Options (Applied After Analysis)')
    filter_group.add_argument("--min-salary", type=int, help="Minimum desired annual salary.")
    filter_group.add_argument("--max-salary", type=int, help="Maximum desired annual salary.")
    filter_group.add_argument("--filter-work-models", help="Standard work models (e.g., 'Remote,Hybrid').")
    filter_group.add_argument("--filter-job-types", help="Comma-separated job types (e.g., 'Full-time')")

    adv_loc_group = parser.add_argument_group('Advanced Location Filtering')
    adv_loc_group.add_argument("--filter-remote-country", help="Filter REMOTE jobs within specific country.")
    adv_loc_group.add_argument("--filter-proximity-location", help="Reference location for proximity filtering.")
    adv_loc_group.add_argument("--filter-proximity-range", type=float, help="Distance in miles for proximity.")
    adv_loc_group.add_argument("--filter-proximity-models", default="Hybrid,On-site", help="Work models for proximity.")

    args = parser.parse_args()

    # Logging is already set up by setup_logging() at the top of the module.
    # The --verbose flag can be used to adjust the console handler's level if desired,
    # or the root logger's level if setup_logging is called after arg parsing.
    # For simplicity, setup_logging() is called once at import time.
    # If args.verbose needs to override, setup_logging would need to be callable with a level.
    # For now, config.yaml and the default setup_logging behavior will control levels.
    if args.verbose:
        # If verbose, ensure the root logger (and thus console by RichHandler default) is at least DEBUG
        # This might be redundant if setup_logging already sets root to DEBUG and console to INFO/DEBUG
        logging.getLogger().setLevel(logging.DEBUG) 
        logger.info("Verbose mode enabled. Root logger level set to DEBUG.")
    
    logger.info(
        f"Starting ASYNC Pipeline Run ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
    )

    try:
        # Argument validation and location determination (remains the same)
        if not args.location and not args.filter_remote_country and not args.filter_proximity_location:
            parser.error("Ambiguous location: Specify --location OR --filter-remote-country OR --filter-proximity-location.")
        if args.filter_proximity_location and args.filter_remote_country:
            parser.error("Conflicting filters: Cannot use --filter-proximity-location and --filter-remote-country.")
        if args.filter_proximity_location and args.filter_proximity_range is None:
            parser.error("--filter-proximity-range required with --filter-proximity-location.")
        if args.filter_proximity_range is not None and not args.filter_proximity_location:
            parser.error("--filter-proximity-location required with --filter-proximity-range.")
        scrape_location = None
        if args.filter_remote_country:
            scrape_location = args.filter_remote_country.strip()
            logger.info(f"Using country '{scrape_location}' as primary scrape location.")
        elif args.filter_proximity_location:
            scrape_location = args.filter_proximity_location.strip()
            logger.info(f"Using proximity target '{scrape_location}' as primary scrape location.")
        elif args.location:
            scrape_location = args.location
            logger.info(
                f"Using provided --location '{scrape_location}' as primary scrape location."
            )

        # --- Step 1: Scrape Jobs ---
        scraper_sites = [site.strip().lower() for site in args.sites.split(',')]
        proxy_list = [p.strip() for p in args.proxies.split(',')] if args.proxies else None
        
        # Convert days_old to hours_old for JobSpy
        hours_old_for_jobspy = args.days_old * 24 if args.days_old is not None and args.days_old > 0 else None
        
        # Parse linkedin_company_ids from string to list of ints
        linkedin_company_ids_list = []
        if args.linkedin_company_ids:
            try:
                linkedin_company_ids_list = [int(cid.strip()) for cid in args.linkedin_company_ids.split(',') if cid.strip()]
            except ValueError:
                logger.warning(f"Could not parse --linkedin-company-ids '{args.linkedin_company_ids}'. Ensure it's a comma-separated list of numbers.")
                linkedin_company_ids_list = []


        jobs_df = await scrape_jobs_with_jobspy(
            sites=scraper_sites, 
            search_terms=args.search, 
            # google_search_term=args.google_search_term, # Removed
            location=scrape_location,
            distance=args.distance,
            is_remote=args.is_remote,
            job_type=args.job_type,
            easy_apply=args.easy_apply,
            results_wanted=args.results,
            hours_old=hours_old_for_jobspy, 
            country_indeed=args.country_indeed,
            proxies=proxy_list,
            ca_cert=args.ca_cert,
            offset=args.offset,
            verbose=(2 if args.verbose else scrape_cfg.get('jobspy_verbose_level', 2)),
            # description_format is hardcoded to markdown in scrape_jobs_with_jobspy call below
            linkedin_fetch_description=getattr(settings.scraping, 'linkedin_fetch_description', True), # Default to True if not in config
            linkedin_company_ids=linkedin_company_ids_list,
            enforce_annual_salary=args.enforce_annual_salary,
            cli_args=args, # Pass the parsed CLI arguments
        )

        # The jobs_df is processed and saved within scrape_jobs_with_jobspy (via convert_and_save_scraped)
        # or the script exits if scraping truly yields no results from jobspy.scrape_jobs.
        # If jobs_df was None or empty initially, scrape_jobs_with_jobspy would have exited.
        # If it proceeds, convert_and_save_scraped is called within scrape_jobs_with_jobspy.
        # We need the jobs_list that convert_and_save_scraped would produce.
        # For now, we assume scrape_jobs_with_jobspy returns the DataFrame, and we convert it here
        # if it's not None. The exit logic for no results is handled inside scrape_jobs_with_jobspy.

        if jobs_df is None: # This check might be redundant if scrape_jobs_with_jobspy exits on None
            logger.warning("scrape_jobs_with_jobspy returned None. Exiting pipeline.")
            sys.exit(0) # Or handle as an error

        jobs_list = convert_and_save_scraped(jobs_df, args.scraped_jobs_file)
        
        if not jobs_list: # If convert_and_save_scraped returned an empty list (e.g. df was empty)
            logger.warning("No jobs to analyze after scraping and conversion. Exiting pipeline.")
            # Ensure empty analysis file is created if not already handled by scrape_jobs_with_jobspy's exit
            output_settings = getattr(settings, 'output', {})
            output_dir = getattr(output_settings, 'directory', 'output')
            output_file = getattr(output_settings, 'analysis_output_file', 'analysis_results.json')
            analysis_output_file_path = args.analysis_output or os.path.join(output_dir, output_file)
            try:
                os.makedirs(os.path.dirname(analysis_output_file_path), exist_ok=True)
                with open(analysis_output_file_path, 'w', encoding='utf-8') as f:
                    json.dump([], f, indent=4)
                logger.info(f"Empty analysis results file created at {analysis_output_file_path}")
            except Exception as e:
                log_exception(f"Error saving empty analysis results file: {e}", e)
            sys.exit(0)


        # --- Steps 2-6: Analysis, Filtering, and Saving ---
        # Get LLM configuration from settings
        llm_config = getattr(settings, 'llm', None)
        
        if not llm_config:
            logger.warning("No LLM configuration found in settings, using defaults")
            from myjobspyai.config import LLMConfig
            llm_config = LLMConfig()
        
        # Log the configuration
        logger.info(f"Using LLM configuration: {llm_config.model_dump()}")
        
        # Initialize LLM provider
        try:
            from myjobspyai.analysis.factory import get_factory
            
            # Convert llm_config to a dictionary if it's a Pydantic model
            if hasattr(llm_config, 'model_dump'):
                llm_config_dict = llm_config.model_dump(exclude_unset=True)
            else:
                llm_config_dict = dict(llm_config)
            
            # Get provider and model with defaults
            provider = llm_config_dict.pop('provider', 'ollama')
            
            # Initialize the provider with the config including the model
            factory = get_factory()
            llm_provider = await factory.get_or_create_provider(
                provider_name=provider,
                config_overrides=llm_config_dict  # model is included in the config
            )
            logger.info(f"Initialized LLM provider: {llm_provider.__class__.__name__}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM provider: {e}", exc_info=True)
            sys.exit(1)
        
        # Load and parse the resume with the LLM configuration
        structured_resume = await load_and_extract_resume_async(
            resume_path=args.resume,
            config=llm_config
        )
        
        if not structured_resume:
            logger.error("Failed to load/extract resume data. Exiting.")
            sys.exit(1)
            
        # Analyze jobs with the initialized LLM provider and configuration
        analyzed_results = await analyze_jobs_async(
            structured_resume_data=structured_resume,
            job_list=jobs_list,
            llm_provider=llm_provider,
            config=llm_config
        )
        if not analyzed_results:
            logger.warning("Analysis step produced no results.")
        filter_args_dict = {}
        if args.min_salary is not None:
            filter_args_dict['salary_min'] = args.min_salary
        if args.max_salary is not None:
            filter_args_dict['salary_max'] = args.max_salary
        if args.filter_work_models:
            filter_args_dict['work_models'] = [wm.strip().lower() for wm in args.filter_work_models.split(',')]
        if args.filter_job_types:
            filter_args_dict['job_types'] = [jt.strip().lower() for jt in args.filter_job_types.split(',')]
        if args.filter_remote_country:
            filter_args_dict['filter_remote_country'] = args.filter_remote_country.strip()
        if args.filter_proximity_location:
            filter_args_dict['filter_proximity_location'] = args.filter_proximity_location.strip()
            filter_args_dict['filter_proximity_range'] = args.filter_proximity_range
            filter_args_dict['filter_proximity_models'] = [pm.strip().lower() for pm in args.filter_proximity_models.split(',')]
        # Ensure we have a valid output path
        output_dir = getattr(settings.output, 'directory', 'output')
        output_file = getattr(settings.output, 'analysis_output_file', 'analysis_results.json')
        default_output_path = os.path.join(output_dir, output_file)
        
        # Use the provided output path or fall back to the default
        output_path = args.analysis_output or default_output_path
        
        # Ensure the output directory exists and output_path is valid
        if not output_path or not isinstance(output_path, str):
            output_path = default_output_path
            logger.warning(f"Invalid output path provided, using default: {output_path}")
            
        try:
            # Ensure the output directory exists
            output_dir = os.path.dirname(os.path.abspath(output_path))
            if output_dir:  # Only try to create directory if there is a path component
                os.makedirs(output_dir, exist_ok=True)
            
            logger.info(f"Saving analysis results to: {output_path}")
            final_results_list_dict = apply_filters_sort_and_save(
                analyzed_results, output_path, filter_args_dict
            )
        except (TypeError, OSError) as _:
            logger.error(f"Failed to create output directory or invalid path: {output_path}", exc_info=True)
            raise
        logger.info("Pipeline Summary:")
        print_summary_table(final_results_list_dict, top_n=10) 
        detailed_count = getattr(getattr(settings, 'output', {}), 'detailed_analysis_count', 3)
        print_detailed_analysis(final_results_list_dict, top_n=detailed_count) # Print detailed for top N
        logger.info("Pipeline Run Finished Successfully")

    except KeyboardInterrupt:
        print() # Keep for immediate feedback
        logger.warning("Pipeline interrupted by user (Ctrl+C).")
        sys.exit(130)
    except Exception as e:
        log_exception(f"Unexpected critical error in run_pipeline_async: {e}", e)
        sys.exit(1)


async def main():
    """Main async function to run the pipeline."""
    try:
        await run_pipeline_async()
    except KeyboardInterrupt:
        logger.warning("Execution cancelled by user (Ctrl+C at top level).")
        return 130
    except Exception as e:
        logger.critical(f"Unhandled critical error at top level: {e}", exc_info=True)
        return 1
    return 0

# --- Update entry point to run the async function ---
if __name__ == "__main__":
    # config.settings is already loaded when config.py is imported.
    # setup_logging() is called at the top of this file after config import.
    import sys
    sys.exit(asyncio.run(main()))
