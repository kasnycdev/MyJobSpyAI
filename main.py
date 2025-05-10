
import logging
import json
import os
import sys
from typing import List, Dict, Any, Optional
import pandas as pd
import argparse
import asyncio
from datetime import datetime
from analysis.analyzer import ResumeAnalyzer
from colorama import init  # Import colorama
import traceback

# Ensure config module is imported
import config

# Initialize colorama
init(autoreset=True)

from rich.console import Console
from rich.logging import RichHandler

# Initialize rich console
console = Console()

# Update logging configuration to use RichHandler
logging.basicConfig(
    level=config.settings.get('logging', {}).get('level', 'INFO').upper(),
    format=config.settings.get('logging', {}).get('format', '%(message)s'),
    datefmt=config.settings.get('logging', {}).get('date_format', '[%X]'),
    handlers=[RichHandler(rich_tracebacks=True, show_path=False)]
)

# Replace os.system("color") with a rich console message
console.print("[green]Rich console initialized successfully.[/green]")

# Use the jobspy library for scraping
try:
    from jobspy import scrape_jobs
except ImportError:
    console.print("[red]CRITICAL ERROR: 'jobspy' library not found.[/red]")
    sys.exit(1)

# Import analysis components
try:
    from main_matcher import load_and_extract_resume_async, analyze_jobs_async, apply_filters_sort_and_save
except ImportError as e:
    console.print(f"[red]CRITICAL ERROR: Could not import analysis functions: {e}[/red]")
    sys.exit(1)

# Import the loaded settings dictionary from config.py
try:
    from config import settings
except ImportError:
    console.print("[red]CRITICAL ERROR: config.py not found or cannot be imported.[/red]")
    sys.exit(1)
except AttributeError:
    console.print("[red]CRITICAL ERROR: 'settings' dictionary not found in config.py.[/red]")
    sys.exit(1)

# Rich for UX needs settings for logging setup
try:
    from rich.table import Table
except ImportError:
    console.print("[yellow]WARNING: 'rich' library not found. Console output will be basic.[/yellow]")
    class Table:
        print = staticmethod(print)

# Helper function for logging exceptions
def log_exception(message, exception):
    console.log(message)
    console.log(traceback.format_exc())

# --- scrape_jobs_with_jobspy function ---
async def scrape_jobs_with_jobspy(
    search_terms: str,
    location: Optional[str],
    sites: list[str],
    results_wanted: int,
    hours_old: Optional[int],
    country_indeed: str,
    proxies: Optional[list[str]] = None,
    offset: int = 0,
    google_search_term: Optional[str] = None,
    distance: Optional[int] = None,
    is_remote: bool = False,
    job_type: Optional[str] = None,
    easy_apply: Optional[bool] = None,
    ca_cert: Optional[str] = None,
    linkedin_company_ids: Optional[list[int]] = None,
    enforce_annual_salary: bool = False,
    verbose: int = 0,
) -> Optional[pd.DataFrame]:
    """Uses the jobspy library to scrape jobs, with better logging."""
    console.log("[blue]Starting job scraping via JobSpy...[/blue]")
    console.log(
        f"Search: '[cyan]{search_terms}[/cyan]' | Location: '[cyan]{location}[/cyan]' | "
        f"Sites: {sites}"
    )
    console.log(
        f"Params: Results ≈{results_wanted}, Max Age={hours_old}h, "
        f"Indeed Country='{country_indeed}', Offset={offset}, "
        f"Is Remote={is_remote}, Job Type={job_type}, Easy Apply={easy_apply}, "
        f"Distance={distance}, Enforce Annual Salary={enforce_annual_salary}"
    )
    if google_search_term:
        console.log(f"Google Search Term: '{google_search_term}'")
    if proxies:
        console.log(f"Using {len(proxies)} proxies.")
    if linkedin_company_ids:
        console.log(f"LinkedIn Company IDs: {linkedin_company_ids}")
    if ca_cert:
        console.log(f"Using CA Cert: {ca_cert}")

    try:
        # Run the synchronous scrape_jobs in a separate thread
        jobs_df = await asyncio.to_thread(
            scrape_jobs,
            site_name=sites,
            search_term=search_terms,
            google_search_term=google_search_term,
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
            linkedin_fetch_description=settings.get('scraping', {}).get('linkedin_fetch_description'), # Keep from config
            linkedin_company_ids=linkedin_company_ids,
            enforce_annual_salary=enforce_annual_salary,
        )
        if jobs_df is None or jobs_df.empty:
            console.log("[yellow]Jobspy scraping returned no results or failed.[/yellow]")
            return None
        else:
            console.log(f"Jobspy scraping successful. Found {len(jobs_df)} jobs.")
            console.log(f"DataFrame columns: {jobs_df.columns.tolist()}")
            essential_scrape_cols = ['title', 'company', 'location', 'description', 'job_url', 'date_posted', 'job_type']
            for col in essential_scrape_cols:
                if col not in jobs_df.columns:
                    console.log(f"[yellow]Essential column '{col}' missing, adding empty.[/yellow]")
                    jobs_df[col] = ''
            return jobs_df
    except ImportError as ie:
        log_exception(f"[red]Import error during scraping: {ie}.[/red]", ie)
        return None
    except TypeError as te:  # Catch TypeError specifically
        log_exception(f"[red]TypeError during jobspy scrape call: {te}. Check argument names.[/red]", te)
        return None
    except Exception as e:
        log_exception(f"[red]An error occurred during jobspy scraping: {e}[/red]", e)
        return None


# --- convert_and_save_scraped function (no changes needed here) ---
def convert_and_save_scraped(jobs_df: pd.DataFrame, output_path: str) -> List[Dict[str, Any]]:
    console.log(f"Converting DataFrame to list and saving to {output_path}")
    rename_map = {'job_url': 'url', 'job_type': 'employment_type', 'salary': 'salary_text', 'benefits': 'benefits_text'}
    actual_rename_map = {k: v for k, v in rename_map.items() if k in jobs_df.columns}
    jobs_df = jobs_df.rename(columns=actual_rename_map)
    possible_date_columns = ['date_posted', 'posted_date', 'date']
    for col in possible_date_columns:
        if col in jobs_df.columns and (pd.api.types.is_datetime64_any_dtype(jobs_df[col]) or jobs_df[col].dtype == 'object'):
            try:
                jobs_df[col] = pd.to_datetime(jobs_df[col], errors='coerce')
                jobs_df[col] = jobs_df[col].dt.strftime('%Y-%m-%d')
            except Exception:
                jobs_df[col] = jobs_df[col].astype(str)
    essential_cols = ['title', 'company', 'location', 'description', 'url', 'salary_text', 'employment_type',
                      'benefits_text', 'skills', 'date_posted']
    for col in essential_cols:
        if col not in jobs_df.columns:
            console.log(f"[yellow]Column '{col}' missing, adding empty.[/yellow]")
            jobs_df[col] = ''
    jobs_df = jobs_df.fillna('')
    jobs_list = jobs_df.to_dict('records')
    if output_dir := os.path.dirname(output_path):
        os.makedirs(output_dir, exist_ok=True)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(jobs_list, f, indent=4)
        console.log(
            f"[green]Successfully saved {len(jobs_list)} scraped jobs to {output_path}[/green]"
        )
        return jobs_list
    except Exception as e:
        console.log(f"[red]Error saving scraped jobs JSON: {e}[/red]", exc_info=True)
        return []


# --- print_summary_table function (no changes needed here) ---
def print_summary_table(results_json: List[Dict[str, Any]], top_n: int = 10):
    if not results_json:
        console.print("[yellow]No analysis results to summarize.[/yellow]")
        return

    table = Table(
        title=f"Top {min(top_n, len(results_json))} Job Matches",
        show_header=True,
        header_style="bold magenta",
        show_lines=False
    )
    table.add_column("Score", style="dim", width=6, justify="right")
    table.add_column("Title", style="bold", min_width=20)
    table.add_column("Company")
    table.add_column("Location")
    table.add_column("URL", overflow="fold", style="cyan")

    count = 0
    for result in results_json:
        analysis = result.get('analysis', {})
        original = result.get('original_job_data', {})
        score = analysis.get('suitability_score', -1)
        if score is None or score == 0:
            continue
        score_str = f"{score}%"
        table.add_row(
            score_str,
            original.get('title', 'N/A'),
            original.get('company', 'N/A'),
            original.get('location', 'N/A'),
            original.get('url', '#')
        )
        count += 1
        if count >= top_n:
            break
    try:
        console.print(
            table
        )
    except Exception:
        console.print(
            "[yellow]No successfully analyzed jobs with score > 0 to display.[/yellow]"
        )


# --- Main ASYNC Execution Function ---
async def run_pipeline_async():
    parser = argparse.ArgumentParser(
        description="Run Job Scraping (JobSpy) & LLM Analysis Pipeline (using LM Studio/OpenAI-compatible API).", # Updated description
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    scrape_cfg = settings.get('scraping', {})

    parser = argparse.ArgumentParser(
        description="Run Job Scraping (JobSpy) & LLM Analysis Pipeline (using LM Studio/OpenAI-compatible API).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    scrape_group = parser.add_argument_group('Scraping Options (JobSpy)')
    scrape_group.add_argument("--search", required=True, help="Job title, keywords, or company.")
    scrape_group.add_argument("--location", default=scrape_cfg.get('location'), help="Primary location for scraping. Overridden if --filter-remote-country.")
    scrape_group.add_argument("--sites", default=",".join(scrape_cfg.get('default_sites', [])), help="Comma-separated sites.")
    scrape_group.add_argument("--results", type=int, default=scrape_cfg.get('default_results_limit', 20), help="Approx total jobs per site.")
    scrape_group.add_argument("--hours_old", type=int, default=scrape_cfg.get('default_hours_old', 3), help="Max job age in hours (0=disable).")
    scrape_group.add_argument("--country-indeed", default=scrape_cfg.get('default_country_indeed', 'usa'), help="Country for Indeed search.")
    scrape_group.add_argument("--proxies", default=scrape_cfg.get('proxies'), help="Comma-separated proxies.")
    scrape_group.add_argument("--offset", type=int, default=scrape_cfg.get('offset', 0), help="Search results offset.")
    scrape_group.add_argument("--scraped-jobs-file", default=settings.get("scraped_jobs_path"), help="Intermediate file for scraped jobs.")
    scrape_group.add_argument("--google-search-term", default=scrape_cfg.get('google_search_term'), help="Specific Google search term (if using Google scraper).")
    scrape_group.add_argument("--distance", type=int, default=scrape_cfg.get('distance'), help="Distance from location in miles (for supported sites).")
    scrape_group.add_argument("--is-remote", type=bool, default=scrape_cfg.get('is_remote', False), help="Filter for remote jobs.")
    scrape_group.add_argument("--job-type", default=scrape_cfg.get('job_type'), help="Job type (e.g., 'fulltime', 'parttime', 'contract', 'internship').")
    scrape_group.add_argument("--easy-apply", type=bool, default=scrape_cfg.get('easy_apply'), help="Filter for jobs with easy apply option (for supported sites).")
    scrape_group.add_argument("--ca-cert", default=scrape_cfg.get('ca_cert'), help="Path to CA certificate file.")
    scrape_group.add_argument("--linkedin-company-ids", default=scrape_cfg.get('linkedin_company_ids'), help="Comma-separated LinkedIn company IDs.")
    scrape_group.add_argument("--enforce-annual-salary", type=bool, default=scrape_cfg.get('enforce_annual_salary', False), help="Enforce conversion of salary to annual.")


    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument("--resume", required=True, help="Path to the resume file.")
    analysis_group.add_argument("--analysis-output", default=settings.get("analysis_output_path"), help="Final analysis output JSON.")
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

    # Setup Logging Level
    log_level_name = "DEBUG" if args.verbose else settings.get('logging', {}).get('level', 'INFO').upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    logging.getLogger().setLevel(log_level)
    console.log(f"Log level set to: {log_level_name}")
    console.log(
        f"[green]Starting ASYNC Pipeline Run[/green] "
        f"({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
    )

    try:
        # Argument validation and location determination (remains the same)
        # Prioritize command line args over config for location
        scrape_location = args.location
        if args.filter_remote_country:
            scrape_location = args.filter_remote_country.strip()
            console.log(f"Using country '{scrape_location}' as primary scrape location.")
        elif args.filter_proximity_location:
            scrape_location = args.filter_proximity_location.strip()
            console.log(f"Using proximity target '{scrape_location}' as primary scrape location.")
        elif args.location:
             console.log(
                f"Using provided --location '{scrape_location}' as primary scrape location."
            )
        elif scrape_cfg.get('location'):
             scrape_location = scrape_cfg.get('location')
             console.log(
                f"Using config location '{scrape_location}' as primary scrape location."
            )
        else:
             parser.error("Ambiguous location: Specify --location OR --filter-remote-country OR --filter-proximity-location, or set 'location' in config.yaml.")


        if args.filter_proximity_location and args.filter_remote_country:
            parser.error("Conflicting filters: Cannot use --filter-proximity-location and --filter-remote-country.")
        if args.filter_proximity_location and args.filter_proximity_range is None:
            parser.error("--filter-proximity-range required with --filter-proximity-location.")
        if args.filter_proximity_range is not None and not args.filter_proximity_location:
            parser.error("--filter-proximity-location required with --filter-proximity-location.")


        # --- Step 1: Scrape Jobs ---
        scraper_sites = [site.strip().lower() for site in args.sites.split(',')] if args.sites else scrape_cfg.get('default_sites', [])
        proxy_list = [p.strip() for p in args.proxies.split(',')] if args.proxies else scrape_cfg.get('proxies')
        linkedin_company_ids_list = [int(id.strip()) for id in args.linkedin_company_ids.split(',')] if args.linkedin_company_ids else scrape_cfg.get('linkedin_company_ids')


        jobs_df = await scrape_jobs_with_jobspy(
            search_terms=args.search,
            location=scrape_location,
            sites=scraper_sites,
            results_wanted=args.results,
            hours_old=args.hours_old,
            country_indeed=args.country_indeed,
            proxies=proxy_list,
            offset=args.offset,
            google_search_term=args.google_search_term,
            distance=args.distance,
            is_remote=args.is_remote,
            job_type=args.job_type,
            easy_apply=args.easy_apply,
            ca_cert=args.ca_cert,
            linkedin_company_ids=linkedin_company_ids_list,
            enforce_annual_salary=args.enforce_annual_salary,
            verbose=args.verbose # Pass verbose from args
        )

        if jobs_df is None or jobs_df.empty:
            console.log("[yellow]Scraping yielded no results. Pipeline cannot continue.[/yellow]")
            try:
                if analysis_output_dir := os.path.dirname(args.analysis_output):
                    os.makedirs(analysis_output_dir, exist_ok=True)
                empty_df = pd.DataFrame()
                empty_df.to_json(settings.get('output', {}).get('scraped_jobs_file'), orient='records')
                console.log(f"[green]Empty analysis results file created at {args.analysis_output}[/green]")
                sys.exit(0)  # Exit after creating empty file
            except Exception as e:
                log_exception(f"[red]Error saving empty analysis results file: {e}[/red]", e)
                sys.exit(1)  # Exit with error code

        # --- Steps 2-6 remain unchanged ---
        jobs_list = convert_and_save_scraped(jobs_df, args.scraped_jobs_file)
        if not jobs_list:
            console.log("[red]Failed to convert/save scraped data. Exiting.[/red]")
            sys.exit(1)
        try:
            analyzer = ResumeAnalyzer()
        except Exception as e:
            log_exception(f"[red]Failed to initialize ResumeAnalyzer: {e}.[/red]", e)
            sys.exit(1)
        structured_resume = await load_and_extract_resume_async(args.resume, analyzer)
        if not structured_resume:
            console.log("[red]Failed to load/extract resume data. Exiting.[/red]")
            sys.exit(1)
        analyzed_results = await analyze_jobs_async(analyzer, structured_resume, jobs_list)
        if not analyzed_results:
            console.log("[yellow]Analysis step produced no results.[/yellow]")
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
        final_results_list_dict = apply_filters_sort_and_save(
            analyzed_results, args.analysis_output, filter_args_dict
        )
        console.log("[blue]Pipeline Summary:[/blue]")
        print_summary_table(final_results_list_dict, top_n=10)
        console.print("[green]Pipeline Run Finished Successfully[/green]")

    except KeyboardInterrupt:
        print()
        console.log("[yellow]Pipeline interrupted by user (Ctrl+C).[/yellow]")
        sys.exit(130)
    except Exception as e:
        log_exception(f"[red]Unexpected critical error: {e}[/red]", e)
        sys.exit(1)


# --- Update entry point to run the async function ---
if __name__ == "__main__":
    # Ensure configuration is loaded before running the pipeline
    config.settings = config.load_config()
    try:
        asyncio.run(run_pipeline_async())
    except KeyboardInterrupt:
        console.print("\nExecution cancelled by user.")
        sys.exit(130)
