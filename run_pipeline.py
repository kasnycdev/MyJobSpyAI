import argparse
import logging
import json
import os
import sys
import asyncio # Import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

# Use the jobspy library for scraping
try:
    import jobspy
    from jobspy import scrape_jobs # Explicit import
except ImportError:
    print("CRITICAL ERROR: 'jobspy' library not found. Please install it via requirements.txt")
    sys.exit(1)

# Import analysis components
try:
    from main_matcher import load_and_extract_resume_async, analyze_jobs_async, apply_filters_sort_and_save # Use async versions
    from analysis.analyzer import ResumeAnalyzer
except ImportError as e:
    # Provide more context on import error
    print(f"CRITICAL ERROR: Could not import analysis functions: {e}")
    print(f"Import attempted from: {__file__}")
    print("Ensure main_matcher.py and analysis/analyzer.py are in the correct path relative to run_pipeline.py and define expected classes/functions.")
    sys.exit(1)

# --- MODIFIED IMPORT & SETUP ---
# Import the loaded settings dictionary and helper function from config.py
try:
    from config import settings, get_setting
except ImportError as e:
     print(f"CRITICAL ERROR: Could not import 'settings' from config.py: {e}")
     print("Ensure config.py exists and loads settings correctly.")
     sys.exit(1)

# Rich for UX needs settings for logging setup
try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    RICH_AVAILABLE = True
except ImportError:
     RICH_AVAILABLE = False
     print("WARNING: 'rich' library not found. Console output will be basic.")
     # Define dummy classes if rich is not available to avoid errors later
     class Console: pass
     class RichHandler(logging.StreamHandler): pass # Inherit from base handler
     class Table: pass

# Setup logging using Rich if available, otherwise basic
# --- MODIFIED LOGGING SETUP ---
log_cfg = settings.get('logging', {}) # Get logging sub-dictionary safely
log_level_name = log_cfg.get('level', 'INFO').upper()
log_level = getattr(logging, log_level_name, logging.INFO)
log_format = log_cfg.get('format', '%(asctime)s - %(levelname)s - %(message)s') # Provide basic default format
log_date_format = log_cfg.get('date_format', '%Y-%m-%d %H:%M:%S')

if RICH_AVAILABLE:
    logging.basicConfig(
        level=log_level,
        format="%(message)s", # Let Rich handle timestamp/level formatting
        datefmt="[%X]", # Rich specific date format
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)] )
else:
     logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=log_date_format)

logging.getLogger("httpx").setLevel(logging.WARNING) # Reduce httpx verbosity always
log = logging.getLogger(__name__) # Use project logger
console = Console() if RICH_AVAILABLE else None # Instantiate Console only if Rich is available
# --- END MODIFIED LOGGING SETUP ---


# --- scrape_jobs_with_jobspy function remains unchanged ---
def scrape_jobs_with_jobspy(
    search_terms: str, location: str, sites: list[str], results_wanted: int, hours_old: int,
    country_indeed: str, proxies: Optional[list[str]] = None, offset: int = 0
    ) -> Optional[pd.DataFrame]:
    # (Content remains the same - uses jobspy.scrape_jobs)
    log.info(f"[bold blue]Starting job scraping via JobSpy...[/bold blue]" if RICH_AVAILABLE else "Starting job scraping via JobSpy...")
    log.info(f"Search: '[cyan]{search_terms}[/cyan]' | Location: '[cyan]{location}[/cyan]' | Sites: {sites}" if RICH_AVAILABLE else f"Search: '{search_terms}' | Location: '{location}' | Sites: {sites}")
    log.info(f"Params: Results ≈{results_wanted}, Max Age={hours_old}h, Indeed Country='{country_indeed}', Offset={offset}")
    if proxies: log.info(f"Using {len(proxies)} proxies.")
    try:
        # Pass linkedin specific args from settings if they exist
        scrape_kwargs = settings.get('scraping', {})
        linkedin_fetch = scrape_kwargs.get('linkedin_fetch_description', True) # Default to true
        linkedin_co_ids = scrape_kwargs.get('linkedin_company_ids', None) # Default to None (Jobspy handles empty list vs None)

        jobs_df = scrape_jobs(
            site_name=sites, search_term=search_terms, location=location, results_wanted=results_wanted,
            hours_old=hours_old, country_indeed=country_indeed, proxies=proxies, offset=offset,
            verbose=1, description_format="markdown",
            linkedin_fetch_description=linkedin_fetch, # Pass the value
            linkedin_company_ids=linkedin_co_ids # Pass the value (Jobspy expects list or None)
        )
        if jobs_df is None or jobs_df.empty: log.warning("Jobspy scraping returned no results or failed."); return None
        else:
            log.info(f"Jobspy scraping successful. Found {len(jobs_df)} jobs.")
            log.debug(f"DataFrame columns: {jobs_df.columns.tolist()}")
            essential_scrape_cols = ['title', 'company', 'location', 'description', 'job_url', 'date_posted', 'job_type']
            for col in essential_scrape_cols:
                 if col not in jobs_df.columns: log.warning(f"Essential column '{col}' missing from JobSpy output, adding as empty."); jobs_df[col] = ''
            return jobs_df
    except ImportError as ie: log.critical(f"Import error during scraping: {ie}."); return None
    # Catch specific pydantic error from incorrect jobspy args
    except Exception as e:
         if "validation error for ScraperInput" in str(e):
              log.error(f"JobSpy validation error: {e}. Check arguments passed vs JobSpy expectations (e.g., linkedin_company_ids should be a list or None).")
         else:
              log.error(f"An error occurred during jobspy scraping: {e}", exc_info=True)
         return None

# --- convert_and_save_scraped function remains unchanged ---
def convert_and_save_scraped(jobs_df: pd.DataFrame, output_path: str) -> List[Dict[str, Any]]:
    # (Content remains the same)
    log.info(f"Converting DataFrame to list and saving to {output_path}")
    rename_map = {'job_url': 'url','job_type': 'employment_type','salary': 'salary_text','benefits': 'benefits_text'}
    actual_rename_map = {k: v for k, v in rename_map.items() if k in jobs_df.columns}; jobs_df = jobs_df.rename(columns=actual_rename_map)
    possible_date_columns = ['date_posted', 'posted_date', 'date']
    for col in possible_date_columns:
        if col in jobs_df.columns:
            if pd.api.types.is_datetime64_any_dtype(jobs_df[col]) or jobs_df[col].dtype == 'object':
                 try: jobs_df[col] = pd.to_datetime(jobs_df[col], errors='coerce'); jobs_df[col] = jobs_df[col].dt.strftime('%Y-%m-%d')
                 except Exception: jobs_df[col] = jobs_df[col].astype(str)
    essential_cols = ['title','company','location','description','url','salary_text','employment_type','benefits_text','skills','date_posted']
    for col in essential_cols:
         if col not in jobs_df.columns: log.warning(f"Column '{col}' missing, adding empty."); jobs_df[col] = ''
    jobs_df = jobs_df.fillna(''); jobs_list = jobs_df.to_dict('records'); log.debug(f"Converted DataFrame to list of {len(jobs_list)} dictionaries.")
    output_dir = os.path.dirname(output_path);
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    try:
        with open(output_path, 'w', encoding='utf-8') as f: json.dump(jobs_list, f, indent=4)
        log.info(f"Saved {len(jobs_list)} scraped jobs to {output_path}"); return jobs_list
    except Exception as e: log.error(f"Error saving scraped jobs JSON: {e}", exc_info=True); return []


# --- print_summary_table function remains unchanged ---
def print_summary_table(results_json: List[Dict[str, Any]], top_n: int = 10):
     # (Content remains the same)
     if not results_json: console.print("[yellow]No analysis results to summarize.[/yellow]") if RICH_AVAILABLE else print("No results."); return
     count = 0; header = ["Score", "Title", "Company", "Location", "URL"]
     rows = []
     for result in results_json:
         if count >= top_n: break
         analysis = result.get('analysis', {}); original = result.get('original_job_data', {})
         score = analysis.get('suitability_score', -1)
         if score is None or score == 0: continue
         rows.append([f"{score}%", original.get('title', 'N/A'), original.get('company', 'N/A'), original.get('location', 'N/A'), original.get('url', '#')])
         count += 1
     if count == 0: console.print("[yellow]No jobs with score > 0.[/yellow]") if RICH_AVAILABLE else print("No jobs with score > 0.")
     elif RICH_AVAILABLE:
         table = Table(title=f"Top {count} Job Matches", show_header=True, header_style="bold magenta", show_lines=False)
         table.add_column(header[0], style="dim", width=6, justify="right"); table.add_column(header[1], style="bold", min_width=20);
         table.add_column(header[2]); table.add_column(header[3]); table.add_column(header[4], overflow="fold", style="cyan")
         for row in rows: table.add_row(*row)
         console.print(table)
     else: # Basic print fallback
          print(f"\n--- Top {count} Job Matches ---")
          print(f"{header[0]:>6} | {header[1]:<30} | {header[2]:<20} | {header[3]:<20}")
          print("-" * 80)
          for row in rows: print(f"{row[0]:>6} | {row[1]:<30.30} | {row[2]:<20.20} | {row[3]:<20.20}")

# --- Main ASYNC Execution Function ---
async def run_pipeline_async():
    # --- Argument Parsing (Use defaults from settings) ---
    parser = argparse.ArgumentParser( description="Run Job Scraping & GenAI Analysis Pipeline.", formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    scrape_cfg = settings.get('scraping', {}) # Use loaded settings
    analysis_cfg = settings.get('analysis', {})

    scrape_group = parser.add_argument_group('Scraping Options (JobSpy)')
    scrape_group.add_argument("--search", required=True, help="Job title, keywords, or company.")
    scrape_group.add_argument("--location", default=None, help="Primary location for scraping. Overridden if --filter-remote-country.")
    scrape_group.add_argument("--sites", default=",".join(scrape_cfg.get('default_sites', [])), help="Comma-separated sites.")
    scrape_group.add_argument("--results", type=int, default=scrape_cfg.get('default_results_limit', 20), help="Approx total jobs per site.")
    scrape_group.add_argument("--hours-old", type=int, default=scrape_cfg.get('default_hours_old', 72), help="Max job age in hours (0=disable).")
    scrape_group.add_argument("--country-indeed", default=scrape_cfg.get('default_country_indeed', 'usa'), help="Country for Indeed search.")
    scrape_group.add_argument("--proxies", help="Comma-separated proxies.")
    scrape_group.add_argument("--offset", type=int, default=0, help="Search results offset.")
    scrape_group.add_argument("--scraped-jobs-file", default=settings.get("scraped_jobs_path"), help="Intermediate file for scraped jobs.")

    analysis_group = parser.add_argument_group('Analysis Options')
    analysis_group.add_argument("--resume", required=True, help="Path to the resume file.")
    analysis_group.add_argument("--analysis-output", default=settings.get("analysis_output_path"), help="Final analysis output JSON.")
    analysis_group.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging (overrides config).")

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

    # --- Setup Logging Level based on verbosity flag ---
    # Already done globally based on settings, override if -v present
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        log.info("Verbose logging enabled.")
    log.info(f"[bold green]Starting ASYNC Pipeline Run[/bold green] ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})" if RICH_AVAILABLE else f"Starting Pipeline Run ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

    # --- Main Pipeline Logic Wrapped in try...except ---
    try:
        # Output dir is ensured by config loading
        # Validate argument combinations
        if not args.location and not args.filter_remote_country and not args.filter_proximity_location: parser.error("Specify --location OR --filter-remote-country OR --filter-proximity-location.")
        if args.filter_proximity_location and args.filter_remote_country: parser.error("Cannot use --filter-proximity-location and --filter-remote-country simultaneously.")
        if args.filter_proximity_location and args.filter_proximity_range is None: parser.error("--filter-proximity-range required with --filter-proximity-location.")
        if args.filter_proximity_range is not None and not args.filter_proximity_location: parser.error("--filter-proximity-location required with --filter-proximity-range.")

        # Determine scrape location
        scrape_location = None
        if args.filter_remote_country: scrape_location = args.filter_remote_country.strip(); log.info(f"Using country '{scrape_location}' as primary scrape location.")
        elif args.filter_proximity_location: scrape_location = args.filter_proximity_location.strip(); log.info(f"Using proximity target '{scrape_location}' as primary scrape location.")
        elif args.location: scrape_location = args.location; log.info(f"Using provided --location '{scrape_location}' as primary scrape location.")

        # --- Step 1: Scrape Jobs ---
        scraper_sites = [site.strip().lower() for site in args.sites.split(',')]
        proxy_list = [p.strip() for p in args.proxies.split(',')] if args.proxies else None
        jobs_df = scrape_jobs_with_jobspy( search_terms=args.search, location=scrape_location, sites=scraper_sites, results_wanted=args.results, hours_old=args.hours_old, country_indeed=args.country_indeed, proxies=proxy_list, offset=args.offset )
        if jobs_df is None or jobs_df.empty: log.warning("Scraping yielded no results. Exiting."); sys.exit(0)

        # --- Step 2: Convert and Save ---
        jobs_list = convert_and_save_scraped(jobs_df, args.scraped_jobs_file)
        if not jobs_list: log.error("Failed convert/save scraped data. Exiting."); sys.exit(1)

        # --- Step 3: Initialize Analyzer and Load Resume ---
        try: analyzer = ResumeAnalyzer()
        except Exception as e: log.critical(f"Failed to initialize ResumeAnalyzer: {e}.", exc_info=True); sys.exit(1)
        structured_resume = await load_and_extract_resume_async(args.resume, analyzer)
        if not structured_resume: log.critical("Failed load/extract resume data. Exiting."); sys.exit(1)

        # --- Step 4: Analyze Jobs ---
        analyzed_results = await analyze_jobs_async(analyzer, structured_resume, jobs_list)
        if not analyzed_results: log.warning("Analysis step produced no results.")

        # --- Step 5: Apply Filters, Sort, and Save ---
        filter_args_dict = {}
        if args.min_salary is not None: filter_args_dict['salary_min'] = args.min_salary
        if args.max_salary is not None: filter_args_dict['salary_max'] = args.max_salary
        if args.filter_work_models: filter_args_dict['work_models'] = [wm.strip().lower() for wm in args.filter_work_models.split(',')]
        if args.filter_job_types: filter_args_dict['job_types'] = [jt.strip().lower() for jt in args.filter_job_types.split(',')]
        if args.filter_remote_country: filter_args_dict['filter_remote_country'] = args.filter_remote_country.strip()
        if args.filter_proximity_location:
             filter_args_dict['filter_proximity_location'] = args.filter_proximity_location.strip()
             filter_args_dict['filter_proximity_range'] = args.filter_proximity_range
             filter_args_dict['filter_proximity_models'] = [pm.strip().lower() for pm in args.filter_proximity_models.split(',')]

        final_results_list_dict = apply_filters_sort_and_save( analyzed_results, args.analysis_output, filter_args_dict )

        # --- Step 6: Print Summary Table ---
        log.info("[bold blue]Pipeline Summary:[/bold blue]" if RICH_AVAILABLE else "--- Pipeline Summary ---")
        print_summary_table(final_results_list_dict, top_n=10)
        log.info(f"[bold green]Pipeline Run Finished Successfully[/bold green]" if RICH_AVAILABLE else "Pipeline Run Finished Successfully")

    except KeyboardInterrupt: print(); log.warning("[yellow]Pipeline interrupted by user (Ctrl+C).[/yellow]" if RICH_AVAILABLE else "Pipeline interrupted."); sys.exit(130)
    except Exception as e: log.critical(f"Unexpected critical error: {e}", exc_info=True); sys.exit(1)

# --- Entry point ---
if __name__ == "__main__":
    try: asyncio.run(run_pipeline_async())
    except KeyboardInterrupt: print("\nExecution cancelled by user."); sys.exit(130)