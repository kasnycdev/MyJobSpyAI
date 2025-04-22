import argparse
import logging
import json
import os
import sys
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import colorama

# Rich for UX - Import Console early
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# === Initialize Colorama FIRST ===
colorama.init(autoreset=True)
# === Initialize Rich Console ===
console = Console() # Global console object for direct prints

# Use the jobspy library for scraping
try:
    import jobspy
    from jobspy import scrape_jobs
except ImportError:
    # Use console for early critical errors
    console.print("[bold red]CRITICAL ERROR: 'jobspy' library not found. Please install it via requirements.txt[/bold red]")
    sys.exit(1)

# Import analysis components
try:
    from main_matcher import load_and_extract_resume_async, analyze_jobs_async, apply_filters_sort_and_save
    from analysis.analyzer import ResumeAnalyzer
except ImportError as e:
    console.print(f"[bold red]CRITICAL ERROR: Could not import analysis functions:[/bold red] {e}")
    console.print("[yellow]Ensure main_matcher.py and analysis/analyzer.py are in the correct path and define expected classes/functions.[/yellow]")
    sys.exit(1)

# Import the loaded settings dictionary from config.py
try:
    from config import settings
except ImportError as e:
    console.print(f"[bold red]CRITICAL ERROR: Could not import configuration:[/bold red] {e}")
    console.print("[yellow]Ensure config.py and config.yaml exist and are correctly configured.[/yellow]")
    sys.exit(1)


# Setup logging using Rich - Use settings loaded from config
# Ensure this happens AFTER settings are loaded but BEFORE parser potentially exits
log_level_name = settings.get('logging', {}).get('level', 'INFO').upper()
log_level = getattr(logging, log_level_name, logging.INFO)

logging.basicConfig(
    level=log_level, # Use level from settings
    format=settings.get('logging', {}).get('format', '%(message)s'),
    datefmt=settings.get('logging', {}).get('date_format', '[%X]'),
    handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=True, console=console)] # Pass console to handler
)
logging.getLogger("httpx").setLevel(logging.WARNING) # Silence noisy httpx
# Get logger *after* basicConfig is called
log = logging.getLogger(__name__)


# === Helper Functions (scrape, convert, print_summary - using rich markup) ===

def scrape_jobs_with_jobspy(
    search_terms: str, location: str, sites: list[str], results_wanted: int, hours_old: int,
    country_indeed: str, proxies: Optional[list[str]] = None, offset: int = 0,
    linkedin_fetch_description: bool = True,
    linkedin_company_ids: Optional[List[int]] = None
    ) -> Optional[pd.DataFrame]:
    """Uses jobspy, logs with rich markup."""
    log.info(f"[bold blue]Starting job scraping via JobSpy...[/bold blue]")
    log.info(f"Search: '[cyan]{search_terms}[/cyan]' | Location: '[cyan]{location}[/cyan]' | Sites: [yellow]{sites}[/yellow]")
    log.info(f"Params: Results ≈{results_wanted}, Max Age={hours_old}h, Indeed Country='{country_indeed}', Offset={offset}")
    if proxies: log.info(f"[yellow]Using {len(proxies)} proxies.[/yellow]")
    if linkedin_company_ids: log.info(f"Filtering by LinkedIn Company IDs: {linkedin_company_ids}")
    try:
        jobs_df = scrape_jobs( site_name=sites, search_term=search_terms, location=location, results_wanted=results_wanted, hours_old=hours_old, country_indeed=country_indeed, proxies=proxies, offset=offset, linkedin_fetch_description=linkedin_fetch_description, linkedin_company_ids=linkedin_company_ids if linkedin_company_ids else [], verbose=1, description_format="markdown" )
        if jobs_df is None or jobs_df.empty:
            log.warning("[yellow]Jobspy scraping returned no results or failed.[/yellow]"); return None
        else:
            log.info(f"[green]Jobspy scraping successful.[/green] Found [bold]{len(jobs_df)}[/bold] jobs.")
            log.debug(f"DataFrame columns: {jobs_df.columns.tolist()}"); required_cols = ['title', 'company', 'location', 'description', 'job_url']
            missing_cols = [col for col in required_cols if col not in jobs_df.columns]
            if missing_cols: log.warning(f"[yellow]Jobspy output missing columns:[/yellow] {missing_cols}")
            essential_scrape_cols = ['title','company','location','description','job_url','date_posted','job_type']
            for col in essential_scrape_cols:
                 if col not in jobs_df.columns: log.warning(f"[yellow]Essential '{col}' missing.[/yellow]"); jobs_df[col] = ''
            return jobs_df
    except ImportError as ie: log.critical(f"[bold red]Import error during scraping:[/bold red] {ie}."); return None
    except Exception as e:
        if "linkedin_company_ids" in str(e) and "Input should be a valid list" in str(e): log.error(f"[bold red]JobSpy Config Error:[/bold red] LinkedIn Company IDs must be list."); log.error(f"Details: {e}", exc_info=False)
        else: log.error(f"[bold red]Error during jobspy scraping:[/bold red] {e}", exc_info=True)
        return None

def convert_and_save_scraped(jobs_df: pd.DataFrame, output_path: str) -> List[Dict[str, Any]]:
    """Converts DataFrame, logs with rich markup."""
    log.info(f"Converting DataFrame and saving to [cyan]{output_path}[/cyan]")
    rename_map = {'job_url':'url','job_type':'employment_type','salary':'salary_text','benefits':'benefits_text'}
    actual_rename_map = {k: v for k, v in rename_map.items() if k in jobs_df.columns};
    if actual_rename_map: jobs_df = jobs_df.rename(columns=actual_rename_map); log.debug(f"Renamed columns: {actual_rename_map}")
    for col in ['date_posted', 'posted_date', 'date']:
        if col in jobs_df.columns:
            if pd.api.types.is_datetime64_any_dtype(jobs_df[col]) or jobs_df[col].dtype == 'object':
                 try: jobs_df[col] = pd.to_datetime(jobs_df[col], errors='coerce'); jobs_df[col] = jobs_df[col].dt.strftime('%Y-%m-%d')
                 except Exception as date_err: log.warning(f"[yellow]Date convert warning for {col}:[/yellow] {date_err}"); jobs_df[col] = jobs_df[col].astype(str)
            else: log.debug(f"Col '{col}' not date/obj type ({jobs_df[col].dtype}), skipping.")
    for col in ['title','company','location','description','url','salary_text','employment_type','benefits_text','skills','date_posted']:
         if col not in jobs_df.columns: log.warning(f"[yellow]Column '{col}' missing, adding empty.[/yellow]"); jobs_df[col] = ''
    jobs_df = jobs_df.fillna(''); log.debug("Filled NA/NaN values.")
    jobs_list = jobs_df.to_dict('records'); log.debug(f"Converted to {len(jobs_list)} dicts.")
    output_dir = os.path.dirname(output_path);
    if output_dir: os.makedirs(output_dir, exist_ok=True)
    try:
        with open(output_path, 'w', encoding='utf-8') as f: json.dump(jobs_list, f, indent=4)
        log.info(f"[green]Saved {len(jobs_list)} scraped jobs[/green] to [cyan]{output_path}[/cyan]")
        return jobs_list
    except TypeError as json_err:
         log.error(f"[bold red]JSON Serialization Error:[/bold red] {json_err}", exc_info=True)
         for i, record in enumerate(jobs_list):
              try: json.dumps(record)
              except TypeError: log.error(f"[red]Problem record index {i}:[/red] {record}"); break
         return []
    except Exception as e: log.error(f"[bold red]Error saving scraped jobs JSON:[/bold red] {e}", exc_info=True); return []


def print_summary_table(results_json: List[Dict[str, Any]], top_n: int = 10):
    """Prints summary table using rich."""
    # (Function content remains the same - uses rich Table styles)
    if not results_json: console.print("[yellow]No analysis results to summarize.[/yellow]"); return
    table = Table(title=f"Top {min(top_n, len(results_json))} Job Matches", show_header=True, header_style="bold magenta", show_lines=False, border_style="dim")
    table.add_column("Score", style="bold cyan", width=6, justify="right"); table.add_column("Title", style="bold white", min_width=20, no_wrap=False); table.add_column("Company", style="green"); table.add_column("Location", style="dim blue"); table.add_column("URL", style="underline blue", overflow="fold")
    count = 0
    for result in results_json:
        if count >= top_n: break
        analysis = result.get('analysis', {}); original = result.get('original_job_data', {})
        score = analysis.get('suitability_score', 0)
        if score == 0: continue
        score_str = f"{score}%"; table.add_row(score_str, original.get('title', 'N/A'), original.get('company', 'N/A'), original.get('location', 'N/A'), original.get('url', '#')); count += 1
    if count == 0: console.print("[yellow]No analyzed jobs with score > 0 to display.[/yellow]")
    else: console.print(table)


# === Main ASYNC Execution Function ===
async def run_pipeline_async():
    # --- Argument Parser Setup (Remains the same) ---
    parser = argparse.ArgumentParser( description="Run Job Scraping (JobSpy) & GenAI Analysis Pipeline.", formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    scrape_cfg = settings.get('scraping', {})
    scrape_group = parser.add_argument_group('Scraping Options (JobSpy)'); scrape_group.add_argument("--search", required=True, help="Job title, keywords, or company."); scrape_group.add_argument("--location", default=None, help="Primary location for scraping. Overridden if --filter-remote-country."); scrape_group.add_argument("--sites", default=",".join(scrape_cfg.get('default_sites', [])), help="Comma-separated sites."); scrape_group.add_argument("--results", type=int, default=scrape_cfg.get('default_results_limit', 20), help="Approx total jobs per site."); scrape_group.add_argument("--hours-old", type=int, default=scrape_cfg.get('default_hours_old', 72), help="Max job age in hours (0=disable)."); scrape_group.add_argument("--country-indeed", default=scrape_cfg.get('default_country_indeed', 'usa'), help="Country for Indeed search."); scrape_group.add_argument("--proxies", help="Comma-separated proxies."); scrape_group.add_argument("--offset", type=int, default=0, help="Search results offset."); scrape_group.add_argument("--linkedin-fetch-description", type=lambda x: (str(x).lower() == 'true'), default=scrape_cfg.get('linkedin_fetch_description', True), help="Fetch full description from LinkedIn (boolean)."); scrape_group.add_argument("--linkedin-company-ids", type=str, default=None, help="Comma-separated LinkedIn company IDs."); scrape_group.add_argument("--scraped-jobs-file", default=settings.get("scraped_jobs_path"), help="Intermediate file for scraped jobs.")
    analysis_group = parser.add_argument_group('Analysis Options'); analysis_group.add_argument("--resume", required=True, help="Path to the resume file."); analysis_group.add_argument("--analysis-output", default=settings.get("analysis_output_path"), help="Final analysis output JSON."); analysis_group.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG level logging.")
    filter_group = parser.add_argument_group('Filtering Options (Applied After Analysis)'); filter_group.add_argument("--min-salary", type=int, help="Minimum desired annual salary."); filter_group.add_argument("--max-salary", type=int, help="Maximum desired annual salary."); filter_group.add_argument("--filter-work-models", help="Standard work models (e.g., 'Remote,Hybrid')."); filter_group.add_argument("--filter-job-types", help="Comma-separated job types (e.g., 'Full-time')")
    adv_loc_group = parser.add_argument_group('Advanced Location Filtering'); adv_loc_group.add_argument("--filter-remote-country", help="Filter REMOTE jobs within specific country."); adv_loc_group.add_argument("--filter-proximity-location", help="Reference location for proximity filtering."); adv_loc_group.add_argument("--filter-proximity-range", type=float, help="Distance in miles for proximity."); adv_loc_group.add_argument("--filter-proximity-models", default="Hybrid,On-site", help="Work models for proximity.")
    args = parser.parse_args()

    # --- Setup Logging Level ---
    log_level_name = "DEBUG" if args.verbose else settings.get('logging', {}).get('level', 'INFO').upper()
    log_level_val = getattr(logging, log_level_name, logging.INFO)
    logging.getLogger().setLevel(log_level_val)
    log.info(f"Log level set to: [yellow]{log_level_name}[/yellow]") # Added color

    log.info(f"[bold green]Starting ASYNC Pipeline Run[/bold green] ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

    # --- Main Pipeline Logic Wrapped in try...except ---
    try:
        # Argument validation
        if not args.location and not args.filter_remote_country and not args.filter_proximity_location: parser.error("Ambiguous location: Specify --location OR --filter-remote-country OR --filter-proximity-location.")
        if args.filter_proximity_location and args.filter_remote_country: parser.error("Conflicting filters: Cannot use --filter-proximity-location and --filter-remote-country.")
        if args.filter_proximity_location and args.filter_proximity_range is None: parser.error("--filter-proximity-range required with --filter-proximity-location.")
        if args.filter_proximity_range is not None and not args.filter_proximity_location: parser.error("--filter-proximity-location required with --filter-proximity-range.")

        # Determine scrape location
        scrape_location = args.location # Default if not overridden
        if args.filter_remote_country: scrape_location = args.filter_remote_country.strip(); log.info(f"Using country '[cyan]{scrape_location}[/cyan]' as primary scrape location.")
        elif args.filter_proximity_location: scrape_location = args.filter_proximity_location.strip(); log.info(f"Using proximity target '[cyan]{scrape_location}[/cyan]' as primary scrape location.")
        elif args.location: log.info(f"Using provided --location '[cyan]{scrape_location}[/cyan]' as primary scrape location.")

        # Parse linkedin_company_ids
        linkedin_company_ids_list = None
        if args.linkedin_company_ids:
            try:
                linkedin_company_ids_list = [int(cid.strip()) for cid in args.linkedin_company_ids.split(',') if cid.strip().isdigit()]
                if not linkedin_company_ids_list: log.warning("[yellow]--linkedin-company-ids provided but contained no valid integers.[/yellow]")
            except ValueError: parser.error("Invalid format for --linkedin-company-ids. Must be comma-separated integers.")

        # --- Step 1: Scrape Jobs ---
        jobs_df = scrape_jobs_with_jobspy(
            search_terms=args.search, location=scrape_location, sites=[s.strip().lower() for s in args.sites.split(',')],
            results_wanted=args.results, hours_old=args.hours_old, country_indeed=args.country_indeed,
            proxies=[p.strip() for p in args.proxies.split(',')] if args.proxies else None, offset=args.offset,
            linkedin_fetch_description=args.linkedin_fetch_description,
            linkedin_company_ids=linkedin_company_ids_list )
        if jobs_df is None or jobs_df.empty:
            log.warning("[yellow]Scraping yielded no results. Exiting.[/yellow]")
            analysis_output_dir = os.path.dirname(args.analysis_output);
            if analysis_output_dir: os.makedirs(analysis_output_dir, exist_ok=True)
            with open(args.analysis_output, 'w', encoding='utf-8') as f: json.dump([], f)
            log.info(f"Empty analysis results file created at [cyan]{args.analysis_output}[/cyan]")
            sys.exit(0)

        # --- Step 2: Convert and Save ---
        jobs_list = convert_and_save_scraped(jobs_df, args.scraped_jobs_file)
        if not jobs_list: log.error("[bold red]Failed convert/save scraped data. Exiting.[/bold red]"); sys.exit(1)

        # --- Step 3: Initialize Analyzer and Load Resume ---
        try: analyzer = ResumeAnalyzer()
        except Exception as e: log.critical(f"[bold red]Failed init ResumeAnalyzer:[/bold red] {e}.", exc_info=True); sys.exit(1)
        structured_resume = await load_and_extract_resume_async(args.resume, analyzer)
        if not structured_resume: log.critical("[bold red]Failed load/extract resume data. Exiting.[/bold red]"); sys.exit(1)

        # --- Step 4: Analyze Jobs ---
        analyzed_results = await analyze_jobs_async(analyzer, structured_resume, jobs_list)
        if not analyzed_results: log.warning("[yellow]Analysis step produced no results.[/yellow]")

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
        # apply_filters_sort_and_save uses internal logging
        final_results_list_dict = apply_filters_sort_and_save( analyzed_results, args.analysis_output, filter_args_dict )

        # --- Step 6: Print Summary Table ---
        log.info("[bold blue]Pipeline Summary:[/bold blue]")
        print_summary_table(final_results_list_dict, top_n=10) # Uses rich table
        log.info(f"[bold green]Pipeline Run Finished Successfully[/bold green]")

    except KeyboardInterrupt:
        # Use console.print for interrupt message
        console.print("\n[bold yellow]Pipeline execution interrupted by user (Ctrl+C). Exiting gracefully.[/bold yellow]")
        sys.exit(130)
    except Exception as e:
         # Use logger for unexpected errors
         log.critical(f"[bold red]Unexpected critical error during pipeline execution:[/bold red] {e}", exc_info=True)
         sys.exit(1)

# --- Entry point ---
if __name__ == "__main__":
    try: asyncio.run(run_pipeline_async())
    except KeyboardInterrupt:
        # Catch Ctrl+C if it happens before async loop starts or after it ends
        console.print("\n[yellow]Execution cancelled by user.[/yellow]")
        sys.exit(130)