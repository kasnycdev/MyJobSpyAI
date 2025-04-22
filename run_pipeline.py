# run_pipeline.py

import argparse
import logging
import json
import os
import sys
import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import colorama # Import colorama

# Use the jobspy library for scraping
try:
    import jobspy
    from jobspy import scrape_jobs
except ImportError: print("CRITICAL ERROR: 'jobspy' library not found."); sys.exit(1)

# Import analysis components
try:
    from main_matcher import load_and_extract_resume_async, analyze_jobs_async, apply_filters_sort_and_save
    from analysis.analyzer import ResumeAnalyzer
except ImportError as e: print(f"CRITICAL ERROR: Could not import analysis functions: {e}"); sys.exit(1)

# Import the loaded settings dictionary from config.py
from config import settings # Assuming config.py loads YAML into 'settings' dict

# Rich for UX
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# === Initialize Colorama ===
# This call wraps stdout/stderr on Windows to make ANSI codes work.
# autoreset=True automatically adds Style.RESET_ALL after each print.
colorama.init(autoreset=True)
# === End Colorama Init ===

# Setup logging using Rich - Use settings loaded from config
log_level_name = settings.get('logging', {}).get('level', 'INFO').upper()
log_level = getattr(logging, log_level_name, logging.INFO)

logging.basicConfig(
    level=log_level,
    format=settings.get('logging', {}).get('format', '%(message)s'),
    datefmt=settings.get('logging', {}).get('date_format', '[%X]'),
    handlers=[RichHandler(rich_tracebacks=True, show_path=False, markup=True)] # Keep markup=True
)
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger(__name__)
console = Console() # For direct console output like tables


# === Helper Functions (scrape, convert, print_summary - No changes needed here, keep rich markup) ===

def scrape_jobs_with_jobspy(
    search_terms: str, location: str, sites: list[str], results_wanted: int, hours_old: int,
    country_indeed: str, proxies: Optional[list[str]] = None, offset: int = 0,
    linkedin_fetch_description: bool = True,
    linkedin_company_ids: Optional[List[int]] = None
    ) -> Optional[pd.DataFrame]:
    """Uses jobspy, logs with rich markup."""
    # (Function content remains the same - uses rich markup like [bold blue] etc.)
    log.info(f"[bold blue]Starting job scraping via JobSpy...[/bold blue]")
    log.info(f"Search: '[cyan]{search_terms}[/cyan]' | Location: '[cyan]{location}[/cyan]' | Sites: [yellow]{sites}[/yellow]")
    log.info(f"Params: Results ≈{results_wanted}, Max Age={hours_old}h, Indeed Country='{country_indeed}', Offset={offset}")
    if proxies: log.info(f"[yellow]Using {len(proxies)} proxies.[/yellow]")
    if linkedin_company_ids: log.info(f"Filtering by LinkedIn Company IDs: {linkedin_company_ids}")
    try:
        jobs_df = scrape_jobs( site_name=sites, search_term=search_terms, location=location, results_wanted=results_wanted, hours_old=hours_old, country_indeed=country_indeed, proxies=proxies, offset=offset, linkedin_fetch_description=linkedin_fetch_description, linkedin_company_ids=linkedin_company_ids if linkedin_company_ids else [], verbose=1, description_format="markdown" )
        if jobs_df is None or jobs_df.empty: log.warning("[yellow]Jobspy scraping returned no results or failed.[/yellow]"); return None
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
    # (Function content remains the same - uses rich markup like [cyan] etc.)
    log.info(f"Converting DataFrame and saving to [cyan]{output_path}[/cyan]")
    rename_map = {'job_url':'url','job_type':'employment_type','salary':'salary_text','benefits':'benefits_text'}
    actual_rename_map = {k: v for k, v in rename_map.items() if k in jobs_df.columns};
    if actual_rename_map: jobs_df = jobs_df.rename(columns=actual_rename_map); log.debug(f"Renamed columns: {actual_rename_map}")
    for col in ['date_posted', 'posted_date', 'date']:
        if col in jobs_df.columns:
            if pd.api.types.is_datetime64_any_dtype(jobs_df[col]) or jobs_df[col].dtype == 'object':
                 try: jobs_df[col] = pd.to_datetime(jobs_df[col], errors='coerce'); jobs_df[col] = jobs_df[col].dt.strftime('%Y-%m-%d')
                 except Exception as date_err: log.warning(f"Date convert warning for {col}: {date_err}"); jobs_df[col] = jobs_df[col].astype(str)
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
    # (Argument Parsing remains the same)
    parser = argparse.ArgumentParser( description="Run Job Scraping (JobSpy) & GenAI Analysis Pipeline.", formatter_class=argparse.ArgumentDefaultsHelpFormatter )
    # ... all argument definitions ...
    args = parser.parse_args()

    # --- Setup Logging Level ---
    # (Logging level setup remains the same)
    log_level_name = "DEBUG" if args.verbose else settings.get('logging', {}).get('level', 'INFO').upper()
    log_level_val = getattr(logging, log_level_name, logging.INFO)
    logging.getLogger().setLevel(log_level_val)
    log.info(f"Log level set to: {log_level_name}")

    log.info(f"[bold green]Starting ASYNC Pipeline Run[/bold green] ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")

    try:
        # --- Main pipeline steps remain the same ---
        # Argument validation
        # Determine scrape location
        # Step 1: Scrape Jobs (call scrape_jobs_with_jobspy)
        # Step 2: Convert and Save (call convert_and_save_scraped)
        # Step 3: Initialize Analyzer and Load Resume (instantiate ResumeAnalyzer, call load_and_extract_resume_async)
        # Step 4: Analyze Jobs (call analyze_jobs_async)
        # Step 5: Apply Filters, Sort, and Save (populate filter_args_dict, call apply_filters_sort_and_save)
        # Step 6: Print Summary Table (call print_summary_table)

        # --- Example snippet for Step 1 call ---
        # ... (validation and scrape_location logic) ...
        linkedin_company_ids_list = None # Parse args.linkedin_company_ids if provided
        # ...
        jobs_df = scrape_jobs_with_jobspy( search_terms=args.search, location=scrape_location, sites=[s.strip().lower() for s in args.sites.split(',')], results_wanted=args.results, hours_old=args.hours_old, country_indeed=args.country_indeed, proxies=[p.strip() for p in args.proxies.split(',')] if args.proxies else None, offset=args.offset, linkedin_fetch_description=args.linkedin_fetch_description, linkedin_company_ids=linkedin_company_ids_list )
        # ... (rest of pipeline steps) ...

        log.info(f"[bold green]Pipeline Run Finished Successfully[/bold green]")

    except KeyboardInterrupt: print(); log.warning("[bold yellow]Pipeline interrupted by user (Ctrl+C).[/bold yellow]"); sys.exit(130)
    except Exception as e: log.critical(f"[bold red]Unexpected critical error during pipeline:[/bold red] {e}", exc_info=True); sys.exit(1)


# --- Entry point ---
if __name__ == "__main__":
    try: asyncio.run(run_pipeline_async())
    except KeyboardInterrupt: print("\nExecution cancelled by user."); sys.exit(130)