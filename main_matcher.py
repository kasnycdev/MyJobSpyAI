import logging
import json
import os
import argparse
import asyncio
from typing import List, Dict, Any, Optional

# Use ResumeAnalyzer which now has async methods
from analysis.analyzer import ResumeAnalyzer
from parsers.resume_parser import parse_resume
from parsers.job_parser import load_job_mandates
from analysis.models import ResumeData, AnalyzedJob, JobAnalysisResult
from filtering.filter import apply_filters # filter functions remain synchronous
import config

# Rich for UX (used for progress bar)
try:
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Setup logger
log = logging.getLogger(__name__)

# --- ASYNC Resume Loading/Extraction ---
async def load_and_extract_resume_async(resume_path: str, analyzer: ResumeAnalyzer) -> Optional[ResumeData]:
    """ASYNC: Loads resume, parses text, and extracts structured data."""
    log.info(f"Processing resume file: [cyan]{resume_path}[/cyan]")
    try: resume_text = parse_resume(resume_path)
    except Exception as e: log.error(f"[bold red]Failed to parse resume text from {resume_path}:[/bold red] {e}", exc_info=True); return None
    if not resume_text: log.error("[bold red]Parsed resume text is empty.[/bold red]"); return None
    # Extraction uses LLM, make it async (logs within analyzer are colored)
    structured_resume_data = await analyzer.extract_resume_data_async(resume_text)
    if not structured_resume_data: log.error("[bold red]Failed to extract structured data from resume.[/bold red]"); return None
    log.info("[green]Successfully extracted structured data from resume.[/green]")
    return structured_resume_data

# --- ASYNC Job Analysis ---
async def analyze_jobs_async(
    analyzer: ResumeAnalyzer,
    structured_resume_data: ResumeData,
    job_list: List[Dict[str, Any]]
) -> List[AnalyzedJob]:
    """ASYNC: Analyzes a list of jobs concurrently against the resume data."""
    analyzed_results: list[AnalyzedJob] = []
    total_jobs = len(job_list)
    log.info(f"Starting ASYNC analysis of [bold]{total_jobs}[/bold] jobs...")

    tasks = [asyncio.create_task(analyzer.analyze_suitability_async(structured_resume_data, job_dict)) for job_dict in job_list]

    # Setup rich progress bar
    progress_context = None
    if RICH_AVAILABLE:
        progress_context = Progress( SpinnerColumn(), "[progress.description]{task.description}", BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%", TextColumn("[progress.completed]{task.completed} of {task.total} jobs"), TimeElapsedColumn() )
    else: log.warning("[yellow]Rich library not found, progress bar disabled.[/yellow]")

    results_or_exceptions = []
    if progress_context:
         with progress_context as progress:
              analysis_task_tracker = progress.add_task("[cyan]Analyzing Jobs...", total=len(tasks))
              results_or_exceptions = await asyncio.gather(*tasks, return_exceptions=True)
              progress.update(analysis_task_tracker, completed=len(tasks), description="[green]Analysis Complete")
    else:
         log.info(f"Running {len(tasks)} analysis tasks concurrently (no progress bar)...")
         results_or_exceptions = await asyncio.gather(*tasks, return_exceptions=True)

    log.info("Processing analysis results...")
    for i, result_or_exc in enumerate(results_or_exceptions):
        job_dict = job_list[i]; job_title = job_dict.get('title', 'N/A'); analysis_result = None
        if isinstance(result_or_exc, Exception): log.warning(f"[yellow]Analysis task for '{job_title}' failed:[/yellow] {result_or_exc}", exc_info=False); log.debug("Exc details:", exc_info=result_or_exc)
        elif result_or_exc is None: log.warning(f"[yellow]Analysis task for '{job_title}' returned None (LLM fail/no description).[/yellow]")
        else: analysis_result = result_or_exc

        if analysis_result is None:
            try:
                analysis_result = JobAnalysisResult( suitability_score=0, justification="Analysis task failed or job data insufficient.", skill_match=None, experience_match=None, qualification_match=None, salary_alignment="N/A", benefit_alignment="N/A", missing_keywords=[] )
            except Exception as placeholder_err: log.critical(f"[bold red]Failed create placeholder for failed analysis of '{job_title}':[/bold red] {placeholder_err}", exc_info=True); continue
        analyzed_job = AnalyzedJob(original_job_data=job_dict, analysis=analysis_result)
        analyzed_results.append(analyzed_job)

    log.info(f"ASYNC analysis complete. Processed results for {len(analyzed_results)} jobs.")
    return analyzed_results


# --- apply_filters_sort_and_save remains synchronous ---
def apply_filters_sort_and_save(
    analyzed_results: List[AnalyzedJob], output_path: str, filter_args: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Applies filters, sorts, and saves the final results."""
    # (Function content remains the same - internal logs use rich markup via root logger)
    jobs_to_filter = [res.original_job_data for res in analyzed_results]
    if filter_args:
        log.info("Applying post-analysis filters...")
        filtered_original_jobs = apply_filters(jobs_to_filter, **filter_args) # Uses filter.py logging
        log.info(f"{len(filtered_original_jobs)} jobs passed filters.")
        filtered_keys = {
            (
                job.get('url', job.get('job_url')),
                job.get('title'),
                job.get('company'),
                job.get('location'),
            )
            for job in filtered_original_jobs
        }
        final_filtered_results = [res for res in analyzed_results if (res.original_job_data.get('url', res.original_job_data.get('job_url')), res.original_job_data.get('title'), res.original_job_data.get('company'), res.original_job_data.get('location')) in filtered_keys]
    else: final_filtered_results = analyzed_results
    log.info("Sorting results by suitability score...")
    final_filtered_results.sort(key=lambda x: x.analysis.suitability_score if x.analysis else 0, reverse=True )
    final_results_json = [result.model_dump(mode='json') for result in final_filtered_results]
    if output_dir := os.path.dirname(output_path):
        os.makedirs(output_dir, exist_ok=True)
    try:
        with open(output_path, 'w', encoding='utf-8') as f: json.dump(final_results_json, f, indent=4)
        log.info(f"[green]Successfully saved {len(final_results_json)} analyzed jobs[/green] to [cyan]{output_path}[/cyan]")
    except Exception as e: log.error(f"[bold red]Error writing output file {output_path}:[/bold red] {e}", exc_info=True); log.debug(f"Problem data (first item): {final_results_json[0] if final_results_json else 'N/A'}")
    return final_results_json


# --- Main execution block updated for async ---
async def main_async():
    """Async main function for standalone execution."""
    # (Argument parsing unchanged)
    parser = argparse.ArgumentParser(description="Analyze pre-existing job JSON against a resume.") # ... add args ...
    args = parser.parse_args()
    # (Standalone logging setup - might not use Rich unless explicitly configured here)
    log_level = logging.DEBUG if args.verbose else config.settings.get('logging', {}).get('level', 'INFO').upper()
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'); logging.getLogger("httpx").setLevel(logging.WARNING)
    log.info("Starting standalone ASYNC analysis process...")
    try: analyzer = ResumeAnalyzer()
    except Exception as e: log.error(f"Failed to initialize analyzer: {e}", exc_info=True); return
    structured_resume = await load_and_extract_resume_async(args.resume, analyzer)
    if not structured_resume: log.error("Exiting due to resume processing failure."); return
    log.info(f"Loading jobs from Parquet file: {args.jobs}"); job_list = load_job_mandates(args.jobs)
    if not job_list: log.error("No jobs loaded from JSON file. Exiting."); return
    analyzed_results = await analyze_jobs_async(analyzer, structured_resume, job_list)
    filter_args_dict = {} # ... populate filter_args_dict from args ...
    apply_filters_sort_and_save(analyzed_results, args.output, filter_args_dict)
    log.info("Standalone ASYNC analysis finished.")

if __name__ == "__main__":
    try: asyncio.run(main_async())
    except KeyboardInterrupt: console.print("\n[yellow]Standalone analysis interrupted.[/yellow]"); sys.exit(130) # Use console here