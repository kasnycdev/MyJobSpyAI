import logging
import json
import os
import argparse
import asyncio
import sys  # Importing sys to resolve undefined variable issue
from typing import List, Dict, Any, Optional

import config
from analysis.analyzer import ResumeAnalyzer
from parsers.resume_parser import parse_resume
from parsers.job_parser import load_job_mandates
from analysis.models import ResumeData, AnalyzedJob, JobAnalysisResult
from filtering.filter import apply_filters  # filter functions remain synchronous
from rich.console import Console  # Ensure this import is resolved
from colorama import Fore, Style  # Import colorama
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TaskProgressColumn

# Initialize rich console
console = Console()

# Replace logging calls with console.log
console.log("[green]Main matcher initialized with rich logging.[/green]")

# Rich for UX (used for progress bar)
try:
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# --- ASYNC Resume Loading/Extraction ---
async def load_and_extract_resume_async(
    resume_path: str, analyzer: ResumeAnalyzer
) -> Optional[ResumeData]:
    """ASYNC: Loads resume, parses text, and extracts structured data."""
    console.log(f"Processing resume file: [cyan]{resume_path}[/cyan]")
    try:
        resume_text = parse_resume(resume_path)
    except Exception as e:
        console.log(
            f"[bold red]Failed to parse resume text from {resume_path}: {e}"
        )
        return None
    if not resume_text:
        console.log("[bold red]Parsed resume text is empty.[/bold red]")
        return None
    structured_resume_data = await analyzer.extract_resume_data_async(resume_text)
    if not structured_resume_data:
        console.log(
            "[bold red]Failed to extract structured data from resume.[/bold red]"
        )
        return None
    console.log("[green]Successfully extracted structured data from resume.[/green]")
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
    console.log(f"Starting ASYNC analysis of [bold]{total_jobs}[/bold] jobs...")

    analyzed_results: list[AnalyzedJob] = []
    total_jobs = len(job_list)
    console.log(f"Starting ASYNC analysis of [bold]{total_jobs}[/bold] jobs...")

    async def process_single_job(job_dict_item: Dict[str, Any]):
        job_description_text = job_dict_item.get("description", "")
        job_title_for_log = job_dict_item.get("title", "N/A") # Get title for logging

        # Task 1: Extract Job Details
        console.log(f"Creating task: Job Details Extraction for '{job_title_for_log}'") # Added log
        parsed_job_details_task = asyncio.create_task(
            analyzer.extract_job_details_async(job_description_text, job_title=job_title_for_log) # Pass job_title
        )

        # Task 2: Analyze Suitability
        console.log(f"Creating task: Suitability Analysis for '{job_title_for_log}'") # Added log
        # Task 2: Analyze Suitability
        suitability_analysis_task = asyncio.create_task(
            analyzer.analyze_suitability_async(structured_resume_data, job_dict_item)
        )
        
        parsed_job_details = await parsed_job_details_task
        suitability_result = await suitability_analysis_task
        
        return job_dict_item, parsed_job_details, suitability_result

    tasks = [process_single_job(job_dict) for job_dict in job_list]

    # Setup rich progress bar with overall progress
    progress_context = Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("[progress.completed]{task.completed}/{task.total} jobs"),
        TimeElapsedColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%"
    )

    with progress_context as progress:
        overall_task = progress.add_task("[cyan]Processing Jobs...", total=len(tasks))

        for future in asyncio.as_completed(tasks):
            try:
                original_job_data, parsed_details, analysis_result_obj = await future
                job_title = original_job_data.get("title", "N/A")

                if analysis_result_obj is None:
                    console.log(
                        f"[yellow]Suitability analysis for '{job_title}' returned None or failed. Creating placeholder.[/yellow]"
                    )
                    # Create a default/placeholder JobAnalysisResult if suitability failed
                    analysis_result_obj = JobAnalysisResult(
                        suitability_score=0,
                        justification="Suitability analysis failed or job data insufficient for analysis.",
                        pros=[],
                        cons=[],
                        skill_match_summary="N/A",
                        experience_match_summary="N/A",
                        education_match_summary="N/A",
                        missing_keywords=[]
                    )
                
                if parsed_details is None:
                    console.log(
                        f"[yellow]Job detail extraction for '{job_title}' returned None or failed.[/yellow]"
                        # parsed_details will remain None in AnalyzedJob
                    )

                analyzed_job = AnalyzedJob(
                    original_job_data=original_job_data,
                    parsed_job_details=parsed_details, # Can be None
                    analysis=analysis_result_obj # Should always be a JobAnalysisResult instance
                )
                analyzed_results.append(analyzed_job)

            except Exception as e:
                # This catches errors from process_single_job itself or unhandled ones from tasks
                console.log(f"[bold red]Error processing a job: {e}[/bold red]", exc_info=True)
                # Optionally, create a placeholder AnalyzedJob for jobs that hit this broader error
                # For now, we just log and it won't be added to analyzed_results if it errors here.
            
            progress.update(overall_task, advance=1)
        
        progress.update(overall_task, description="[green]Job Processing Complete[/green]")

    console.log(
        f"[green]ASYNC processing complete. Results for {len(analyzed_results)} jobs.[/green]"
    )
    return analyzed_results


# --- apply_filters_sort_and_save remains synchronous ---
def apply_filters_sort_and_save(
    analyzed_results: List[AnalyzedJob], output_path: str, filter_args: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Applies filters, sorts, and saves the final results."""
    # (Function content remains the same - internal logs use rich markup via root logger)
    jobs_to_filter = [res.original_job_data for res in analyzed_results]
    if filter_args:
        console.log("Applying post-analysis filters...")
        filtered_original_jobs = apply_filters(jobs_to_filter, **filter_args)  # Uses filter.py logging
        console.log(f"{len(filtered_original_jobs)} jobs passed filters.")
        filtered_keys = {
            (
                job.get('url', job.get('job_url')),
                job.get('title'),
                job.get('company'),
                job.get('location'),
            )
            for job in filtered_original_jobs
        }
        final_filtered_results = [
            res for res in analyzed_results if (
                res.original_job_data.get('url', res.original_job_data.get('job_url')),
                res.original_job_data.get('title'),
                res.original_job_data.get('company'),
                res.original_job_data.get('location')
            ) in filtered_keys
        ]
    else:
        final_filtered_results = analyzed_results
    console.log("Sorting results by suitability score...")
    final_filtered_results.sort(key=lambda x: x.score, reverse=True)  # Use the score property
    final_results_list = [
        result.model_dump() for result in final_filtered_results if result
    ]  # Convert to list of dicts

    # Save to JSON
    if output_dir := os.path.dirname(output_path):
        os.makedirs(output_dir, exist_ok=True)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_results_list, f, indent=4)  # Save as JSON
        console.log(
            f"[green]Successfully saved {len(final_results_list)} analyzed jobs "
            f"to [cyan]{output_path}[/cyan]"
        )
    except Exception as e:
        console.log(f"[bold red]Error saving JSON file:[/bold red] {e}", exc_info=True)

    return final_results_list  # Return list of dictionaries


# --- Main execution block updated for async ---
async def main_async():
    """Async main function for standalone execution."""
    # (Argument parsing unchanged)
    parser = argparse.ArgumentParser(
        description="Analyze pre-existing job JSON against a resume."
    )
    args = parser.parse_args()
    # (Standalone logging setup - might not use Rich unless explicitly configured here)
    log_level = logging.DEBUG if args.verbose else config.settings.get(
        "logging", {}
    ).get("level", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console.log("Starting standalone ASYNC analysis process...")
    try:
        analyzer = ResumeAnalyzer()
    except Exception as e:
        console.log(f"Failed to initialize analyzer: {e}", exc_info=True)
        return
    structured_resume = await load_and_extract_resume_async(args.resume, analyzer)
    if not structured_resume:
        console.log("Exiting due to resume processing failure.")
        return
    console.log(f"Loading jobs from Parquet file: {args.jobs}")
    job_list = load_job_mandates(args.jobs)
    if not job_list:
        console.log("No jobs loaded from JSON file. Exiting.")
        return
    analyzed_results = await analyze_jobs_async(analyzer, structured_resume, job_list)
    filter_args_dict = {}  # ... populate filter_args_dict from args ...
    apply_filters_sort_and_save(analyzed_results, args.output, filter_args_dict)
    console.log("Standalone ASYNC analysis finished.")
    console.log("[green]Standalone analysis finished successfully.[/green]")


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        console.log(
            "\n[yellow]Standalone analysis interrupted.[/yellow]"
        )
        console.log("\nExecution cancelled by user.")
        sys.exit(130)
