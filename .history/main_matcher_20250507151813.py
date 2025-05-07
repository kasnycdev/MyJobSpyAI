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

    tasks = [
        asyncio.create_task(
            analyzer.analyze_suitability_async(structured_resume_data, job_dict)
        )
        for job_dict in job_list
    ]

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

    # Add an overall progress task
    with progress_context as progress:
        overall_task = progress.add_task("[cyan]Overall Progress...", total=len(tasks))
        analysis_task_tracker = progress.add_task("[cyan]Analyzing Jobs...", total=len(tasks))

        results_or_exceptions = []
        for task in tasks:
            result = await task
            progress.update(analysis_task_tracker, advance=1)
            progress.update(overall_task, advance=1)
            results_or_exceptions.append(result)

        progress.update(
            analysis_task_tracker,
            completed=len(tasks),
            description="[green]Analysis Complete"
        )
        progress.update(
            overall_task,
            completed=len(tasks),
            description="[green]All Tasks Complete"
        )

    console.log("Processing analysis results...")
    for i, result_or_exc in enumerate(results_or_exceptions):
        job_dict = job_list[i]
        job_title = job_dict.get("title", "N/A")
        analysis_result = None
        if isinstance(result_or_exc, Exception):
            console.log(
                f"[yellow]Analysis task for '{job_title}' failed: "
                f"[yellow]{result_or_exc}[/yellow]",
                exc_info=False,
            )
        elif result_or_exc is None:
            console.log(
                f"[yellow]Analysis task for '{job_title}' returned None "
                f"(LLM fail/no description).[/yellow]"
            )
        else:
            analysis_result = result_or_exc

        if analysis_result is None:
            try:
                analysis_result = JobAnalysisResult(
                    suitability_score=0,
                    justification="Analysis task failed or job data insufficient.",
                    skill_match=None,
                    experience_match=None,
                    qualification_match=None,
                    salary_alignment="N/A",
                    benefit_alignment="N/A",
                    missing_keywords=[],
                )
            except Exception as placeholder_err:
                console.log(
                    f"[bold red]Failed create placeholder for failed analysis of '{job_title}':[/bold red] {placeholder_err}",
                    exc_info=True,
                )
                continue
        analyzed_job = AnalyzedJob(
            original_job_data=job_dict, analysis=analysis_result
        )
        analyzed_results.append(analyzed_job)

    console.log(
        f"[green]ASYNC analysis complete. Processed results for {len(analyzed_results)} jobs.[/green]"
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
