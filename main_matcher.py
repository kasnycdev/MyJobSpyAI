import logging
import json
from filtering.filter_utils import DateEncoder # Import DateEncoder
import os
import argparse
import asyncio
import sys  # Importing sys to resolve undefined variable issue
from typing import List, Dict, Any, Optional

import config
from parsers.resume_parser import parse_resume
from parsers.job_parser import load_job_mandates
from analysis.models import ResumeData, AnalyzedJob, JobAnalysisResult # Removed SkillDetail
from filtering.filter import apply_filters
from analysis.analyzer import ResumeAnalyzer, JobAnalyzer, BaseAnalyzer # Added BaseAnalyzer
# from rich.console import Console # Unused
# from colorama import Fore, Style # Unused
from rich.progress import TaskProgressColumn # Others imported in try/except

# Get a logger for this module
logger = logging.getLogger(__name__)

# Import tracer from logging_utils
try:
    from logging_utils import tracer as global_tracer
    if global_tracer is None: # Check if OTEL was disabled in logging_utils
        from opentelemetry import trace
        # Fallback to a NoOpTracer if OTEL is not configured
        tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
        logger.warning("OpenTelemetry not configured in logging_utils, using NoOpTracer for main_matcher.")
    else:
        tracer = global_tracer
except ImportError:
    from opentelemetry import trace
    tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
    logger.error("Could not import global_tracer from logging_utils. Using NoOpTracer.", exc_info=True)


# console = Console() # Replaced by logger for general messages

logger.info("Main matcher initialized.")

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
@tracer.start_as_current_span("load_and_extract_resume_async")
async def load_and_extract_resume_async(
    resume_path: str, force_reparse: bool = False
) -> Optional[ResumeData]:
    """ASYNC: Loads resume, parses text, and extracts structured data with caching."""
    current_span = trace.get_current_span()
    current_span.set_attribute("resume_path", resume_path)
    current_span.set_attribute("force_reparse", force_reparse)
    logger.info(f"Processing resume file: {resume_path}")

    cache_dir = config.get_setting('output.resume_cache_dir', 'cache/resume_cache/')
    os.makedirs(cache_dir, exist_ok=True)
    
    # Use file modification time for cache key
    try:
        mtime = os.path.getmtime(resume_path)
        cache_key = f"{os.path.basename(resume_path)}_{mtime}.json"
        cache_path = os.path.join(cache_dir, cache_key)
    except Exception as e:
        logger.warning(f"Could not get resume file modification time for caching: {e}. Skipping cache.")
        cache_path = None # Disable caching if mtime fails

    if cache_path and os.path.exists(cache_path) and not force_reparse:
        with tracer.start_as_current_span("load_resume_from_cache"):
            logger.info(f"Loading structured resume data from cache: {cache_path}")
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                # Validate cached data structure if necessary, for now assume it's correct
                structured_resume_data = ResumeData(**cached_data)
                logger.info("Successfully loaded structured data from cache.")
                return structured_resume_data
            except Exception as e:
                logger.warning(f"Error loading from cache {cache_path}: {e}. Reparsing.")
                trace.get_current_span().record_exception(e)
                # Fall through to parsing if cache loading fails

    if force_reparse:
        logger.warning("Force reparse requested. Skipping cache.") # Corrected indentation

    # If cache not found, invalid, or force_reparse is True, parse and extract
    logger.info("Cache not found or invalid. Parsing and extracting resume data.")
    try:
        with tracer.start_as_current_span("parse_resume_text"):
            resume_text = parse_resume(resume_path) # Assumes parse_resume uses standard logging
    except Exception as e:
        logger.error(f"Failed to parse resume text from {resume_path}: {e}", exc_info=True)
        trace.get_current_span().record_exception(e)
        return None
    if not resume_text:
        logger.error("Parsed resume text is empty.")
        return None

    try:
        with tracer.start_as_current_span("extract_resume_data_llm"):
            # Instantiate ResumeAnalyzer and extract data
            resume_analyzer = ResumeAnalyzer() # This might also be instrumented if it makes external calls
            structured_resume_data = await resume_analyzer.extract_resume_data_async(resume_text)
    except Exception as e: # Catch errors during instantiation or call
        logger.error(f"Error during resume analysis: {e}", exc_info=True)
        trace.get_current_span().record_exception(e)
        return None
    
    if not structured_resume_data: # Check after attempting extraction
        logger.error("Failed to extract structured data from resume.")
        return None

    logger.info("Successfully extracted structured data from resume.")

    # Save to cache
    if cache_path:
        with tracer.start_as_current_span("save_resume_to_cache"):
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(structured_resume_data.model_dump(), f, indent=4)
                logger.info(f"Successfully saved structured data to cache: {cache_path}")
                # Clean up old cache files for this resume name
                for filename in os.listdir(cache_dir):
                    if filename.startswith(os.path.basename(resume_path) + "_") and filename != cache_key: # Correctly indented
                        try:
                            os.remove(os.path.join(cache_dir, filename))
                            logger.info(f"Removed old cache file: {filename}")
                        except Exception as e:
                            logger.warning(f"Error removing old cache file {filename}: {e}")
                            # Not recording this minor exception to span to avoid noise
            except Exception as e:
                logger.warning(f"Error saving to cache {cache_path}: {e}.")
                trace.get_current_span().record_exception(e)

    return structured_resume_data


# --- ASYNC Job Analysis ---
@tracer.start_as_current_span("analyze_jobs_async")
async def analyze_jobs_async(
    structured_resume_data: ResumeData,
    job_list: List[Dict[str, Any]]
) -> List[AnalyzedJob]:
    """ASYNC: Analyzes a list of jobs concurrently against the resume data."""
    current_span = trace.get_current_span()
    current_span.set_attribute("num_jobs_to_analyze", len(job_list))
    analyzed_results: list[AnalyzedJob] = []
    total_jobs = len(job_list)
    logger.info(f"Starting ASYNC analysis of {total_jobs} jobs...")

    try:
        job_analyzer = JobAnalyzer() # Instantiate JobAnalyzer once
    except Exception as e:
        logger.error(f"Failed to instantiate JobAnalyzer: {e}. Aborting job analysis.", exc_info=True)
        return [] # Return empty list as job analysis cannot proceed

    @tracer.start_as_current_span("process_single_job_analysis")
    async def process_single_job(job_dict_item: Dict[str,Any]):
        job_description_text = job_dict_item.get("description", "")
        job_title_for_log = job_dict_item.get("title", "N/A")
        trace.get_current_span().set_attribute("job_title", job_title_for_log)

        # Task 1: Extract Job Details using JobAnalyzer instance
        with tracer.start_as_current_span("extract_job_details_task_creation"):
            logger.debug(f"Creating task: Job Details Extraction for '{job_title_for_log}'")
            parsed_job_details_task = asyncio.create_task(
                job_analyzer.extract_job_details_async(job_description_text, job_title=job_title_for_log)
            )

        # Task 2: Analyze Suitability using JobAnalyzer instance
        with tracer.start_as_current_span("analyze_suitability_task_creation"):
            logger.debug(f"Creating task: Suitability Analysis for '{job_title_for_log}'")
            suitability_analysis_task = asyncio.create_task(
                job_analyzer.analyze_resume_suitability(structured_resume_data, job_dict_item) # Pass full resume_data and job_dict
            )
        
        parsed_job_details = await parsed_job_details_task
        suitability_result = await suitability_analysis_task
        
        return job_dict_item, parsed_job_details, suitability_result

    tasks = [process_single_job(job_dict) for job_dict in job_list] # This will create child spans for each job

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
                    logger.warning(
                        f"Suitability analysis for '{job_title}' returned None or failed. Creating placeholder."
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
                    logger.warning(
                        f"Job detail extraction for '{job_title}' returned None or failed."
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
                logger.error(f"Error processing a job: {e}", exc_info=True)
                # Optionally, create a placeholder AnalyzedJob for jobs that hit this broader error
            
            progress.update(overall_task, advance=1)
        
        progress.update(overall_task, description="[green]Job Processing Complete[/green]") # Keep Rich markup for progress

    logger.info(
        f"ASYNC processing complete. Results for {len(analyzed_results)} jobs."
    )
    
    # Log LLM call statistics summary
    BaseAnalyzer.log_llm_call_summary()
    
    return analyzed_results


# --- apply_filters_sort_and_save remains synchronous ---
@tracer.start_as_current_span("apply_filters_sort_and_save")
def apply_filters_sort_and_save(
    analyzed_results: List[AnalyzedJob], output_path: str, filter_args: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Applies filters, sorts, and saves the final results."""
    current_span = trace.get_current_span()
    current_span.set_attribute("num_analyzed_results", len(analyzed_results))
    current_span.set_attribute("output_path", output_path)

    jobs_to_filter = [res.original_job_data for res in analyzed_results]
    if filter_args:
        with tracer.start_as_current_span("apply_filters_call"):
            logger.info("Applying post-analysis filters...")
            filtered_original_jobs = apply_filters(jobs_to_filter, **filter_args)  # Assumes apply_filters uses standard logging
            logger.info(f"{len(filtered_original_jobs)} jobs passed filters.")
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
    logger.info("Sorting results by suitability score...")
    final_filtered_results.sort(key=lambda x: x.score, reverse=True)  # Use the score property
    final_results_list = [
        result.model_dump() for result in final_filtered_results if result
    ]  # Convert to list of dicts

    # Save to JSON
    with tracer.start_as_current_span("save_final_results_to_json"):
        if output_dir := os.path.dirname(output_path):
            os.makedirs(output_dir, exist_ok=True)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_results_list, f, indent=4, cls=DateEncoder)  # Use DateEncoder
            logger.info(
                f"Successfully saved {len(final_results_list)} analyzed jobs to {output_path}"
            )
        except Exception as e:
            logger.error(f"Error saving JSON file: {e}", exc_info=True)
            trace.get_current_span().record_exception(e)

    return final_results_list  # Return list of dictionaries


# --- Main execution block updated for async ---
@tracer.start_as_current_span("main_matcher_main_async")
async def main_async():
    """Async main function for standalone execution."""
    # Argument parsing unchanged
    parser = argparse.ArgumentParser(
        description="Analyze pre-existing job JSON against a resume."
    )
    # Add arguments as before...
    # For brevity, assuming args are parsed correctly.
    # Example: args = parser.parse_args() 
    # This part needs to be filled in if running standalone, but for library use, it's not critical here.
    # For now, we'll assume args is populated if this __main__ block is hit.
    args = parser.parse_args() # This will fail if not run as script with args.

    # Standalone logging setup - this might be redundant if main.py calls setup_logging
    # However, if run directly, this ensures logging is configured.
    # Consider if this block is truly needed or if main.py is the sole entry point.
    if not logging.getLogger().hasHandlers(): # Setup only if not already configured by main.py
        log_level_standalone = logging.DEBUG if getattr(args, 'verbose', False) else config.settings.get(
            "logging", {}
        ).get("level", "INFO").upper()
        logging.basicConfig( # Basic config for standalone, RichHandler might not be present
            level=log_level_standalone,
            format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        logger.info(f"Standalone main_matcher.py: Basic logging configured at {log_level_standalone}.")


    logger.info("Starting standalone ASYNC analysis process (main_matcher.py)...")
    # Ensure args has 'resume' and 'jobs' attributes if this is run
    if not hasattr(args, 'resume') or not hasattr(args, 'jobs') or not hasattr(args, 'output'):
        logger.error("Standalone execution requires --resume, --jobs, and --output arguments.")
        return

    structured_resume = await load_and_extract_resume_async(args.resume)
    if not structured_resume:
        logger.error("Exiting due to resume processing failure.")
        return
    
    logger.info(f"Loading jobs from file: {args.jobs}") # Assuming jobs is a path
    job_list = load_job_mandates(args.jobs) # Assumes load_job_mandates handles its own logging
    if not job_list:
        logger.error("No jobs loaded. Exiting.")
        return
    
    analyzed_results = await analyze_jobs_async(structured_resume, job_list)
    filter_args_dict = {}  # Populate from args if needed for standalone
    apply_filters_sort_and_save(analyzed_results, args.output, filter_args_dict)
    logger.info("Standalone ASYNC analysis finished successfully (main_matcher.py).")


if __name__ == "__main__":
    # This block is for when main_matcher.py is run directly.
    # Ensure logging is set up before asyncio.run
    # This might be complex if main.py is the intended entry point that sets up logging.
    # For now, let's assume if __name__ == "__main__", we need a basic logging setup.
    if not logging.getLogger().hasHandlers():
        # A very basic config if not already set up by an importer (like main.py)
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        logger.info("Basic logging configured for direct main_matcher.py execution.")

    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.warning("\nStandalone analysis interrupted by user (Ctrl+C).")
        sys.exit(130)
    except Exception as e:
        logger.critical(f"Critical error in main_matcher standalone: {e}", exc_info=True)
        sys.exit(1)
