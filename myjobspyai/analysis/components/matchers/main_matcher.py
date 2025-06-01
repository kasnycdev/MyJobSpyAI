import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar

from opentelemetry import trace

from ...providers.base import BaseProvider
from ...factory import _get_provider
from ..analyzers.resume_analyzer import ResumeAnalyzer
from myjobspyai.config import settings as config
from ...exceptions import ConfigurationError, LLMError, ProviderError
from ...components.models.models import AnalyzedJob, ResumeData

# Set up logging
logger = logging.getLogger(__name__)

# Lazy import JobAnalyzer to avoid circular imports
JobAnalyzer = None

# Import the resume parser
try:
    from myjobspyai.parsers.resume_parser import parse_resume
except ImportError as e:
    logger.error(f"Failed to import resume parser: {e}")
    raise

# Import apply_filters if it exists in your codebase
try:
    from myjobspyai.analysis.filters import apply_filters
except ImportError:
    def apply_filters(jobs: List[Dict[str, Any]], **filters) -> List[Dict[str, Any]]:
        """Filter jobs based on the provided criteria.
        
        Args:
            jobs: List of job dictionaries to filter
            **filters: Filter criteria including:
                - min_salary: Minimum salary threshold
                - location: Location to filter by (case-insensitive partial match)
                - remote: Whether to include only remote jobs (True/False/None)
                - job_type: Job type to filter by (e.g., 'full-time', 'part-time')
                
        Returns:
            List of filtered job dictionaries
        """
        if not jobs:
            return []
            
        filtered_jobs = jobs
        
        # Apply salary filter if specified
        if 'min_salary' in filters and filters['min_salary'] is not None:
            min_salary = float(filters['min_salary'])
            filtered_jobs = [
                job for job in filtered_jobs 
                if job.get('salary') and float(job['salary']) >= min_salary
            ]
        
        # Apply location filter if specified
        if 'location' in filters and filters['location']:
            location = filters['location'].lower()
            filtered_jobs = [
                job for job in filtered_jobs
                if location in (job.get('location', '').lower() or '')
            ]
        
        # Apply remote filter if specified
        if 'remote' in filters and filters['remote'] is not None:
            is_remote = bool(filters['remote'])
            filtered_jobs = [
                job for job in filtered_jobs
                if job.get('is_remote', False) == is_remote
            ]
        
        # Apply job type filter if specified
        if 'job_type' in filters and filters['job_type']:
            job_type = filters['job_type'].lower()
            filtered_jobs = [
                job for job in filtered_jobs
                if job_type in (job.get('job_type', '').lower() or '')
            ]
        
        logger.info(f"Filtered from {len(jobs)} to {len(filtered_jobs)} jobs")
        return filtered_jobs

# Import load_job_mandates if it exists in your codebase
try:
    from myjobspyai.analysis.job_mandates import load_job_mandates
except ImportError:
    def load_job_mandates(*args, **kwargs):
        return {}  # Return an empty dictionary instead of raising an error

# Custom JSON encoder for handling datetime objects
class DateEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        return super().default(obj)

# Type variable for generic provider types
T = TypeVar('T', bound=BaseProvider)

logger = logging.getLogger(__name__)

# Configure OpenTelemetry tracer
try:
    # Try to get the global tracer
    tracer = trace.get_tracer(__name__)
    logger.info("Successfully configured OpenTelemetry tracer for main_matcher")
except Exception as e:
    # Fallback to a NoOpTracer if there's an error
    tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
    logger.warning(
        f"Failed to configure OpenTelemetry tracer: {e}. "
        "Using NoOpTracer for main_matcher."
    )


logger = logging.getLogger(__name__)


# --- ASYNC Resume Loading/Extraction ---
@tracer.start_as_current_span("load_and_extract_resume_async")
async def load_and_extract_resume_async(
    resume_path: str, 
    force_reparse: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> Optional[ResumeData]:
    """ASYNC: Loads resume, parses text, and extracts structured data with caching.
    
    Args:
        resume_path: Path to the resume file
        force_reparse: If True, ignore cache and re-parse the resume
        config: Optional configuration dictionary for the resume analyzer
    """
    # Only use tracing if it's available
    current_span = None
    try:
        current_span = trace.get_current_span()
        if current_span and hasattr(current_span, 'set_attribute'):
            current_span.set_attribute("resume_path", str(resume_path))
            current_span.set_attribute("force_reparse", str(force_reparse))
    except Exception as e:
        logger.debug(f"Could not set trace attributes: {e}")
    
    logger.info(f"Processing resume file: {resume_path}")

    # Ensure the resume file exists
    if not os.path.exists(resume_path):
        logger.error(f"Resume file not found: {resume_path}")
        return None

    # Initialize config if not provided
    config_dict = config.dict() if hasattr(config, 'dict') else (config or {})
    
    # Get cache directory from config or use default
    cache_dir = Path(config_dict.get('cache_dir', 'cache/resume_cache/'))
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate cache key based on file modification time
    cache_key = None
    cache_path = None
    
    if not force_reparse:
        try:
            mtime = os.path.getmtime(resume_path)
            cache_key = f"{Path(resume_path).stem}_{mtime}.json"
            cache_path = cache_dir / cache_key
            
            # Try to load from cache
            if cache_path.exists():
                logger.info(f"Loading structured resume data from cache: {cache_path}")
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                    if isinstance(cached_data, dict):
                        structured_resume_data = ResumeData(**cached_data)
                        logger.info("Successfully loaded structured data from cache.")
                        return structured_resume_data
                except Exception as e:
                    logger.warning(f"Error loading from cache {cache_path}: {e}")
                    if current_span:
                        current_span.record_exception(e)
        except Exception as e:
            logger.warning(f"Could not generate cache key: {e}")

    # If we get here, we need to parse the resume
    logger.info("Parsing and extracting resume data...")
    
    # Parse the resume text
    try:
        with tracer.start_as_current_span("parse_resume_text"):
            resume_text = parse_resume(resume_path)
            if not resume_text or not isinstance(resume_text, str):
                logger.error("Failed to extract text from resume or empty content")
                return None
    except Exception as e:
        logger.error(f"Failed to parse resume text from {resume_path}: {e}", exc_info=True)
        if current_span:
            current_span.record_exception(e)
        return None

    # Initialize the resume analyzer with the provided config
    try:
        logger.debug(f"Raw config received: {config}")
        if not config:
            raise ConfigurationError("No configuration provided")
            
        if 'llm' not in config:
            available_keys = list(config.keys()) if isinstance(config, dict) else []
            logger.error(f"LLM configuration not found. Available keys in config: {available_keys}")
            raise ConfigurationError("No LLM configuration found in config. Make sure the config contains an 'llm' key.")
        
        llm_config = config['llm']
        
        # Get provider name and config
        provider_name = str(llm_config.get('provider', '')).lower()
        if not provider_name:
            raise ConfigurationError("No provider specified in LLM configuration")
            
        provider_config = llm_config.get(provider_name, {})
        
        # Create provider instance using the factory
        from myjobspyai.analysis.factory import get_factory
        
        # Get or create the factory with the current config
        factory = get_factory({
            'providers': {
                provider_name: provider_config
            }
        })
        
        # Create the provider instance
        provider = await factory.create_provider(provider_name)
        
        # Initialize ResumeAnalyzer with provider, model, and config
        model = llm_config.get('model')
        if not model:
            raise ConfigurationError("No model specified in LLM configuration")
            
        resume_analyzer = ResumeAnalyzer(
            provider=provider,
            model=model,
            config=config
        )
        logger.info(f"Analyzing resume with model: {resume_analyzer.model}")
    except Exception as e:
        logger.error(f"Failed to initialize ResumeAnalyzer: {e}")
        if current_span:
            current_span.record_exception(e)
        return None

    # Extract structured data using ResumeAnalyzer
    try:
        with tracer.start_as_current_span("extract_resume_data_llm"):
            
            # Extract structured data using the analyze method
            structured_resume_data = await resume_analyzer.analyze_resume(resume_text)
            
            if not structured_resume_data:
                logger.error("Failed to extract structured data from resume")
                return None
                
            logger.info("Successfully extracted structured data from resume.")
            
            # Save to cache if we have a valid cache path
            if cache_path:
                try:
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(
                            structured_resume_data.model_dump(), 
                            f, 
                            indent=2,
                            cls=DateEncoder
                        )
                    logger.info(f"Saved structured data to cache: {cache_path}")
                    
                    # Clean up old cache files for this resume
                    for old_file in cache_dir.glob(f"{Path(resume_path).stem}_*.json"):
                        if old_file != cache_path:
                            try:
                                old_file.unlink()
                                logger.debug(f"Removed old cache file: {old_file}")
                            except Exception as e:
                                logger.warning(f"Error removing old cache file {old_file}: {e}")
                except Exception as e:
                    logger.warning(f"Error saving to cache {cache_path}: {e}")
                    if current_span:
                        current_span.record_exception(e)
            
            return structured_resume_data
            
    except Exception as e:
        logger.error(f"Error during resume analysis: {e}", exc_info=True)
        if current_span:
            current_span.record_exception(e)
        return None


# --- ASYNC Job Analysis ---
@tracer.start_as_current_span("analyze_jobs_async")
async def analyze_jobs_async(
    structured_resume_data: ResumeData,
    job_list: List[Dict[str, Any]],
    llm_provider: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None
) -> List[AnalyzedJob]:
    """ASYNC: Analyzes a list of jobs concurrently against the resume data.
    
    Args:
        structured_resume_data: The structured resume data
        job_list: List of job dictionaries to analyze
        llm_provider: Optional pre-initialized LLM provider to use
        config: Optional configuration dictionary for the analyzer
        
    Returns:
        List of AnalyzedJob objects with analysis results
    """
    current_span = trace.get_current_span()
    current_span.set_attribute("num_jobs_to_analyze", len(job_list))
    total_jobs = len(job_list)
    logger.info(f"Starting ASYNC analysis of {total_jobs} jobs...")
    
    # Lazy import to avoid circular imports
    global JobAnalyzer
    if JobAnalyzer is None:
        from myjobspyai.analysis.analyzer import JobAnalyzer as JA
        JobAnalyzer = JA
    
    # Initialize the job analyzer with the provided or default LLM provider and config
    try:
        # Log the configuration being used
        logger.debug(f"Initializing JobAnalyzer with config: {config}")
        
        # If no LLM provider was provided, use the config to create one
        if llm_provider is None:
            from myjobspyai.analysis.factory import get_factory
            try:
                factory = get_factory()
                # Get provider and model from config, raise error if not found
                if not config or not config.get('llm'):
                    raise ValueError("No LLM configuration found. Please provide 'llm' configuration.")
                    
                provider_name = config['llm'].get('provider')
                model_name = config['llm'].get('model')
                
                if not provider_name:
                    raise ValueError("No provider specified in LLM configuration. Please set 'llm.provider'.")
                if not model_name:
                    raise ValueError("No model specified in LLM configuration. Please set 'llm.model'.")
                
                logger.debug(f"Creating LLM provider: {provider_name} with model: {model_name}")
                llm_provider = await factory.get_provider(provider_name, model=model_name)
                logger.info(f"Initialized LLM provider: {provider_name} with model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize LLM provider: {str(e)}", exc_info=True)
                raise RuntimeError(f"Failed to initialize LLM provider: {str(e)}") from e
        
        # Initialize the job analyzer with the provider and config
        job_analyzer = JobAnalyzer(provider=llm_provider, config=config or {})
        logger.info(f"Initialized JobAnalyzer with provider: {llm_provider.__class__.__name__}")
        
    except Exception as e:
        logger.error(f"Failed to initialize job analyzer: {str(e)}", exc_info=True)
        # Return empty results if we can't initialize the analyzer
        return []

    @tracer.start_as_current_span("process_single_job_analysis")
    async def process_single_job(
        job_dict_item: Dict[str, Any], 
        resume_data: Dict[str, Any], 
        analyzer: Any
    ) -> Optional[Dict[str, Any]]:
        """Process a single job asynchronously.
        
        Args:
            job_dict_item: The job to process
            resume_data: The structured resume data
            analyzer: The JobAnalyzer instance to use
            
        Returns:
            The analysis result or None if processing failed
        """
        job_id = job_dict_item.get('job_id', 'unknown')
        job_title = job_dict_item.get('title', 'Untitled Position')
        company = job_dict_item.get('company', 'Unknown Company')
        
        logger.info(f"Processing job: {job_title} at {company} (ID: {job_id})")
        
        try:
            # Log basic job info for debugging
            logger.debug(f"Job data keys: {list(job_dict_item.keys())}")
            
            # Extract job details from the description
            logger.debug("Extracting job details...")
            try:
                job_details = await analyzer.extract_job_details_async(
                    job_description=job_dict_item.get('description', ''),
                    job_title=job_title
                )
                logger.debug(f"Extracted job details with keys: {list(job_details.keys()) if isinstance(job_details, dict) else 'Not a dict'}")
            except Exception as e:
                logger.error(f"Failed to extract job details for {job_title} at {company}: {str(e)}", exc_info=True)
                job_details = {
                    'title': job_title,
                    'company': company,
                    'error': f"Failed to extract job details: {str(e)}",
                    'description': job_dict_item.get('description', '')[:500] + '...'  # Include first 500 chars of description
                }
            
            # Analyze resume suitability for this job
            logger.debug("Analyzing resume suitability...")
            try:
                analysis_result = await analyzer.analyze_resume_suitability(
                    resume_data=resume_data,
                    job_dict=job_details
                )
                logger.debug(f"Analysis result keys: {list(analysis_result.keys()) if isinstance(analysis_result, dict) else 'Not a dict'}")
            except Exception as e:
                logger.error(f"Failed to analyze resume suitability for {job_title} at {company}: {str(e)}", exc_info=True)
                analysis_result = {
                    'match_score': 0.0,
                    'matching_skills': [],
                    'missing_skills': [f"Analysis failed: {str(e)[:200]}"],
                    'analysis': f"Error during analysis: {str(e)[:500]}",
                    'error': str(e),
                    'error_type': type(e).__name__
                }
            
            # Ensure we have a dictionary result
            if not isinstance(analysis_result, dict):
                logger.warning(f"Unexpected analysis result type: {type(analysis_result)}")
                analysis_result = {
                    'match_score': 0.0,
                    'matching_skills': [],
                    'missing_skills': ['Unexpected analysis result format'],
                    'analysis': 'Analysis completed but with unexpected format',
                    'raw_result': str(analysis_result)[:1000]  # Include first 1000 chars of raw result
                }
            
            # Add job metadata to the result
            analysis_result.update({
                'job_id': job_id,
                'title': job_title,
                'company': company,
                'processed_at': datetime.utcnow().isoformat()
            })
            
            logger.info(f"Completed analysis for {job_title} at {company} - Score: {analysis_result.get('match_score', 0.0):.1f}")
            return analysis_result
            
        except Exception as e:
            error_msg = f"Unexpected error processing job {job_id} ({job_title} at {company}): {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Return a minimal error response
            return {
                'job_id': job_id,
                'title': job_title,
                'company': company,
                'match_score': 0.0,
                'matching_skills': [],
                'missing_skills': [f"Processing error: {str(e)[:200]}"],
                'analysis': f"Error during job processing: {str(e)[:500]}",
                'error': str(e),
                'error_type': type(e).__name__,
                'processed_at': datetime.utcnow().isoformat()
            }

    try:
        # Import settings if not provided
        if config is None:
            from myjobspyai import settings
            config = getattr(settings, 'llm', {})
        
        # Initialize the job analyzer with the provided LLM provider or config
        if llm_provider is not None:
            logger.info(f"Using provided LLM provider: {llm_provider.__class__.__name__}")
            job_analyzer = JobAnalyzer(llm_provider=llm_provider, config=config)
        else:
            logger.info("Initializing JobAnalyzer with config")
            job_analyzer = JobAnalyzer(config=config)
        
        # Log the provider and model being used
        provider_info = job_analyzer.provider.__class__.__name__ if hasattr(job_analyzer, 'provider') else 'default'
        model_info = getattr(job_analyzer, 'model', 'default')
        logger.info(f"Initialized JobAnalyzer with provider: {provider_info}, model: {model_info}")
        
        # Ensure the analyzer is properly initialized
        if hasattr(job_analyzer, 'initialize') and callable(job_analyzer.initialize):
            await job_analyzer.initialize()
            logger.info("Successfully initialized JobAnalyzer")
        
        # Process each job asynchronously
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(process_single_job(job, structured_resume_data, job_analyzer))
                for job in job_list
            ]
        
        # Get results from completed tasks
        results = [task.result() for task in tasks if not task.cancelled() and task.result() is not None]
        
        # Log LLM call statistics summary if available
        try:
            # Create a temporary JobAnalyzer instance to access the class method
            job_analyzer.log_llm_call_summary()
        except Exception as e:
            logger.warning(f"Failed to log LLM call summary: {e}")
        
        # Log completion and return results
        logger.info(f"Completed analysis of {len(results)} jobs")
        return results
        
    except Exception as e:
        logger.error(f"Error in analyze_jobs_async: {str(e)}", exc_info=True)
        # Return an empty list on error to prevent NoneType issues
        return []


# --- apply_filters_sort_and_save remains synchronous ---
@tracer.start_as_current_span("apply_filters_sort_and_save")
def apply_filters_sort_and_save(
    analyzed_results: List[AnalyzedJob], output_path: str, filter_args: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Apply filters, sort results, and save to file.
    
    Args:
        analyzed_results: List of analyzed job results
        output_path: Path to save the results to
        filter_args: Dictionary of filter arguments
        
    Returns:
        List of filtered and sorted job results as dictionaries
    """
    """Applies filters, sorts, and saves the final results."""
    current_span = trace.get_current_span()
    current_span.set_attribute("num_analyzed_results", len(analyzed_results))
    current_span.set_attribute("output_path", output_path)

    # Handle both dictionary and object access for analyzed_results
    jobs_to_filter = []
    for res in analyzed_results:
        if hasattr(res, 'original_job_data'):
            jobs_to_filter.append(res.original_job_data)
        elif isinstance(res, dict) and 'original_job_data' in res:
            jobs_to_filter.append(res['original_job_data'])
        else:
            # If no original_job_data is found, use the result itself
            jobs_to_filter.append(res)
    
    if filter_args:
        with tracer.start_as_current_span("apply_filters_call"):
            logger.info("Applying post-analysis filters...")
            filtered_original_jobs = apply_filters(jobs_to_filter, **filter_args)
            logger.info(f"{len(filtered_original_jobs)} jobs passed filters.")
        
        # Create a set of unique job identifiers for filtering
        filtered_keys = set()
        for job in filtered_original_jobs:
            # Handle both dictionary and object access for job data
            if hasattr(job, 'get'):
                # It's a dictionary
                job_url = job.get('url', job.get('job_url'))
                title = job.get('title')
                company = job.get('company')
                location = job.get('location')
            else:
                # It's an object, try to access attributes
                job_url = getattr(job, 'url', getattr(job, 'job_url', None))
                title = getattr(job, 'title', None)
                company = getattr(job, 'company', None)
                location = getattr(job, 'location', None)
            
            filtered_keys.add((job_url, title, company, location))
        final_filtered_results = []
        for res in analyzed_results:
            # Get the job data, handling both dictionary and object access
            if hasattr(res, 'original_job_data'):
                job_data = res.original_job_data
            elif isinstance(res, dict) and 'original_job_data' in res:
                job_data = res['original_job_data']
            else:
                job_data = res
            
            # Get job identifiers, handling both dictionary and object access
            if hasattr(job_data, 'get'):
                # It's a dictionary
                job_url = job_data.get('url', job_data.get('job_url'))
                title = job_data.get('title')
                company = job_data.get('company')
                location = job_data.get('location')
            else:
                # It's an object, try to access attributes
                job_url = getattr(job_data, 'url', getattr(job_data, 'job_url', None))
                title = getattr(job_data, 'title', None)
                company = getattr(job_data, 'company', None)
                location = getattr(job_data, 'location', None)
            
            if (job_url, title, company, location) in filtered_keys:
                final_filtered_results.append(res)
    else:
        final_filtered_results = analyzed_results
    logger.info("Sorting results by suitability score...")
    
    def get_sort_key(item):
        # Handle both dictionary and object access for sorting
        if hasattr(item, 'overall_suitability_score'):
            # Object access
            return (
                getattr(item, 'overall_suitability_score', 0),
                getattr(item, 'skills_match_score', 0),
            )
        elif isinstance(item, dict):
            # Dictionary access
            return (
                item.get('overall_suitability_score', 0),
                item.get('skills_match_score', 0),
            )
        else:
            # Fallback to default values
            return (0, 0)
    
    # Sort the results using the sort key
    final_filtered_results.sort(key=get_sort_key, reverse=True)
    
    # Convert to list of dicts, handling both objects and dictionaries
    final_results_list = []
    for result in final_filtered_results:
        if result is None:
            continue
            
        if hasattr(result, 'model_dump'):
            # Object with model_dump method (e.g., Pydantic model)
            final_results_list.append(result.model_dump())
        elif hasattr(result, 'dict'):
            # Object with dict method (e.g., Pydantic model)
            final_results_list.append(result.dict())
        elif hasattr(result, '__dict__'):
            # Regular object with __dict__
            final_results_list.append(vars(result))
        elif isinstance(result, dict):
            # Already a dictionary
            final_results_list.append(result)
        else:
            logger.warning(f"Could not convert result to dict: {result}")

    # Save to JSON if output_path is provided and valid
    if not output_path or not isinstance(output_path, str):
        logger.warning("No valid output path provided, skipping file save")
        return final_results_list
        
    try:
        with tracer.start_as_current_span("save_final_results_to_json"):
            # Ensure the output directory exists
            output_dir = os.path.dirname(os.path.abspath(output_path))
            if output_dir:  # Only try to create directory if there is a path component
                os.makedirs(output_dir, exist_ok=True)
            
            # Write the results to the output file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(final_results_list, f, indent=2, cls=DateEncoder)
            logger.info(f"Successfully saved {len(final_results_list)} analyzed jobs to {output_path}")
    except (IOError, OSError, TypeError, ValueError) as e:
        error_msg = f"Failed to save results to {output_path}: {e}"
        logger.error(error_msg, exc_info=True)
        current_span = trace.get_current_span()
        if current_span:
            current_span.record_exception(e)
            current_span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
        # Don't raise the exception, just log it and continue

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

    try:
        structured_resume = await load_and_extract_resume_async(args.resume)
        if not structured_resume:
            raise ConfigurationError("Failed to process resume data")
    except ProviderError as e:
        logger.error(f"Provider error during resume processing: {e}", exc_info=True)
        raise
    except LLMError as e:
        logger.error(f"LLM error during resume processing: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Unexpected error during resume processing: {e}", exc_info=True)
        raise ConfigurationError("Resume processing failed", original_exception=e) from e
    
    try:
        job_list = load_job_mandates(args.jobs)
        if not job_list:
            raise ConfigurationError("No jobs loaded from input file")
    except ProviderError as e:
        logger.error(f"Provider error loading jobs: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Error loading jobs: {e}", exc_info=True)
        raise ConfigurationError("Failed to load jobs", original_exception=e) from e
    
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
