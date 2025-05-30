import logging
from typing import Dict, List, Any, Optional

# Third-party imports
from opentelemetry import trace

# Local application imports
from .filter_utils import parse_salary

# Get a logger for this module
logger = logging.getLogger(__name__)

# Initialize tracer
try:
    from myjobspyai.utils.logging_utils import tracer as global_tracer_instance
    if global_tracer_instance is None:  # Check if OTEL was disabled in logging_utils
        # Fallback to a NoOpTracer if OTEL is not configured
        tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
        logger.warning("OpenTelemetry not configured in logging_utils (global_tracer_instance is None), using NoOpTracer for filtering/filter.")
    else:
        tracer = global_tracer_instance  # Use the instance from logging_utils
        logger.info("Using global_tracer_instance from myjobspyai.utils.logging_utils for filtering/filter.")
except ImportError:
    # Fallback to a NoOpTracer if logging_utils or its tracer cannot be imported
    tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
    logger.error(
        "Could not import global_tracer_instance from myjobspyai.utils.logging_utils. "
        "Using NoOpTracer for filtering/filter.", 
        exc_info=True
    )

def apply_filters(
    jobs: List[Dict[str, Any]],
    salary_min: Optional[int] = None,
    salary_max: Optional[int] = None,
    work_models: Optional[List[str]] = None,
    job_titles: Optional[List[str]] = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Apply filters to a list of job listings.
    
    Args:
        jobs: List of job dictionaries to filter
        salary_min: Minimum salary threshold
        salary_max: Maximum salary threshold
        work_models: List of acceptable work models (e.g., ['remote', 'hybrid', 'onsite'])
        job_titles: List of job titles to include (case-insensitive)
        **kwargs: Additional filter parameters (currently unused)
        
    Returns:
        List of filtered job dictionaries
    """
    filtered_jobs = []
    
    for job in jobs:
        # Initialize job_span for tracing
        with tracer.start_as_current_span("filter_job") as job_span:
            passes_filters = True
            
            # Apply salary filter if specified
            if salary_min is not None or salary_max is not None:
                job_salary = job.get('salary')
                if job_salary:
                    salary_value = parse_salary(job_salary)
                    if salary_value:
                        if salary_min is not None and salary_value < salary_min:
                            job_span.set_attribute("filter_failed_reason", "salary_below_min")
                            passes_filters = False
                        if salary_max is not None and salary_value > salary_max:
                            job_span.set_attribute("filter_failed_reason", "salary_above_max")
                            passes_filters = False
            
            # Apply work model filter if specified
            if work_models and passes_filters:
                job_work_model = job.get('work_model', '').lower()
                if job_work_model not in [wm.lower() for wm in work_models]:
                    job_span.set_attribute("filter_failed_reason", "work_model_mismatch")
                    passes_filters = False
            
            # Apply job title filter if specified
            if job_titles and passes_filters:
                job_title = job.get('title', '').lower()
                if not any(title.lower() in job_title for title in job_titles):
                    job_span.set_attribute("filter_failed_reason", "job_title_mismatch")
                    passes_filters = False
            
            if passes_filters:
                filtered_jobs.append(job)
    
    return filtered_jobs
