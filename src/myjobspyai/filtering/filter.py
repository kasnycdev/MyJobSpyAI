import logging
from typing import Dict, List, Any, Optional

# Get a logger for this module
logger = logging.getLogger(__name__)

# Import OpenTelemetry trace module and tracer from myjobspyai.utils.logging_utils
from opentelemetry import trace

try:
    from myjobspyai.utils.logging_utils import tracer as global_tracer_instance

    if global_tracer_instance is None:  # Check if OTEL was disabled in logging_utils
        # Fallback to a NoOpTracer if OTEL is not configured
        tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
        logger.warning(
            "OpenTelemetry not configured in myjobspyai.utils.logging_utils (global_tracer_instance is None), using NoOpTracer for filtering/filter."
        )
    else:
        tracer = global_tracer_instance  # Use the instance from logging_utils
        logger.info(
            "Using global_tracer_instance from myjobspyai.utils.logging_utils for filtering/filter."
        )
except ImportError:
    # Fallback to a NoOpTracer if logging_utils or its tracer cannot be imported
    tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
    logger.error(
        "Could not import global_tracer_instance from logging_utils. Using NoOpTracer for filtering/filter.",
        exc_info=True,
    )

from .filter_utils import parse_salary, normalize_string


# --- Main Filter Function ---
@tracer.start_as_current_span("apply_filters")
def apply_filters(
    jobs: List[Dict[str, Any]],
    salary_min: Optional[int] = None,
    salary_max: Optional[int] = None,
    work_models: Optional[List[str]] = None,
    job_types: Optional[List[str]] = None,
    locations: Optional[List[str]] = None,  # Kept for backward compatibility, not used
) -> List[Dict[str, Any]]:
    """
    Filters jobs based on standard criteria.
    """
    filtered_jobs = []
    normalized_work_models = (
        [normalize_string(wm) for wm in work_models] if work_models else []
    )
    normalized_job_types = (
        [normalize_string(jt) for jt in job_types] if job_types else []
    )

    logger.info("Applying filters to job list...")
    initial_count = len(jobs)
    trace.get_current_span().set_attribute("initial_job_count", initial_count)
    jobs_processed_count = 0

    for job_idx, job in enumerate(jobs):
        with tracer.start_as_current_span(f"filter_job_{job_idx}") as job_span:
            jobs_processed_count += 1
            job_title = job.get("title", "N/A")
            job_span.set_attribute("job_title", job_title)
            passes_all_filters = True

            logger.debug(
                f"--- Checking Job {jobs_processed_count}/{initial_count}: '{job_title}' ---"
            )

            # Salary
            salary_text = job.get("salary_text")
            if isinstance(salary_text, str) and salary_text:
                job_min_salary, job_max_salary = parse_salary(salary_text)
            else:
                job_min_salary, job_max_salary = None, None

            salary_passes_check = True
            if salary_min is not None and (
                (job_max_salary is not None and job_max_salary < salary_min)
                or (job_min_salary is not None and job_min_salary < salary_min)
            ):
                salary_passes_check = False

            if (
                salary_passes_check
                and salary_max is not None
                and (
                    (job_min_salary is not None and job_min_salary > salary_max)
                    or (job_max_salary is not None and job_max_salary > salary_max)
                )
            ):
                salary_passes_check = False

            if not salary_passes_check:
                job_span.set_attribute("filter_failed_reason", "salary")
                passes_all_filters = False

            # Work Model (Standard)
            if passes_all_filters and normalized_work_models:
                job_model = normalize_string(job.get("work_model")) or normalize_string(
                    job.get("remote")
                )
                if not job_model:
                    job_loc_wm = normalize_string(job.get("location", ""))
                    if "remote" in job_loc_wm:
                        job_model = "remote"
                    elif "hybrid" in job_loc_wm:
                        job_model = "hybrid"
                    elif "on-site" in job_loc_wm or "office" in job_loc_wm:
                        job_model = "on-site"
                    else:
                        job_model = None
                if not job_model or job_model not in normalized_work_models:
                    job_span.set_attribute("filter_failed_reason", "work_model")
                    passes_all_filters = False

            # Job Type
            if passes_all_filters and normalized_job_types:
                job_type_text = normalize_string(job.get("employment_type", ""))
                if all(
                    jt_filter not in job_type_text
                    for jt_filter in normalized_job_types
                    if job_type_text
                ):
                    job_span.set_attribute("filter_failed_reason", "job_type")
                    passes_all_filters = False

            job_span.set_attribute("passes_all_filters", passes_all_filters)
            if passes_all_filters:
                filtered_jobs.append(job)

    final_count = len(filtered_jobs)
    logger.info(
        f"Filtering complete. {final_count} out of {initial_count} jobs passed active filters."
    )
    return filtered_jobs
