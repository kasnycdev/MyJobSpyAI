import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Union

# Get a logger for this module
logger = logging.getLogger(__name__)

# Get a logger for this module
logger = logging.getLogger(__name__)

# Import tracer from logging_utils
try:
    from myjobspyai.utils.logging_utils import tracer as global_tracer
    if global_tracer is None: # Check if OTEL was disabled in logging_utils
        from opentelemetry import trace
        tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
        logger.warning("OpenTelemetry not configured in logging_utils, using NoOpTracer for job_parser.")
    else:
        tracer = global_tracer
except ImportError:
    from opentelemetry import trace
    tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
    logger.error("Could not import global_tracer from myjobspyai.utils.logging_utils. Using NoOpTracer for job_parser.", exc_info=True)


# Module-level cache for job mandates and prompt fields
_loaded_job_mandates = None
_extracted_prompt_fields = None

# Default job schema for validation
DEFAULT_JOB_SCHEMA = {
    "job_title_extracted": str,
    "key_responsibilities": list,
    "required_skills": list,
    "preferred_skills": list,
    "required_experience_years": (int, float, type(None)),
    "preferred_experience_years": (int, float, type(None)),
    "required_education": (str, type(None)),
    "preferred_education": (str, type(None)),
    "salary_range_extracted": (str, type(None)),
    "work_model_extracted": (str, type(None)),
    "company_culture_hints": list,
    "tools_technologies": list,
    "job_type": (str, type(None)),
    "industry": (str, type(None)),
    "company_name": (str, type(None)),
    "company_size": (str, type(None)),
    "location": (str, type(None)),
    "job_description": (str, type(None)),
    "required_certifications": list,
    "preferred_certifications": list,
    "security_clearance": (str, type(None)),
    "travel_requirements": (str, type(None)),
    "job_id": (str, type(None)),
    "source": (str, type(None)),
    "posting_date": (str, type(None))
}

@tracer.start_as_current_span("load_job_mandates")
def load_job_mandates(json_file_path: Union[str, Path], use_cache: bool = True) -> List[Dict[str, Any]]:
    """
    Loads job mandates from a JSON file, with caching and validation.

    Args:
        json_file_path: Path to the JSON file containing a list of job objects.
        use_cache: Whether to use the in-memory cache. Defaults to True.

    Returns:
        A list of validated job dictionaries, or an empty list on error.
    """
    current_span = trace.get_current_span()
    json_file_path = Path(json_file_path)
    
    # Set trace attributes
    current_span.set_attributes({
        "file.path": str(json_file_path),
        "use_cache": use_cache
    })
    
    # Check cache if enabled
    global _loaded_job_mandates
    if use_cache and _loaded_job_mandates is not None:
        logger.info("Returning job mandates from cache.")
        current_span.set_attribute("result_source", "cache")
        current_span.set_attribute("num_jobs_loaded", len(_loaded_job_mandates))
        return _loaded_job_mandates

    # Validate file exists and is accessible
    if not json_file_path.exists():
        error_msg = f"Job mandates file not found: {json_file_path}"
        logger.error(error_msg)
        current_span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
        return []

    try:
        # Read and parse JSON
        with json_file_path.open('r', encoding='utf-8') as f:
            data = json.load(f)

        # Validate top-level structure
        if not isinstance(data, list):
            error_msg = "JSON file root is not a list."
            logger.error(error_msg)
            current_span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
            return []

        # Validate and clean each job entry
        validated_jobs = []
        for i, job in enumerate(data, 1):
            if not isinstance(job, dict):
                logger.warning(f"Skipping non-dictionary item at index {i-1}")
                continue
                
            # Clean and validate the job data
            cleaned_job = _clean_job_data(job)
            if _validate_job_schema(cleaned_job):
                validated_jobs.append(cleaned_job)
            else:
                logger.warning(f"Skipping invalid job at index {i-1}")

        # Update cache if we loaded valid data
        if validated_jobs:
            _loaded_job_mandates = validated_jobs
            current_span.set_attributes({
                "result_source": "file_load",
                "num_jobs_loaded": len(validated_jobs),
                "num_jobs_skipped": len(data) - len(validated_jobs)
            })
            logger.info(f"Successfully loaded {len(validated_jobs)} valid jobs from {json_file_path}")
            return validated_jobs
        else:
            error_msg = "No valid job entries found in the file."
            logger.error(error_msg)
            current_span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
            return []

    except json.JSONDecodeError as e:
        error_msg = f"Error decoding JSON file {json_file_path}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        current_span.record_exception(e)
        current_span.set_status(trace.Status(trace.StatusCode.ERROR, "JSON decode error"))
        return []
    except Exception as e:
        error_msg = f"Unexpected error loading {json_file_path}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        current_span.record_exception(e)
        current_span.set_status(trace.Status(trace.StatusCode.ERROR, "Unexpected error"))
        return []


def _clean_job_data(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and standardize job data."""
    cleaned = {}
    
    # Basic cleaning for all fields
    for key, value in job_data.items():
        if isinstance(value, str):
            # Clean string values
            cleaned_value = ' '.join(value.split())
            cleaned[key] = cleaned_value.strip()
        elif isinstance(value, (list, tuple)):
            # Clean list items if they're strings
            cleaned_list = []
            for item in value:
                if isinstance(item, str):
                    cleaned_item = ' '.join(item.split()).strip()
                    if cleaned_item:  # Skip empty strings
                        cleaned_list.append(cleaned_item)
                elif item is not None:  # Keep non-None, non-string items
                    cleaned_list.append(item)
            cleaned[key] = cleaned_list
        elif value is not None:  # Keep other non-None values as-is
            cleaned[key] = value
    
    return cleaned


def _validate_job_schema(job_data: Dict[str, Any]) -> bool:
    """Validate job data against the expected schema."""
    if not isinstance(job_data, dict):
        return False
    
    # Check required fields
    required_fields = ["title", "description"]
    for field in required_fields:
        if field not in job_data or not job_data[field]:
            return False
    
    # Type checking for all fields in the schema
    for field, expected_type in DEFAULT_JOB_SCHEMA.items():
        if field in job_data and job_data[field] is not None:
            if not isinstance(job_data[field], expected_type):
                return False
    
    return True

@tracer.start_as_current_span("extract_fields_from_prompt")
def extract_fields_from_prompt(prompt_path: str) -> list[str]:
    """
    Extracts field names from the JSON schema in the job_extraction.prompt file,
    using an in-memory cache.

    Args:
        prompt_path: Path to the job_extraction.prompt file.

    Returns:
        A list of field names defined in the JSON schema.
    """
    current_span = trace.get_current_span()
    current_span.set_attribute("prompt.path", prompt_path)
    global _extracted_prompt_fields
    if _extracted_prompt_fields is not None:
        logger.info("Returning extracted prompt fields from cache.")
        current_span.set_attribute("result_source", "cache")
        current_span.set_attribute("num_fields_extracted", len(_extracted_prompt_fields))
        return _extracted_prompt_fields

    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_content = f.read()
        current_span.set_attribute("prompt.content_length", len(prompt_content))

        # Extract JSON schema block
        schema_match = re.search(r'```json\n({.*?})\n```', prompt_content, re.DOTALL)
        if not schema_match:
            logger.error("Failed to extract JSON schema from prompt.")
            return []

        schema_content = schema_match.group(1)
        # Extract field names from the JSON schema
        field_names = re.findall(r'"(.*?)":', schema_content)
        _extracted_prompt_fields = field_names # Cache the extracted fields
        current_span.set_attribute("num_fields_extracted", len(field_names))
        current_span.set_attribute("result_source", "file_parse")
        return field_names
    except Exception as e:
        logger.error(f"Error extracting fields from prompt: {e}", exc_info=True)
        current_span.record_exception(e)
        current_span.set_status(trace.Status(trace.StatusCode.ERROR, "Prompt field extraction failed"))
        return []

@tracer.start_as_current_span("parse_job_data")
def parse_job_data(job_data: Dict[str, Any], validate: bool = True) -> Dict[str, Any]:
    """
    Parses and validates raw job data to match the expected schema.

    Args:
        job_data: A dictionary containing raw job data.
        validate: Whether to validate the output against the schema. Defaults to True.

    Returns:
        A dictionary conforming to the expected job data schema.
        
    Raises:
        ValueError: If validation fails and validate=True
    """
    current_span = trace.get_current_span()
    
    # Set basic span attributes
    current_span.set_attributes({
        "job_data.title_present": "title" in job_data,
        "job_data.keys_present": str(list(job_data.keys())),
        "validate_output": validate
    })
    
    # Clean the input data
    cleaned_data = _clean_job_data(job_data)
    
    # Map fields to the expected schema
    parsed_data = {
        "job_title_extracted": cleaned_data.get("title"),
        "key_responsibilities": cleaned_data.get("responsibilities") or [],
        "required_skills": cleaned_data.get("required_skills") or [],
        "preferred_skills": cleaned_data.get("preferred_skills") or [],
        "required_experience_years": cleaned_data.get("min_experience"),
        "preferred_experience_years": cleaned_data.get("preferred_experience"),
        "required_education": cleaned_data.get("education"),
        "preferred_education": cleaned_data.get("preferred_education"),
        "salary_range_extracted": cleaned_data.get("salary"),
        "work_model_extracted": cleaned_data.get("work_model"),
        "company_culture_hints": cleaned_data.get("culture_hints") or [],
        "tools_technologies": cleaned_data.get("tools") or [],
        "job_type": cleaned_data.get("job_type"),
        "industry": cleaned_data.get("industry"),
        "company_name": cleaned_data.get("company"),
        "company_size": cleaned_data.get("company_size"),
        "location": cleaned_data.get("location"),
        "job_description": cleaned_data.get("description"),
        "required_certifications": cleaned_data.get("required_certifications") or [],
        "preferred_certifications": cleaned_data.get("preferred_certifications") or [],
        "security_clearance": cleaned_data.get("security_clearance"),
        "travel_requirements": cleaned_data.get("travel_requirements"),
        "job_id": cleaned_data.get("id"),
        "source": cleaned_data.get("site"),
        "posting_date": cleaned_data.get("date_posted"),
    }
    
    # Validate against schema if requested
    if validate and not _validate_job_schema(parsed_data):
        error_msg = "Parsed job data does not match the expected schema"
        logger.error(error_msg)
        current_span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
        if validate == 'strict':
            raise ValueError(error_msg)
    
    return parsed_data

@tracer.start_as_current_span("parse_job_data_dynamic")
def parse_job_data_dynamic(job_data: dict, fields: list[str]) -> dict:
    """
    Dynamically parses raw job data to match the fields extracted from the prompt schema.

    Args:
        job_data: A dictionary containing raw job data.
        fields: A list of field names to extract from the job data.

    Returns:
        A dictionary containing the extracted fields.
    """
    parsed_data = {}
    for field in fields:
        parsed_data[field] = job_data.get(field)
    return parsed_data

# Example usage
if __name__ == "__main__":
    # Example of how to use the caching functions
    # Setup basic logging if run standalone for example to work
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    prompt_path = os.path.join(os.path.dirname(__file__), '../analysis/prompts/job_extraction.prompt')
    fields = extract_fields_from_prompt(prompt_path) # First call reads the file and caches
    logger.info(f"Extracted fields from prompt: {fields}")

    fields_cached = extract_fields_from_prompt(prompt_path) # Second call should use the cache
    logger.info(f"Extracted fields from prompt (cached): {fields_cached}")


    # Example job data
    job_data = {
        "title": "Software Engineer",
        "description": "Develop and maintain software applications.",
        "location": "Remote",
        "salary": "$100k - $120k",
        "company": "TechCorp",
    }

    parsed_job = parse_job_data_dynamic(job_data, fields)
    logger.info(f"Parsed job data: {parsed_job}")
