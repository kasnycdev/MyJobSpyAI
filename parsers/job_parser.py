import json
import os
import html
import re
import logging # Added for standard logging
# from colorama import Fore, Style # Likely no longer needed
# from rich.console import Console # Replaced by logger

# Get a logger for this module
logger = logging.getLogger(__name__)

# Import tracer from logging_utils
try:
    from logging_utils import tracer as global_tracer
    if global_tracer is None: # Check if OTEL was disabled in logging_utils
        from opentelemetry import trace
        tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
        logger.warning("OpenTelemetry not configured in logging_utils, using NoOpTracer for job_parser.")
    else:
        tracer = global_tracer
except ImportError:
    from opentelemetry import trace
    tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
    logger.error("Could not import global_tracer from logging_utils. Using NoOpTracer for job_parser.", exc_info=True)


# Module-level cache for job mandates and prompt fields
_loaded_job_mandates = None
_extracted_prompt_fields = None

@tracer.start_as_current_span("load_job_mandates")
def load_job_mandates(json_file_path: str) -> list[dict]:
    """
    Loads job mandates from a JSON file, using an in-memory cache.

    Args:
        json_file_path: Path to the JSON file containing a list of job objects.

    Returns:
        A list of job dictionaries, or an empty list on error.
    """
    current_span = trace.get_current_span()
    current_span.set_attribute("file.path", json_file_path)
    global _loaded_job_mandates
    if _loaded_job_mandates is not None:
        logger.info("Returning job mandates from cache.")
        current_span.set_attribute("result_source", "cache")
        current_span.set_attribute("num_jobs_loaded", len(_loaded_job_mandates))
        return _loaded_job_mandates

    if not os.path.exists(json_file_path):
        logger.error(f"Job mandates file not found: {json_file_path}")
        current_span.set_attribute("error", "file_not_found")
        current_span.set_status(trace.Status(trace.StatusCode.ERROR, "File not found"))
        return []

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            num_jobs = len(data)
            logger.info(
                f"Successfully loaded {num_jobs} job mandates from {html.escape(json_file_path)}"
            )
            current_span.set_attribute("num_jobs_loaded", num_jobs)
            # Basic validation: ensure items are dictionaries
            if all(isinstance(job, dict) for job in data):
                _loaded_job_mandates = data # Cache the loaded data
                current_span.set_attribute("result_source", "file_load")
                return data
            else:
                logger.error(
                    "JSON file does not contain a list of job objects (dictionaries)."
                )
                current_span.set_attribute("error", "invalid_json_structure_not_list_of_dicts")
                current_span.set_status(trace.Status(trace.StatusCode.ERROR, "Invalid JSON structure"))
                return []
        else:
            logger.error("JSON file root is not a list.")
            current_span.set_attribute("error", "invalid_json_structure_not_list")
            current_span.set_status(trace.Status(trace.StatusCode.ERROR, "JSON root not a list"))
            return []
    except json.JSONDecodeError as e:
        logger.error(
            f"Error decoding JSON file {html.escape(json_file_path)}: {str(e)}"
        )
        current_span.record_exception(e)
        current_span.set_attribute("error", "json_decode_error")
        current_span.set_status(trace.Status(trace.StatusCode.ERROR, "JSON decode error"))
        return []
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while loading {html.escape(json_file_path)}: {str(e)}",
            exc_info=True
        )
        current_span.record_exception(e)
        current_span.set_attribute("error", "unexpected_load_error")
        current_span.set_status(trace.Status(trace.StatusCode.ERROR, "Unexpected load error"))
        return []

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

@tracer.start_as_current_span("parse_job_data_static") # Renamed for clarity vs dynamic
def parse_job_data(job_data: dict) -> dict:
    """
    Parses raw job data to match the schema defined in `job_extraction.prompt`.

    Args:
        job_data: A dictionary containing raw job data.

    Returns:
        A dictionary conforming to the schema defined in `job_extraction.prompt`.
    """
    # This function doesn't directly read files, so no change needed here for I/O reduction
    # Adding basic span attributes for context
    current_span = trace.get_current_span()
    current_span.set_attribute("job_data.title_present", "title" in job_data)
    current_span.set_attribute("job_data.keys_present", str(list(job_data.keys())))

    return {
        "job_title_extracted": job_data.get("title"),
        "key_responsibilities": job_data.get("responsibilities", []),
        "required_skills": job_data.get("required_skills", []),
        "preferred_skills": job_data.get("preferred_skills", []),
        "required_experience_years": job_data.get("min_experience"),
        "preferred_experience_years": job_data.get("preferred_experience"),
        "required_education": job_data.get("education"),
        "preferred_education": job_data.get("preferred_education"),
        "salary_range_extracted": job_data.get("salary"),
        "work_model_extracted": job_data.get("work_model"),
        "company_culture_hints": job_data.get("culture_hints", []),
        "tools_technologies": job_data.get("tools", []),
        "job_type": job_data.get("job_type"),
        "industry": job_data.get("industry"),
        "company_name": job_data.get("company"),
        "company_size": job_data.get("company_size"),
        "location": job_data.get("location"),
        "job_description": job_data.get("description"),
        "required_certifications": job_data.get("required_certifications", []),
        "preferred_certifications": job_data.get("preferred_certifications", []),
        "security_clearance": job_data.get("security_clearance"),
        "travel_requirements": job_data.get("travel_requirements"),
        "job_id": job_data.get("id"),
        "source": job_data.get("site"),
        "posting_date": job_data.get("date_posted"),
    }

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
