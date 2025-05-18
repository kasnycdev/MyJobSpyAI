import logging
import time
import json
import os
# import logging # Added for standard logging - This is a redefinition
from typing import Dict, Optional, List, Any
# from colorama import Fore, Style # Likely no longer needed
# from rich.console import Console # Replaced by logger

# Get a logger for this module
logger = logging.getLogger(__name__)

# Import OpenTelemetry trace module and tracer from logging_utils
from opentelemetry import trace # Ensure trace module is always imported

try:
    from logging_utils import tracer as global_tracer_instance
    if global_tracer_instance is None: # Check if OTEL was disabled in logging_utils
        # Fallback to a NoOpTracer if OTEL is not configured
        tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
        logger.warning("OpenTelemetry not configured in logging_utils (global_tracer_instance is None), using NoOpTracer for filtering/filter.")
    else:
        tracer = global_tracer_instance # Use the instance from logging_utils
        logger.info("Using global_tracer_instance from logging_utils for filtering/filter.")
except ImportError:
    # Fallback to a NoOpTracer if logging_utils or its tracer cannot be imported
    tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
    logger.error("Could not import global_tracer_instance from logging_utils. Using NoOpTracer for filtering/filter.", exc_info=True)

# Import geopy and specific exceptions
try:
    from geopy.geocoders import Nominatim
    from geopy.distance import geodesic
    from geopy.exc import GeocoderTimedOut, GeocoderServiceError
    GEOPY_AVAILABLE = True
except ImportError:
    GEOPY_AVAILABLE = False

from .filter_utils import parse_salary, normalize_string
from config import settings

# --- Geocoding Setup ---
GEOCODER = None
if GEOPY_AVAILABLE:
    user_agent = settings.get('geocoding', {}).get('geopy_user_agent', 'MyJobSpyAI/1.0 (DEFAULT)')
    if not user_agent or 'PLEASE_UPDATE' in user_agent or 'example.com' in user_agent:
        logger.warning("GEOPY_USER_AGENT not set/default in config.yaml. Geocoding may fail.")
    GEOCODER = Nominatim(user_agent=user_agent, timeout=10)
else:
    logger.warning("Geopy library not installed. Location filtering disabled.")

_geocode_cache = {}
_geocode_fail_cache = set() # type: ignore

def get_geocode_cache_path():
    """Gets the geocode cache file path from settings with a default."""
    path = settings.get('output', {}).get('geocode_cache_file', 'cache/geocode_cache.json')
    logger.info(f"Resolved geocode_cache_file path: '{path}' (Absolute: '{os.path.abspath(path)}')")
    return path

def load_geocode_cache():
    """Loads the geocode cache from a JSON file."""
    global _geocode_cache, _geocode_fail_cache
    cache_file_path = get_geocode_cache_path()
    if os.path.exists(cache_file_path):
        try:
            with open(cache_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                _geocode_cache = data.get('cache', {})
                _geocode_fail_cache = set(data.get('fail_cache', [])) # type: ignore
            logger.info(f"Loaded geocode cache from {cache_file_path}")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading geocode cache from {cache_file_path}: {e}")

def save_geocode_cache():
    """Saves the geocode cache to a JSON file."""
    cache_file_path = get_geocode_cache_path()
    cache_dir = os.path.dirname(cache_file_path)
    
    if cache_dir: # Ensure directory exists if path includes one
        abs_cache_dir = os.path.abspath(cache_dir)
        logger.info(f"Ensuring geocode cache directory exists: '{cache_dir}' (Absolute: '{abs_cache_dir}')")
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create directory '{cache_dir}' (Absolute: '{abs_cache_dir}'): {e}", exc_info=True)
            # Potentially re-raise or handle if directory creation is critical and fails
            return # Do not proceed if directory cannot be made
    else:
        # This case means cache_file_path is a filename in the CWD, cache_dir is ''
        logger.info(f"Geocode cache file '{cache_file_path}' will be saved in the current working directory ('{os.getcwd()}').")

    try:
        with open(cache_file_path, 'w', encoding='utf-8') as f:
            json.dump({'cache': _geocode_cache, 'fail_cache': list(_geocode_fail_cache)}, f, indent=4)
        logger.info(f"Saved geocode cache to {cache_file_path}")
    except IOError as e:
        logger.error(f"Error saving geocode cache to {cache_file_path}: {e}")

# Load cache on startup
load_geocode_cache()


@tracer.start_as_current_span("get_lat_lon_country")
def get_lat_lon_country(location_str: str) -> Optional[tuple[float, float, str]]:
    """
    Geocodes a location string using Nominatim with caching and rate limiting.
    """
    current_span = trace.get_current_span()
    current_span.set_attribute("location_input", location_str)

    if not GEOPY_AVAILABLE or not GEOCODER or not location_str:
        current_span.set_attribute("geocoding_skipped_reason", "geopy_unavailable_or_no_location")
        return None

    normalized_loc = location_str.lower().strip()
    current_span.set_attribute("normalized_location", normalized_loc)
    if not normalized_loc:
        current_span.set_attribute("geocoding_skipped_reason", "empty_normalized_location")
        return None

    if normalized_loc in _geocode_cache:
        current_span.set_attribute("geocoding_result_source", "cache_success")
        return _geocode_cache[normalized_loc]

    if normalized_loc in _geocode_fail_cache:
        logger.debug(f"Skipping geocode for previously failed: '{location_str}'")
        current_span.set_attribute("geocoding_result_source", "cache_fail_skip")
        return None

    logger.info(f"Geocoding location: '{location_str}'")
    try:
        with tracer.start_as_current_span("nominatim_geocode_call") as geocode_span:
            geocode_span.set_attribute("location_to_geocode", normalized_loc)
            time.sleep(1.0)  # Rate limit
            location_data = GEOCODER.geocode(normalized_loc, addressdetails=True, language='en')
            geocode_span.set_attribute("geocode_api_returned_data", bool(location_data))

        if location_data and location_data.latitude and location_data.longitude:
            lat, lon = location_data.latitude, location_data.longitude
            address = location_data.raw.get('address', {})
            current_span.set_attribute("geocoded_lat", lat)
            current_span.set_attribute("geocoded_lon", lon)
            country_code = address.get('country_code')
            country_name = address.get('country')

            if country_code and not country_name:
                cc_map = {
                    'us': 'United States',
                    'usa': 'United States',
                    'ca': 'Canada',
                    'gb': 'United Kingdom',
                    'uk': 'United Kingdom',
                    'de': 'Germany',
                    'fr': 'France',
                    'au': 'Australia',
                }
                country_name = cc_map.get(country_code.lower())
                if country_name:
                    logger.debug(f"Inferred country name '{country_name}' from code '{country_code}'")

            if country_name:
                logger.info(f"Geocoded '{location_str}' to ({lat:.4f}, {lon:.4f}), Country: {country_name}")
                current_span.set_attribute("geocoded_country", country_name)
                result = (lat, lon, country_name)
                _geocode_cache[normalized_loc] = result
                save_geocode_cache() # Save cache after successful geocode
                current_span.set_attribute("geocoding_result_source", "api_success")
                return result
            else:
                logger.warning(
                    f"Geocoded '{location_str}' but couldn't extract country name. Address: {address}"
                )
                current_span.set_attribute("geocoding_error", "country_extraction_failed")
                _geocode_fail_cache.add(normalized_loc) # type: ignore
                save_geocode_cache() # Save cache after failed geocode
                return None
        else:
            logger.warning(f"Failed to geocode '{location_str}' - No results.")
            current_span.set_attribute("geocoding_error", "no_results_from_api")
            _geocode_fail_cache.add(normalized_loc) # type: ignore
            save_geocode_cache() # Save cache after failed geocode
            return None
    except (GeocoderTimedOut, GeocoderServiceError) as geo_err:
        logger.error(f"Geocoding error for '{location_str}': {geo_err}")
        current_span.record_exception(geo_err)
        current_span.set_attribute("geocoding_error", str(geo_err))
        _geocode_fail_cache.add(normalized_loc) # type: ignore
        save_geocode_cache() # Save cache after geocoding error
        return None
    except Exception as e:
        logger.error(f"Unexpected geocoding error for '{location_str}': {e}", exc_info=True)
        current_span.record_exception(e)
        current_span.set_attribute("geocoding_error", "unexpected_exception")
        _geocode_fail_cache.add(normalized_loc) # type: ignore
        save_geocode_cache() # Save cache after unexpected error
        return None

# --- Main Filter Function ---
@tracer.start_as_current_span("apply_filters")
def apply_filters(
    jobs: List[Dict[str, Any]],
    salary_min: Optional[int] = None,
    salary_max: Optional[int] = None,
    work_models: Optional[List[str]] = None,
    job_types: Optional[List[str]] = None,
    filter_remote_country: Optional[str] = None,
    filter_proximity_location: Optional[str] = None,
    filter_proximity_range: Optional[float] = None,
    filter_proximity_models: Optional[List[str]] = None,
    locations: Optional[List[str]] = None,  # Ignored
) -> List[Dict[str, Any]]:
    """
    Filters jobs based on standard criteria PLUS advanced location filters.
    """
    filtered_jobs = []
    normalized_work_models = [normalize_string(wm) for wm in work_models] if work_models else []
    normalized_job_types = [normalize_string(jt) for jt in job_types] if job_types else []
    normalized_remote_country = normalize_string(filter_remote_country) if filter_remote_country else None
    normalized_proximity_models = [normalize_string(pm) for pm in filter_proximity_models] if filter_proximity_models else []

    target_lat_lon = None
    if filter_proximity_location and filter_proximity_range is not None:
        logger.info(f"Attempting to geocode target proximity location: '{filter_proximity_location}'")
        if (target_geo_result := get_lat_lon_country(filter_proximity_location)):
            target_lat_lon = (target_geo_result[0], target_geo_result[1])
            logger.info(f"Target proximity location geocoded to: {target_lat_lon}")
        else:
            logger.error(
                f"Could not geocode target proximity location "
                f"'{filter_proximity_location}'. Proximity filter disabled."
            )
            filter_proximity_location = None # type: ignore
            filter_proximity_range = None # type: ignore

    logger.info("Applying filters to job list...")
    initial_count = len(jobs)
    trace.get_current_span().set_attribute("initial_job_count", initial_count)
    jobs_processed_count = 0

    for job_idx, job in enumerate(jobs):
        with tracer.start_as_current_span(f"filter_job_{job_idx}") as job_span:
            jobs_processed_count += 1
            job_title = job.get('title', 'N/A')
            job_span.set_attribute("job_title", job_title)
            passes_all_filters = True

            logger.debug(f"--- Checking Job {jobs_processed_count}/{initial_count}: '{job_title}' ---")

            # Salary
            salary_text = job.get('salary_text') # Line 271
            if isinstance(salary_text, str) and salary_text: # Line 272
                job_min_salary, job_max_salary = parse_salary(salary_text)
            else:
                job_min_salary, job_max_salary = None, None

            salary_passes_check = True
            if salary_min is not None and (
                (job_max_salary is not None and job_max_salary < salary_min) or
                (job_min_salary is not None and job_min_salary < salary_min)
            ):
                salary_passes_check = False

            if salary_passes_check and salary_max is not None and (
                (job_min_salary is not None and job_min_salary > salary_max) or
                (job_max_salary is not None and job_max_salary > salary_max)
            ):
                salary_passes_check = False

            if not salary_passes_check:
                job_span.set_attribute("filter_failed_reason", "salary")
                passes_all_filters = False
                # continue # This continue should be outside the 'with job_span' if we want the span to end.
                         # However, it's more natural for the span to cover the whole attempt for this job.
                         # So, if it fails a filter, the span will end, and the loop continues.

            # Work Model (Standard)
            if passes_all_filters and normalized_work_models: # Check passes_all_filters before continuing
                job_model = normalize_string(job.get('work_model')) or normalize_string(job.get('remote'))
                if not job_model:
                    job_loc_wm = normalize_string(job.get('location'))
                    if 'remote' in job_loc_wm:
                        job_model = 'remote'
                    elif 'hybrid' in job_loc_wm:
                        job_model = 'hybrid'
                    elif 'on-site' in job_loc_wm or 'office' in job_loc_wm:
                        job_model = 'on-site'
                    else:
                        job_model = None
                if not job_model or job_model not in normalized_work_models:
                    job_span.set_attribute("filter_failed_reason", "work_model")
                    passes_all_filters = False

            # Job Type
            if passes_all_filters and normalized_job_types:
                job_type_text = normalize_string(job.get('employment_type'))
                if all(jt_filter not in job_type_text for jt_filter in normalized_job_types if job_type_text):
                    job_span.set_attribute("filter_failed_reason", "job_type")
                    passes_all_filters = False

            # Advanced Location Filters
            job_location_str = job.get('location', '')
            job_geo_result = None # Reset for each job inside the loop
            
            # Filter 1: Remote Job in Specific Country
            if passes_all_filters and normalized_remote_country:
                job_model_rc = normalize_string(job.get('work_model')) or normalize_string(job.get('remote'))
                loc_text_rc = normalize_string(job_location_str)
                is_remote = job_model_rc == 'remote' or 'remote' in loc_text_rc

                if is_remote:
                    if not job_geo_result: # Geocode only if needed
                        job_geo_result = get_lat_lon_country(job_location_str)

                    if job_geo_result:
                        job_country = job_geo_result[2]
                        if not job_country or normalize_string(job_country) != normalized_remote_country:
                            passes_all_filters = False
                            job_span.set_attribute("filter_failed_reason", "remote_country_mismatch")
                    else: 
                        logger.warning(
                            f"Geocoding failed for '{job_location_str}'. Cannot confirm country for remote filter. Job will likely fail this filter."
                        )
                        passes_all_filters = False 
                        job_span.set_attribute("filter_failed_reason", "remote_country_geocode_fail")
                else: 
                    passes_all_filters = False
                    job_span.set_attribute("filter_failed_reason", "not_remote_for_country_filter")
            
            # Filter 2: Proximity
            if passes_all_filters and filter_proximity_location and target_lat_lon:
                job_model_prox = normalize_string(job.get('work_model')) or normalize_string(job.get('remote'))
                loc_text_prox = normalize_string(job_location_str)

                if not job_model_prox: # Infer model if not explicit
                    if 'remote' in loc_text_prox: job_model_prox = 'remote'
                    elif 'hybrid' in loc_text_prox: job_model_prox = 'hybrid'
                    elif 'on-site' in loc_text_prox or 'office' in loc_text_prox: job_model_prox = 'on-site'
                    else: job_model_prox = None

                if not job_model_prox or job_model_prox not in normalized_proximity_models:
                    passes_all_filters = False
                    job_span.set_attribute("filter_failed_reason", "proximity_work_model_mismatch")
                else: # Work model matches for proximity, now check distance
                    if not job_geo_result: # Geocode only if needed
                        job_geo_result = get_lat_lon_country(job_location_str)

                    if not job_geo_result:
                        passes_all_filters = False
                        job_span.set_attribute("filter_failed_reason", "proximity_geocode_fail")
                    else:
                        job_lat_lon = (job_geo_result[0], job_geo_result[1])
                        try:
                            distance_miles = geodesic(target_lat_lon, job_lat_lon).miles
                            job_span.set_attribute("proximity_calculated_distance_miles", distance_miles)
                            if distance_miles > filter_proximity_range: # type: ignore
                                passes_all_filters = False
                                job_span.set_attribute("filter_failed_reason", "proximity_too_far")
                        except Exception as dist_err:
                            logger.warning(
                                f"Could not calculate distance for '{job_title}': {dist_err}"
                            )
                            passes_all_filters = False
                            job_span.set_attribute("filter_failed_reason", "proximity_distance_calc_error")
                            job_span.record_exception(dist_err)

            job_span.set_attribute("passes_all_filters", passes_all_filters)
            if passes_all_filters:
                filtered_jobs.append(job)
            # End of 'with job_span'

    final_count = len(filtered_jobs)
    logger.info(f"Filtering complete. {final_count} out of {initial_count} jobs passed active filters.")
    # Save cache at the end of filtering
    save_geocode_cache()
    return filtered_jobs
