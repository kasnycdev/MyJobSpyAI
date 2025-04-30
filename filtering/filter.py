import logging
import time
from typing import Dict, Optional, List, Any

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

# Use root logger
log = logging.getLogger(__name__)

# --- Geocoding Setup ---
GEOCODER = None
if GEOPY_AVAILABLE:
    user_agent = settings.get('geocoding', {}).get('geopy_user_agent', 'MyJobSpyAI/1.0 (DEFAULT)')
    if not user_agent or 'PLEASE_UPDATE' in user_agent or 'example.com' in user_agent:
        log.warning("[yellow]GEOPY_USER_AGENT not set/default in config.yaml. Geocoding may fail.[/yellow]")
    GEOCODER = Nominatim(user_agent=user_agent, timeout=10)
else:
    log.warning("[yellow]Geopy library not installed. Location filtering disabled.[/yellow]")

_geocode_cache = {}
_geocode_fail_cache = set()

def get_lat_lon_country(location_str: str) -> Optional[tuple[float, float, str]]:
    """
    Geocodes a location string using Nominatim with caching and rate limiting.
    """
    if not GEOPY_AVAILABLE or not GEOCODER or not location_str:
        return None

    normalized_loc = location_str.lower().strip()
    if not normalized_loc:
        return None

    if normalized_loc in _geocode_cache:
        return _geocode_cache[normalized_loc]

    if normalized_loc in _geocode_fail_cache:
        log.debug(f"Skipping geocode for previously failed: '{location_str}'")
        return None

    log.debug(f"Geocoding location: '{location_str}'")
    try:
        time.sleep(1.0)  # Rate limit
        location_data = GEOCODER.geocode(normalized_loc, addressdetails=True, language='en')
        if location_data and location_data.latitude and location_data.longitude:
            lat, lon = location_data.latitude, location_data.longitude
            address = location_data.raw.get('address', {})
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
                    log.debug(f"Inferred country name '{country_name}' from code '{country_code}'")

            if country_name:
                log.debug(f"Geocoded '{location_str}' to ({lat:.4f}, {lon:.4f}), Country: {country_name}")
                result = (lat, lon, country_name)
                _geocode_cache[normalized_loc] = result
                return result
            else:
                log.warning(
                    f"[yellow]Geocoded '{location_str}' but couldn't extract country name. Address: {address}[/yellow]"
                )
                _geocode_fail_cache.add(normalized_loc)
                return None
        else:
            log.warning(f"[yellow]Failed to geocode '{location_str}' - No results.[/yellow]")
            _geocode_fail_cache.add(normalized_loc)
            return None
    except (GeocoderTimedOut, GeocoderServiceError) as geo_err:
        log.error(f"[red]Geocoding error for '{location_str}': {geo_err}[/red]")
        _geocode_fail_cache.add(normalized_loc)
        return None
    except Exception as e:
        log.error(f"[red]Unexpected geocoding error for '{location_str}': {e}[/red]", exc_info=True)
        _geocode_fail_cache.add(normalized_loc)
        return None

# --- Main Filter Function ---
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
        log.info(f"Attempting to geocode target proximity location: '{filter_proximity_location}'")
        if (target_geo_result := get_lat_lon_country(filter_proximity_location)):
            target_lat_lon = (target_geo_result[0], target_geo_result[1])
            log.info(f"Target proximity location geocoded to: {target_lat_lon}")
        else:
            log.error(
                f"[red]Could not geocode target proximity location "
                f"'{filter_proximity_location}'. Proximity filter disabled.[/red]"
            )
            filter_proximity_location = None
            filter_proximity_range = None

    log.info("Applying filters to job list...")
    initial_count = len(jobs)
    jobs_processed_count = 0

    for job in jobs:
        jobs_processed_count += 1
        job_title = job.get('title', 'N/A')
        passes_all_filters = True

        log.debug(f"--- Checking Job {jobs_processed_count}/{initial_count}: '{job_title}' ---")

        # Salary
        salary_text = job.get('salary_text')
        if isinstance(salary_text, str) and salary_text:
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
            passes_all_filters = False
            continue

        # Work Model (Standard)
        if normalized_work_models:
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
                passes_all_filters = False
                continue

        # Job Type
        if passes_all_filters and normalized_job_types:
            job_type_text = normalize_string(job.get('employment_type'))
            if all(jt_filter not in job_type_text for jt_filter in normalized_job_types if job_type_text):
                passes_all_filters = False
                continue

        # Advanced Location Filters
        job_location_str = job.get('location', '')
        job_geo_result = None
        # Filter 1: Remote Job in Specific Country
        if passes_all_filters and normalized_remote_country:
            job_model_rc = normalize_string(job.get('work_model')) or normalize_string(job.get('remote'))
            loc_text_rc = normalize_string(job_location_str)
            is_remote = job_model_rc == 'remote' or 'remote' in loc_text_rc

            if is_remote:
                if not job_geo_result:
                    job_geo_result = get_lat_lon_country(job_location_str)

                if job_geo_result:
                    job_country = job_geo_result[2]
                    if not job_country or normalize_string(job_country) != normalized_remote_country:
                        passes_all_filters = False
                else:
                    log.warning(
                        f"[yellow]Geocoding failed for '{job_location_str}'. Assuming it matches the remote country filter.[/yellow]"
                    )
            else:
                passes_all_filters = False

            if not passes_all_filters:
                continue

        # Filter 2: Proximity
        if passes_all_filters and filter_proximity_location and target_lat_lon:
            job_model_prox = normalize_string(job.get('work_model')) or normalize_string(job.get('remote'))
            loc_text_prox = normalize_string(job_location_str)

            if not job_model_prox:
                if 'remote' in loc_text_prox:
                    job_model_prox = 'remote'
                elif 'hybrid' in loc_text_prox:
                    job_model_prox = 'hybrid'
                elif 'on-site' in loc_text_prox or 'office' in loc_text_prox:
                    job_model_prox = 'on-site'
                else:
                    job_model_prox = None

            if not job_model_prox or job_model_prox not in normalized_proximity_models:
                passes_all_filters = False
            else:
                if not job_geo_result:
                    job_geo_result = get_lat_lon_country(job_location_str)

                if not job_geo_result:
                    passes_all_filters = False
                else:
                    job_lat_lon = (job_geo_result[0], job_geo_result[1])
                    try:
                        distance_miles = geodesic(target_lat_lon, job_lat_lon).miles
                        if distance_miles > filter_proximity_range:
                            passes_all_filters = False
                    except Exception as dist_err:
                        log.warning(
                            f"[yellow]Could not calculate distance for '{job_title}': {dist_err}[/yellow]"
                        )
                        passes_all_filters = False

        if passes_all_filters:
            filtered_jobs.append(job)

    final_count = len(filtered_jobs)
    log.info(f"Filtering complete. {final_count} out of {initial_count} jobs passed active filters.")
    return filtered_jobs