"""Factory module for creating scraper instances.

This module provides a factory function to create instances of different scraper
implementations based on the provided configuration.
"""

import importlib
import logging
from typing import Any, Dict, List, Optional, TypeVar

from . import BaseJobScraper

logger = logging.getLogger(__name__)

# Type variable for the scraper class

# Mapping of built-in scraper names to their module and class names
BUILTIN_SCRAPERS = {
    "linkedin": ("myjobspyai.scrapers.linkedin", "LinkedInScraper"),
    "indeed": ("myjobspyai.scrapers.indeed", "IndeedScraper"),
    "jobspy": ("myjobspyai.scrapers.jobspy_scraper", "JobSpyScraper"),
}


def create_scraper(
    scraper_type: str,
    config: Optional[Dict[str, Any]] = None,
    custom_scrapers: Optional[Dict[str, str]] = None,
) -> BaseJobScraper:
    """
    Create a scraper instance based on the provided type and configuration.

    Args:
        scraper_type: Type of scraper to create (e.g., 'linkedin', 'indeed')
        config: Configuration dictionary for the scraper
        custom_scrapers: Optional mapping of custom scraper types to their import paths

    Returns:
        An instance of the specified scraper

    Raises:
        ValueError: If the specified scraper type is not found
        ImportError: If there's an error importing the scraper module or class
    """
    config = config or {}
    custom_scrapers = custom_scrapers or {}

    # Check if it's a built-in scraper
    if scraper_type in BUILTIN_SCRAPERS:
        module_name, class_name = BUILTIN_SCRAPERS[scraper_type]
    elif scraper_type in (custom_scrapers or {}):
        # Custom scraper specified in the config
        module_class_path = custom_scrapers[scraper_type]
        if "." not in module_class_path:
            raise ValueError(f"Invalid custom scraper path: {module_class_path}")
        module_name, class_name = module_class_path.rsplit(".", 1)
    else:
        raise ValueError(f"Unknown scraper type: {scraper_type}")

    try:
        # Import the module
        module = importlib.import_module(module_name)
        # Get the class
        scraper_class = getattr(module, class_name)
        # Create an instance with the provided config
        return scraper_class(config)
    except ImportError as e:
        logger.error(f"Error importing scraper module {module_name}: {e}")
        raise ImportError(f"Could not import scraper module: {module_name}") from e
    except AttributeError as e:
        logger.error(
            f"Error finding scraper class {class_name} in module {module_name}: {e}"
        )
        raise ImportError(f"Could not find scraper class: {class_name}") from e


def get_available_scrapers() -> List[str]:
    """
    Get a list of available built-in scraper types.

    Returns:
        List of available scraper type names
    """
    return list(BUILTIN_SCRAPERS.keys())


def register_scraper(
    scraper_type: str,
    module_path: str,
    class_name: str,
    custom_scrapers: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Register a custom scraper type.

    Args:
        scraper_type: Unique identifier for the scraper type
        module_path: Full import path to the module containing the scraper class
        class_name: Name of the scraper class
        custom_scrapers: Optional dictionary to update with the new scraper

    Returns:
        Updated dictionary of custom scrapers
    """
    custom_scrapers = custom_scrapers or {}
    full_path = f"{module_path}.{class_name}"
    custom_scrapers[scraper_type] = full_path
    logger.info(f"Registered custom scraper: {scraper_type} -> {full_path}")
    return custom_scrapers
