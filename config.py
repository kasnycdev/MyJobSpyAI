# Removed the config.yaml data in config.py
# config.py
import yaml
import os
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# --- Constants defining paths ---
PROJECT_ROOT = Path(__file__).parent.resolve()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" # Default if not in YAML

# --- Default Settings (Fallback if YAML is missing fields) ---
DEFAULT_SETTINGS = {
    "output_dir": str(DEFAULT_OUTPUT_DIR),
    "scraped_jobs_filename": "scraped_jobs.parquet",
    "analysis_filename": "analyzed_jobs.json",
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "llama3:instruct",
        "request_timeout": 450,
        "max_retries": 2,
        "retry_delay": 5,
    },
    "analysis": {
        "prompts_dir": "analysis/prompts", # Relative to project root
        "resume_prompt_file": "resume_extraction.prompt",
        "suitability_prompt_file": "suitability_analysis.prompt",
        "job_extraction_prompt_file": "job_extraction.prompt", # <-- Add this
        "max_prompt_chars": 24000,
    },
    "scraping": {
        "default_sites": ["linkedin", "indeed"],
        "default_results_limit": 25,
        "default_hours_old": 72,
        "default_country_indeed": "usa",
        "linkedin_fetch_description": True,
        "linkedin_company_ids": []
    },
    "geocoding": {
        "geopy_user_agent": "MyJobSpyAI/1.0 (PLEASE_UPDATE_EMAIL@example.com)" # <-- User should update this in config.yaml
    },
    "logging": {
        "level": "INFO",
        # --- Use minimal format - RichHandler adds timestamp, level etc. ---
        "format": "%(message)s",
        "date_format": "[%X]", # RichHandler uses this
    }
}

def _deep_merge_dicts(base, update):
    """Recursively merges update dict into base dict."""
    for key, value in update.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_merge_dicts(base[key], value)
        else:
            base[key] = value
    return base

def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Loads configuration from YAML file, merging with defaults."""
    settings = DEFAULT_SETTINGS.copy() # Start with defaults

    if config_path.exists():
        # Use basic logging until config is loaded if logging config itself is in YAML
        temp_logger = logging.getLogger(f"{__name__}.config_loader")
        temp_logger.info(f"Loading configuration from {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)

            if user_config and isinstance(user_config, dict):
                settings = _deep_merge_dicts(settings, user_config)
                temp_logger.info("Successfully loaded and merged configuration.")
            elif user_config is None:
                 temp_logger.warning(f"Config file {config_path} empty. Using defaults.")
            else:
                 temp_logger.error(f"Config file {config_path} invalid format. Using defaults.")
        except yaml.YAMLError as e:
            temp_logger.error(f"Error parsing YAML {config_path}: {e}. Using defaults.")
        except Exception as e:
            temp_logger.error(f"Error loading config {config_path}: {e}. Using defaults.", exc_info=True)
    else:
        logging.warning(f"Config file not found at {config_path}. Using defaults.") # Use standard logging here

    # --- Make paths absolute relative to PROJECT_ROOT ---
    settings["output_dir"] = str(PROJECT_ROOT / settings.get("output_dir", "output"))
    if "analysis" in settings and "prompts_dir" in settings["analysis"]:
        # Ensure prompts_dir is treated as relative to project root if not absolute
        prompts_dir_path = Path(settings["analysis"]["prompts_dir"])
        if not prompts_dir_path.is_absolute():
             settings["analysis"]["prompts_dir"] = str(PROJECT_ROOT / prompts_dir_path)
        else:
             settings["analysis"]["prompts_dir"] = str(prompts_dir_path) # Keep absolute path

    # --- Ensure output dir exists ---
    try:
        os.makedirs(settings["output_dir"], exist_ok=True)
    except OSError as e:
        logging.error(f"Could not create output directory '{settings['output_dir']}': {e}") # Use standard logging

    # --- Add derived absolute paths ---
    settings["scraped_jobs_path"] = os.path.join(settings["output_dir"], settings.get("scraped_jobs_filename", "scraped_jobs.json"))
    output_settings = settings.get('output', {})
    analysis_filename = output_settings.get('analysis_output_file', 'analyzed_jobs.json') # Get from 'output' section
    settings["analysis_output_path"] = os.path.join(settings["output_dir"], analysis_filename)
    debug_filename = output_settings.get('debug_output_file', 'debug_info.json')
    settings["debug_output_path"] = os.path.join(settings["output_dir"], debug_filename)
    if "analysis" in settings:
        prompts_dir = settings["analysis"].get("prompts_dir", str(PROJECT_ROOT / "analysis/prompts"))
        settings["analysis"]["resume_prompt_path"] = os.path.join(prompts_dir, settings["analysis"].get("resume_prompt_file", "resume_extraction.prompt"))
        settings["analysis"]["suitability_prompt_path"] = os.path.join(prompts_dir, settings["analysis"].get("suitability_prompt_file", "suitability_analysis.prompt"))
        settings["analysis"]["job_extraction_prompt_path"] = os.path.join(prompts_dir, settings["analysis"].get("job_extraction_prompt_file", "job_extraction.prompt")) # <-- Add this

    logging.debug(f"Final configuration settings: {settings}") # Use standard logging
    return settings

# --- Load settings ONCE ---
settings = load_config()

# --- Optional: Helper function ---
def get_setting(key_path: str, default: Any = None) -> Any:
    keys = key_path.split('.'); value = settings
    try:
        for key in keys: value = value[key]
        return value
    except (KeyError, TypeError): return default