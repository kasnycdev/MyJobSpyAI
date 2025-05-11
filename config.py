# Removed the config.yaml data in config.py
# config.py
import yaml
import os
import logging
from pathlib import Path
from typing import Any
from opentelemetry.sdk.trace.sampling import ALWAYS_ON, TraceIdRatioBasedSampler # Updated for OTEL config

log = logging.getLogger(__name__)

# --- Constants defining paths ---
PROJECT_ROOT = Path(__file__).parent.resolve()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"  # Default if not in YAML

# --- Default Settings (Fallback if YAML is missing fields) ---
DEFAULT_SETTINGS = {
    "output_dir": str(DEFAULT_OUTPUT_DIR),
    "scraped_jobs_filename": "scraped_jobs.json",
    "analysis_filename": "analyzed_jobs.json",

    # --- LLM Provider Selection ---
    "llm_provider": "openai", # Default provider ("openai", "ollama", "gemini")

    # --- Provider Specific Settings ---
    "openai": {
        "base_url": "http://localhost:1234/v1", # Default for LM Studio
        "model": "loaded-model-name-in-lm-studio", # Placeholder - User should update
        "api_key": "lm-studio", # Placeholder
        "request_timeout": 600,
        "max_retries": 2,
        "retry_delay": 5,
    },
    "ollama": {
        "base_url": "http://localhost:11434",
        "model": "llama3:instruct", # Example default
        "request_timeout": 600,
        "max_retries": 2,
        "retry_delay": 5,
    },
    "gemini": {
        "model": "gemini-1.5-flash-latest",
        "api_key": "YOUR_GEMINI_API_KEY", # Placeholder - User MUST update
        "request_timeout": 600, # Note: May be handled by client library
        "max_retries": 2,       # Note: May be handled by client library
        "retry_delay": 5,       # Note: May be handled by client library
    },

    # --- Analysis Settings ---
    "analysis": {
        "prompts_dir": "analysis/prompts",  # Relative to project root
        "resume_prompt_file": "resume_extraction.prompt",
        "suitability_prompt_file": "suitability_analysis.prompt",
        "job_extraction_prompt_file": "job_extraction.prompt",  # <-- Add this
        "max_prompt_chars": 24000,
    },
    "scraping": {
        "default_sites": ["linkedin"],
        "default_results_limit": 25,
        "default_hours_old": 72,
        "default_country_indeed": "usa",
        "linkedin_fetch_description": True,
        "linkedin_company_ids": []
    },
    "geocoding": {
        "geopy_user_agent": (
            "MyJobSpyAI/1.0 (kasnycdev@gmail.com)"  # User should update this
        )
    },
    "logging": {
        "level": "INFO",  # Overall minimum level for the root logger if not overridden by handlers
        "format": "%(message)s", # Basic format for RichHandler, file handlers will use more detail
        "date_format": "[%X]",  # RichHandler uses this for its timestamp
        "log_dir": "logs",      # Directory for log files, relative to project root
        "info_log_file": "info.log",
        "debug_log_file": "debug.log",
        "error_log_file": "error.log",
        "log_to_console": True,
        "console_log_level": "INFO", # Level for console output via RichHandler
        "file_log_level_debug": "DEBUG", # Level for the debug log file
        "file_log_level_info": "INFO",   # Level for the info log file
        "file_log_level_error": "ERROR",  # Level for the error log file
        "model_output_log_file": "model_output.log" # New setting for model responses
    },
    # --- OpenTelemetry Settings ---
    "opentelemetry": {
        "OTEL_ENABLED": True, # Added enable/disable flag
        "OTEL_SERVICE_NAME": "MyJobSpyAI",
        "OTEL_EXPORTER_OTLP_ENDPOINT": os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317"),
        "OTEL_TRACES_SAMPLER": "always_on", # Default to string
        "OTEL_TRACES_SAMPLER_CONFIG": { # Config for samplers like traceidratio
             "ratio": 0.5 # Default ratio if traceidratio is chosen
        },
        "OTEL_RESOURCE_ATTRIBUTES": {
            "environment": os.environ.get("ENVIRONMENT", "development"),
            "version": "0.1.0",
        },
    }
}

def _deep_merge_dicts(base, update):
    """Recursively merges update dict into base dict."""
    for key, value in update.items():
        if isinstance(value, dict) and key in base \
                and isinstance(base[key], dict):
            _deep_merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    """Loads configuration from YAML file, merging with defaults."""
    settings = DEFAULT_SETTINGS.copy()  # Start with defaults

    if config_path.exists():
        # Use basic logging until config is loaded if logging config itself is in YAML
        temp_logger = logging.getLogger(f"{__name__}.config_loader")
        temp_logger.info(f"Loading configuration from {config_path}")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = yaml.safe_load(f)

            if user_config and isinstance(user_config, dict):
                settings = _deep_merge_dicts(settings, user_config)
                temp_logger.info(
                    "Successfully loaded and merged configuration."
                )
            elif user_config is None:
                temp_logger.warning(
                    f"Config file {config_path} empty. Using defaults."
                )
            else:
                temp_logger.error(
                    f"Config file {config_path} invalid format. "
                    "Using defaults."
                )
        except yaml.YAMLError as e:
            temp_logger.error(
                f"Error parsing YAML {config_path}: {e}. Using defaults."
            )
        except Exception as e:
            temp_logger.error(
                f"Error loading config {config_path}: {e}. Using defaults.",
                exc_info=True,
            )
    else:
        logging.warning(
            f"Config file not found at {config_path}. Using defaults."
        )  # Use standard logging here

    # --- Make paths absolute relative to PROJECT_ROOT ---
    settings["output_dir"] = str(
        PROJECT_ROOT / settings.get("output_dir", "output")
    )
    if "analysis" in settings and "prompts_dir" in settings["analysis"]:
        # Ensure prompts_dir is treated as relative to project root if not absolute
        prompts_dir_path = Path(settings["analysis"]["prompts_dir"])
        if not prompts_dir_path.is_absolute():
            settings["analysis"]["prompts_dir"] = str(PROJECT_ROOT / prompts_dir_path)
        else:
            settings["analysis"]["prompts_dir"] = str(prompts_dir_path)  # Keep absolute path

    # --- Ensure output dir exists ---
    try:
        os.makedirs(settings["output_dir"], exist_ok=True)
    except OSError as e:
        logging.error(
            f"Could not create output directory "
            f"'{settings['output_dir']}': {e}"
        )
    
    # --- Process OpenTelemetry Settings with Environment Variable Overrides ---
    otel_settings = settings.get("opentelemetry", {})

    # Handle OTEL_ENABLED flag, respecting OTEL_SDK_DISABLED environment variable
    # OTEL_SDK_DISABLED=true means OTEL is disabled.
    if os.environ.get("OTEL_SDK_DISABLED", "false").lower() == "true":
        otel_settings["OTEL_ENABLED"] = False
        logging.info("OpenTelemetry disabled via OTEL_SDK_DISABLED environment variable.")
    else:
        # If not disabled by env var, use the value from settings (YAML or default)
        # Ensure it's a boolean
        enabled_from_settings = otel_settings.get("OTEL_ENABLED", True)
        if isinstance(enabled_from_settings, str):
            otel_settings["OTEL_ENABLED"] = enabled_from_settings.lower() == "true"
        else:
            otel_settings["OTEL_ENABLED"] = bool(enabled_from_settings)
        if not otel_settings["OTEL_ENABLED"]:
            logging.info("OpenTelemetry disabled via config (OTEL_ENABLED=false).")
        else:
            logging.info("OpenTelemetry is ENABLED.")
            
    # Only process other OTEL settings if OTEL_ENABLED is true
    if otel_settings.get("OTEL_ENABLED", False):
        # OTEL_SERVICE_NAME
        otel_settings["OTEL_SERVICE_NAME"] = os.environ.get(
            "OTEL_SERVICE_NAME", 
            otel_settings.get("OTEL_SERVICE_NAME", "MyJobSpyAI-Default")
        )

        # OTEL_EXPORTER_OTLP_ENDPOINT
        otel_settings["OTEL_EXPORTER_OTLP_ENDPOINT"] = os.environ.get(
            "OTEL_EXPORTER_OTLP_ENDPOINT", 
            otel_settings.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        )

        # OTEL_TRACES_SAMPLER (string value)
        sampler_str_from_env = os.environ.get("OTEL_TRACES_SAMPLER")
        sampler_str = sampler_str_from_env.lower() if sampler_str_from_env else otel_settings.get("OTEL_TRACES_SAMPLER", "always_on").lower()
        otel_settings["OTEL_TRACES_SAMPLER"] = sampler_str # Store the final choice

        # OTEL_TRACES_SAMPLER_ARG (for traceidratio ratio)
        sampler_config = otel_settings.get("OTEL_TRACES_SAMPLER_CONFIG", {})
        sampler_arg_from_env = os.environ.get("OTEL_TRACES_SAMPLER_ARG")
        if sampler_arg_from_env:
            try:
                sampler_config["ratio"] = float(sampler_arg_from_env)
                logging.info(f"OTEL_TRACES_SAMPLER_ARG from env: set ratio to {sampler_config['ratio']}")
            except ValueError:
                logging.warning(f"Invalid OTEL_TRACES_SAMPLER_ARG '{sampler_arg_from_env}' from env. Not a float. Using config/default ratio.")
        otel_settings["OTEL_TRACES_SAMPLER_CONFIG"] = sampler_config # Store updated config

        # OTEL_RESOURCE_ATTRIBUTES
        # Start with defaults, merge YAML, then merge ENV VARS
        default_resource_attrs = DEFAULT_SETTINGS.get("opentelemetry", {}).get("OTEL_RESOURCE_ATTRIBUTES", {})
        yaml_resource_attrs = settings.get("opentelemetry", {}).get("OTEL_RESOURCE_ATTRIBUTES", {}) # This already has merged defaults
        
        final_resource_attrs = default_resource_attrs.copy()
        final_resource_attrs.update(yaml_resource_attrs) # Merge YAML over defaults

        env_resource_attrs_str = os.environ.get("OTEL_RESOURCE_ATTRIBUTES")
        if env_resource_attrs_str:
            try:
                env_attrs = dict(pair.split("=", 1) for pair in env_resource_attrs_str.split(","))
                final_resource_attrs.update(env_attrs) # Env vars override
                logging.info(f"Loaded OTEL_RESOURCE_ATTRIBUTES from env: {env_attrs}")
            except ValueError:
                logging.warning(f"Invalid format for OTEL_RESOURCE_ATTRIBUTES env var: '{env_resource_attrs_str}'. Expected 'key1=value1,key2=value2'.")
        otel_settings["OTEL_RESOURCE_ATTRIBUTES"] = final_resource_attrs
        
        # Ensure 'environment' from OTEL_RESOURCE_ATTRIBUTES can still be overridden by a specific 'ENVIRONMENT' env var as a convenience
        otel_settings["OTEL_RESOURCE_ATTRIBUTES"]["environment"] = os.environ.get(
            "ENVIRONMENT", 
            otel_settings["OTEL_RESOURCE_ATTRIBUTES"].get("environment", "development")
        )

        # Create Sampler Instance based on final string and config
        if sampler_str == "always_on":
            otel_settings["OTEL_TRACES_SAMPLER_INSTANCE"] = ALWAYS_ON
        elif sampler_str == "traceidratio":
            ratio = otel_settings.get("OTEL_TRACES_SAMPLER_CONFIG", {}).get("ratio", 0.5) # Use the potentially updated ratio
            try:
                ratio = float(ratio)
                if not (0.0 <= ratio <= 1.0):
                    logging.warning(f"Invalid OTEL_TRACES_SAMPLER_CONFIG ratio '{ratio}'. Must be between 0.0 and 1.0. Defaulting to ALWAYS_ON.")
                    otel_settings["OTEL_TRACES_SAMPLER_INSTANCE"] = ALWAYS_ON
                else:
                    otel_settings["OTEL_TRACES_SAMPLER_INSTANCE"] = TraceIdRatioBasedSampler(ratio)
                    logging.info(f"Using TraceIdRatioBasedSampler with ratio: {ratio}")
            except ValueError:
                logging.warning(f"Invalid OTEL_TRACES_SAMPLER_CONFIG ratio '{ratio}'. Not a float. Defaulting to ALWAYS_ON.")
                otel_settings["OTEL_TRACES_SAMPLER_INSTANCE"] = ALWAYS_ON
        else:
            logging.warning(f"Unsupported OTEL_TRACES_SAMPLER '{sampler_str}'. Defaulting to ALWAYS_ON.")
            otel_settings["OTEL_TRACES_SAMPLER_INSTANCE"] = ALWAYS_ON
    else: # OTEL is disabled
        # Ensure instance is None or a NoOpSampler if OTEL is disabled, so logging_utils can handle it
        otel_settings["OTEL_TRACES_SAMPLER_INSTANCE"] = None # Or specific NoOpSampler if preferred
        # Clear other potentially sensitive or unnecessary settings if disabled
        otel_settings["OTEL_EXPORTER_OTLP_ENDPOINT"] = None 
        # Resource attributes might still be useful for non-OTEL purposes, or clear them too.
        # For now, leave them as they might be used by other parts of the app.

    settings["opentelemetry"] = otel_settings # Ensure the main settings dict has the processed otel_settings

    # --- Ensure log dir exists and make log paths absolute ---
    log_cfg = settings.get("logging", {})
    log_dir_path_str = log_cfg.get("log_dir", "logs")
    log_dir_path = Path(log_dir_path_str)
    if not log_dir_path.is_absolute():
        log_dir_path = PROJECT_ROOT / log_dir_path
    
    settings["logging"]["log_dir_abs"] = str(log_dir_path) # Store absolute log dir path
    
    try:
        os.makedirs(log_dir_path, exist_ok=True)
        log.info(f"Ensured log directory exists: {log_dir_path}")
    except OSError as e:
        logging.error(f"Could not create log directory '{log_dir_path}': {e}")

    settings["logging"]["info_log_path"] = str(log_dir_path / log_cfg.get("info_log_file", "info.log"))
    settings["logging"]["debug_log_path"] = str(log_dir_path / log_cfg.get("debug_log_file", "debug.log"))
    settings["logging"]["error_log_path"] = str(log_dir_path / log_cfg.get("error_log_file", "error.log"))
    settings["logging"]["model_output_log_path"] = str(log_dir_path / log_cfg.get("model_output_log_file", "model_output.log")) # New path


    # --- Add derived absolute paths for other outputs ---
    settings["scraped_jobs_path"] = os.path.join(
        settings["output_dir"],
        settings.get("scraped_jobs_filename", "scraped_jobs.json")
    )
    output_settings = settings.get('output', {})
    analysis_filename = output_settings.get(
        'analysis_output_file', 'analyzed_jobs.json'
    )  # Get from 'output' section
    settings["analysis_output_path"] = os.path.join(
        settings["output_dir"], analysis_filename
    )
    debug_filename = output_settings.get(
        'debug_output_file', 'debug_info.json'
    )
    settings["debug_output_path"] = os.path.join(
        settings["output_dir"], debug_filename
    )
    if "analysis" in settings:
        prompts_dir = settings["analysis"].get(
            "prompts_dir", str(PROJECT_ROOT / "analysis/prompts")
        )
        settings["analysis"]["resume_prompt_path"] = os.path.join(
            prompts_dir,
            settings["analysis"].get(
                "resume_prompt_file", "resume_extraction.prompt"
            )
        )
        settings["analysis"]["suitability_prompt_path"] = os.path.join(
            prompts_dir,
            settings["analysis"].get(
                "suitability_prompt_file", "suitability_analysis.prompt"
            )
        )
        settings["analysis"]["job_extraction_prompt_path"] = os.path.join(
            prompts_dir,
            settings["analysis"].get(
                "job_extraction_prompt_file", "job_extraction.prompt"
            ),
        )  # <-- Add this

    logging.debug(
        f"Final configuration settings: {settings}"
    )  # Use standard logging
    return settings


# --- Load settings ONCE ---
settings = load_config()


# --- Optional: Helper function ---
def get_setting(key_path: str, default: Any = None) -> Any:
    keys = key_path.split(".")
    value = settings
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default
