"""MyJobSpy AI - A powerful job search and analysis tool."""

# Initialize logger with comprehensive configuration
import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Import config first to ensure it's available
from .config import config as _config
from .config import load_config as _load_config
from .config import save_config as _save_config

# Create logs directory if it doesn't exist
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

# Configure root logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG if os.environ.get('MYJOBSPYAI_DEBUG') else logging.INFO)

# Create formatters
console_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S'
)

file_formatter = logging.Formatter(
    '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}',
    datefmt='%Y-%m-%d %H:%M:%S',
)

# Console handler for general output
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(console_formatter)
console_handler.setLevel(logging.INFO)

# Debug file handler for detailed logs
debug_handler = logging.handlers.RotatingFileHandler(
    log_dir / 'debug.log',
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
    encoding='utf-8',
)
debug_handler.setFormatter(file_formatter)
debug_handler.setLevel(logging.DEBUG)

# Error file handler for error logs
error_handler = logging.handlers.RotatingFileHandler(
    log_dir / 'error.log',
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
    encoding='utf-8',
)
error_handler.setFormatter(file_formatter)
error_handler.setLevel(logging.WARNING)

# HTTP request logging
http_handler = logging.handlers.RotatingFileHandler(
    log_dir / 'http_requests.log',
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
    encoding='utf-8',
)
http_handler.setFormatter(file_formatter)
http_handler.setLevel(logging.DEBUG)

# Add handlers to root logger
logger.addHandler(console_handler)
logger.addHandler(debug_handler)
logger.addHandler(error_handler)

# Configure HTTP client logging
http_client_logger = logging.getLogger('http.client')
http_client_logger.setLevel(logging.DEBUG)
http_client_logger.addHandler(http_handler)

# Configure urllib3 logging
urllib3_logger = logging.getLogger('urllib3')
urllib3_logger.setLevel(logging.DEBUG)
urllib3_logger.addHandler(http_handler)

# Module logger
logger = logging.getLogger(__name__)

# Default configuration paths
DEFAULT_CONFIG_PATHS = [
    Path('~/.config/myjobspyai/config.yaml').expanduser(),
    Path('config.yaml').absolute(),
    Path('/etc/myjobspyai/config.yaml'),
]

# Add debug logging for config paths
if os.environ.get('MYJOBSPYAI_DEBUG'):
    logger.setLevel(logging.DEBUG)
    logger.debug("Debug logging enabled for myjobspyai.__init__")
    for i, path in enumerate(DEFAULT_CONFIG_PATHS, 1):
        exists = "✓" if path.exists() else "✗"
        logger.debug(f"  {i}. {path} [{exists}]")


def load_config(
    config_path: Optional[Union[str, Path]] = None, verbose: bool = True
) -> 'AppConfig':
    """Load configuration from a file.

    Args:
        config_path: Path to the configuration file. If None, searches in default locations.
        verbose: Whether to log information about the loading process.

    Returns:
        The loaded AppConfig instance.
    """
    global _config

    if verbose:
        logger.info("Starting configuration loading process...")

    # If a specific config path was provided
    if config_path is not None:
        config_path = Path(config_path).expanduser().resolve()
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            if verbose:
                logger.info("Using default configuration")
            return _config
        if verbose:
            logger.info(f"Using specified config file: {config_path}")
    else:
        # Try default paths
        if verbose:
            logger.debug("No config path specified, searching default locations...")

        for path in DEFAULT_CONFIG_PATHS:
            if path.exists():
                config_path = path
                if verbose:
                    logger.info(f"Found config file at: {config_path}")
                break
        else:
            logger.warning("No configuration file found in default locations")
            if verbose:
                logger.info("Using default configuration")
            return _config

    try:
        if verbose:
            logger.info(f"Loading configuration from: {config_path}")

        # Load the configuration
        _config = _load_config(config_path)

        if verbose:
            # Log basic config info
            logger.info(f"Successfully loaded configuration from: {config_path}")

            # Log LLM providers if available
            if hasattr(_config, 'llm_providers') and _config.llm_providers:
                providers = list(_config.llm_providers.keys())
                logger.info(
                    f"Loaded {len(providers)} LLM providers: {', '.join(providers)}"
                )

                # Log details of each provider
                for name, provider in _config.llm_providers.items():
                    logger.debug(f"Provider '{name}': {provider}")

                # Log default provider
                default_provider = getattr(_config, 'default_llm_provider', None)
                if default_provider and default_provider in _config.llm_providers:
                    logger.info(f"Default LLM provider: {default_provider}")
                elif default_provider:
                    logger.warning(
                        f"Configured default LLM provider '{default_provider}' not found in providers"
                    )
                elif _config.llm_providers:
                    logger.warning("No default LLM provider configured")
            else:
                logger.warning("No LLM providers found in configuration")

    except Exception as e:
        logger.error(
            f"Failed to load configuration from {config_path}: {e}", exc_info=True
        )
        if verbose:
            logger.info("Falling back to default configuration")

    return _config


# Load config when the package is imported
config = load_config(verbose=os.environ.get('MYJOBSPYAI_DEBUG') == '1')

# For backward compatibility
settings = config


# Expose save_config
def save_config(config_path: Optional[Union[str, Path]] = None):
    """Save the current configuration to a file.

    Args:
        config_path: Path to save the configuration to. If None, uses the current config path.
    """
    try:
        _save_config(config_path)
        logger.info(f"Saved configuration to: {config_path or 'default location'}")
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}", exc_info=True)
        raise


__all__ = ['config', 'settings', 'load_config', 'save_config']
