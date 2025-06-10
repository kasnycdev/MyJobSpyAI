"""Main module for MyJobSpy AI application."""

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# Initialize console for Rich output
console = Console()


# Create a reusable table creation function
def _create_table(show_details: bool = False) -> Table:
    """Create a table for displaying job listings.

    Args:
        show_details: Whether to show detailed columns

    Returns:
        Table: Rich Table object
    """
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#", style="dim", width=3)
    table.add_column("Title", style="cyan", no_wrap=True)
    table.add_column("Company", style="green")
    table.add_column("Location", style="yellow")
    table.add_column("Type", style="blue")
    table.add_column("Posted", style="purple")

    if show_details:
        table.add_column("Salary", style="magenta")
        table.add_column("Experience", style="cyan")
        table.add_column("Skills", style="green")
        table.add_column("Requirements", style="yellow")

    return table


# Helper functions for formatting job data
def _format_job_type(job_type: str) -> str:
    """Format job type string for display.

    Args:
        job_type: Job type string

    Returns:
        str: Formatted job type
    """
    if not job_type:
        return "N/A"
    return job_type.title()


def _format_posted_date(posted_date: Any) -> str:
    """Format posted date for display.

    Args:
        posted_date: Date object or string

    Returns:
        str: Formatted date string
    """
    if not posted_date:
        return "N/A"
    if isinstance(posted_date, str):
        return posted_date
    return posted_date.strftime("%Y-%m-%d")


def _format_description(description: str, show_details: bool = False) -> str:
    """Format job description for display.

    Args:
        description: Job description
        show_details: Whether to show full description

    Returns:
        str: Formatted description
    """
    if not description:
        return "N/A"
    if not show_details:
        return description[:100] + "..." if len(description) > 100 else description
    return description


from myjobspyai.analysis.analyzer import JobAnalyzer, LangChainProvider, ResumeAnalyzer
from myjobspyai.analysis.models import ResumeData
from myjobspyai.config_manager import get_config, load_config
from myjobspyai.main_matcher import load_and_extract_resume_async
from myjobspyai.scrapers.factory import create_scraper
from myjobspyai.utils.logging_config import LoggingConfig

# Set up logging
logging_config = LoggingConfig(debug=get_config().debug)
logging_config.setup()
logger = logging_config.get_logger()

# Set up rich console
console = Console()


def parse_config_overrides(overrides: List[str]) -> Dict[str, Any]:
    """Parse configuration overrides from command line arguments.

    Args:
        overrides: List of key=value strings.

    Returns:
        Dictionary of configuration overrides.
    """
    result = {}
    for override in overrides:
        if "=" not in override:
            logger.warning(f"Invalid config override (missing '='): {override}")
            continue

        key, value = override.split("=", 1)
        keys = key.split(".")

        # Handle nested keys
        current = result
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        # Convert value to appropriate type
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.isdigit():
            value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass  # Keep as string

        current[keys[-1]] = value

    return result


def create_file_handler_config(name, level, log_path, mode='a'):
    """Create file handler configuration."""
    handler_config = {
        'level': level,
        'formatter': 'standard',
        'filename': str(log_path),
        'encoding': 'utf-8',
        'delay': True,
        'mode': mode,
    }
    return handler_config


def setup_logging_custom(debug: bool = False) -> None:
    """Configure logging for the application.

    Args:
        debug: Enable debug logging if True.
    """
    import logging
    import logging.config
    import os
    from pathlib import Path

    # Set up basic logging first to capture any errors
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("myjobspyai")

    # Set log level based on debug flag
    log_level = logging.DEBUG if debug else logging.INFO

    try:
        # Get logging config from app_config
        log_config = app_config.logging

        # Set up log directory
        log_dir = (
            Path(str(getattr(log_config, 'log_dir', 'logs'))).expanduser().absolute()
        )
        _setup_log_directory(log_dir, logger)
        logger.info(f"Log directory: {log_dir}")

        # Set default log level
        default_log_level = _get_log_level(debug, log_config)

        # Define log format
        log_format = _get_log_format(log_config)
        date_format = _get_date_format(log_config)

        # Get log files configuration
        log_files = _get_log_files(log_config, log_dir, default_log_level)

        # Get log file configuration
        log_file_mode = _get_log_file_mode(log_config)
        max_size = _get_log_file_size(log_config)

        # Get rolling strategy configuration
        rolling_strategy = _get_rolling_strategy(log_config)
        when = _get_rotation_when(log_config)
        interval = _get_rotation_interval(log_config)
        utc = _get_rotation_utc(log_config)
        at_time = _get_rotation_at_time(log_config)

        # Create handlers
        try:
            handlers = _create_handlers(
                log_level,
                log_files,
                log_file_mode,
                rolling_strategy,
                max_size,
                when,
                interval,
                utc,
                at_time,
            )
        except Exception as e:
            logger.error(f"Failed to create log handlers: {str(e)}", exc_info=True)
            handlers = {
                'console': create_file_handler_config('console', 'INFO', 'stderr')
            }
            logger.warning("Using default console handler due to configuration error")

        # Configure loggers
        loggers = _configure_loggers(debug, log_files)

        # Configure logging
        logging_config = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': _create_formatters(date_format),
            'handlers': handlers,
            'loggers': loggers,
            'root': {'handlers': ['console'], 'level': 'DEBUG' if debug else 'INFO'},
        }

        # Apply the logging configuration
        logging.config.dictConfig(logging_config)

        # Set up error tracking for uncaught exceptions
        _setup_exception_handler(logger)

        # Log Python warnings
        logging.captureWarnings(True)

        # Log the configuration
        _log_configuration(logger, log_dir, debug, log_files)

    except Exception as e:
        # Fallback to basic config if logging setup fails
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger("myjobspyai")
        logger.error(f"Failed to configure logging: {e}", exc_info=True)
        logger.info("Falling back to basic logging configuration")


def _setup_log_directory(log_dir: Path, logger) -> None:
    """Set up the log directory with appropriate permissions."""
    try:
        log_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
        if not os.access(log_dir, os.W_OK):
            logger.warning(
                f"Log directory {log_dir} is not writable. Falling back to default location."
            )
            log_dir = Path.cwd() / 'logs'
            log_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
    except Exception as e:
        logger.warning(
            f"Failed to create log directory at {log_dir}: {e}. Using default location."
        )
        log_dir = Path.cwd() / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True, mode=0o755)


def _get_log_level(debug: bool, log_config) -> int:
    """Get the log level from configuration or default."""
    return (
        logging.DEBUG
        if debug
        else getattr(
            logging,
            str(getattr(log_config, 'log_level', 'INFO')).upper(),
            logging.INFO,
        )
    )


def _get_log_format(log_config) -> str:
    """Get the log format from configuration or default."""
    log_format = getattr(
        log_config,
        'format',
        '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
    )
    return log_format


def _get_date_format(log_config) -> str:
    """Get the date format from configuration or default."""
    return str(getattr(log_config, 'date_format', "%Y-%m-%d %H:%M:%S"))


def _get_log_files(log_config, log_dir: Path, default_log_level: int) -> dict:
    """Get log files configuration."""
    if hasattr(log_config, 'files') and log_config.files:
        return {
            log_name: {
                'path': log_dir / str(log_spec.get('path', f"{log_name}.log")),
                'level': getattr(
                    logging,
                    str(log_spec.get('level', 'INFO')).upper(),
                    logging.INFO,
                ),
            }
            for log_name, log_spec in log_config.files.items()
        }
    else:
        return {
            'app': {
                'path': log_dir / str(getattr(log_config, 'info_log_file', 'app.log')),
                'level': default_log_level,
            },
            'debug': {
                'path': log_dir
                / str(getattr(log_config, 'debug_log_file', 'debug.log')),
                'level': logging.DEBUG,
            },
            'error': {
                'path': log_dir
                / str(getattr(log_config, 'error_log_file', 'error.log')),
                'level': logging.WARNING,
            },
            'llm': {
                'path': log_dir
                / str(getattr(log_config, 'model_output_log_file', 'llm.log')),
                'level': logging.INFO,
            },
        }


def _get_log_file_mode(log_config) -> str:
    """Get the log file mode from configuration or default."""
    return str(getattr(log_config, 'log_file_mode', 'a'))


def _get_log_file_size(log_config) -> int:
    """Get the log file size from configuration or default."""
    return int(getattr(log_config, 'max_size', 10 * 1024 * 1024))


def _get_rolling_strategy(log_config) -> str:
    """Get the rolling strategy from configuration or default."""
    return str(getattr(log_config, 'rolling_strategy', 'size')).lower()


def _get_rotation_when(log_config) -> str:
    """Get the rotation when from configuration or default."""
    return str(getattr(log_config, 'when', 'midnight'))


def _get_rotation_interval(log_config) -> int:
    """Get the rotation interval from configuration or default."""
    return int(getattr(log_config, 'interval', 1))


def _get_rotation_utc(log_config) -> bool:
    """Get the rotation UTC from configuration or default."""
    return bool(getattr(log_config, 'utc', False))


def _get_rotation_at_time(log_config):
    """Get the rotation at time from configuration."""
    try:
        at_time = getattr(log_config, 'at_time', None)
        if at_time:
            # Validate time format if provided
            try:
                datetime.strptime(at_time, '%H:%M')
            except ValueError:
                logger.warning(
                    f"Invalid time format for rotation at_time: {at_time}. Using default."
                )
                return None
        return at_time
    except Exception as e:
        logger.error(f"Error processing rotation at_time: {str(e)}")
        return None


def _create_handlers(
    log_level: int,
    log_files: dict,
    log_file_mode: str,
    rolling_strategy: str,
    max_size: int,
    when: str,
    interval: int,
    utc: bool,
    at_time,
) -> dict:
    """Create handlers configuration."""
    handlers = {
        'console': {
            'class': 'rich.logging.RichHandler',
            'level': log_level,
            'formatter': 'standard',
            'rich_tracebacks': True,
            'show_path': debug,
            'markup': True,
            'show_time': True,
            'show_level': True,
        }
    }

    if hasattr(log_config, 'info_log_file') and getattr(
        log_config, 'info_log_file', None
    ):
        handlers['info_file'] = create_file_handler_config(
            'info_file', 'INFO', 'standard', log_files['info']
        )

    for name, log_spec in log_files.items():
        handlers[f'file_{name}'] = create_file_handler_config(
            name=name, level=log_spec['level'], log_path=log_spec['path']
        )

    return handlers


def _configure_loggers(debug: bool, log_files: dict) -> dict:
    """Configure loggers."""
    return {
        '': {  # root logger
            'handlers': ['console'] + [f'file_{name}' for name in log_files.keys()],
            'level': 'DEBUG' if debug else 'INFO',
            'propagate': True,
        },
        'myjobspyai': {
            'handlers': ['console', 'file_app', 'file_error', 'file_debug'],
            'level': 'DEBUG' if debug else 'INFO',
            'propagate': False,
        },
        'llm': {
            'handlers': ['file_llm'],
            'level': 'DEBUG' if debug else 'INFO',
            'propagate': False,
        },
        'debug': {'handlers': ['file_debug'], 'level': 'DEBUG', 'propagate': False},
        'error': {
            'handlers': ['file_error'],
            'level': 'WARNING',
            'propagate': False,
        },
    }


def _create_formatters(date_format: str) -> dict:
    """Create formatters configuration."""
    try:
        return {
            'standard': {
                'format': _get_log_format(log_config),
                'datefmt': date_format,
                'style': '%',
            },
            'detailed': {
                'format': '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
                'datefmt': date_format,
            },
            'simple': {'format': '%(levelname)-8s %(message)s'},
            'json': {
                'class': 'pythonjsonlogger.jsonlogger.JsonFormatter',
                'format': '''{
                    "timestamp": "%(asctime)s",
                    "level": "%(levelname)s",
                    "logger": "%(name)s",
                    "module": "%(module)s",
                    "function": "%(funcName)s",
                    "line": %(lineno)d,
                    "message": "%(message)s",
                    "process": %(process)d,
                    "thread": %(thread)d,
                    "process_name": "%(processName)s",
                    "thread_name": "%(threadName)s"
                }''',
                'datefmt': date_format,
            },
        }
    except Exception as e:
        logger.error(f"Failed to create formatters: {str(e)}")
        return {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'datefmt': date_format,
                'style': '%',
            },
            'simple': {'format': '%(levelname)s %(message)s'},
        }


def _setup_exception_handler(logger) -> None:
    """Set up exception handler."""

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.critical(
            "Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = handle_exception


def _log_configuration(logger, log_dir: Path, debug: bool, log_files: dict) -> None:
    """Log the configuration."""
    logger.info("=" * 50)
    logger.info(f"Logging configured. Log directory: {log_dir.absolute()}")
    logger.info(f"Log level: {'DEBUG' if debug else 'INFO'}")
    log_files_str = ', '.join(
        [f"{name}: {log_spec['path'].name}" for name, log_spec in log_files.items()]
    )
    logger.info(f"Log files: {log_files_str}")

    config_path = Path('~/.config/myjobspyai/config.yaml').expanduser()
    if config_path.exists():
        logger.info(f"Using configuration from: {config_path}")
    else:
        logger.warning(
            f"Configuration file not found at {config_path}. Using default settings."
        )
    logger.info("-" * 50)


def parse_args(args: list[str]) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        args: Command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="MyJobSpy AI - Job Search and Analysis Tool"
    )
    # Basic arguments
    # Basic search arguments
    parser.add_argument('--search', type=str, help='Search term for job titles')
    parser.add_argument('--min-salary', type=float, help='Minimum salary requirement')
    parser.add_argument('--max-salary', type=float, help='Maximum salary requirement')
    parser.add_argument('--resume', type=str, help='Path to resume PDF file')
    parser.add_argument(
        '--analyze', action='store_true', help='Analyze jobs against resume'
    )
    parser.add_argument(
        '--details', action='store_true', help='Show detailed job information'
    )
    parser.add_argument('--save', action='store_true', help='Save results to file')
    parser.add_argument('--output', type=str, help='Output file name')
    parser.add_argument(
        '--format',
        type=str,
        choices=['json', 'csv', 'xlsx', 'markdown'],
        default='json',
        help='Output format',
    )
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument(
        '--quiet', action='store_true', help='Suppress all output except errors'
    )
    parser.add_argument(
        '--version', action='store_true', help='Show version information'
    )
    parser.add_argument(
        '--interactive', action='store_true', help='Enable interactive mode'
    )
    parser.add_argument(
        '--no-interactive', action='store_true', help='Disable interactive mode'
    )

    # Scraper selection
    parser.add_argument(
        '--scraper',
        type=str,
        default='jobspy',
        choices=['jobspy', 'linkedin', 'indeed'],
        help='Scraper to use (default: jobspy)',
    )

    # Search options - specialized search parameters
    search_group = parser.add_argument_group('Search Options')
    search_group.add_argument(
        '--site-name',
        type=str,
        nargs='+',
        choices=[
            'linkedin',
            'indeed',
            'glassdoor',
            'google',
            'zip_recruiter',
            'bayt',
            'naukri',
        ],
        help='Job sites to search (default: all supported sites)',
    )
    search_group.add_argument(
        '--search-term',
        type=str,
        help='Search term (e.g., "software engineer" or "data scientist")',
    )
    search_group.add_argument(
        '--google-search-term',
        type=str,
        help='Search term specifically for Google Jobs (overrides --search-term for Google)',
    )
    search_group.add_argument(
        '--distance',
        type=int,
        default=50,
        help='Distance in miles from location (default: 50)',
    )
    search_group.add_argument(
        '--results-wanted',
        type=int,
        default=15,
        help='Number of results wanted per site (default: 15)',
    )
    search_group.add_argument(
        '--hours-old',
        type=int,
        help='Maximum age of job postings in hours',
    )
    search_group.add_argument(
        '--easy-apply',
        action='store_true',
        help='Filter for jobs that support quick apply',
    )
    search_group.add_argument(
        '--offset',
        type=int,
        default=0,
        help='Start results from this offset (default: 0)',
    )
    search_group.add_argument(
        '--description-format',
        type=str,
        choices=['markdown', 'html'],
        default='markdown',
        help='Format for job descriptions (default: markdown)',
    )
    search_group.add_argument(
        '--proxies',
        type=str,
        nargs='+',
        help='List of proxies to use (format: user:pass@host:port)',
    )
    search_group.add_argument(
        '--enforce-annual-salary',
        action='store_true',
        help='Convert wages to annual salary',
    )
    search_group.add_argument(
        '--country-indeed',
        type=str,
        help='Country for Indeed/Glassdoor searches (e.g., "USA", "Canada")',
    )

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory to save output files (default: output/)',
    )
    output_group.add_argument(
        '--output-format',
        type=str,
        choices=['json', 'csv', 'xlsx', 'markdown'],
        default='json',
        help='Output format for results (default: json)',
    )
    output_group.add_argument(
        '--no-save',
        action='store_true',
        help="Don't save results to disk (only show in console)",
    )

    # Proxy settings
    proxy_group = parser.add_argument_group('Proxy Settings')
    proxy_group.add_argument(
        '--proxy',
        type=str,
        action='append',
        help='Proxy server to use (format: user:pass@host:port or localhost)',
    )
    proxy_group.add_argument(
        '--ca-cert',
        type=str,
        help='Path to CA certificate file for SSL verification',
    )

    # LinkedIn specific options
    linkedin_group = parser.add_argument_group('LinkedIn Options')
    linkedin_group.add_argument(
        '--linkedin-fetch-description',
        action='store_true',
        help='Fetch full job descriptions from LinkedIn (increases requests)',
    )
    linkedin_group.add_argument(
        '--linkedin-company-ids',
        type=int,
        nargs='+',
        help='Filter LinkedIn jobs by company IDs',
    )

    # Runtime options
    runtime_group = parser.add_argument_group('Runtime Options')
    verbose_group = runtime_group.add_mutually_exclusive_group()
    verbose_group.add_argument(
        '-v',
        '--verbose',
        action='count',
        default=2,
        help='Increase verbosity (can be used multiple times, e.g., -vv for debug)',
    )
    verbose_group.add_argument(
        '-q',
        '--quiet',
        action='store_true',
        help='Suppress all output except errors',
    )
    runtime_group.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (more verbose output)',
    )
    runtime_group.add_argument(
        '--version',
        action='store_true',
        help='Show version and exit',
    )
    runtime_group.add_argument(
        '--interactive',
        action='store_true',
        help='Enable interactive mode to view job details',
    )

    return parser.parse_args(args)


async def _get_openai_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get OpenAI configuration."""
    try:
        return {
            'base_url': config.get('base_url', ''),
            'model': config.get('model', 'gpt-4-turbo-preview'),
            'api_key': config.get('api_key') or os.getenv('OPENAI_API_KEY'),
            'temperature': config.get('temperature', 0.7),
            'max_tokens': config.get('max_tokens', 1000),
            'streaming': config.get('streaming', True),
            'request_timeout': config.get('timeout', 60),
            'organization': config.get('organization') or os.getenv('OPENAI_ORG_ID'),
        }
    except Exception as e:
        logger.error(f"Failed to get OpenAI configuration: {str(e)}")
        return {
            'model': 'gpt-4-turbo-preview',
            'temperature': 0.7,
            'max_tokens': 1000,
            'streaming': True,
            'request_timeout': 60,
        }


def _get_anthropic_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get Anthropic configuration."""
    try:
        return {
            'model': config.get('model', 'claude-3-opus-20240229'),
            'api_key': config.get('api_key') or os.getenv('ANTHROPIC_API_KEY'),
            'temperature': config.get('temperature', 0.7),
            'max_tokens': config.get('max_tokens', 1000),
        }
    except Exception as e:
        logger.error(f"Failed to get Anthropic configuration: {str(e)}")
        return {
            'model': 'claude-3-opus-20240229',
            'temperature': 0.7,
            'max_tokens': 1000,
        }


def _get_google_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get Google configuration."""
    try:
        return {
            'model': config.get('model', 'gemini-pro'),
            'google_api_key': config.get('api_key') or os.getenv('GOOGLE_API_KEY'),
            'temperature': config.get('temperature', 0.7),
        }
    except Exception as e:
        logger.error(f"Failed to get Google configuration: {str(e)}")
        return {
            'model': 'gemini-pro',
            'temperature': 0.7,
        }


def _get_ollama_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get Ollama configuration."""
    try:
        return {
            'base_url': config.get('base_url', 'http://localhost:11434'),
            'model': config.get('model', 'llama2'),
            'temperature': config.get('temperature', 0.7),
            'top_p': config.get('top_p', 1.0),
            'timeout': config.get('timeout', 60),
            'num_predict': config.get('max_tokens', 1000),
        }
    except Exception as e:
        logger.error(f"Failed to get Ollama configuration: {str(e)}")
        return {
            'base_url': 'http://localhost:11434',
            'model': 'llama2',
            'temperature': 0.7,
            'top_p': 1.0,
            'timeout': 60,
        }


def _initialize_openai(config: Dict[str, Any]) -> Optional[Any]:
    """Initialize OpenAI provider."""
    try:
        from langchain_openai import ChatOpenAI

        if config['base_url'].startswith(('http://', 'https://')):
            return ChatOpenAI(
                base_url=config['base_url'],
                model=config['model'],
                api_key=config['api_key'],
                temperature=config['temperature'],
                max_tokens=config['max_tokens'],
                streaming=config['streaming'],
                request_timeout=config['request_timeout'],
            )
        return ChatOpenAI(
            model=config['model'],
            api_key=config['api_key'],
            temperature=config['temperature'],
            max_tokens=config['max_tokens'],
            streaming=config['streaming'],
            request_timeout=config['request_timeout'],
            organization=config['organization'],
        )
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI provider: {str(e)}")
        return None


def _initialize_anthropic(config: Dict[str, Any]) -> Optional[Any]:
    """Initialize Anthropic provider."""
    try:
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(**config)
    except Exception as e:
        logger.error(f"Failed to initialize Anthropic provider: {str(e)}")
        return None


def _initialize_google(config: Dict[str, Any]) -> Optional[Any]:
    """Initialize Google provider."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(**config)
    except Exception as e:
        logger.error(f"Failed to initialize Google provider: {str(e)}")
        return None


def _initialize_ollama(config: Dict[str, Any]) -> Optional[Any]:
    """Initialize Ollama provider."""
    try:
        from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
        from langchain_community.llms import Ollama
        from langchain_core.callbacks.manager import CallbackManager

        callbacks = [StreamingStdOutCallbackHandler()] if config['streaming'] else []
        return Ollama(
            base_url=config['base_url'],
            model=config['model'],
            temperature=config['temperature'],
            top_p=config['top_p'],
            callback_manager=CallbackManager(callbacks) if callbacks else None,
            timeout=config['timeout'],
            num_predict=config['num_predict'],
        )
    except ImportError as e:
        logger.error(
            "Failed to import Ollama dependencies. Install with: pip install langchain-community"
        )
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Ollama provider: {e}", exc_info=True)
        return None


def initialize_llm_provider(provider_config: Dict[str, Any]) -> Optional[Any]:
    """Initialize an LLM provider from configuration.

    Args:
        provider_config: Provider configuration.

    Returns:
        Initialized LLM provider or None if initialization failed.
    """
    provider_type = provider_config.get("type", "").lower()
    config = provider_config.get("config", {})

    if not provider_type:
        logger.error("Provider type not specified in configuration")
        return None

    try:
        if provider_type == "openai":
            return _initialize_openai(_get_openai_config(config))
        elif provider_type == "anthropic":
            return _initialize_anthropic(_get_anthropic_config(config))
        elif provider_type == "google":
            return _initialize_google(_get_google_config(config))
        elif provider_type == "langchain":
            return LangChainProvider(config)
        elif provider_type == "ollama":
            return _initialize_ollama(_get_ollama_config(config))
        else:
            logger.warning(f"Unsupported LLM provider: {provider_type}")
            return None

    except ImportError as e:
        logger.error(f"Failed to import dependencies for {provider_type} provider: {e}")
        logger.info("You may need to install additional packages. For example:")
        if provider_type == "openai":
            logger.info("pip install langchain-openai")
        elif provider_type == "anthropic":
            logger.info("pip install langchain-anthropic")
        elif provider_type == "google":
            logger.info("pip install langchain-google-genai")
        return None


def display_jobs_table(
    jobs: List[Dict[str, Any]], console: Console, show_details: bool = False
) -> None:
    """Display jobs in a formatted table.

    Args:
        jobs: List of job dictionaries.
        console: Rich console for output.
        show_details: Whether to show detailed information.
    """
    try:
        if not jobs:
            console.print("[yellow]No jobs found![/]")
            return

        table = _create_table(show_details)
        for i, job in enumerate(jobs):
            # Get job properties
            title = str(job.get('title', ''))
            company = str(job.get('company', ''))
            location = str(job.get('location', ''))
            job_type = _format_job_type(job.get('job_type', 'Full-time'))
            posted_date = _format_posted_date(job.get('posted_date', 'N/A'))
            description = _format_description(job.get('description', ''), show_details)

            # Format text to fit columns
            title = title[:50] + ("..." if len(title) > 50 else "")
            company = company[:20] + ("..." if len(company) > 20 else "")
            location = location[:20] + ("..." if len(location) > 20 else "")

            # Add row to table
            table.add_row(
                str(i), title, company, location, job_type, posted_date, description
            )

        # Print the table
        console.print(table)

        # Print analysis legend if we have analysis
        has_analysis = any('_analysis' in job for job in jobs)
        if has_analysis:
            console.print("\n[bold]Match Score Legend:[/bold]")
            console.print("  [green]75-100%:[/green] Strong match")
            console.print("  [yellow]50-74%:[/yellow] Moderate match")
            console.print("  [red]0-49%:[/red] Weak match")

        console.print(
            f"\n[yellow]Found {len(jobs)} jobs. Use the job number to view details.[/yellow]"
        )
    except Exception as e:
        logger.error(f"Error displaying jobs table: {str(e)}")
        console.print(
            "[red]Error displaying jobs table. Please check the logs for details.[/red]"
        )


def display_job_details(job):
    """Display detailed information about a job with analysis if available."""
    try:
        console = Console()

        # Handle both dict and object job formats
        is_dict = isinstance(job, dict)

        # Get job details
        title = job.get('title') if is_dict else getattr(job, 'title', 'N/A')
        company = job.get('company') if is_dict else getattr(job, 'company', 'N/A')
        location = job.get('location') if is_dict else getattr(job, 'location', 'N/A')
        job_type = job.get('job_type') if is_dict else getattr(job, 'job_type', 'N/A')
        remote = job.get('is_remote') if is_dict else getattr(job, 'is_remote', False)
        posted_date = (
            job.get('posted_date') if is_dict else getattr(job, 'posted_date', None)
        )
        description = (
            job.get('description') if is_dict else getattr(job, 'description', '')
        )
        url = job.get('url') if is_dict else getattr(job, 'url', '')

        # Get analysis if available
        analysis = job.get('_analysis') if is_dict and '_analysis' in job else None

        # Create a panel for the job details
        console.print(f"\n[bold blue]{'=' * 80}[/bold blue]")
        console.print(f"[bold]{title}[/bold]")
        console.print(f"[bold cyan]{company}[/bold]")
        console.print(f"[yellow]{location}[/yellow]")
    except Exception as e:
        logger.error(f"Error displaying job details: {str(e)}")
        console.print(
            "[red]Error displaying job details. Please check the logs for details.[/red]"
        )
        return

    try:
        # Job type and remote status
        job_type_str = str(job_type or "Not specified").replace("_", " ").title()
        remote_status = "✅ Remote" if remote else "❌ On-site"
        console.print(f"{job_type_str} • {remote_status}")

        # Display match score if analysis is available
        if (
            analysis
            and 'suitability_score' in analysis
            and analysis['suitability_score'] is not None
        ):
            score = analysis['suitability_score']
            # Color code the score
            if score >= 75:
                score_display = f"[green]{score}% Match[/green]"
            elif score >= 50:
                score_display = f"[yellow]{score}% Match[/yellow]"
            else:
                score_display = f"[red]{score}% Match[/red]"
            console.print(f"\n[bold]Match Score:[/bold] {score_display}")

            # Display strengths if available
            if 'pros' in analysis and analysis['pros']:
                console.print("\n[bold green]Strengths:[/bold green]")
                for strength in analysis['pros'][:3]:  # Limit to top 3 strengths
                    console.print(f"  • {strength}")

            # Display areas for improvement if available
            if 'cons' in analysis and analysis['cons']:
                console.print("\n[bold yellow]Areas for Improvement:[/bold yellow]")
                for con in analysis['cons'][:3]:  # Limit to top 3 areas
                    console.print(f"  • {con}")

        # Posted date
        if posted_date:
            if hasattr(posted_date, 'strftime'):
                posted_date = posted_date.strftime('%Y-%m-%d')
            console.print(f"\n[dim]Posted: {posted_date}[/dim]")

        # Job description
        if description:
            console.print("\n[bold]Description:[/bold]")
            console.print(
                description[:1000] + ("..." if len(description) > 1000 else "")
            )

        # Salary information if available
        salary = (
            job.get('salary')
            if is_dict
            else (getattr(job, 'salary', None) if hasattr(job, 'salary') else None)
        )
        if salary:
            salary_str = []

            # Handle both dict and object salary formats
            min_amount = (
                salary.get('min_amount')
                if isinstance(salary, dict)
                else getattr(salary, 'min_amount', None)
            )
            max_amount = (
                salary.get('max_amount')
                if isinstance(salary, dict)
                else getattr(salary, 'max_amount', None)
            )
            currency = (
                salary.get('currency')
                if isinstance(salary, dict)
                else getattr(salary, 'currency', None)
            )
            period = (
                salary.get('period')
                if isinstance(salary, dict)
                else getattr(salary, 'period', None)
            )

            if min_amount is not None:
                salary_str.append(f"${min_amount:,.0f}")
            if max_amount is not None:
                if salary_str:
                    salary_str.append("-")
                salary_str.append(f"${max_amount:,.0f}")
            if currency:
                salary_str.append(currency)
            if period:
                salary_str.append(f"per {period}")

            if salary_str:
                console.print("\n[bold]Salary:[/bold]", " ".join(salary_str))

        # Application URL if available
        if url:
            console.print(f"\n[bold]Apply:[/bold] {url}")

        console.print(f"[bold blue]{'=' * 80}[/bold blue]\n")
    except Exception as e:
        logger.error(f"Error displaying job details (part 2): {str(e)}")
        console.print(
            "[red]Error displaying job details. Please check the logs for details.[/red]"
        )
        return


async def analyze_jobs_with_resume(
    jobs: List[Any], resume_data: ResumeData
) -> List[Dict[str, Any]]:
    """Analyze jobs against a resume.

    Args:
        jobs: List of job objects to analyze
        resume_data: Parsed resume data

    Returns:
        List of job dicts with analysis results
    """
    if not jobs:
        return []

    # Initialize LLM provider if configured
    llm_provider = None
    try:
        from myjobspyai.config import config as app_config

        if hasattr(app_config, 'llm') and app_config.llm:
            llm_config = app_config.llm
            default_provider = llm_config.get('default_provider')
            providers = llm_config.get('providers', {})

            if default_provider and default_provider in providers:
                provider_config = providers[default_provider]
                if provider_config.get('enabled', False):
                    llm_provider = await initialize_llm_provider(provider_config)
                    if llm_provider:
                        logger.info(f"Initialized LLM provider: {default_provider}")
                    else:
                        logger.warning(
                            f"Failed to initialize LLM provider: {default_provider}"
                        )
    except Exception as e:
        logger.warning(f"Error initializing LLM provider: {e}")

    # Initialize job analyzer with the LLM provider
    job_analyzer = JobAnalyzer(llm_provider=llm_provider)
    analyzed_jobs = []

    for job in jobs:
        try:
            # Convert job to dict if it's an object
            job_dict = job.model_dump() if hasattr(job, 'model_dump') else dict(job)

            # Add analysis results
            analysis = await job_analyzer.analyze_resume_suitability(
                resume_data=resume_data, job_data=job_dict
            )

            # Add analysis to job data
            if hasattr(analysis, 'dict'):
                job_dict['_analysis'] = analysis.dict()
            elif isinstance(analysis, dict):
                job_dict['_analysis'] = analysis
            else:
                job_dict['_analysis'] = {'error': 'Invalid analysis result format'}

            analyzed_jobs.append(job_dict)

        except Exception as e:
            logger.error(
                f"Error analyzing job {getattr(job, 'title', 'Unknown')}: {e}",
                exc_info=True,
            )
            # Add job without analysis if analysis fails
            job_dict = job.model_dump() if hasattr(job, 'model_dump') else dict(job)
            job_dict['_analysis'] = {'error': str(e)}
            analyzed_jobs.append(job_dict)

    return analyzed_jobs


async def search_jobs(args) -> int:
    """Perform job search and analysis.

    Args:
        args: Command line arguments

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    try:
        # Load and analyze resume if provided
        resume_data = None
        analyzed_jobs = []  # Initialize to empty list
        if hasattr(args, 'resume') and args.resume:
            try:
                console.print(
                    f"[blue]Loading and analyzing resume: {args.resume}[/blue]"
                )
                resume_data = await load_and_extract_resume_async(args.resume)
                if not resume_data:
                    console.print(
                        "[yellow]Warning: Could not parse resume. Analysis will be limited.[/yellow]"
                    )
            except Exception as e:
                logger.error(f"Error loading resume: {e}", exc_info=True)
                console.print(
                    f"[yellow]Warning: Error loading resume: {e}. Analysis will be limited.[/yellow]"
                )

        # Create scraper based on arguments
        scraper_type = getattr(args, 'scraper', 'jobspy')
        scraper = create_scraper(scraper_type)
        if not scraper:
            console.print(f"[red]Error: Invalid scraper type: {scraper_type}[/red]")
            return 1

        # Get scraper-specific configuration
        scraper_config = getattr(app_config, scraper_type, {})

        # Get jobspy-specific parameters
        jobspy_params = {
            'job_title': getattr(args, 'job_title', None),
            'location': getattr(args, 'location', None),
            'radius': getattr(args, 'radius', None),
            'date_posted': getattr(args, 'date_posted', None),
            'experience_level': getattr(args, 'experience_level', None),
            'job_type': getattr(args, 'job_type', None),
            'remote': getattr(args, 'remote', None),
            'full_time': getattr(args, 'full_time', None),
            'part_time': getattr(args, 'part_time', None),
            'internship': getattr(args, 'internship', None),
            'contract': getattr(args, 'contract', None),
            'min_salary': getattr(args, 'min_salary', None),
            'max_salary': getattr(args, 'max_salary', None),
            'salary_period': getattr(args, 'salary_period', None),
            'company': getattr(args, 'company', None),
            'linkedin_fetch_description': getattr(
                args,
                'linkedin_fetch_description',
                scraper_config.get('linkedin_fetch_description', False),
            ),
            'linkedin_company_ids': getattr(
                args, 'linkedin_company_ids', scraper_config.get('linkedin_company_ids')
            ),
        }

        # Initialize search parameters
        search_params = {}
        if hasattr(app_config, 'search'):
            search_params.update(app_config.search)

        # Add search query from args
        search_query = getattr(args, 'search', '')
        if search_query:
            search_params['query'] = search_query

        # Override with command line arguments
        search_params.update(jobspy_params)

        # Remove None values and empty strings
        search_params = {
            k: v for k, v in search_params.items() if v is not None and v != ''
        }

        # Execute search with progress
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            ) as progress:
                task = progress.add_task(
                    description="Searching for jobs...", total=None
                )
                # Execute the job search
                jobs = await scraper.search_jobs(**search_params)
                if not jobs:
                    console.print("[yellow]No jobs found.[/yellow]")
                    return 0

                # Convert jobs to dicts for easier handling
                jobs_list = []
                for job in jobs:
                    if hasattr(job, 'model_dump'):
                        # Pydantic v2 model
                        job_dict = job.model_dump()
                    elif hasattr(job, 'dict'):
                        # Pydantic v1 model (fallback)
                        job_dict = job.dict()
                    elif isinstance(job, dict):
                        # Already a dictionary
                        job_dict = job
                    else:
                        # For other objects, convert to dict safely
                        job_dict = {}
                        for attr in dir(job):
                            if not attr.startswith('_') and not callable(
                                getattr(job, attr)
                            ):
                                try:
                                    value = getattr(job, attr)
                                    # Handle nested Pydantic models
                                    if hasattr(value, 'model_dump'):
                                        value = value.model_dump()
                                    elif hasattr(value, 'dict'):
                                        value = value.dict()
                                    job_dict[attr] = value
                                except Exception as e:
                                    logger.warning(
                                        f"Could not get attribute {attr} from job object: {e}"
                                    )

                    jobs_list.append(job_dict)

        except Exception as e:
            logger.error(f"Error during job search: {str(e)}", exc_info=True)
            console.print(f"[red]Error: {str(e)}[/red]")
            return 1

        # Apply min-salary filter if specified
        min_salary = getattr(args, 'min_salary', None)
        if min_salary is not None:
            try:
                original_count = len(jobs_list)
                filtered_jobs = []
                for job in jobs_list:
                    if not job.get('salary') or not job['salary'].get('min_amount'):
                        filtered_jobs.append(job)
                    elif job['salary']['min_amount'] >= min_salary:
                        filtered_jobs.append(job)

                filtered_count = original_count - len(filtered_jobs)
                jobs_list = filtered_jobs
                if filtered_count > 0:
                    logger.info(
                        f"Filtered out {filtered_count} jobs with confirmed salaries below ${min_salary:,.0f}"
                    )
                    logger.info(
                        f"Kept {len(jobs_list)} jobs (including {len([j for j in jobs_list if not j.get('salary') or not j['salary'].get('min_amount')])} with unknown salaries)"
                    )
            except Exception as e:
                logger.error(
                    f"Error applying min-salary filter: {str(e)}", exc_info=True
                )
                console.print(
                    f"[yellow]Warning: Failed to apply min-salary filter: {str(e)}[/yellow]"
                )

        # Handle job display and analysis
        if not getattr(args, 'analyze', False):
            # Display jobs without analysis
            display_title = f"{scraper_type.upper()} Job Search Results"
            display_jobs_table(
                jobs_list, console, show_details=getattr(args, 'details', False)
            )

        # Display analyzed jobs
        if analyzed_jobs:
            display_title = f"{scraper_type.upper()} Job Search Results (with Analysis)"
            display_jobs_table(analyzed_jobs, console, show_details=True)

        # Interactive mode if explicitly enabled
        if getattr(args, 'interactive', False):
            save_results = False  # Initialize save_results flag
            while True:
                try:
                    console.print(
                        "\n[bold]Enter job number to view details, 's' to save results, or 'q' to quit:[/bold]"
                    )
                    choice = input("> ").strip().lower()
                    if choice == 'q':
                        break
                    elif choice == 's':
                        # Save results
                        save_results = True
                        break
                    elif choice.isdigit():
                        job_index = int(choice) - 1
                        if 0 <= job_index < len(analyzed_jobs):
                            display_job_details(analyzed_jobs[job_index])
                        else:
                            console.print(
                                "[yellow]Invalid job number. Please try again.[/yellow]"
                            )
                    else:
                        console.print(
                            "[yellow]Invalid input. Please enter a number, 's', or 'q'.[/yellow]"
                        )
                except Exception as e:
                    logger.error(f"Error in interactive mode: {str(e)}")
                    console.print("[red]An error occurred. Please try again.[/red]")

            # Save results if requested
            if save_results:
                output_file = getattr(args, 'output', 'jobs_results.json')
                output_format = getattr(args, 'format', 'json')
                try:
                    # Convert jobs to DataFrame for saving
                    jobs_data = []
                    for job in analyzed_jobs:
                        if hasattr(job, 'model_dump'):
                            job_dict = job.model_dump()
                        else:
                            job_dict = dict(job)

                            # Flatten analysis if it exists
                            if '_analysis' in job_dict and job_dict['_analysis']:
                                analysis = job_dict.pop('_analysis', {})
                                if 'suitability_score' in analysis:
                                    job_dict['match_score'] = analysis[
                                        'suitability_score'
                                    ]
                                if 'pros' in analysis:
                                    job_dict['strengths'] = (
                                        "; ".join(analysis['pros'][:3])
                                        if analysis['pros']
                                        else ""
                                    )
                                if 'cons' in analysis:
                                    job_dict['areas_for_improvement'] = (
                                        "; ".join(analysis['cons'][:3])
                                        if analysis['cons']
                                        else ""
                                    )

                        jobs_data.append(job_dict)

                    df = pd.DataFrame(jobs_data)

                    if output_format == 'csv':
                        df.to_csv(output_file, index=False)
                    elif output_format == 'xlsx':
                        df.to_excel(output_file, index=False)
                    elif output_format == 'markdown':
                        with open(output_file, 'w', encoding='utf-8') as f:
                            f.write(df.to_markdown(index=False))
                    else:  # default to json
                        df.to_json(
                            output_file, orient='records', indent=2, force_ascii=False
                        )

                    console.print(f"\n[green]Results saved to: {output_file}[/green]")
                    return 0
                except Exception as e:
                    logger.error(f"Failed to save results: {str(e)}")
                    console.print(
                        f"[yellow]Warning: Failed to save results: {str(e)}[/yellow]"
                    )
                    return 1

            return 0

        return 0

    except Exception as e:
        logger.error(f"Error during job analysis and display: {str(e)}", exc_info=True)
        console.print(f"[red]Error: {str(e)}[/red]")
        return 1


async def main_async() -> int:
    """Async main entry point for the application."""
    try:
        # Parse command line arguments
        args = parse_args(sys.argv[1:])

        # Handle version flag
        if getattr(args, 'show_version', False):
            console.print(f"MyJobSpy AI v{app_config.version}")
            return 0

        # Set log level based on verbosity
        verbose_level = getattr(args, 'verbose', 0)
        if verbose_level > 1:
            logger.setLevel(logging.DEBUG)
        elif verbose_level == 1:
            logger.setLevel(logging.INFO)
        else:
            logger.setLevel(logging.WARNING)

        # Log command line arguments (safely, without sensitive data)
        safe_args = {
            k: v
            for k, v in vars(args).items()
            if k not in ['linkedin_password', 'proxy', 'api_key']
            and not k.startswith('_')
        }
        logger.debug(f"Command line arguments: {safe_args}")

        logger.info("Starting MyJobSpy AI")

        # Perform job search with the provided arguments
        return await search_jobs(args)

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        console.print(f"[red]Error: {str(e)}[/red]")
        return 1


def main() -> int:
    """Main entry point for the application.

    Returns:
        int: Exit code (0 for success, non-zero for error)
    """
    # Set up initial console logging
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console.setFormatter(formatter)
    logging.basicConfig(handlers=[console], level=logging.INFO)

    # Get logger
    logger = logging.getLogger(__name__)

    try:
        # Parse command line arguments
        args = parse_args(sys.argv[1:])

        # Set up debug mode from args or environment
        debug_mode = getattr(args, 'debug', False) or os.environ.get(
            'MYJOBSPYAI_DEBUG', ''
        ).lower() in ('1', 'true', 'yes')

        # Configure logging with debug mode
        setup_logging_custom(debug_mode)

        # Get logger after setup
        logger = logging.getLogger(__name__)

        # Log startup information
        logger.info("=" * 50)
        logger.info(f"Starting MyJobSpy AI (Debug: {debug_mode})")
        logger.info("-" * 50)

    except Exception as e:
        logger.error(f"Failed to initialize application: {str(e)}", exc_info=True)
        return 1

    # Initialize the application
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("\nApplication interrupted by user")
        return 0
    except Exception as e:
        logger.exception("Fatal error in main function")
        print(f"Fatal error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
