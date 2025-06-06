"""Logging utilities for the MyJobSpyAI application."""

import logging
import logging.config
import os
from pathlib import Path
from typing import Dict, Optional, Union

from opentelemetry import metrics, trace

from myjobspyai.config import config

# Initialize OpenTelemetry meter and tracer
meter = metrics.get_meter(__name__)
tracer = trace.get_tracer(__name__)

# Logger names
offset = 0
MODEL_OUTPUT_LOGGER_NAME = "model_output"
REQUEST_LOGGER_NAME = "request"
ANALYSIS_LOGGER_NAME = "analysis"

# Default log format
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    log_level: Optional[Union[str, int]] = None,
    log_file: Optional[Union[str, Path]] = None,
    log_format: Optional[str] = None,
    date_format: Optional[str] = None,
) -> None:
    """
    Set up logging configuration for the application.

    Args:
        log_level: The log level to use (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Path to the log file
        log_format: Format string for log messages
        date_format: Format string for dates in log messages
    """
    log_level = log_level or getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    log_format = log_format or DEFAULT_LOG_FORMAT
    date_format = date_format or DEFAULT_DATE_FORMAT

    # Default logging configuration
    logging_config: Dict = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": log_format,
                "datefmt": date_format,
            },
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s",
                "datefmt": date_format,
            },
        },
        "handlers": {
            "console": {
                "level": log_level,
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "": {  # root logger
                "handlers": ["console"],
                "level": log_level,
                "propagate": True,
            },
            "myjobspyai": {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
            MODEL_OUTPUT_LOGGER_NAME: {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
            REQUEST_LOGGER_NAME: {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
            ANALYSIS_LOGGER_NAME: {
                "handlers": ["console"],
                "level": log_level,
                "propagate": False,
            },
        },
    }

    # Add file handler if log_file is provided
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logging_config["handlers"]["file"] = {
            "level": log_level,
            "class": "logging.handlers.RotatingFileHandler",
            "filename": str(log_file),
            "maxBytes": 10 * 1024 * 1024,  # 10MB
            "backupCount": 5,
            "formatter": "detailed",
            "encoding": "utf8",
        }
        for logger in logging_config["loggers"].values():
            logger["handlers"].append("file")

    # Apply the logging configuration
    logging.config.dictConfig(logging_config)

    # Set log level for third-party libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: The name of the logger. If None, returns the root logger.

    Returns:
        A configured logger instance.
    """
    return logging.getLogger(name)


def get_model_output_logger() -> logging.Logger:
    """Get the model output logger."""
    return get_logger(MODEL_OUTPUT_LOGGER_NAME)


def get_request_logger() -> logging.Logger:
    """Get the request logger."""
    return get_logger(REQUEST_LOGGER_NAME)


def get_analysis_logger() -> logging.Logger:
    """Get the analysis logger."""
    return get_logger(ANALYSIS_LOGGER_NAME)
