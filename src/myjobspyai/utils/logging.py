"""Logging configuration for MyJobSpy AI."""

import logging
import logging.config
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Union

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    log_level: Union[str, int] = logging.INFO,
    log_file: Optional[Union[str, Path]] = None,
    log_format: str = "%(message)s",
    date_format: str = "[%X]",
) -> None:
    """Configure logging for the application.

    Args:
        log_level: Logging level (e.g., 'DEBUG', 'INFO', 'WARNING').
        log_file: Optional file path to write logs to.
        log_format: Format string for log messages.
        date_format: Format string for timestamps.
    """
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper())

    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # Configure console handler with Rich
    console_handler = RichHandler(
        console=Console(stderr=True),
        markup=True,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
    )
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    root_logger.addHandler(console_handler)

    # Configure file handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt=date_format,
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Configure third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    # Set up a default logger for this module
    logger = logging.getLogger(__name__)
    logger.info("Logging configured at level %s", logging.getLevelName(log_level))
