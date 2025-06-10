"""Logging configuration for MyJobSpyAI."""

import logging
import logging.handlers
from pathlib import Path
from typing import Any, Dict, Optional

from pythonjsonlogger import jsonlogger

from . import settings


def setup_logging() -> None:
    """Configure logging based on settings."""
    log_config = settings.settings.logging
    log_dir = Path(log_config.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_level = getattr(logging, log_config.log_level.upper())
    log_file_mode = log_config.log_file_mode

    # Base formatter
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(levelname)s %(name)s %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%SZ',
        json_ensure_ascii=False,
        json_indent=2 if settings.settings.debug else None,
    )

    # Configure each logger
    for name, logger_config in log_config.files.items():
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, logger_config['level'].upper()))

        # Clear existing handlers
        logger.handlers = []

        # Create file handler
        file_path = log_dir / logger_config['path']
        if log_config.rolling_strategy == 'size':
            handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=log_config.max_size,
                backupCount=log_config.backup_count,
                mode=log_file_mode,
            )
        else:  # time-based rotation
            handler = logging.handlers.TimedRotatingFileHandler(
                file_path,
                when='midnight',
                interval=1,
                backupCount=log_config.backup_count,
            )

        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # Add console handler in development
        if settings.settings.debug:
            console = logging.StreamHandler()
            console.setLevel(logging.DEBUG)
            console.setFormatter(formatter)
            logger.addHandler(console)

    # Set root logger level
    logging.getLogger().setLevel(log_level)

    # Configure third-party loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)

    # Log configuration
    logger = logging.getLogger('app')
    logger.info("Logging configured", extra={"config": log_config.model_dump()})
