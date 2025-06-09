"""
Unified logging configuration for MyJobSpy AI.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

from rich.console import Console
from rich.logging import RichHandler

from ..config import config as app_config


class LoggingConfig:
    """Unified logging configuration handler."""

    def __init__(self, debug: bool = False):
        """Initialize logging configuration.

        Args:
            debug: Enable debug logging if True
        """
        self.debug = debug
        self.log_dir = Path(app_config.logging.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _get_log_level(self) -> str:
        """Get the appropriate log level based on debug mode."""
        return "DEBUG" if self.debug else app_config.logging.level.upper()

    def _get_handlers(self) -> Dict[str, Dict[str, Any]]:
        """Get configured handlers."""
        handlers = {
            "console": {
                "class": "rich.logging.RichHandler",
                "formatter": "console",
                "level": self._get_log_level(),
            },
        }

        # Add file handlers if configured
        if app_config.logging.file:
            handlers["file"] = {
                "class": "logging.FileHandler",
                "formatter": "detailed",
                "filename": str(app_config.logging.file),
                "level": self._get_log_level(),
            }

        return handlers

    def _get_formatters(self) -> Dict[str, Dict[str, Any]]:
        """Get configured formatters."""
        return {
            "console": {
                "format": "%(message)s",
                "datefmt": "[%X]",
            },
            "detailed": {
                "format": app_config.logging.format,
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        }

    def _get_loggers(self) -> Dict[str, Dict[str, Any]]:
        """Get configured loggers."""
        return {
            "": {  # root logger
                "handlers": ["console"],
                "level": self._get_log_level(),
                "propagate": True,
            },
            "myjobspyai": {
                "handlers": ["console"],
                "level": self._get_log_level(),
                "propagate": False,
            },
        }

    def setup(self) -> None:
        """Set up the logging configuration."""
        logging_config = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": self._get_formatters(),
            "handlers": self._get_handlers(),
            "loggers": self._get_loggers(),
        }

        try:
            logging.config.dictConfig(logging_config)
        except Exception as e:
            # Fallback to basic config if setup fails
            logging.basicConfig(
                level=self._get_log_level(),
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
            logger = logging.getLogger("myjobspyai")
            logger.error(f"Failed to configure logging: {e}", exc_info=True)
            logger.info("Falling back to basic logging configuration")

    @staticmethod
    def get_logger(name: Optional[str] = None) -> logging.Logger:
        """Get a named logger or root logger."""
        return logging.getLogger(name or "myjobspyai")

    @staticmethod
    def get_analysis_logger() -> logging.Logger:
        """Get the analysis logger."""
        return logging.getLogger("myjobspyai.analysis")

    @staticmethod
    def get_model_output_logger() -> logging.Logger:
        """Get the model output logger."""
        return logging.getLogger("myjobspyai.model")

    @staticmethod
    def get_request_logger() -> logging.Logger:
        """Get the request logger."""
        return logging.getLogger("myjobspyai.request")
