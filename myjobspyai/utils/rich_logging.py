"""Rich console logging and progress utilities with exception handling.

This module provides a centralized way to handle logging, progress bars, and exceptions
using the Rich library for beautiful console output.
"""
from __future__ import annotations

import logging
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Type, TypeVar, Union

from rich.console import Console, ConsoleRenderable, Group
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.syntax import Syntax
from rich.table import Column
from rich.text import Text

# Type variable for generic exception handling
T = TypeVar("T")


class RichLoggingConfig:
    """Configuration for Rich logging and progress bars."""

    def __init__(
        self,
        log_level: Union[int, str] = logging.INFO,
        log_format: str = "%(message)s",
        date_format: str = "[%X]",
        show_path: bool = True,
        show_time: bool = True,
        show_level: bool = True,
        rich_tracebacks: bool = True,
        traceback_show_locals: bool = True,
        traceback_extra_lines: int = 2,
        traceback_theme: str = "monokai",
        width: Optional[int] = None,
        color_system: Optional[str] = "auto",
    ):
        """Initialize Rich logging configuration.

        Args:
            log_level: Logging level (e.g., logging.INFO, "INFO")
            log_format: Log message format
            date_format: Date format for log messages
            show_path: Show file path in log messages
            show_time: Show time in log messages
            show_level: Show log level in log messages
            rich_tracebacks: Enable rich traceback formatting
            traceback_show_locals: Show local variables in tracebacks
            traceback_extra_lines: Number of extra lines to show in tracebacks
            traceback_theme: Color theme for tracebacks
            width: Console width (None for auto)
            color_system: Color system to use (None, "auto", "standard", "256", "truecolor", "windows")
        """
        self.log_level = log_level
        self.log_format = log_format
        self.date_format = date_format
        self.show_path = show_path
        self.show_time = show_time
        self.show_level = show_level
        self.rich_tracebacks = rich_tracebacks
        self.traceback_show_locals = traceback_show_locals
        self.traceback_extra_lines = traceback_extra_lines
        self.traceback_theme = traceback_theme
        self.width = width
        self.color_system = color_system


class RichProgress(Progress):
    """Custom progress bar with multiple columns and task tracking."""

    def __init__(self, config: Optional[RichLoggingConfig] = None, **kwargs):
        """Initialize the progress bar with custom columns.

        Args:
            config: Rich logging configuration
            **kwargs: Additional arguments for rich.progress.Progress
        """
        self.config = config or RichLoggingConfig()
        
        # Define progress bar columns
        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
        ]
        
        # Initialize the progress bar
        super().__init__(
            *columns,
            console=Console(color_system=self.config.color_system, width=self.config.width),
            **kwargs
        )


class RichLogger:
    """Rich logging and progress utility class."""

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[RichLoggingConfig] = None):
        """Initialize the Rich logger.

        Args:
            config: Rich logging configuration
        """
        if self._initialized:
            return

        self.config = config or RichLoggingConfig()
        self._setup_console()
        self._setup_logging()
        self._setup_progress()
        self._setup_exception_handler()
        self._initialized = True

    def _setup_console(self) -> None:
        """Set up the Rich console."""
        self.console = Console(
            color_system=self.config.color_system,
            width=self.config.width,
            log_time_format=self.config.date_format,
        )

    def _setup_logging(self) -> None:
        """Set up Python logging with Rich handler."""
        # Remove all existing handlers
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Configure rich handler
        rich_handler = RichHandler(
            console=self.console,
            markup=True,
            rich_tracebacks=self.config.rich_tracebacks,
            tracebacks_show_locals=self.config.traceback_show_locals,
            tracebacks_extra_lines=self.config.traceback_extra_lines,
            tracebacks_theme=self.config.traceback_theme,
            show_time=self.config.show_time,
            show_path=self.config.show_path,
            show_level=self.config.show_level,
        )

        # Configure root logger
        logging.basicConfig(
            level=self.config.log_level,
            format=self.config.log_format,
            datefmt=self.config.date_format,
            handlers=[rich_handler],
        )

        # Get logger for this module
        self.logger = logging.getLogger(__name__)

    def _setup_progress(self) -> None:
        """Set up the progress bar display."""
        self.progress = RichProgress(config=self.config)

    def _setup_exception_handler(self) -> None:
        """Set up global exception handler for uncaught exceptions."""
        if self.config.rich_tracebacks:
            import sys
            from rich.traceback import install
            
            install(
                console=self.console,
                show_locals=self.config.traceback_show_locals,
                extra_lines=self.config.traceback_extra_lines,
                theme=self.config.traceback_theme,
            )
            
            # Override sys.excepthook to use rich traceback
            def exception_handler(type_, value, traceback_):
                self.console.print(
                    "[bold red]Unhandled exception:[/]\n",
                    style="red",
                    highlight=False,
                )
                self.console.print(
                    Panel(
                        Syntax(
                            "\n".join(traceback.format_exception(type_, value, traceback_)),
                            "python",
                            theme=self.config.traceback_theme,
                            line_numbers=True,
                        ),
                        title="Traceback",
                        border_style="red",
                    )
                )
                
                # Log the exception
                self.logger.exception("Unhandled exception", exc_info=(type_, value, traceback_))
            
            sys.excepthook = exception_handler

    @contextmanager
    def progress_bar(self, **kwargs) -> Generator[RichProgress, None, None]:
        """Context manager for a progress bar.
        
        Args:
            **kwargs: Additional arguments for the progress bar
            
        Yields:
            RichProgress: The progress bar instance
        """
        try:
            with self.progress as progress:
                yield progress
        except Exception as e:
            self.logger.exception("Error in progress bar")
            raise

    def log_exception(
        self,
        exception: Exception,
        message: str = "An error occurred",
        level: int = logging.ERROR,
        **kwargs: Any,
    ) -> None:
        """Log an exception with rich formatting.
        
        Args:
            exception: The exception to log
            message: Custom message to include with the exception
            level: Logging level (default: ERROR)
            **kwargs: Additional context to include in the log
        """
        self.logger.log(
            level,
            f"{message}: {str(exception)}\n"
            f"Type: {type(exception).__name__}\n"
            f"Args: {exception.args}",
            exc_info=exception,
            extra={"extra": kwargs},
        )

    def print_panel(
        self,
        renderable: ConsoleRenderable,
        title: str = "",
        border_style: str = "blue",
        **kwargs: Any,
    ) -> None:
        """Print a renderable in a panel.
        
        Args:
            renderable: The content to display in the panel
            title: Panel title
            border_style: Panel border style
            **kwargs: Additional arguments for Panel
        """
        self.console.print(
            Panel(
                renderable,
                title=title,
                border_style=border_style,
                **kwargs,
            )
        )


# Global instance
rich_logger = RichLogger()


def get_rich_logger() -> RichLogger:
    """Get the global Rich logger instance.
    
    Returns:
        RichLogger: The global Rich logger instance
    """
    return rich_logger


def setup_rich_logging(config: Optional[RichLoggingConfig] = None) -> RichLogger:
    """Set up Rich logging with the given configuration.
    
    Args:
        config: Rich logging configuration
        
    Returns:
        RichLogger: The configured Rich logger instance
    """
    global rich_logger
    rich_logger = RichLogger(config)
    return rich_logger


def log_exception(
    exception: Exception,
    message: str = "An error occurred",
    level: int = logging.ERROR,
    **kwargs: Any,
) -> None:
    """Log an exception with rich formatting.
    
    Args:
        exception: The exception to log
        message: Custom message to include with the exception
        level: Logging level (default: ERROR)
        **kwargs: Additional context to include in the log
    """
    rich_logger.log_exception(exception, message, level, **kwargs)


@contextmanager
def progress_bar(**kwargs) -> Generator[RichProgress, None, None]:
    """Context manager for a progress bar.
    
    Args:
        **kwargs: Additional arguments for the progress bar
        
    Yields:
        RichProgress: The progress bar instance
    """
    with rich_logger.progress_bar(**kwargs) as progress:
        yield progress


def print_panel(
    renderable: ConsoleRenderable,
    title: str = "",
    border_style: str = "blue",
    **kwargs: Any,
) -> None:
    """Print a renderable in a panel.
    
    Args:
        renderable: The content to display in the panel
        title: Panel title
        border_style: Panel border style
        **kwargs: Additional arguments for Panel
    """
    rich_logger.print_panel(renderable, title, border_style, **kwargs)


# Example usage
if __name__ == "__main__":
    # Configure logging
    config = RichLoggingConfig(
        log_level=logging.DEBUG,
        rich_tracebacks=True,
        traceback_show_locals=True,
    )
    logger = setup_rich_logging(config)
    
    # Example logging
    logger.logger.info("This is an info message")
    logger.logger.warning("This is a warning")
    
    # Example progress bar
    import time
    
    with progress_bar() as progress:
        task = progress.add_task("Processing...", total=100)
        for i in range(100):
            time.sleep(0.02)  # Simulate work
            progress.update(task, advance=1)
    
    # Example exception handling
    try:
        1 / 0
    except Exception as e:
        log_exception(e, "Math error occurred")
