"""MyJobSpy AI - A powerful job search and analysis tool."""

__version__ = "0.1.0"
__author__ = "Your Name <your.email@example.com>"
__license__ = "MIT"

# Import key components to make them available at the package level
from .config import AppConfig, load_config  # noqa: F401

# Note: Don't import main here to avoid circular imports
__all__ = ["AppConfig", "load_config"]
