"""
MyJobSpyAI - AI-powered job search and analysis tool.

This package provides tools for job search, resume analysis, and job matching
using AI and machine learning techniques.
"""

import importlib.metadata
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import configuration at the module level
from .config import settings, configure


# Version handling
try:
    __version__ = importlib.metadata.version("myjobspyai")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"  # Fallback version for development

__author__ = "Kenneth Johnson <kennethjohnson521@gmail.com>"
__license__ = "MIT"
__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "settings",
    "configure",
    "config",
    "get_config",
    "update_config",
    "save_config",
    "setup_logging",
    "log_duration",
    "get_tracer",
]

# Suppress noisy warnings
warnings.filterwarnings(
    "ignore",
    message=".*was never awaited.*",
    category=RuntimeWarning,
)

# Lazy imports to improve startup time
_import_structure: Dict[str, List[str]] = {
    "utils.config_utils": ["config", "get_config", "update_config", "save_config"],
    "utils.logging_utils": ["setup_logging", "log_duration", "get_tracer"],
}


def __getattr__(name: str) -> Any:
    """Lazy import of module attributes.

    This function allows for lazy loading of submodules and their attributes,
    which can significantly improve import time for large packages.

    Args:
        name: The name of the attribute to import.

    Returns:
        The requested attribute.

    Raises:
        AttributeError: If the requested attribute is not found.
    """
    if name in _import_structure:
        module = importlib.import_module(f"myjobspyai.{name}")
        return module

    for module_path, attrs in _import_structure.items():
        if name in attrs:
            module = importlib.import_module(f"myjobspyai.{module_path}")
            return getattr(module, name)

    raise AttributeError(f"module 'myjobspyai' has no attribute '{name}'")


# Add settings to __all__
__all__.extend(['settings', 'configure'])

# Maintain backward compatibility with old config system
try:
    from .utils.config_utils import config, get_config, update_config, save_config
except ImportError:
    # Fallback implementations if config_utils is not available
    config = {}
    
    def get_config() -> Dict[str, Any]:
        return dict(config)
    
    def update_config(new_config: Dict[str, Any]) -> None:
        config.update(new_config)
    
    def save_config(path: Optional[Path] = None) -> None:
        pass

# Initialize logging when the package is imported
from .utils.logging_utils import setup_logging  # noqa: E402

# Setup basic logging if not already configured
if not logging.getLogger().hasHandlers():
    setup_logging()
# This file is intentionally left with just the exports and version information