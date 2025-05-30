"""
Test script to verify logging functionality.
"""
import logging
from pathlib import Path
from myjobspyai.utils.logging_utils import setup_logging

def test_logging():
    """Test logging configuration."""
    # Setup logging
    setup_logging()
    
    # Get logger
    logger = logging.getLogger(__name__)
    
    # Test log messages
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    # Check if log files were created
    project_root = Path(__file__).parent
    logs_dir = project_root / "logs"
    debug_log = logs_dir / "debug.log"
    error_log = logs_dir / "error.log"
    
    print(f"Project root: {project_root}")
    print(f"Logs directory: {logs_dir}")
    print(f"Debug log exists: {debug_log.exists()}")
    print(f"Error log exists: {error_log.exists()}")
    
    if debug_log.exists():
        print("\nDebug log contents:")
        print(debug_log.read_text(encoding='utf-8'))
    
    if error_log.exists():
        print("\nError log contents:")
        print(error_log.read_text(encoding='utf-8'))

if __name__ == "__main__":
    test_logging()
