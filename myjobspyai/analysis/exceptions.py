"""
Custom exceptions for the analysis system.
"""

from typing import Optional

class AnalysisError(Exception):
    """Base class for analysis-related exceptions."""
    pass

class ResumeProcessingError(AnalysisError):
    """Raised when there's an error processing a resume."""
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.original_exception = original_exception

class JobProcessingError(AnalysisError):
    """Raised when there's an error processing a job."""
    def __init__(self, message: str, job_title: str, original_exception: Optional[Exception] = None):
        super().__init__(f"Job '{job_title}': {message}")
        self.job_title = job_title
        self.original_exception = original_exception

class LLMError(AnalysisError):
    """Raised when there's an error with LLM calls."""
    def __init__(self, message: str, provider: Optional[str] = None, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.provider = provider
        self.original_exception = original_exception

class FilterError(AnalysisError):
    """Raised when there's an error applying filters."""
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.original_exception = original_exception

class ConfigurationError(AnalysisError):
    """Raised when there's a configuration error."""
    def __init__(self, message: str, config_key: Optional[str] = None, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.config_key = config_key
        self.original_exception = original_exception


class ProviderError(AnalysisError):
    """Raised when there's an error with a provider."""
    def __init__(self, message: str, provider: Optional[str] = None, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.provider = provider
        self.original_exception = original_exception
