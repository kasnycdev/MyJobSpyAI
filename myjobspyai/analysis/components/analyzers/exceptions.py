"""
Custom exceptions for the analysis components.

This module defines custom exceptions used throughout the analysis pipeline.
"""
from typing import Optional, Dict, Any, Union


class AnalysisError(Exception):
    """Base exception for analysis-related errors.
    
    This exception is raised when there is an error during the analysis process,
    such as invalid input, processing failures, or unexpected results.
    
    Attributes:
        message: A human-readable error message
        details: Optional dictionary with additional error details
        error_code: Optional error code for programmatic handling
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        error_code: Optional[Union[str, int]] = None,
        cause: Optional[Exception] = None
    ) -> None:
        """Initialize the AnalysisError.
        
        Args:
            message: A human-readable error message
            details: Optional dictionary with additional error details
            error_code: Optional error code for programmatic handling
            cause: The original exception that caused this error, if any
        """
        self.message = message
        self.details = details or {}
        self.error_code = error_code
        self.cause = cause
        
        # Format the error message with details if available
        if details:
            details_str = ", ".join(f"{k}={v!r}" for k, v in details.items())
            message = f"{message} ({details_str})"
        
        if error_code:
            message = f"[{error_code}] {message}"
            
        if cause:
            message = f"{message}. Caused by: {str(cause)}"
            
        super().__init__(message)


class ValidationError(AnalysisError):
    """Raised when input data fails validation."""
    pass


class ParsingError(AnalysisError):
    """Raised when there is an error parsing data."""
    pass


class ProviderError(AnalysisError):
    """Raised when there is an error with an LLM provider."""
    pass


class ConfigurationError(AnalysisError):
    """Raised when there is a configuration error."""
    pass


class RetryableError(AnalysisError):
    """Raised when an operation fails but can be retried."""
    
    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        max_retries: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """Initialize a retryable error.
        
        Args:
            message: Error message
            retry_after: Suggested time to wait before retrying (in seconds)
            max_retries: Maximum number of retries allowed
            **kwargs: Additional arguments to pass to AnalysisError
        """
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        self.max_retries = max_retries


class FatalError(AnalysisError):
    """Raised when a fatal error occurs that cannot be recovered from."""
    pass


class ResourceExhaustedError(AnalysisError):
    """Raised when a resource limit has been reached."""
    pass
