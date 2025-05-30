"""Custom exceptions for MyJobSpyAI."""

class LLMError(Exception):
    """Base exception for LLM-related errors."""
    pass

class RateLimitExceeded(LLMError):
    """Raised when rate limits are exceeded."""
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = None):
        self.retry_after = retry_after
        if retry_after:
            message = f"{message}. Please try again in {retry_after} seconds."
        super().__init__(message)

class ModelUnavailable(LLMError):
    """Raised when the specified model is not available."""
    pass

class ConfigurationError(LLMError):
    """Raised when there is a configuration error."""
    pass

class AuthenticationError(LLMError):
    """Raised when authentication fails."""
    pass

class TimeoutError(LLMError):
    """Raised when a request times out."""
    pass

class RetryError(LLMError):
    """Raised when maximum retries are exceeded."""
    def __init__(self, message: str = "Maximum retries exceeded", last_exception: Exception = None):
        self.last_exception = last_exception
        if last_exception:
            message = f"{message}: {str(last_exception)}"
        super().__init__(message)

class ProviderError(LLMError):
    """Raised when there is an error with a provider."""
    pass

class ServiceUnavailable(LLMError):
    """Raised when a service is unavailable."""
    pass
