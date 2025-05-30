"""
Custom exception classes for configuration management.
"""

class ConfigurationError(Exception):
    """Base exception for configuration errors."""
    pass

class ValidationError(ConfigurationError):
    """Raised when configuration validation fails."""
    pass

class MissingRequiredFieldError(ValidationError):
    """Raised when a required configuration field is missing."""
    pass

class InvalidValueError(ValidationError):
    """Raised when a configuration value is invalid."""
    pass
