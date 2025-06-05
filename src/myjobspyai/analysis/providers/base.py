"""Base classes for LLM providers with common functionality."""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, Type, TypeVar

from opentelemetry import metrics, trace
from opentelemetry.trace import Span, Status, StatusCode

# Type variable for the response type
T = TypeVar('T')

logger = logging.getLogger(__name__)


class ProviderError(Exception):
    """Base exception for provider-related errors."""

    def __init__(
        self,
        message: str,
        provider: Optional[str] = None,
        error_type: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the error.

        Args:
            message: Error message
            provider: Name of the provider that raised the error
            error_type: Type of error (e.g., 'api_error', 'validation_error')
            status_code: HTTP status code, if applicable
            details: Additional error details
        """
        self.message = message
        self.provider = provider
        self.error_type = error_type or 'provider_error'
        self.status_code = status_code
        self.details = details or {}

        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the error message."""
        parts = []
        if self.provider:
            parts.append(f"Provider '{self.provider}':")
        parts.append(self.message)
        if self.status_code:
            parts.append(f"(Status: {self.status_code})")
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the error to a dictionary."""
        return {
            'message': str(self),
            'provider': self.provider,
            'error_type': self.error_type,
            'status_code': self.status_code,
            'details': self.details
        }

    def __str__(self) -> str:
        """Return a string representation of the error."""
        return self._format_message()

    def __repr__(self) -> str:
        """Return a detailed string representation of the error."""
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"provider='{self.provider}', "
            f"error_type='{self.error_type}', "
            f"status_code={self.status_code}, "
            f"details={self.details}"
            ")"
        )


class BaseProvider(ABC, Generic[T]):
    """Base class for all LLM providers with common functionality.

    This class provides a consistent interface for all LLM providers, including
    configuration management, error handling, and telemetry.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        provider_name: Optional[str] = None,
        **kwargs
    ) -> None:
        """Initialize the provider with configuration.

        Args:
            config: Provider-specific configuration
            provider_name: Name of the provider for logging and tracing
            **kwargs: Additional keyword arguments
                - name: Alias for provider_name (for backward compatibility)
                - provider_type: Type of the provider (defaults to class name)
        """
        self.config = config
        self.provider_name = provider_name or kwargs.get('name') or self.__class__.__name__
        self.provider_type = kwargs.get('provider_type', self.provider_name.lower())

        # Initialize telemetry
        self.tracer = trace.get_tracer(__name__)
        self.meter = metrics.get_meter(__name__)

        # Initialize metrics
        self._initialize_metrics()

        # Initialize the provider
        self._initialize_provider()

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        **kwargs: Any
    ) -> T:
        """Generate a response to the given prompt.

        Args:
            prompt: The input prompt
            **kwargs: Additional arguments specific to the provider

        Returns:
            The generated response

        Raises:
            ProviderError: If there's an error generating the response
        """
        pass

    def get_config_value(
        self,
        key: str,
        default: Any = None,
        required: bool = False
    ) -> Any:
        """Get a value from the config, supporting dot notation for nested keys.

        Args:
            key: The key to retrieve, can use dot notation for nested keys
            default: Default value to return if key is not found
            required: If True, raises KeyError if the key is not found

        Returns:
            The value from the config, or default if not found and not required

        Raises:
            KeyError: If the key is not found and required is True
        """
        keys = key.split('.')
        value = self.config

        try:
            for k in keys:
                if not isinstance(value, dict) or k not in value:
                    if required:
                        raise KeyError(f"Required config key not found: {key}")
                    return default
                value = value[k]
            return value
        except (KeyError, AttributeError) as e:
            if required:
                raise KeyError(f"Error accessing config key {key}: {e}") from e
            return default

    def _initialize_metrics(self) -> None:
        """Initialize OpenTelemetry metrics for this provider."""
        # Request metrics
        self.request_counter = self.meter.create_counter(
            name=f"{self.provider_type}.requests.total",
            description=f"Total number of {self.provider_type} API requests"
        )

        self.request_duration = self.meter.create_histogram(
            name=f"{self.provider_type}.request.duration.seconds",
            description=f"Duration of {self.provider_type} API requests in seconds",
            unit="s"
        )

        self.error_counter = self.meter.create_counter(
            name=f"{self.provider_type}.errors.total",
            description=f"Total number of {self.provider_type} API errors"
        )

        # Token usage metrics
        self.token_counter = self.meter.create_counter(
            name=f"{self.provider_type}.tokens.total",
            description=f"Total number of tokens used by {self.provider_type}",
            unit="tokens"
        )

    def _initialize_provider(self) -> None:
        """Initialize the provider with the given configuration.

        Subclasses should override this method to perform any necessary
        initialization, such as setting up API clients.
        """
        pass

    async def close(self) -> None:
        """Clean up resources used by the provider.

        Subclasses should override this method to clean up any resources,
        such as closing HTTP sessions or database connections.
        """
        pass

    def __str__(self) -> str:
        """Return a string representation of the provider."""
        return f"{self.__class__.__name__}(name='{self.provider_name}')"

    def __repr__(self) -> str:
        """Return a detailed string representation of the provider."""
        return f"<{self.__class__.__name__} name='{self.provider_name}' type='{self.provider_type}'>"


class SyncProvider(BaseProvider[T], ABC):
    """Base class for synchronous LLM providers.

    This class provides a synchronous interface that wraps the asynchronous
    methods of the base provider.
    """

    def generate_sync(
        self,
        prompt: str,
        **kwargs: Any
    ) -> T:
        """Synchronous version of generate.

        Args:
            prompt: The input prompt
            **kwargs: Additional arguments specific to the provider

        Returns:
            The generated response

        Raises:
            ProviderError: If there's an error generating the response
        """
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.generate(prompt, **kwargs))


class BaseLLMProvider(BaseProvider[str], ABC):
    """Base class for LLM providers with common functionality.

    This class provides a consistent interface for all LLM providers, including
    configuration management, error handling, and telemetry.
    """

    @abstractmethod
    async def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """Generate text using the provider's LLM.

        Args:
            prompt: The prompt to generate text from
            model: Override the default model
            **kwargs: Additional provider-specific arguments

        Returns:
            Generated text
        """
        pass

    def _start_span(
        self, name: str, attributes: Optional[Dict[str, Any]] = None
    ) -> Span:
        """Start a new OpenTelemetry span.

        Args:
            name: Name of the span
            attributes: Optional attributes to add to the span

        Returns:
            The started span
        """
        return self.tracer.start_span(
            name=name, attributes=attributes or {}, kind=trace.SpanKind.CLIENT
        )

    def _end_span(
        self,
        span: Span,
        status: StatusCode = StatusCode.OK,
        error: Optional[Exception] = None,
    ):
        """End a span with the given status and optional error.

        Args:
            span: The span to end
            status: Status code for the span
            error: Optional exception that occurred
        """
        if error:
            span.record_exception(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
        else:
            span.set_status(Status(status))
        span.end()

    def _log_llm_call(
        self,
        prompt: str,
        response: str,
        duration: float,
        error: Optional[Exception] = None,
        **kwargs,
    ):
        """Log an LLM call with relevant metrics and telemetry.

        This method provides detailed logging and telemetry for LLM calls, including:
        - Basic call information (duration, model, provider)
        - Token usage metrics (if available)
        - Error tracking (if any)
        - Response characteristics

        Args:
            prompt: The prompt sent to the LLM (may be truncated for logging)
            response: The response from the LLM (may be truncated for logging)
            duration: Duration of the call in seconds
            error: Optional exception that occurred
            **kwargs: Additional metadata including:
                - model: Model name
                - prompt_tokens: Number of tokens in the prompt
                - completion_tokens: Number of tokens in the completion
                - total_tokens: Total tokens used
                - output_schema: Output schema if used
        """
        try:
            # Prepare base log data
            model = kwargs.get("model") or self.config.get("model")
            prompt_length = len(prompt) if prompt else 0
            response_length = len(response) if response else 0

            # Create tags for metrics
            tags = {
                "provider": self.provider_name,
                "model": model or "unknown",
                "has_error": str(error is not None).lower(),
                "has_schema": str("output_schema" in kwargs).lower(),
            }

            # Log basic call info
            if error:
                log_level = logging.ERROR
                log_msg = f"LLM call failed after {duration:.2f}s"
            else:
                log_level = logging.INFO
                log_msg = f"LLM call completed in {duration:.2f}s"

            # Prepare log data with sanitized values
            log_data = {
                "provider": self.provider_name,
                "model": model,
                "duration_seconds": round(duration, 2),
                "prompt_length": prompt_length,
                "response_length": response_length,
                "has_error": error is not None,
                "timestamp": datetime.utcnow().isoformat(),
                **{
                    k: v
                    for k, v in kwargs.items()
                    if v is not None and not k.startswith("_")
                },
            }

            # Log the main event
            logger.log(
                log_level,
                log_msg,
                extra={"llm_call": log_data, "error": str(error) if error else None},
            )

            # Log detailed debug info if enabled
            if logger.isEnabledFor(logging.DEBUG):
                prompt_preview = prompt[:200] + ("..." if len(prompt) > 200 else "")
                response_preview = response[:200] + (
                    "..." if response and len(response) > 200 else ""
                )

                logger.debug(
                    "LLM call details",
                    extra={
                        "prompt_preview": prompt_preview,
                        "response_preview": response_preview,
                        **log_data,
                    },
                )

            # Update metrics if available
            try:
                if hasattr(self, "llm_calls_counter"):
                    self.llm_calls_counter.add(1, tags)

                if hasattr(self, "llm_call_duration"):
                    self.llm_call_duration.record(duration, tags)

                # Record token usage if available
                if hasattr(self, "prompt_tokens") and "prompt_tokens" in kwargs:
                    self.prompt_tokens.record(kwargs["prompt_tokens"], tags)
                if hasattr(self, "completion_tokens") and "completion_tokens" in kwargs:
                    self.completion_tokens.record(kwargs["completion_tokens"], tags)
                if hasattr(self, "total_tokens") and "total_tokens" in kwargs:
                    self.total_tokens.record(kwargs["total_tokens"], tags)

            except Exception as metric_error:
                logger.warning(
                    f"Failed to update metrics: {metric_error}", exc_info=True
                )

        except Exception as log_error:
            # Don't let logging errors break the main flow
            logger.error(f"Error in _log_llm_call: {log_error}", exc_info=True)

        if error:
            log_data["error"] = str(error)
            logger.error(f"LLM call failed: {error}", extra=log_data)
        else:
            logger.debug("LLM call completed", extra=log_data)
