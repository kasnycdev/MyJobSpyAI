"""OpenTelemetry integration for LangChain.

This module provides utilities for instrumenting LangChain with OpenTelemetry
to collect traces, metrics, and logs.
"""

import inspect
import logging
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, cast

from opentelemetry import metrics, trace

T = TypeVar("T")


def with_otel_tracing(
    span_name: Optional[str] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to add OpenTelemetry tracing to a function.

    Args:
        span_name: Name for the span. If None, uses the function name.
        attributes: Additional attributes to add to the span.

    Returns:
        Decorated function with OTEL tracing.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Get the tracer for the current module
            tracer = trace.get_tracer(func.__module__)

            # Use provided span name or function name
            name = span_name or f"{func.__module__}.{func.__name__}"

            # Create attributes from function arguments
            func_attrs = {}
            if attributes:
                func_attrs.update(attributes)

            # Add parameter values as attributes
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            for param, value in bound_args.arguments.items():
                # Skip 'self' and long values
                if param == "self" or value is None:
                    continue

                # Convert value to string and truncate if too long
                str_value = str(value)
                if len(str_value) > 256:
                    str_value = str_value[:253] + "..."

                func_attrs[f"langchain.{param}"] = str_value

            # Start a new span
            with tracer.start_as_current_span(
                name=name, kind=trace.SpanKind.CLIENT, attributes=func_attrs
            ) as span:
                try:
                    # Execute the function
                    result = func(*args, **kwargs)

                    # Record the result if it's a string (e.g., LLM response)
                    if isinstance(result, str):
                        span.set_attribute(
                            SpanAttributes.LLM_CONTENT,
                            result[:1000],  # Truncate long responses
                        )

                    return result

                except Exception as e:
                    # Record the exception and re-raise
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    raise

        return wrapper

    return decorator


class LangChainOTELMetrics:
    """OpenTelemetry metrics for LangChain operations."""

    def __init__(self, meter: metrics.Meter):
        """Initialize the metrics collector.

        Args:
            meter: OpenTelemetry meter instance
        """
        self.meter = meter

        # Initialize counters
        self.llm_calls = self.meter.create_counter(
            name="langchain.llm.calls",
            description="Total number of LLM calls",
            unit="1",
        )

        self.llm_errors = self.meter.create_counter(
            name="langchain.llm.errors", description="Number of LLM errors", unit="1"
        )

        # Initialize histograms
        self.llm_duration = self.meter.create_histogram(
            name="langchain.llm.duration",
            description="Duration of LLM calls in seconds",
            unit="s",
        )

        self.llm_prompt_tokens = self.meter.create_histogram(
            name="langchain.llm.prompt_tokens",
            description="Number of tokens in prompts",
            unit="token",
        )

        self.llm_completion_tokens = self.meter.create_histogram(
            name="langchain.llm.completion_tokens",
            description="Number of tokens in completions",
            unit="token",
        )

    def record_llm_call(
        self,
        model: str,
        duration: float,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        error: bool = False,
    ) -> None:
        """Record metrics for an LLM call.

        Args:
            model: Name of the model used
            duration: Duration of the call in seconds
            prompt_tokens: Number of tokens in the prompt
            completion_tokens: Number of tokens in the completion
            error: Whether the call resulted in an error
        """
        attributes = {"model": model}

        # Record the call
        self.llm_calls.add(1, attributes=attributes)
        self.llm_duration.record(duration, attributes=attributes)

        # Record tokens if available
        if prompt_tokens is not None:
            self.llm_prompt_tokens.record(prompt_tokens, attributes=attributes)

        if completion_tokens is not None:
            self.llm_completion_tokens.record(completion_tokens, attributes=attributes)

        # Record error if applicable
        if error:
            self.llm_errors.add(1, attributes=attributes)
