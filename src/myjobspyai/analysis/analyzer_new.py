"""
Analyzer module for MyJobSpyAI with support for multiple LLM providers.

This module provides the BaseAnalyzer class which serves as the foundation for
all analysis tasks, with built-in support for multiple LLM providers including
OpenAI, Ollama, Gemini, and LangChain backends.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

import google.api_core.exceptions
import google.generativeai as genai
import ollama

# Import LLM providers
import openai
from analysis.providers.base import BaseProvider as LLMProvider

# Import provider factory and base provider
from analysis.providers.factory import ProviderFactory

# Import Jinja2 components
try:
    from jinja2 import (
        Environment,
        FileSystemLoader,
        TemplateNotFound,
        TemplateSyntaxError,
        select_autoescape,
    )

    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

# Import models and utilities
from analysis.models import JobAnalysisResult, ParsedJobData, ResumeData
from config import settings

from myjobspyai.utils.logging_utils import MODEL_OUTPUT_LOGGER_NAME

# Get a logger for this module
logger = logging.getLogger(__name__)
model_output_logger = logging.getLogger(MODEL_OUTPUT_LOGGER_NAME)

# Import OpenTelemetry trace and metrics modules
from opentelemetry import metrics, trace

# Import tracer and meter instances from myjobspyai.utils.logging_utils
try:
    from myjobspyai.utils.logging_utils import meter as global_meter_instance
    from myjobspyai.utils.logging_utils import tracer as global_tracer_instance

    if (
        global_tracer_instance is None
    ):  # Check if OTEL was disabled in logging_utils for tracer
        tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
        logger.warning(
            "OpenTelemetry tracer not configured in logging_utils (global_tracer_instance is None), using NoOpTracer for analyzer."
        )
    else:
        tracer = global_tracer_instance  # Use the instance from logging_utils
        logger.info("Using global_tracer_instance from myjobspyai.utils.logging_utils for analyzer.")

    if (
        global_meter_instance is None
    ):  # Check if OTEL was disabled in logging_utils for meter
        meter = metrics.get_meter(__name__, meter_provider=metrics.NoOpMeterProvider())
        logger.warning(
            "OpenTelemetry meter not configured in logging_utils (global_meter_instance is None), using NoOpMeter for analyzer."
        )
    else:
        meter = global_meter_instance  # Use the instance from logging_utils
        logger.info("Using global_meter_instance from myjobspyai.utils.logging_utils for analyzer.")

except ImportError:
    # Fallback to NoOp versions if logging_utils or its tracer/meter cannot be imported
    tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
    meter = metrics.get_meter(__name__, meter_provider=metrics.NoOpMeterProvider())
    logger.error(
        "Could not import global_tracer_instance/global_meter_instance from myjobspyai.utils.logging_utils. Using NoOp versions for analyzer.",
        exc_info=True,
    )


# Helper function for logging exceptions
def log_exception(message, exception):
    logger.error(message, exc_info=True)


class BaseAnalyzer:
    """
    Base class for all analyzers with support for multiple LLM providers.

    This class provides common functionality for interacting with LLM providers,
    including initialization, connection checking, and request handling with
    retries and OpenTelemetry instrumentation.
    """

    def __init__(self, provider_name: Optional[str] = None):
        """Initialize the analyzer with the specified LLM provider.

        Args:
            provider_name: Optional name of the provider to use. If not specified,
                         uses the default provider from settings.
        """
        # Get provider configuration from settings
        llm_config = settings.get("llm", {})

        # Use specified provider or default from settings
        self.provider_name = provider_name or llm_config.get(
            "default_provider", "langchain_default"
        )
        self.provider_type = None
        self.llm_provider: Optional[LLMProvider] = None

        logger.info(f"Initializing LLM client for provider: {self.provider_name}")

        self.model_name: Optional[str] = None
        self.request_timeout: Optional[float] = None
        self.max_retries: int = 2
        self.retry_delay: int = 5

        with tracer.start_as_current_span("analyzer_init") as span:
            span.set_attribute("provider_name", self.provider_name)
            self._initialize_llm_provider()
            self._check_connection_and_model()

    # LLM Call Metrics
    llm_calls_counter = meter.create_counter(
        name="llm.calls.total", description="Total number of LLM calls.", unit="1"
    )

    llm_successful_calls_counter = meter.create_counter(
        name="llm.calls.successful",
        description="Number of successful LLM calls.",
        unit="1",
    )

    llm_failed_calls_counter = meter.create_counter(
        name="llm.calls.failed",
        description="Number of failed LLM calls (after retries).",
        unit="1",
    )

    llm_call_duration_histogram = meter.create_histogram(
        name="llm.call.duration", description="Duration of LLM calls.", unit="s"
    )

    llm_prompt_chars_histogram = meter.create_histogram(
        name="llm.prompt.chars",
        description="Number of characters in LLM prompts.",
        unit="char",
    )

    llm_response_chars_histogram = meter.create_histogram(
        name="llm.response.chars",
        description="Number of characters in LLM responses.",
        unit="char",
    )

    # Track call statistics (legacy, consider removing if using OTEL metrics exclusively)
    _llm_call_stats: Dict[str, Any] = {
        "total_calls": 0,
        "successful_calls": 0,
        "failed_calls": 0,
        "total_prompt_chars": 0,
        "total_response_chars": 0,
        "total_duration_seconds": 0.0,
        "errors_by_type": defaultdict(int),
        "calls_by_task": defaultdict(lambda: defaultdict(int)),
    }

    @classmethod
    def log_llm_call_summary(cls):
        """Log a summary of LLM call statistics."""
        stats = cls._llm_call_stats
        logger.info("\n=== LLM Call Summary ===")
        logger.info(f"Total Calls: {stats['total_calls']}")
        logger.info(f"Successful: {stats['successful_calls']}")
        logger.info(f"Failed: {stats['failed_calls']}")
        logger.info(f"Total Prompt Chars: {stats['total_prompt_chars']}")
        logger.info(f"Total Response Chars: {stats['total_response_chars']}")
        logger.info(f"Total Duration: {stats['total_duration_seconds']:.2f}s")

        if stats["errors_by_type"]:
            logger.info("\nErrors by Type:")
            for error_type, count in stats["errors_by_type"].items():
                logger.info(f"  {error_type}: {count}")

        if stats["calls_by_task"]:
            logger.info("\nCalls by Task:")
            for task, task_stats in stats["calls_by_task"].items():
                total = task_stats.get("total", 0)
                success = task_stats.get("success", 0)
                errors = task_stats.get("errors", 0)
                logger.info(
                    f"  {task}: {total} total, {success} success, {errors} errors"
                )

    def _initialize_llm_provider(self):
        """Initialize the LLM provider from configuration."""
        try:
            # Get the provider configuration
            llm_config = settings.get("llm", {})
            providers_config = llm_config.get("providers", {})

            if self.provider_name not in providers_config:
                raise ValueError(
                    f"Provider '{self.provider_name}' not found in configuration"
                )

            provider_config = providers_config[self.provider_name]
            self.provider_type = provider_config.get("type")

            if not self.provider_type:
                raise ValueError(
                    f"Provider '{self.provider_name}' is missing 'type' field"
                )

            # Initialize the provider using the factory
            self.llm_provider = ProviderFactory.create_provider(
                provider_type=self.provider_type,
                config=provider_config,
                name=self.provider_name,
            )

            # Set common attributes
            self.model_name = provider_config.get("model") or provider_config.get(
                "params", {}
            ).get("model_name")
            self.request_timeout = provider_config.get("request_timeout", 60.0)
            self.max_retries = provider_config.get("max_retries", 2)
            self.retry_delay = provider_config.get("retry_delay", 5)

            logger.info(
                f"Initialized {self.provider_type} provider '{self.provider_name}' with model: {self.model_name}"
            )

        except Exception as e:
            logger.error(
                f"Failed to initialize LLM provider '{self.provider_name}': {str(e)}"
            )
            raise

    def _check_connection_and_model(self):
        """
        Check the connection to the LLM provider and verify the model is available.
        For LangChain providers, we'll do a simple test generation to verify connectivity.
        """
        with tracer.start_as_current_span("check_connection_and_model") as span:
            span.set_attributes(
                {
                    "provider_name": self.provider_name,
                    "provider_type": self.provider_type or "unknown",
                    "model": self.model_name or "unknown",
                }
            )

            if not self.llm_provider:
                raise RuntimeError("LLM provider not initialized")

            # For LangChain providers, we'll do a simple test generation
            try:
                # Use a simple test prompt to verify connectivity
                test_prompt = "Test connection - respond with 'OK'"

                # For synchronous check, we'll use asyncio to run the async method
                async def test_connection():
                    response = await self.llm_provider.generate(test_prompt)
                    return response

                # Run the async test
                response = asyncio.run(test_connection())

                if not response or not response.strip():
                    raise ValueError("Empty response from LLM provider")

                logger.info(
                    f"Successfully connected to {self.provider_name} provider with model: {self.model_name}"
                )

            except Exception as e:
                error_msg = (
                    f"Failed to connect to {self.provider_name} provider: {str(e)}"
                )
                logger.error(error_msg, exc_info=True)
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
                raise

    async def _call_llm_async(self, prompt: str, task_name: str, **kwargs):
        """
        Call the LLM asynchronously with retry logic and tracing.

        Args:
            prompt: The prompt to send to the LLM
            task_name: Name of the task for logging and metrics
            **kwargs: Additional arguments to pass to the provider

        Returns:
            The LLM response as a string
        """
        if not self.llm_provider:
            raise RuntimeError("LLM provider not initialized")

        span = tracer.start_span(f"llm_call_{task_name}")
        span.set_attributes(
            {
                "llm.provider": self.provider_name,
                "llm.provider_type": self.provider_type or "unknown",
                "llm.model": self.model_name or "unknown",
                "llm.prompt_length": len(prompt),
                "llm.task": task_name,
            }
        )

        # Log the prompt (truncated for logging)
        logger.debug(
            f"Sending prompt to {self.provider_name} ({task_name}): {prompt[:200]}..."
        )

        # Record metrics for the prompt
        self.llm_prompt_chars_histogram.record(
            len(prompt),
            {
                "provider": self.provider_name,
                "provider_type": self.provider_type or "unknown",
                "task": task_name,
            },
        )

        # Track call stats
        self._llm_call_stats["total_calls"] += 1
        self._llm_call_stats["calls_by_task"][task_name]["total"] += 1

        start_time = time.time()
        last_error = None

        for attempt in range(
            self.max_retries + 1
        ):  # +1 because we try once plus max_retries
            try:
                # Call the provider's generate method
                response = await self.llm_provider.generate(
                    prompt=prompt, model=self.model_name, **kwargs
                )

                # If we get here, the call was successful
                duration = time.time() - start_time

                # Record metrics
                self.llm_successful_calls_counter.add(
                    1,
                    {
                        "provider": self.provider_name,
                        "provider_type": self.provider_type or "unknown",
                        "task": task_name,
                    },
                )

                self.llm_call_duration_histogram.record(
                    duration,
                    {
                        "provider": self.provider_name,
                        "provider_type": self.provider_type or "unknown",
                        "task": task_name,
                    },
                )

                self.llm_response_chars_histogram.record(
                    len(response),
                    {
                        "provider": self.provider_name,
                        "provider_type": self.provider_type or "unknown",
                        "task": task_name,
                    },
                )

                # Update call stats
                self._llm_call_stats["successful_calls"] += 1
                self._llm_call_stats["calls_by_task"][task_name]["success"] += 1
                self._llm_call_stats["total_prompt_chars"] += len(prompt)
                self._llm_call_stats["total_response_chars"] += len(response)
                self._llm_call_stats["total_duration_seconds"] += duration

                # Log the response (truncated)
                logger.debug(
                    f"Received response from {self.provider_name} ({task_name}) in {duration:.2f}s: {response[:200]}..."
                )

                # Log the full prompt and response to the model output logger
                model_output_logger.info(
                    "\n=== LLM CALL ===\n"
                    f"Task: {task_name}\n"
                    f"Provider: {self.provider_name} ({self.provider_type or 'unknown'})\n"
                    f"Model: {self.model_name or 'default'}\n"
                    f"Duration: {duration:.2f}s\n"
                    "\n--- PROMPT ---\n"
                    f"{prompt}\n"
                    "\n--- RESPONSE ---\n"
                    f"{response}\n"
                    "=== END LLM CALL ===\n"
                )

                # Set span attributes
                span.set_attributes(
                    {
                        "llm.response_length": len(response),
                        "llm.duration_ms": duration * 1000,
                        "llm.retry_attempts": attempt,
                        "llm.success": True,
                    }
                )

                return response

            except Exception as e:
                last_error = e
                error_type = type(e).__name__

                # Update error stats
                self._llm_call_stats["calls_by_task"][task_name]["errors"] += 1
                self._llm_call_stats["errors_by_type"][error_type] += 1

                # Log the error
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{self.max_retries + 1}): {str(e)}",
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )

                # Record the error in the span
                span.record_exception(e)

                # If we've exhausted retries, record the final error and re-raise
                if attempt >= self.max_retries:
                    self.llm_failed_calls_counter.add(
                        1,
                        {
                            "provider": self.provider_name,
                            "provider_type": self.provider_type or "unknown",
                            "task": task_name,
                            "error_type": error_type,
                        },
                    )

                    self._llm_call_stats["failed_calls"] += 1

                    span.set_attributes(
                        {
                            "llm.error": str(e),
                            "llm.error_type": error_type,
                            "llm.retry_attempts": attempt,
                            "llm.success": False,
                        }
                    )

                    # Log the failed call details
                    model_output_logger.error(
                        "\n=== LLM CALL FAILED ===\n"
                        f"Task: {task_name}\n"
                        f"Provider: {self.provider_name} ({self.provider_type or 'unknown'})\n"
                        f"Model: {self.model_name or 'default'}\n"
                        f"Error: {error_type}: {str(e)}\n"
                        "\n--- PROMPT ---\n"
                        f"{prompt}\n"
                        "=== END FAILED LLM CALL ===\n"
                    )

                    raise

                # Wait before retrying
                await asyncio.sleep(
                    self.retry_delay * (attempt + 1)
                )  # Exponential backoff

        # This should never be reached due to the raise above, but just in case
        span.end()
        raise RuntimeError("Unexpected error in _call_llm_async")
