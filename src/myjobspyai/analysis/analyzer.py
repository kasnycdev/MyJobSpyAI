import openai
import logging

# Standard library imports
import asyncio
import inspect
import json
from traceback import format_exc
from typing import Dict, List, Any, Optional, Union
import re
import os
import time
from collections import defaultdict
from contextlib import suppress
from typing import Dict, Optional, Any, Union, List, Type, TYPE_CHECKING

# Third-party imports
import google.api_core.exceptions  # Added for DeadlineExceeded
import google.generativeai as genai  # Corrected import alias
import ollama

# cspell:ignore generativeai, genai, ollama, Jinja2, autoescape, lstrip_blocks, api_core

# Import provider factory and base provider
from myjobspyai.analysis.providers.factory import ProviderFactory
from myjobspyai.analysis.providers.base import BaseProvider as LLMProvider

if TYPE_CHECKING:
    from myjobspyai.analysis.providers import LangChainProvider  # For type checking only

# Import Jinja2 components
try:
    from jinja2 import (
        Environment,
        FileSystemLoader,
        select_autoescape,
        TemplateNotFound,
        TemplateSyntaxError,
    )

    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

from myjobspyai.analysis.models import ResumeData, JobAnalysisResult, ParsedJobData, SkillDetail
from myjobspyai.filtering.filter_utils import DateEncoder
from myjobspyai.config import config
from myjobspyai.utils.logging_utils import (
    MODEL_OUTPUT_LOGGER_NAME,
    tracer as global_tracer_instance,
    meter as global_meter_instance,
)  # Import the constant for model output logger

# Get a logger for this module
logger = logging.getLogger(__name__)
# Get the specific logger for model outputs
model_output_logger = logging.getLogger(MODEL_OUTPUT_LOGGER_NAME)

# Import OpenTelemetry trace and metrics modules
from opentelemetry import (
    trace,
    metrics,
)  # Ensure trace and metrics modules are always imported

# Import tracer and meter instances from myjobspyai.utils.logging_utils
try:
    from myjobspyai.utils.logging_utils import (
        tracer as global_tracer_instance,
        meter as global_meter_instance,
    )

    # Use the global tracer instance if available
    if global_tracer_instance is not None:
        tracer = global_tracer_instance  # Use the instance from logging_utils
        logger.info("Using global_tracer_instance from myjobspyai.utils.logging_utils for analyzer.")
    else:
        tracer = trace.NoOpTracer()
        logger.warning("global_tracer_instance is None, using NoOpTracer for analyzer.")

    # Use the global meter instance if available
    if global_meter_instance is not None:
        meter = global_meter_instance  # Use the instance from logging_utils
        logger.info("Using global_meter_instance from myjobspyai.utils.logging_utils for analyzer.")
    else:
        meter = metrics.NoOpMeter()
        logger.warning("global_meter_instance is None, using NoOpMeter for analyzer.")

except ImportError as e:
    # Fallback to NoOp implementations if logging_utils or its components cannot be imported
    tracer = trace.NoOpTracer()
    meter = metrics.NoOpMeter()
    logger.warning(
        "Could not import global_tracer_instance/global_meter_instance from myjobspyai.utils.logging_utils. Using NoOp versions for analyzer.",
        exc_info=True,
    )


# Helper function for logging exceptions (now uses standard logger)
def log_exception(
    message, exception
):  # This could also be traced if it becomes complex
    logger.error(message, exc_info=True)


# --- Jinja2 Environment Setup ---
PROMPT_TEMPLATE_LOADER = None
if JINJA2_AVAILABLE:
    try:
        prompts_dir = getattr(config, "prompts_dir", None)
        if prompts_dir and os.path.isdir(prompts_dir):
            logger.info(
                f"Initializing Jinja2 environment for prompts in: {prompts_dir}"
            )
            PROMPT_TEMPLATE_LOADER = Environment(
                loader=FileSystemLoader(prompts_dir),
                autoescape=select_autoescape(["html", "xml"]),
                trim_blocks=True,
                lstrip_blocks=True,
            )
        else:
            logger.error(
                f"Jinja2 prompts directory not found or invalid in config: {prompts_dir}"
            )
            JINJA2_AVAILABLE = False
    except Exception as jinja_err:
        log_exception(
            f"Failed to initialize Jinja2 environment: {jinja_err}", jinja_err
        )
        JINJA2_AVAILABLE = False
else:
    logger.error("Jinja2 library not installed. Prompts cannot be loaded.")


def load_template(template_name_key: str):
    if not JINJA2_AVAILABLE or not PROMPT_TEMPLATE_LOADER:
        raise RuntimeError(
            "Jinja2 environment not available for loading prompt templates."
        )
    template_filename = getattr(config, template_name_key, None)
    if not template_filename:
        raise ValueError(
            f"Missing template filename configuration for key '{template_name_key}'"
        )
    try:
        template = PROMPT_TEMPLATE_LOADER.get_template(template_filename)
        logger.info(f"Successfully loaded Jinja2 template: {template_filename}")
        return template
    except TemplateNotFound:
        logger.error(
            f"Jinja2 template file not found: {template_filename} in {getattr(config, 'prompts_dir', '')}"
        )
        raise
    except TemplateSyntaxError as syn_err:
        logger.error(f"Syntax error in Jinja2 template {template_filename}: {syn_err}")
        raise
    except Exception as e:
        log_exception(f"Error loading Jinja2 template {template_filename}: {e}", e)
        raise


class BaseAnalyzer:
    async def initialize(self, provider_name: Optional[str] = None):
        """Initialize the analyzer with the specified LLM provider.

        Args:
            provider_name: Optional name of the provider to use. If not specified,
                         uses the default provider from config.
        """
        # Get provider configuration from config
        default_provider = getattr(config, 'default_provider', 'openai')
        
        # Use specified provider or default from config
        self.provider_name = provider_name or default_provider
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
            await self._check_connection_and_model_async()

    def __init__(self, provider_name: Optional[str] = None):
        """Initialize the analyzer with the specified LLM provider.

        Note: This is a synchronous wrapper around the async initialize method.
        For proper async initialization, use the initialize() method instead.
        """
        self.provider_name = provider_name
        self.provider_type = None
        self.llm_provider = None
        self.model_name = None
        self.request_timeout = None
        self.max_retries = 2
        self.retry_delay = 5

    # LLM Call Metrics (can be moved to a separate metrics module if it grows)
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
    # Legacy stats, can be phased out or kept for internal logging
    _llm_call_stats: Dict[str, Any] = (
        {  # TODO: Review if this is still needed with OTEL metrics
            "total_calls": 0,  # Covered by llm_calls_counter
            "successful_calls": 0,
            "failed_calls": 0,
            "total_prompt_chars": 0,
            "total_response_chars": 0,
            "total_duration_seconds": 0.0,
            "errors_by_type": defaultdict(int),
            "calls_by_task": defaultdict(lambda: defaultdict(int)),
        }
    )

    @classmethod
    def log_llm_call_summary(cls):
        if cls._llm_call_stats["total_calls"] == 0:
            logger.info("No LLM calls were made in this session.")
            return

        logger.info("--- LLM Call Statistics Summary ---")
        logger.info(f"Total LLM Calls Attempted: {cls._llm_call_stats['total_calls']}")
        logger.info(f"  Successful Calls: {cls._llm_call_stats['successful_calls']}")
        logger.info(
            f"  Failed Calls (after retries): {cls._llm_call_stats['failed_calls']}"
        )

        successful_calls_count = cls._llm_call_stats["successful_calls"]
        if successful_calls_count > 0:
            avg_duration = (
                cls._llm_call_stats["total_duration_seconds"] / successful_calls_count
            )
            logger.info(
                f"Average Call Duration (per successful call): {avg_duration:.2f}s"
            )
        else:
            logger.info(
                "Average Call Duration (per successful call): N/A (no successful calls)"
            )

        logger.info(
            f"Total Prompt Characters Sent: {cls._llm_call_stats['total_prompt_chars']:,}"
        )
        logger.info(
            f"Total Response Characters Received: {cls._llm_call_stats['total_response_chars']:,}"
        )

        if cls._llm_call_stats["errors_by_type"]:
            logger.info("\nErrors by Type (across all attempts):")
            for err_type, count in sorted(
                cls._llm_call_stats["errors_by_type"].items()
            ):
                logger.info(f"  - {err_type}: {count}")

        if cls._llm_call_stats["calls_by_task"]:
            logger.info("\nCalls by Task:")
            for task, stats in sorted(cls._llm_call_stats["calls_by_task"].items()):
                logger.info(f"  - Task: '{task}'")
                logger.info(
                    f"    Successful: {stats.get('success', 0)}, Failed: {stats.get('fail', 0)}"
                )
        logger.info("--- End LLM Call Statistics Summary ---")

    def _load_common_provider_config(self, provider_key: str) -> Dict[str, Any]:
        cfg = getattr(config, provider_key, {})
        self.model_name = cfg.get("model")
        self.request_timeout = cfg.get("request_timeout")
        if self.request_timeout is None:
            self.request_timeout = 120.0
            logger.warning(
                f"No request_timeout in config for '{provider_key}', defaulting to {self.request_timeout}s."
            )
        elif not isinstance(self.request_timeout, (int, float)):
            try:
                self.request_timeout = float(self.request_timeout)
                logger.warning(
                    f"Converted request_timeout for '{provider_key}' to float: {self.request_timeout}s."
                )
            except ValueError:
                logger.error(
                    f"Invalid request_timeout value '{self.request_timeout}' for '{provider_key}'. Using default 120.0s."
                )
                self.request_timeout = 120.0
        self.max_retries = cfg.get("max_retries", self.max_retries)
        self.retry_delay = cfg.get("retry_delay", self.retry_delay)
        return cfg

    def _chunk_text(self, text: str, max_length: int) -> List[str]:
        """Splits text into chunks of a maximum length."""
        chunks = []
        for i in range(0, len(text), max_length):
            chunks.append(text[i : i + max_length])
        return chunks

    def _initialize_openai_client(self, provider_config_key: str):
        cfg = self._load_common_provider_config(provider_config_key)
        base_url = cfg.get("base_url")
        api_key = cfg.get("api_key", "lm-studio")
        if not base_url or not self.model_name:
            raise ValueError(
                f"OpenAI provider ('{provider_config_key}') requires 'base_url' and 'model' in config."
            )
        self.sync_client = openai.OpenAI(
            base_url=base_url, api_key=api_key, timeout=self.request_timeout
        )
        self.async_client = openai.AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=self.request_timeout
        )
        logger.info(
            f"OpenAI client initialized for model '{self.model_name}' at {base_url}."
        )

    def _initialize_ollama_client(self, provider_config_key: str):
        cfg = self._load_common_provider_config(provider_config_key)
        base_url = cfg.get("base_url")
        self.ollama_base_url = base_url
        if not base_url or not self.model_name:
            raise ValueError(
                f"Ollama provider ('{provider_config_key}') requires 'base_url' and 'model' in config."
            )
        self.sync_client = ollama.Client(host=base_url, timeout=self.request_timeout)  # type: ignore
        self.async_client = ollama.AsyncClient(host=base_url, timeout=self.request_timeout)  # type: ignore
        logger.info(
            f"Ollama client initialized for model '{self.model_name}' at {base_url} with timeout {self.request_timeout}s."
        )

    def _initialize_gemini_client(self, provider_config_key: str):
        cfg = self._load_common_provider_config(provider_config_key)
        api_key = cfg.get("api_key") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                f"Gemini provider ('{provider_config_key}') requires 'api_key' in config or GOOGLE_API_KEY env var."
            )
        if not self.model_name:
            raise ValueError(
                f"Gemini provider ('{provider_config_key}') requires 'model' name in config."
            )

        # Configure the Gemini API
        genai.configure(api_key=api_key)

        # Set up safety settings to minimize content filtering
        safety_settings = [
            {"category": c, "threshold": "BLOCK_NONE"}
            for c in [
                "HARM_CATEGORY_HARASSMENT",
                "HARM_CATEGORY_HATE_SPEECH",
                "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "HARM_CATEGORY_DANGEROUS_CONTENT",
            ]
        ]

        # Initialize the synchronous client
        self.sync_client = genai.GenerativeModel(
            model_name=self.model_name, safety_settings=safety_settings
        )

        # For async operations, we'll use the same model with async methods
        self.async_client = self.sync_client

        logger.info(f"Gemini client initialized for model '{self.model_name}'")

    def _initialize_llm_provider(self):
        """Initialize the LLM provider based on the provider_name."""
        try:
            # Get the provider configuration from config
            provider_config = getattr(config, 'llm_providers', {}).get(self.provider_name, {})

            if not provider_config:
                raise ValueError(
                    f"No configuration found for provider: {self.provider_name}"
                )

            # Get the provider type and params
            self.provider_type = provider_config.get("type")
            provider_params = provider_config.get("params", {})

            # Initialize the appropriate provider based on type
            if self.provider_type == "langchain":
                from .providers.langchain_provider import LangChainProvider

                # Format the config for LangChainProvider
                langchain_config = {
                    "class": provider_config.get("class", ""),
                    "params": provider_params,
                }
                self.llm_provider = LangChainProvider(langchain_config)
            elif self.provider_type == "openai":
                from .providers.openai_provider import OpenAIProvider
                self.llm_provider = OpenAIProvider(**provider_params)
            elif self.provider_type == "ollama":
                from .providers.ollama_provider import OllamaProvider
                self.llm_provider = OllamaProvider(**provider_params)
            elif self.provider_type == "gemini":
                from .providers.gemini_provider import GeminiProvider
                self.llm_provider = GeminiProvider(**provider_params)
            else:
                raise ValueError(f"Unsupported provider type: {self.provider_type}")

            logger.info(f"Initialized {self.provider_type} provider: {self.provider_name}")

        except Exception as e:
            logger.error(f"Failed to initialize LLM provider {self.provider_name}: {str(e)}")
            raise

    def _check_connection_and_model(self):
        """
        Synchronous wrapper for checking the connection to the LLM provider.
        This is a blocking call that runs the async version in an event loop.
        """
        import asyncio

        return asyncio.run(self._check_connection_and_model_async())

    async def _call_llm_async(
        self,
        prompt: str,
        task_name: str = "llm_call",
        output_schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[str, Dict[str, Any], List[Any]]:
        """
        Make an async call to the LLM with retries, error handling, and optional JSON schema validation.

        Args:
            prompt: The prompt to send to the LLM
            task_name: Name of the task for logging and tracing
            output_schema: Optional JSON schema for structured output
            **kwargs: Additional arguments to pass to the LLM provider

        Returns:
            The LLM response as a string or parsed JSON (dict/list)

        Raises:
            RuntimeError: If the LLM provider is not initialized or if all retry attempts fail
            ValueError: If the output schema is invalid or output validation fails
        """
        if not self.llm_provider:
            raise RuntimeError("LLM provider not initialized")

        max_retries = self.max_retries
        retry_delay = self.retry_delay
        last_error = None

        for attempt in range(max_retries + 1):
            current_span = None
            try:
                with tracer.start_as_current_span(f"llm.{task_name}") as current_span:
                    span_attrs = {
                        "llm.provider": self.provider_name,
                        "llm.model": self.model_name or "unknown",
                        "llm.attempt": attempt + 1,
                        "llm.prompt_length": len(prompt),
                        "llm.has_output_schema": output_schema is not None,
                    }

                    if output_schema:
                        span_attrs["llm.output_schema"] = str(output_schema)[
                            :500
                        ]  # Truncate long schemas

                    current_span.set_attributes(span_attrs)

                    logger.info(
                        f"{task_name} - Attempt {attempt + 1}/{max_retries + 1} with {self.provider_name}"
                        + (" (with output schema)" if output_schema else "")
                    )

                    # Make the API call with error handling and retries
                    start_time = time.time()

                    # If we have an output schema, pass it to the provider
                    if (
                        output_schema
                        and hasattr(self.llm_provider, "generate")
                        and "output_schema"
                        in inspect.signature(self.llm_provider.generate).parameters
                    ):
                        response = await self.llm_provider.generate(
                            prompt, output_schema=output_schema, **kwargs
                        )
                    else:
                        # Fallback to regular generation
                        response = await self.llm_provider.generate(prompt, **kwargs)

                        # Try to parse as JSON if schema was provided but not natively supported
                        if output_schema and isinstance(response, str):
                            json_str = response.strip()
                            parsing_attempts = [
                                # Attempt 1: Try parsing as-is first
                                lambda s: (s, json.loads(s)),
                                # Attempt 2: Extract from markdown code blocks
                                lambda s: (
                                    (
                                        s.split("```json", 1)[1]
                                        .rsplit("```", 1)[0]
                                        .strip()
                                        if "```json" in s
                                        else (
                                            s.split("```", 1)[1]
                                            .rsplit("```", 1)[0]
                                            .strip()
                                            if "```" in s
                                            else s
                                        )
                                    ),
                                    None,
                                ),
                                # Attempt 3: Extract JSON object/array using regex
                                lambda s: (
                                    (
                                        re.search(
                                            r"[\[\{](?:[^\{\}\[\]]|(?R))*[\}\]]",
                                            s,
                                            re.DOTALL,
                                        ).group(0)
                                        if re.search(
                                            r"[\[\{](?:[^\{\}\[\]]|(?R))*[\}\]]",
                                            s,
                                            re.DOTALL,
                                        )
                                        else s
                                    ),
                                    None,
                                ),
                                # Attempt 4: Fix common JSON syntax errors
                                lambda s: (
                                    s.replace(
                                        "'", '"'
                                    )  # Replace single quotes with double quotes
                                    .replace(
                                        "None", "null"
                                    )  # Replace Python None with JSON null
                                    .replace(
                                        "True", "true"
                                    )  # Replace Python True with JSON true
                                    .replace(
                                        "False", "false"
                                    ),  # Replace Python False with JSON false
                                    None,
                                ),
                            ]

                            last_error = None
                            parsed = None

                            for attempt_num, attempt in enumerate(parsing_attempts, 1):
                                try:
                                    modified_str, result = attempt(json_str)
                                    if result is None:
                                        parsed = json.loads(modified_str)
                                    else:
                                        parsed = result

                                    if isinstance(parsed, (dict, list)):
                                        response = parsed
                                        logger.info(
                                            f"Successfully parsed JSON after {attempt_num} attempt(s)"
                                        )
                                        break

                                except (
                                    json.JSONDecodeError,
                                    ValueError,
                                    IndexError,
                                    AttributeError,
                                ) as e:
                                    last_error = e
                                    continue

                            if not isinstance(parsed, (dict, list)):
                                logger.error(
                                    f"Failed to parse LLM response as JSON after multiple attempts. Last error: {last_error}\nResponse start: {json_str[:500]}"
                                )
                                raise ValueError(
                                    "LLM response could not be parsed as valid JSON"
                                ) from last_error

                    duration = time.time() - start_time

                    # Log metrics
                    current_span.set_attributes(
                        {
                            "llm.duration_ms": duration * 1000,
                            "llm.response_time_ms": duration * 1000,
                            "llm.response_length": (
                                len(str(response)) if response else 0
                            ),
                            "llm.success": True,
                        }
                    )

                    # Update call statistics
                    self.llm_calls_counter.add(1)
                    self.llm_successful_calls_counter.add(1)

                    # Record token usage if available
                    token_usage = (
                        getattr(response, "usage", {})
                        if hasattr(response, "usage")
                        else {}
                    )
                    prompt_tokens = token_usage.get("prompt_tokens")
                    completion_tokens = token_usage.get("completion_tokens")

                    if prompt_tokens is not None:
                        self.llm_prompt_chars_histogram.record(prompt_tokens)
                    if completion_tokens is not None:
                        self.llm_response_chars_histogram.record(completion_tokens)

                    # Log the successful call
                    logger.debug(
                        f"{task_name} - Successfully generated response in {duration:.2f}s"
                        + (
                            f" with {prompt_tokens} prompt tokens and {completion_tokens} completion tokens"
                            if prompt_tokens is not None
                            and completion_tokens is not None
                            else ""
                        )
                    )

                    return response

            except Exception as e:
                duration = time.time() - start_time if "start_time" in locals() else 0
                last_error = e
                error_type = type(e).__name__

                # Log the error
                logger.error(
                    f"{task_name} - Attempt {attempt + 1}/{max_retries + 1} failed after {duration:.2f}s: {str(e)}",
                    exc_info=logger.isEnabledFor(logging.DEBUG),
                )

                # Update error metrics
                if current_span:
                    current_span.record_exception(e)
                    current_span.set_attributes(
                        {
                            "llm.error": True,
                            "llm.error.type": error_type,
                            "llm.error.message": str(e)[
                                :1000
                            ],  # Truncate long error messages
                            "llm.duration_ms": duration * 1000,
                        }
                    )

                self.llm_failed_calls_counter.add(1)

                # If we've exhausted all retries, raise the last error
                if attempt >= max_retries - 1:
                    error_msg = (
                        f"All {max_retries + 1} attempts failed. Last error: {str(e)}"
                    )
                    logger.error(error_msg)
                    raise RuntimeError(error_msg) from e

                # Otherwise, wait before retrying
                logger.info(
                    f"Retrying in {retry_delay} seconds... (Attempt {attempt + 2}/{max_retries + 1})"
                )
                await asyncio.sleep(retry_delay)

                # Exponential backoff for subsequent retries
                retry_delay = min(retry_delay * 2, 60)  # Cap at 60 seconds
                continue  # Continue to the next retry attempt

            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)

                # Log the error with appropriate level based on attempt
                log_level = logger.warning if attempt < max_retries else logger.error
                log_msg = f"{task_name} - Error in LLM call (Attempt {attempt + 1}/{max_retries + 1}): {error_type} - {error_msg}"
                log_level(
                    log_msg, exc_info=attempt == max_retries
                )  # Full traceback on last attempt only

                # Record the error in the span if available
                if current_span:
                    current_span.record_exception(e)
                    current_span.set_attributes(
                        {
                            "error": "true",
                            "error.type": error_type,
                            "error.message": error_msg,
                        }
                    )

                # If we've hit max retries, re-raise the exception
                if attempt >= max_retries:
                    error_summary = f"LLM call failed after {max_retries + 1} attempts. Last error: {error_type}: {error_msg}"
                    logger.error(error_summary)

                    if current_span:
                        current_span.set_status(
                            trace.Status(trace.StatusCode.ERROR, error_summary)
                        )

                    raise ConnectionError(error_summary) from e

                # Wait before retrying (exponential backoff)
                wait_time = retry_delay * (2**attempt)
                logger.info(f"{task_name} - Retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)

        # This should never be reached due to the raise in the except block, but just in case
        raise ConnectionError(f"LLM call failed after {max_retries + 1} attempts")

    async def _check_connection_and_model_async(self):
        """
        Check the connection to the LLM provider and verify the model is available.
        For LangChain providers, we'll do a simple test generation to verify connectivity.
        """
        task_name = f"{self.__class__.__name__}.check_connection"

        try:
            # Use a simple test prompt to verify connectivity
            test_prompt = "Test connection - respond with 'OK'"
            logger.info(
                f"{task_name} - Testing connection to {self.provider_name} ({self.model_name})"
            )

            # Use our new _call_llm_async method which has retry logic
            response = await self._call_llm_async(test_prompt, task_name)

            if response.strip().upper() != "OK":
                logger.warning(
                    f"{task_name} - Unexpected response from {self.provider_name}: {response}"
                )
                return False

            logger.info(
                f"{task_name} - Successfully connected to {self.provider_name} with model: {self.model_name}"
            )
            return True

        except Exception as e:
            logger.error(
                f"{task_name} - Failed to connect to {self.provider_name}: {str(e)}",
                exc_info=True,
            )
            return False


class ResumeAnalyzer(BaseAnalyzer):
    def __init__(self, provider_name: Optional[str] = None):
        default_provider = getattr(config, 'default_provider', 'openai')
        super().__init__(provider_name=provider_name or default_provider)
        # Templates are now loaded dynamically in each method call
        # self._load_prompt_templates() # Removed

    async def initialize(self, provider_name: Optional[str] = None):
        await super().initialize(provider_name or self.provider_name)

    # def _load_prompt_templates(self): # Removed
    #     try:
    #         self.resume_prompt_template = load_template("resume_prompt_file")
    #     except (RuntimeError, ValueError, TemplateNotFound, TemplateSyntaxError) as tmpl_err:
    #         log_exception(f"ResumeAnalyzer: Failed to load resume prompt template: {tmpl_err}", tmpl_err)
    #         raise RuntimeError(f"ResumeAnalyzer prompt template loading failed: {tmpl_err}") from tmpl_err

    @tracer.start_as_current_span("extract_resume_data_async")
    async def extract_resume_data_async(self, resume_text: str) -> Optional[ResumeData]:
        """Extract structured resume data from raw resume text using LLM with JSON schema validation.

        Args:
            resume_text: The raw text content of the resume

        Returns:
            ResumeData object with extracted resume information, or None if extraction fails

        Raises:
            RuntimeError: If the LLM provider is not available or if there's a configuration error
            ValueError: If the input text is empty or invalid
        """
        current_span = trace.get_current_span()
        current_span.set_attribute("resume_text_length", len(resume_text))

        # Get configuration
        max_prompt_chars = getattr(config, 'max_prompt_chars', 15000)

        # Validate input
        if not resume_text or not resume_text.strip():
            logger.warning("Resume text empty for extraction.")
            return None

        # Define the expected JSON schema for resume data
        resume_schema = {
            "type": "object",
            "properties": {
                "contact_information": {
                    "type": "object",
                    "properties": {
                        "name": {"type": ["string", "null"]},
                        "email": {"type": ["string", "null"]},
                        "phone": {"type": ["string", "null"]},
                        "location": {"type": ["string", "null"]},
                        "linkedin_url": {"type": ["string", "null"]},
                        "portfolio_url": {"type": ["string", "null"]},
                        "github_url": {"type": ["string", "null"]},
                    },
                    "required": ["name", "email"],
                    "additionalProperties": True,
                },
                "summary": {"type": ["string", "null"]},
                "skills": {"type": "array", "items": {"type": "string"}, "default": []},
                "experience": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "company": {"type": "string"},
                            "location": {"type": ["string", "null"]},
                            "start_date": {"type": ["string", "null"]},
                            "end_date": {"type": ["string", "null"]},
                            "current": {"type": ["boolean", "null"]},
                            "description": {"type": ["string", "null"]},
                            "achievements": {
                                "type": "array",
                                "items": {"type": "string"},
                                "default": [],
                            },
                        },
                        "required": ["title", "company"],
                        "additionalProperties": True,
                    },
                    "default": [],
                },
                "education": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "degree": {"type": ["string", "null"]},
                            "field_of_study": {"type": ["string", "null"]},
                            "institution": {"type": "string"},
                            "location": {"type": ["string", "null"]},
                            "start_date": {"type": ["string", "null"]},
                            "end_date": {"type": ["string", "null"]},
                            "gpa": {"type": ["string", "number", "null"]},
                            "description": {"type": ["string", "null"]},
                        },
                        "required": ["institution"],
                        "additionalProperties": True,
                    },
                    "default": [],
                },
                "certifications": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "issuer": {"type": "string"},
                            "date_issued": {"type": ["string", "null"]},
                            "expiration_date": {"type": ["string", "null"]},
                            "credential_id": {"type": ["string", "null"]},
                            "credential_url": {"type": ["string", "null"]},
                        },
                        "required": ["name", "issuer"],
                        "additionalProperties": True,
                    },
                    "default": [],
                },
                "languages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": [],
                },
                "projects": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": ["string", "null"]},
                            "technologies": {
                                "type": "array",
                                "items": {"type": "string"},
                                "default": [],
                            },
                            "start_date": {"type": ["string", "null"]},
                            "end_date": {"type": ["string", "null"]},
                            "url": {"type": ["string", "null"]},
                            "achievements": {
                                "type": "array",
                                "items": {"type": "string"},
                                "default": [],
                            },
                        },
                        "required": ["name"],
                        "additionalProperties": True,
                    },
                    "default": [],
                },
            },
            "required": ["contact_information", "skills", "experience", "education"],
            "additionalProperties": True,
        }

        try:
            # Load the prompt template
            resume_prompt_template = load_template("resume_prompt_file")

            # Check if we need to chunk the resume text
            if len(resume_text) > max_prompt_chars:
                logger.warning(
                    f"Resume text length ({len(resume_text)}) exceeds max_prompt_chars ({max_prompt_chars}). "
                    "Chunking resume text."
                )

                # Split the resume into chunks
                text_chunks = self._chunk_text(resume_text, max_prompt_chars)
                all_extracted_data = {}

                # Process each chunk
                for i, chunk in enumerate(text_chunks):
                    try:
                        # Render the prompt with the current chunk
                        prompt = resume_prompt_template.render(
                            resume_text=chunk,
                            is_first_chunk=(i == 0),
                            is_last_chunk=(i == len(text_chunks) - 1),
                            chunk_index=i,
                            total_chunks=len(text_chunks),
                        )

                        # Call the LLM with the schema for this chunk
                        chunk_data = await self._call_llm_async(
                            prompt=prompt,
                            task_name=f"Resume Extraction - Chunk {i+1}/{len(text_chunks)}",
                            output_schema=resume_schema,
                            temperature=0.2,  # Lower temperature for more consistent results
                        )

                        # Merge the chunk data with existing data
                        if isinstance(chunk_data, dict):
                            self._merge_resume_data(all_extracted_data, chunk_data)

                    except Exception as chunk_error:
                        log_exception(
                            f"Error processing resume chunk {i+1}/{len(text_chunks)}: {chunk_error}",
                            chunk_error,
                        )
                        continue

                # If we didn't get any data from any chunk, return None
                if not all_extracted_data:
                    logger.error("Failed to extract any data from resume chunks")
                    return None

                # Ensure required fields exist
                all_extracted_data.setdefault("contact_information", {})
                all_extracted_data.setdefault("skills", [])
                all_extracted_data.setdefault("experience", [])
                all_extracted_data.setdefault("education", [])

                # Transform skills to technical_skills with SkillDetail objects
                if "skills" in all_extracted_data and isinstance(
                    all_extracted_data["skills"], list
                ):
                    technical_skills = []
                    for skill in all_extracted_data["skills"]:
                        if isinstance(skill, str):
                            technical_skills.append(SkillDetail(name=skill))
                        elif isinstance(skill, dict) and "name" in skill:
                            technical_skills.append(SkillDetail(**skill))
                    all_extracted_data["technical_skills"] = technical_skills
                    # Remove the old skills key to avoid validation errors
                    all_extracted_data.pop("skills", None)

                # Create the ResumeData object
                try:
                    resume_data = ResumeData(**all_extracted_data)
                    logger.info("Successfully extracted resume data from chunks")
                    return resume_data

                except Exception as validation_error:
                    log_exception(
                        "Failed to validate combined resume data from chunks",
                        validation_error,
                    )
                    logger.debug(f"Invalid combined resume data: {all_extracted_data}")
                    return None

            else:
                # Process the entire resume at once (fits within max_prompt_chars)
                prompt = resume_prompt_template.render(
                    resume_text=resume_text,
                    is_first_chunk=True,
                    is_last_chunk=True,
                    chunk_index=0,
                    total_chunks=1,
                )

                # Call the LLM with the schema
                extracted_data = await self._call_llm_async(
                    prompt=prompt,
                    task_name="Resume Extraction",
                    output_schema=resume_schema,
                    temperature=0.2,  # Lower temperature for more consistent results
                )

                if not extracted_data or not isinstance(extracted_data, dict):
                    logger.error(
                        "Failed to extract resume data: Invalid response format"
                    )
                    return None

                # Ensure contact_information is a dictionary with string keys and optional string values
                contact_info = {
                    "email": None,
                    "phone": None,
                    "linkedin": None,
                    "address": None,
                    "portfolio": None,
                }

                # Update with any existing contact information, converting values to strings
                if "contact_information" in extracted_data and isinstance(
                    extracted_data["contact_information"], dict
                ):
                    for k, v in extracted_data["contact_information"].items():
                        if v is not None:
                            contact_info[str(k)] = str(v) if v is not None else None

                extracted_data["contact_information"] = contact_info
                extracted_data.setdefault("skills", [])
                extracted_data.setdefault("experience", [])
                extracted_data.setdefault("education", [])

                # Transform skills to technical_skills with SkillDetail objects
                if "skills" in extracted_data and isinstance(
                    extracted_data["skills"], list
                ):
                    technical_skills = []
                    for skill in extracted_data["skills"]:
                        if isinstance(skill, str):
                            technical_skills.append(SkillDetail(name=skill))
                        elif isinstance(skill, dict) and "name" in skill:
                            technical_skills.append(SkillDetail(**skill))
                    extracted_data["technical_skills"] = technical_skills
                    # Remove the old skills key to avoid validation errors
                    extracted_data.pop("skills", None)

                # Ensure work_experience is a list of valid dictionaries
                if "work_experience" in extracted_data and isinstance(
                    extracted_data["work_experience"], list
                ):
                    valid_experiences = []
                    for exp in extracted_data["work_experience"]:
                        if isinstance(exp, dict):
                            # Ensure required fields exist with proper types
                            if "job_title" not in exp:
                                exp["job_title"] = "Unknown Position"
                            if "company" not in exp:
                                exp["company"] = "Unknown Company"
                            if "responsibilities" not in exp or not isinstance(
                                exp["responsibilities"], list
                            ):
                                exp["responsibilities"] = []
                            if (
                                "quantifiable_achievements" not in exp
                                or not isinstance(
                                    exp["quantifiable_achievements"], list
                                )
                            ):
                                exp["quantifiable_achievements"] = []
                            valid_experiences.append(exp)
                        elif isinstance(exp, str):
                            # Handle case where experience is just a string (e.g., "Job at Company")
                            valid_experiences.append(
                                {
                                    "job_title": (
                                        exp.split(" at ")[0] if " at " in exp else exp
                                    ),
                                    "company": (
                                        exp.split(" at ")[1]
                                        if " at " in exp
                                        else "Unknown Company"
                                    ),
                                    "responsibilities": [],
                                    "quantifiable_achievements": [],
                                }
                            )
                    extracted_data["work_experience"] = valid_experiences

                # Ensure education is a list of valid dictionaries
                if "education" in extracted_data and isinstance(
                    extracted_data["education"], list
                ):
                    valid_education = []
                    for edu in extracted_data["education"]:
                        if isinstance(edu, dict):
                            # Ensure required fields exist with proper types
                            valid_edu = {}
                            if "degree" in edu and edu["degree"] is not None:
                                valid_edu["degree"] = str(edu["degree"])
                            if "institution" in edu and edu["institution"] is not None:
                                valid_edu["institution"] = str(edu["institution"])
                            if (
                                "graduation_year" in edu
                                and edu["graduation_year"] is not None
                            ):
                                valid_edu["graduation_year"] = str(
                                    edu["graduation_year"]
                                )
                            valid_education.append(valid_edu)
                        elif isinstance(edu, str):
                            # Handle case where education is just a string (e.g., "BS in CS from University")
                            parts = [
                                p.strip()
                                for p in re.split(r"\s+in\s+|\s+at\s+|\s+from\s+", edu)
                            ]
                            valid_edu = {}
                            if len(parts) > 0:
                                valid_edu["degree"] = parts[0]
                            if len(parts) > 1:
                                valid_edu["institution"] = parts[-1]
                            valid_education.append(valid_edu)
                    extracted_data["education"] = valid_education
                else:
                    extracted_data["education"] = []

                # Create the ResumeData object with validation
                try:
                    resume_data = ResumeData(**extracted_data)
                    logger.info("Successfully extracted and validated resume data")
                    return resume_data

                except Exception as validation_error:
                    log_exception(
                        "Failed to validate extracted resume data", validation_error
                    )
                    logger.debug(f"Invalid resume data structure: {extracted_data}")
                    # Try to create with minimal required fields if validation fails
                    try:
                        # Ensure contact_information has all required fields set to None
                        contact_info = {
                            "email": None,
                            "phone": None,
                            "linkedin": None,
                            "address": None,
                            "portfolio": None,
                        }
                        # Update with any existing contact information
                        if "contact_information" in extracted_data and isinstance(
                            extracted_data["contact_information"], dict
                        ):
                            for k, v in extracted_data["contact_information"].items():
                                if v is not None:
                                    contact_info[str(k)] = (
                                        str(v) if v is not None else None
                                    )

                        minimal_data = {
                            "contact_information": contact_info,
                            "work_experience": extracted_data.get(
                                "work_experience", []
                            ),
                            "education": extracted_data.get("education", []),
                            "technical_skills": extracted_data.get(
                                "technical_skills", []
                            ),
                        }
                        resume_data = ResumeData(**minimal_data)
                        logger.warning(
                            "Created ResumeData with minimal required fields due to validation errors"
                        )
                        return resume_data
                    except Exception as e:
                        logger.error("Completely failed to create ResumeData object")
                        return None

        except Exception as e:
            log_exception(f"Unexpected error in extract_resume_data_async: {e}", e)
            raise

    def _merge_resume_data(self, target: Dict, source: Dict) -> None:
        """Merge source resume data into target dictionary.

        Args:
            target: The target dictionary to merge into
            source: The source dictionary to merge from
        """
        if not isinstance(target, dict) or not isinstance(source, dict):
            return

        for key, value in source.items():
            # Skip None values
            if value is None:
                continue

            # Handle contact information (merge dictionaries)
            if key == "contact_information" and isinstance(value, dict):
                if key not in target or not isinstance(target[key], dict):
                    target[key] = {}
                target[key].update({k: v for k, v in value.items() if v is not None})

            # Handle arrays (concatenate and deduplicate)
            elif isinstance(value, list) and value:
                if key not in target or not isinstance(target.get(key), list):
                    target[key] = []

                # For experience and education, we need to handle deduplication by unique identifiers
                if key in ["experience", "education", "certifications", "projects"]:
                    existing_items = {
                        self._get_item_identifier(item): item for item in target[key]
                    }

                    for item in value:
                        if not isinstance(item, dict):
                            continue

                        item_id = self._get_item_identifier(item)
                        if item_id not in existing_items:
                            target[key].append(item)
                            existing_items[item_id] = item
                else:
                    # For simple arrays like skills and languages, just add unique values
                    target[key].extend([v for v in value if v not in target[key]])

            # Handle other values (overwrite if not None)
            elif key not in target or target[key] is None:
                target[key] = value

    def _get_item_identifier(self, item: Dict) -> str:
        """Generate a unique identifier for a resume item based on its type and key fields.

        Args:
            item: The resume item (experience, education, etc.)

        Returns:
            A string identifier for the item
        """
        if not isinstance(item, dict):
            return str(id(item))

        # For experience items
        if "title" in item and "company" in item:
            return (
                f"exp_{item.get('title', '').lower()}_{item.get('company', '').lower()}"
            )

        # For education items
        if "institution" in item:
            return f"edu_{item.get('institution', '').lower()}_{item.get('degree', '').lower()}"

        # For certifications
        if "name" in item and "issuer" in item:
            return (
                f"cert_{item.get('name', '').lower()}_{item.get('issuer', '').lower()}"
            )

        # For projects
        if "name" in item:
            return f"proj_{item.get('name', '').lower()}"

        # Fallback to a hash of the item's string representation
        return str(hash(json.dumps(item, sort_keys=True)))


class JobAnalyzer(BaseAnalyzer):
    def __init__(self, provider_name: Optional[str] = None):
        default_provider = getattr(config, 'default_provider', 'openai')
        super().__init__(provider_name=provider_name or default_provider)
        self.common_skills = self._load_common_skills()

    def _load_common_skills(self) -> Dict[str, List[str]]:
        """Load a dictionary of common skills by category."""
        return {
            "programming": ["Python", "Java", "JavaScript", "TypeScript", "C#", "C++"],
            "databases": [
                "PostgreSQL",
                "MySQL",
                "MongoDB",
                "Redis",
                "Oracle",
                "SQL Server",
            ],
            "cloud": ["AWS", "Azure", "GCP", "Docker", "Kubernetes"],
            "tools": ["Git", "Jira", "Jenkins", "Docker", "Kubernetes"],
            "methodologies": ["Agile", "Scrum", "DevOps", "CI/CD"],
            "healthcare_it": ["Epic", "Cerner", "HL7", "FHIR", "HIPAA"],
        }

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""

        # Remove markdown formatting
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # Remove bold
        text = re.sub(r"_(.*?)_", r"\1", text)  # Remove italics
        text = re.sub(r"`(.*?)`", r"\1", text)  # Remove code blocks

        # Normalize whitespace and newlines
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Split job description into logical sections."""
        sections = {}
        current_section = "preamble"
        sections[current_section] = []

        section_patterns = {
            "responsibilities": r"(?i)(?:key\s+)?(?:responsibilities|duties|what you\'ll do|key responsibilities)",
            "requirements": r"(?i)(?:requirements|qualifications|what you(?:\'ll)? need|minimum qualifications|must have)",
            "preferred": r"(?i)(?:preferred\s+qualifications?|nice to have|bonus(?:\s+skills)?|pluses)",
            "education": r"(?i)(?:education(?:al)?(?:\s+requirements?)?|degree(?:s)?(?:\s+required)?)",
            "skills": r"(?i)(?:skills?\s+(?:and\s+)?(?:qualifications|requirements)|technical\s+skills?)",
            "about": r"(?i)(?:about\s+(?:us|the\s+company|the\s+team)|who\s+we\s+are|company\s+profile)",
            "benefits": r"(?i)(?:benefits|perks|what we offer|compensation\s+and\s+benefits)",
            "location": r"(?i)(?:location|workplace|work model|work\s+arrangement)",
            "salary": r"(?i)(?:salary|compensation|pay range|pay scale)",
        }

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            matched = False
            for section, pattern in section_patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    current_section = section
                    sections[current_section] = []
                    matched = True
                    break

            if not matched and line:
                sections.setdefault(current_section, []).append(line)

        return {k: "\n".join(v).strip() for k, v in sections.items() if v}

    def _extract_skills(
        self, text: str, skill_type: str = "required"
    ) -> List[Dict[str, Any]]:
        """Extract skills from text with specified type (required/preferred)."""
        if not text:
            return []

        skills = []
        text_lower = text.lower()

        # Check for skills in each category
        for category, skill_list in self.common_skills.items():
            for skill in skill_list:
                if re.search(rf"\b{re.escape(skill.lower())}\b", text_lower):
                    skills.append({"name": skill, "level": None, "type": skill_type})

        return skills

    def _extract_education(self, text: str) -> List[Dict[str, str]]:
        """Extract education requirements from text."""
        if not text:
            return []

        education = []
        patterns = [
            (
                r"(?i)(?:bachelor\'?s?\s*(?:\(?b\.?[a-z]?\.?s?\.?\)?|of\s+science|of\s+arts)|b\.?s?\.?\s*(?:in|/|or)\s*[a-z]+)",
                "Bachelor's degree",
            ),
            (
                r"(?i)(?:master\'?s?\s*(?:\(?m\.?[a-z]?\.?s?\.?\)?|of\s+science|of\s+arts|of\s+business\s+administration)|m\.?s?\.?\s*(?:in|/|or)\s*[a-z]+|mba)",
                "Master's degree",
            ),
            (r"(?i)(?:ph\.?\s*d\.?|doctor\s*of\s*philosophy)", "PhD"),
            (
                r"(?i)(?:associate\'?s?\s*degree|a\.?s\.?|a\.?a\.?)",
                "Associate's degree",
            ),
            (r"(?i)high\s*school\s*diploma", "High School Diploma"),
        ]

        for pattern, degree in patterns:
            if re.search(pattern, text):
                education.append({"name": degree})

        return education

    def _extract_job_type(self, text: str) -> Optional[str]:
        """Extract job type from text."""
        if not text:
            return None

        text_lower = text.lower()
        if re.search(r"\bfull[\s-]?time\b", text_lower):
            return "Full-time"
        elif re.search(r"\bpart[\s-]?time\b", text_lower):
            return "Part-time"
        elif re.search(r"\bcontract(?:or)?\b", text_lower):
            return "Contract"
        elif re.search(r"\bintern(?:ship)?\b", text_lower):
            return "Internship"
        return None

    def _extract_work_model(self, text: str) -> Optional[str]:
        """Extract work model from text."""
        if not text:
            return None

        text_lower = text.lower()
        if re.search(r"\bremote(?:\s+work)?\b", text_lower):
            if re.search(r"\bhybrid\b", text_lower):
                return "Hybrid"
            return "Remote"
        elif re.search(r"\bon[\s-]?site\b", text_lower) or re.search(
            r"\bin[\s-]?office\b", text_lower
        ):
            return "On-site"
        return None

    def _extract_responsibilities(self, text: str) -> List[str]:
        """Extract responsibilities from text."""
        if not text:
            return []

        # Split by bullet points or newlines
        responsibilities = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Handle bullet points or numbered lists
            line = re.sub(r"^[\s\-\*•]\s*", "", line)  # Remove bullet points
            line = re.sub(r"^\d+[.)]\s*", "", line)  # Remove numbered list markers

            # Split by sentence if multiple responsibilities are on one line
            sentences = re.split(r"[.!?]\s+", line)
            for sent in sentences:
                sent = sent.strip()
                if sent and len(sent) > 10:  # Filter out very short fragments
                    responsibilities.append(sent)

        return responsibilities

    def _extract_tools_technologies(self, text: str) -> List[str]:
        """Extract tools and technologies from text."""
        if not text:
            return []

        tools = set()
        text_lower = text.lower()

        # Check for common tools and technologies
        for category, tech_list in self.common_skills.items():
            for tech in tech_list:
                if re.search(rf"\b{re.escape(tech.lower())}\b", text_lower):
                    tools.add(tech)

        return list(tools)

    def _create_parsed_job_data(
        self, rule_based_data: Dict[str, Any], llm_data: Dict[str, Any]
    ) -> ParsedJobData:
        """Create a ParsedJobData object by merging rule-based and LLM extractions.

        This method handles missing or invalid fields by providing sensible defaults and
        normalizing data to the expected format. It's designed to be resilient to malformed
        input from either the rule-based extraction or the LLM.

        Args:
            rule_based_data: Dictionary of data extracted using rule-based methods
            llm_data: Dictionary of data extracted from LLM

        Returns:
            ParsedJobData: A validated ParsedJobData object with merged data

        Raises:
            ValueError: If the input data is invalid and cannot be normalized
        """
        logger.debug("Starting job data normalization and validation")

        # Ensure inputs are dictionaries
        rule_based_data = rule_based_data or {}
        llm_data = llm_data or {}

        # Start with a clean dictionary for the merged data
        merged_data = {}

        def ensure_str(value: Any, default: str = "") -> str:
            """Convert value to string safely with a default if conversion fails."""
            if value is None:
                return default
            try:
                return str(value).strip()
            except (TypeError, ValueError, AttributeError):
                return default

        def ensure_list(value: Any) -> List[Any]:
            """Ensure the value is a list, converting if necessary."""
            if value is None:
                return []
            if isinstance(value, list):
                return value
            return [value]

        def normalize_string_field(
            value: Any, field_name: str, default: str = ""
        ) -> str:
            """Normalize a string field with validation and default value."""
            if value is None:
                return default

            try:
                value_str = str(value).strip()
                if not value_str and field_name in rule_based_data:
                    # Fall back to rule-based data if LLM returned empty
                    value_str = str(rule_based_data[field_name]).strip()
                return value_str or default
            except (TypeError, ValueError, AttributeError) as e:
                logger.warning(f"Failed to normalize field '{field_name}': {e}")
                return default

        def normalize_list_field(
            data: Dict[str, Any], field_name: str
        ) -> List[Dict[str, str]]:
            """Normalize a list field to the expected format [{"name": "value"}]."""
            items = ensure_list(data.get(field_name, []))
            normalized = []

            for item in items:
                if item is None:
                    continue

                if isinstance(item, dict):
                    if "name" in item and item["name"]:
                        normalized.append({"name": ensure_str(item["name"])})
                elif isinstance(item, str) and item.strip():
                    normalized.append({"name": item.strip()})

            return normalized

        def normalize_skills(skills_data: Any) -> List[Dict[str, Any]]:
            """Normalize skills data to a consistent format."""
            skills = ensure_list(skills_data)
            normalized = []

            for skill in skills:
                if not skill:
                    continue

                try:
                    skill_dict = (
                        skill if isinstance(skill, dict) else {"name": str(skill)}
                    )
                    name = ensure_str(skill_dict.get("name") or skill_dict.get("skill"))
                    if not name:
                        continue

                    normalized_skill = {"name": name}

                    # Handle level if present
                    if "level" in skill_dict and skill_dict["level"] is not None:
                        normalized_skill["level"] = ensure_str(skill_dict["level"])

                    # Handle years_experience if present
                    if (
                        "years_experience" in skill_dict
                        and skill_dict["years_experience"] is not None
                    ):
                        try:
                            years = float(str(skill_dict["years_experience"]).strip())
                            normalized_skill["years_experience"] = (
                                int(years) if years.is_integer() else years
                            )
                        except (ValueError, TypeError) as e:
                            logger.debug(
                                f"Invalid years_experience value: {skill_dict['years_experience']}"
                            )

                    normalized.append(normalized_skill)
                except Exception as e:
                    logger.warning(f"Failed to normalize skill: {skill}. Error: {e}")

            return normalized

        try:
            # Merge data sources with LLM data taking precedence
            for source in [rule_based_data, llm_data]:
                if not source:
                    continue

                for key, value in source.items():
                    if value is not None or key not in merged_data:
                        merged_data[key] = value

            # Normalize all fields with appropriate defaults
            normalized_data = {
                "job_title_extracted": normalize_string_field(
                    merged_data.get("job_title_extracted"),
                    "job_title_extracted",
                    rule_based_data.get("job_title_extracted", "N/A"),
                ),
                "job_description": normalize_string_field(
                    merged_data.get("job_description"),
                    "job_description",
                    rule_based_data.get("job_description", ""),
                ),
                "key_responsibilities": normalize_list_field(
                    merged_data, "key_responsibilities"
                ),
                "required_skills": normalize_skills(
                    merged_data.get("required_skills", [])
                ),
                "preferred_skills": normalize_skills(
                    merged_data.get("preferred_skills", [])
                ),
                "required_education": normalize_list_field(
                    merged_data, "required_education"
                ),
                "preferred_education": normalize_list_field(
                    merged_data, "preferred_education"
                ),
                "required_certifications": normalize_list_field(
                    merged_data, "required_certifications"
                ),
                "preferred_certifications": normalize_list_field(
                    merged_data, "preferred_certifications"
                ),
                "company_culture_hints": normalize_list_field(
                    merged_data, "company_culture_hints"
                ),
                "tools_technologies": normalize_list_field(
                    merged_data, "tools_technologies"
                ),
                "job_type": normalize_string_field(
                    merged_data.get("job_type"), "job_type"
                ),
                "work_model": normalize_string_field(
                    merged_data.get("work_model"), "work_model"
                ),
                "experience_years": merged_data.get("experience_years"),
                "location": normalize_string_field(
                    merged_data.get("location"), "location"
                ),
                "salary_range": merged_data.get("salary_range"),
                "company_name": normalize_string_field(
                    merged_data.get("company_name"), "company_name"
                ),
                "industry": normalize_string_field(
                    merged_data.get("industry"), "industry"
                ),
                "job_function": normalize_string_field(
                    merged_data.get("job_function"), "job_function"
                ),
                "employment_type": normalize_string_field(
                    merged_data.get("employment_type"), "employment_type"
                ),
                "seniority_level": normalize_string_field(
                    merged_data.get("seniority_level"), "seniority_level"
                ),
            }

            # Clean up any None values that might cause validation issues
            cleaned_data = {k: v for k, v in normalized_data.items() if v is not None}

            # Create and return the validated model
            return ParsedJobData(**cleaned_data)

        except Exception as e:
            error_msg = f"Failed to create ParsedJobData: {str(e)}"
            logger.error(
                f"{error_msg}\nRule-based data: {json.dumps(rule_based_data, default=str, indent=2)}\n"
                f"LLM data: {json.dumps(llm_data, default=str, indent=2)}"
            )

            # Return a minimal valid object with error information
            return ParsedJobData(
                job_title_extracted=ensure_str(
                    rule_based_data.get("job_title_extracted", "Error in job data")
                ),
                job_description=ensure_str(
                    rule_based_data.get("job_description", "Error processing job data")
                ),
                key_responsibilities=[],
            )

    async def initialize(self, provider_name: Optional[str] = None):
        await super().initialize(provider_name or self.provider_name)

    # def _load_prompt_templates(self): # Removed
    #     try:
    #         self.suitability_prompt_template = load_template("suitability_prompt_file")
    #         self.job_extraction_prompt_template = load_template("job_extraction_prompt_file")
    #     except (RuntimeError, ValueError, TemplateNotFound, TemplateSyntaxError) as tmpl_err:
    #         log_exception(f"JobAnalyzer: Failed to load necessary prompt templates: {tmpl_err}", tmpl_err)
    #         raise RuntimeError(f"JobAnalyzer prompt template loading failed: {tmpl_err}") from tmpl_err

    @tracer.start_as_current_span("extract_job_details_async")
    async def extract_job_details_async(
        self, job_description_text: str, job_title: str = "N/A"
    ) -> Optional[ParsedJobData]:
        """Extract structured job details from a job description text.

        This method uses a combination of rule-based extraction and LLM processing to extract
        structured job details from unstructured text. It first performs rule-based extraction
        of key sections and fields, then uses the LLM to validate and enhance the extracted data.

        Args:
            job_description_text: The job description text to analyze
            job_title: Optional job title to include in the analysis

        Returns:
            ParsedJobData object with extracted job details, or None if extraction fails

        Raises:
            ValueError: If the job description is empty or invalid
            RuntimeError: If the LLM provider is not available or there's a processing error
        """
        current_span = trace.get_current_span()
        current_span.set_attribute("job_title_param", job_title)
        current_span.set_attribute(
            "job_description_length", len(job_description_text or "")
        )

        if not job_description_text or not job_description_text.strip():
            error_msg = (
                f"Job description for '{job_title}' is empty. Cannot extract details."
            )
            logger.warning(error_msg)
            current_span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
            raise ValueError(error_msg)

        # Clean and preprocess the job description
        cleaned_text = self._clean_text(job_description_text)

        # Extract sections using rule-based approach
        sections = self._extract_sections(cleaned_text)

        # Initialize base job data with rule-based extractions
        job_data = {
            "job_title_extracted": job_title,
            "key_responsibilities": [
                {"name": resp}
                for resp in self._extract_responsibilities(
                    sections.get("responsibilities", "")
                )
            ],
            "required_skills": self._extract_skills(
                sections.get("requirements", ""), "required"
            ),
            "preferred_skills": self._extract_skills(
                sections.get("preferred", ""), "preferred"
            ),
            "required_education": self._extract_education(
                sections.get("education", "") or sections.get("requirements", "")
            ),
            "job_type": self._extract_job_type(cleaned_text),
            "work_model_extracted": self._extract_work_model(cleaned_text),
            "tools_technologies": [
                {"name": tech}
                for tech in self._extract_tools_technologies(cleaned_text)
            ],
            "job_description": job_description_text,
        }

        # Prepare for LLM processing
        try:
            # Load the job extraction prompt template
            job_extraction_prompt_template = load_template("job_extraction_prompt_file")

            # Prepare the prompt with the job description and our initial extractions
            prompt = job_extraction_prompt_template.render(
                job_title=job_title,
                job_description=job_description_text,
                initial_extractions=json.dumps(job_data, indent=2),
            )

            # Log the prompt for debugging (but not the full content if it's large)
            logger.debug(
                f"Job extraction prompt for '{job_title}': {prompt[:200]}..."
                if len(prompt) > 200
                else f"Job extraction prompt: {prompt}"
            )

        except Exception as render_err:
            error_msg = f"Failed to load or render job extraction prompt for '{job_title}': {str(render_err)}"
            log_exception(error_msg, render_err)
            current_span.record_exception(render_err)
            current_span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
            # Fall back to just the rule-based extractions
            return self._create_parsed_job_data(job_data, {})

        try:
            # Call the LLM to enhance the job details
            extracted_response = await self._call_llm_async(
                prompt,
                task_name=f"job_details_extraction",
                output_schema={
                    "type": "object",
                    "properties": {
                        "job_title_extracted": {"type": "string"},
                        "key_responsibilities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"name": {"type": "string"}},
                            },
                        },
                        "required_skills": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "level": {"type": ["string", "null"]},
                                    "years_experience": {"type": ["number", "null"]},
                                },
                            },
                        },
                        "preferred_skills": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "level": {"type": ["string", "null"]},
                                    "years_experience": {"type": ["number", "null"]},
                                },
                            },
                        },
                        "required_education": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"name": {"type": "string"}},
                            },
                        },
                        "job_type": {"type": ["string", "null"]},
                        "work_model_extracted": {"type": ["string", "null"]},
                        "tools_technologies": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"name": {"type": "string"}},
                            },
                        },
                        "company_culture_hints": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"name": {"type": "string"}},
                            },
                        },
                        "required_certifications": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"name": {"type": "string"}},
                            },
                        },
                        "preferred_certifications": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {"name": {"type": "string"}},
                            },
                        },
                        "security_clearance": {"type": ["string", "null"]},
                        "travel_requirements": {"type": ["string", "null"]},
                    },
                    "additionalProperties": True,
                },
            )

            if not extracted_response:
                logger.warning(
                    f"Empty LLM response for job details extraction for '{job_title}'. Using rule-based extractions only."
                )
                return self._create_parsed_job_data(job_data, {})

            # Merge LLM extractions with our rule-based extractions
            return self._create_parsed_job_data(job_data, extracted_response)

        except Exception as e:
            error_msg = (
                f"Error during job details extraction for '{job_title}': {str(e)}"
            )
            logger.error(f"{error_msg}\n{format_exc()}")
            current_span.record_exception(e)
            current_span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))

            # Fall back to just the rule-based extractions
            return self._create_parsed_job_data(job_data, {})

        # Process the LLM response and merge with rule-based data
        try:
            # Ensure we have at least some basic fields
            if not isinstance(extracted_response, dict):
                error_msg = f"Expected a JSON object in the response, got: {type(extracted_response)}"
                logger.error(error_msg)
                current_span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
                return self._create_parsed_job_data(job_data, {})

            # Ensure we have at least a title or use the provided one
            if "job_title_extracted" not in extracted_response and job_title != "N/A":
                extracted_response["job_title_extracted"] = job_title

            # Ensure required list fields are properly initialized
            list_fields = [
                "required_certifications",
                "preferred_certifications",
                "required_skills",
                "preferred_skills",
                "key_responsibilities",
                "required_education",
                "preferred_education",
                "company_culture_hints",
                "tools_technologies",
            ]

            # Process list fields to ensure they contain valid dictionaries
            for field in list_fields:
                if field not in extracted_response or extracted_response[field] is None:
                    extracted_response[field] = []
                elif not isinstance(extracted_response[field], list):
                    # Convert single item to a list if it's not already a list
                    extracted_response[field] = (
                        [extracted_response[field]]
                        if extracted_response[field] is not None
                        else []
                    )

                # Special handling for skills to ensure they are in the correct format
                if field in ["required_skills", "preferred_skills"]:
                    valid_skills = []
                    for skill in extracted_response[field]:
                        if isinstance(skill, dict):
                            # Ensure the skill has at least a 'name' field
                            if "name" not in skill and "skill" in skill:
                                skill["name"] = skill["skill"]
                                del skill["skill"]

                            # Ensure required fields are present
                            if "name" in skill:
                                valid_skill = {"name": str(skill["name"])}

                                # Add optional fields if they exist
                                if "level" in skill and skill["level"] is not None:
                                    valid_skill["level"] = str(skill["level"])
                                if (
                                    "years_experience" in skill
                                    and skill["years_experience"] is not None
                                ):
                                    valid_skill["years_experience"] = int(
                                        skill["years_experience"]
                                    )

                                valid_skills.append(valid_skill)

                    extracted_response[field] = valid_skills

                # Ensure other list fields have the correct format
                elif field in [
                    "key_responsibilities",
                    "required_education",
                    "preferred_education",
                    "company_culture_hints",
                    "tools_technologies",
                    "required_certifications",
                    "preferred_certifications",
                ]:
                    valid_items = []
                    for item in extracted_response[field]:
                        if isinstance(item, dict) and "name" in item:
                            valid_items.append({"name": str(item["name"])})
                        elif isinstance(item, str):
                            valid_items.append({"name": item})

                    extracted_response[field] = valid_items

            # Create the final ParsedJobData object
            return self._create_parsed_job_data(job_data, extracted_response)

        except Exception as e:
            error_msg = f"Error processing LLM response for job '{job_title}': {str(e)}"
            logger.error(f"{error_msg}\n{format_exc()}")
            current_span.record_exception(e)
            current_span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))

            # Fall back to just the rule-based extractions
            return self._create_parsed_job_data(job_data, {})

    @tracer.start_as_current_span("analyze_resume_suitability")
    async def analyze_resume_suitability(
        self, resume_data: ResumeData, job_data: Dict[str, Any]
    ) -> Optional[JobAnalysisResult]:
        """Analyze the suitability of a resume for a given job with enhanced validation and error handling.

        This method performs a detailed analysis of how well a candidate's resume matches a job description.
        It uses an LLM to generate a suitability score and detailed analysis, with robust error handling
        and validation of the LLM's response.

        Args:
            resume_data: The structured resume data to analyze. Must be a valid ResumeData object.
            job_data: The job data to analyze against. Must include at least a 'description' field.

        Returns:
            JobAnalysisResult: A structured analysis result with score and detailed feedback,
                            or None if the analysis could not be completed.

        Raises:
            ValueError: If input validation fails.
            RuntimeError: If there's an error during the analysis process.
        """
        current_span = trace.get_current_span()
        job_title = job_data.get("title", "N/A")
        job_description = job_data.get("description")
        current_span.set_attribute("job_title", job_title)
        current_span.set_attribute("analysis_started_at", datetime.utcnow().isoformat())

        # Enhanced input validation with detailed error messages
        if not resume_data:
            error_msg = "Resume data cannot be None"
            logger.error(error_msg)
            current_span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
            raise ValueError(error_msg)

        if not resume_data.summary and not (
            resume_data.experience or resume_data.education or resume_data.skills
        ):
            error_msg = (
                f"Insufficient resume data for analysis. Missing required fields."
            )
            logger.warning(f"{error_msg} Job: {job_title}")
            current_span.set_status(trace.Status(trace.StatusCode.ERROR, error_msg))
            current_span.set_attribute(
                "suitability_skipped_reason", "insufficient_resume_data"
            )
            return None

        if not job_description:
            error_msg = f"Job description is required for analysis. Job: {job_title}"
            logger.warning(error_msg)
            current_span.set_status(
                trace.Status(trace.StatusCode.ERROR, "Missing job description")
            )
            current_span.set_attribute(
                "suitability_skipped_reason", "missing_job_description"
            )
            return None

        try:
            with tracer.start_as_current_span("prepare_analysis_inputs"):
                # Log basic info about the analysis
                logger.info(
                    f"Starting resume suitability analysis for job: {job_title}"
                )

                # Convert resume data to JSON with error handling
                try:
                    resume_json = resume_data.json(indent=2, exclude_unset=True)
                except Exception as e:
                    error_msg = f"Failed to serialize resume data: {str(e)}"
                    logger.error(error_msg)
                    current_span.record_exception(e)
                    current_span.set_status(
                        trace.Status(trace.StatusCode.ERROR, error_msg)
                    )
                    return None

                # Validate and prepare job data
                try:
                    # Create a sanitized copy of job_data with only the fields we need
                    valid_job_data = {
                        "title": job_data.get("title", "N/A"),
                        "description": job_description,
                        "required_skills": job_data.get("required_skills", []),
                        "preferred_skills": job_data.get("preferred_skills", []),
                        "required_education": job_data.get("required_education", []),
                        "experience_years": job_data.get("experience_years"),
                        "job_type": job_data.get("job_type"),
                        "work_model": job_data.get("work_model"),
                    }

                    # Ensure all list fields are actually lists
                    for field in [
                        "required_skills",
                        "preferred_skills",
                        "required_education",
                    ]:
                        if field in valid_job_data and not isinstance(
                            valid_job_data[field], list
                        ):
                            valid_job_data[field] = (
                                [valid_job_data[field]] if valid_job_data[field] else []
                            )

                    # Convert to JSON string and back to ensure it's serializable
                    job_data_str = json.dumps(
                        valid_job_data, default=str, ensure_ascii=False
                    )
                    job_data_serializable = json.loads(job_data_str)

                except (TypeError, ValueError, json.JSONDecodeError) as e:
                    error_msg = f"Failed to prepare job data for analysis: {str(e)}"
                    logger.error(f"{error_msg} Job: {job_title}")
                    current_span.record_exception(e)
                    current_span.set_status(
                        trace.Status(trace.StatusCode.ERROR, error_msg)
                    )
                    return None

                # Prepare context for the prompt
                context = {
                    "resume_data": resume_data.dict(
                        exclude_unset=True, exclude_none=True
                    ),
                    "job_data": job_data_serializable,
                    "current_date": datetime.utcnow().strftime("%Y-%m-%d"),
                }

                # Load and render the prompt template
                try:
                    with tracer.start_as_current_span("render_prompt"):
                        suitability_prompt_template = load_template(
                            "suitability_prompt_file"
                        )
                        if not suitability_prompt_template:
                            raise ValueError(
                                "Failed to load suitability prompt template"
                            )

                        prompt = suitability_prompt_template.render(context)

                        # Log a sample of the prompt (not the whole thing to avoid log spam)
                        prompt_sample = (
                            (prompt[:500] + "...") if len(prompt) > 500 else prompt
                        )
                        logger.debug(
                            f"Suitability analysis prompt sample: {prompt_sample}"
                        )

                        # Define the expected output schema for the LLM response
                        output_schema = {
                            "type": "object",
                            "required": [
                                "suitability_score",
                                "summary",
                                "strengths",
                                "weaknesses",
                                "recommendations",
                            ],
                            "properties": {
                                "suitability_score": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 100,
                                    "description": "Overall suitability score from 0-100",
                                },
                                "summary": {
                                    "type": "string",
                                    "description": "Brief summary of the candidate's fit for the position",
                                },
                                "strengths": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of key strengths that make the candidate a good fit",
                                },
                                "weaknesses": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of potential weaknesses or gaps in the candidate's profile",
                                },
                                "recommendations": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of recommendations for the candidate to improve their fit",
                                },
                                "key_qualifications": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "requirement": {"type": "string"},
                                            "match_strength": {"type": "string"},
                                            "evidence": {"type": "string"},
                                        },
                                        "required": [
                                            "requirement",
                                            "match_strength",
                                            "evidence",
                                        ],
                                    },
                                },
                            },
                        }

                        # Call the LLM with the prompt and schema validation
                        try:
                            with tracer.start_as_current_span("llm_analysis"):
                                logger.info(
                                    "Calling LLM for resume suitability analysis"
                                )

                                # Make the LLM call with retries and timeout
                                response = await self._call_llm_async(
                                    prompt=prompt,
                                    task_name="resume_suitability_analysis",
                                    output_schema=output_schema,
                                    temperature=0.3,  # Lower temperature for more consistent results
                                    max_tokens=2000,  # Sufficient for detailed analysis
                                    timeout_seconds=60,  # Reasonable timeout for this operation
                                )

                                if not response:
                                    error_msg = "Received empty response from LLM for resume suitability analysis"
                                    logger.error(error_msg)
                                    current_span.set_status(
                                        trace.Status(trace.StatusCode.ERROR, error_msg)
                                    )
                                    return None

                                # Log the response for debugging (truncated)
                                response_str = json.dumps(response, indent=2)
                                logger.debug(
                                    f"LLM analysis response: {response_str[:500]}..."
                                    if len(response_str) > 500
                                    else response_str
                                )

                                # Validate the response structure
                                try:
                                    # Ensure required fields are present and have the correct types
                                    if not (
                                        isinstance(
                                            response.get("suitability_score"),
                                            (int, float),
                                        )
                                        and 0 <= response["suitability_score"] <= 100
                                    ):
                                        raise ValueError(
                                            "Invalid or missing suitability_score (must be 0-100)"
                                        )

                                    if (
                                        not isinstance(response.get("summary"), str)
                                        or not response["summary"].strip()
                                    ):
                                        raise ValueError("Missing or invalid summary")

                                    # Convert any non-list values to single-item lists
                                    for field in [
                                        "strengths",
                                        "weaknesses",
                                        "recommendations",
                                    ]:
                                        if field not in response:
                                            response[field] = []
                                        elif not isinstance(response[field], list):
                                            response[field] = [str(response[field])]

                                    # Ensure all list items are strings
                                    for field in [
                                        "strengths",
                                        "weaknesses",
                                        "recommendations",
                                    ]:
                                        response[field] = [
                                            str(item)
                                            for item in response[field]
                                            if item
                                        ]

                                    # Process key qualifications if present
                                    if "key_qualifications" in response and isinstance(
                                        response["key_qualifications"], list
                                    ):
                                        valid_qualifications = []
                                        for qual in response["key_qualifications"]:
                                            if not isinstance(qual, dict):
                                                continue

                                            valid_qual = {
                                                "requirement": str(
                                                    qual.get("requirement", "")
                                                ),
                                                "match_strength": str(
                                                    qual.get(
                                                        "match_strength", "unknown"
                                                    )
                                                ).lower(),
                                                "evidence": str(
                                                    qual.get("evidence", "")
                                                ),
                                            }

                                            # Only add if we have a requirement and evidence
                                            if (
                                                valid_qual["requirement"]
                                                and valid_qual["evidence"]
                                            ):
                                                valid_qualifications.append(valid_qual)

                                        response["key_qualifications"] = (
                                            valid_qualifications
                                        )
                                    else:
                                        response["key_qualifications"] = []

                                    # Create the JobAnalysisResult object
                                    result = JobAnalysisResult(
                                        suitability_score=float(
                                            response["suitability_score"]
                                        ),
                                        summary=response["summary"].strip(),
                                        strengths=response.get("strengths", []),
                                        weaknesses=response.get("weaknesses", []),
                                        recommendations=response.get(
                                            "recommendations", []
                                        ),
                                        key_qualifications=response.get(
                                            "key_qualifications", []
                                        ),
                                    )

                                    # Log success
                                    logger.info(
                                        f"Successfully analyzed resume suitability. Score: {result.suitability_score}"
                                    )
                                    current_span.set_attribute(
                                        "suitability_score", result.suitability_score
                                    )
                                    current_span.set_status(
                                        trace.Status(
                                            trace.StatusCode.OK,
                                            "Analysis completed successfully",
                                        )
                                    )

                                    return result

                                except (KeyError, ValueError, TypeError) as e:
                                    error_msg = (
                                        f"Invalid response format from LLM: {str(e)}"
                                    )
                                    logger.error(
                                        f"{error_msg}\nResponse: {response_str}"
                                    )
                                    current_span.record_exception(e)
                                    current_span.set_status(
                                        trace.Status(trace.StatusCode.ERROR, error_msg)
                                    )
                                    return None

                        except Exception as e:
                            error_msg = f"Error during LLM call for resume suitability analysis: {str(e)}"
                            logger.error(error_msg)
                            current_span.record_exception(e)
                            current_span.set_status(
                                trace.Status(trace.StatusCode.ERROR, error_msg)
                            )
                            return None

                except Exception as e:
                    error_msg = (
                        f"Failed to render suitability analysis prompt: {str(e)}"
                    )
                    logger.error(error_msg)
                    current_span.set_status(
                        trace.Status(trace.StatusCode.ERROR, error_msg)
                    )
                    return None

        except json.JSONDecodeError as e:
            error_msg = (
                f"JSON decoding error during resume suitability analysis: {str(e)}"
            )
            logger.error(f"{error_msg}\n{format_exc()}")
            current_span.record_exception(e)
            current_span.set_status(
                trace.Status(trace.StatusCode.ERROR, "Invalid JSON in analysis")
            )
            return None

        except asyncio.TimeoutError as e:
            error_msg = "Analysis timed out while processing resume suitability"
            logger.error(f"{error_msg}: {str(e)}")
            current_span.record_exception(e)
            current_span.set_status(
                trace.Status(trace.StatusCode.ERROR, "Analysis timed out")
            )
            return None

        except Exception as e:
            error_msg = f"Unexpected error during resume suitability analysis: {str(e)}"
            logger.error(f"{error_msg}\n{format_exc()}")
            current_span.record_exception(e)
            current_span.set_status(
                trace.Status(trace.StatusCode.ERROR, "Analysis failed unexpectedly")
            )
            return None

        finally:
            # Clean up any resources and finalize tracing
            analysis_duration = (
                datetime.utcnow()
                - datetime.fromisoformat(
                    current_span.attributes.get(
                        "analysis_started_at", datetime.utcnow().isoformat()
                    )
                )
            ).total_seconds()

            current_span.set_attribute("analysis_duration_seconds", analysis_duration)

            # Log completion status
            if current_span.status.status_code == trace.StatusCode.OK:
                logger.info(
                    f"Completed resume suitability analysis in {analysis_duration:.2f} seconds"
                )
            else:
                logger.warning(
                    f"Resume suitability analysis completed with errors in {analysis_duration:.2f} seconds"
                )

        # Handle the LLM response
        try:
            # If we get here, the response should already be a dict due to output_schema validation
            if not isinstance(combined_response, dict):
                logger.error(
                    f"Unexpected response type from LLM for job analysis: {type(combined_response).__name__}"
                )
                current_span.set_status(
                    trace.Status(trace.StatusCode.ERROR, "Invalid LLM response type")
                )
                return None

            # Extract analysis data from the response
            analysis_data = combined_response.get("analysis")

            # If no 'analysis' key, the entire response might be the analysis data
            if analysis_data is None and all(
                key in combined_response
                for key in ["suitability_score", "justification"]
            ):
                analysis_data = combined_response

            if not analysis_data or not isinstance(analysis_data, dict):
                logger.error(
                    f"JobAnalyzer: Could not extract valid analysis data for '{job_title}'."
                )
                logger.debug(
                    f"Response for '{job_title}': {json.dumps(combined_response, indent=2, default=str)}"
                )
                current_span.set_status(
                    trace.Status(trace.StatusCode.ERROR, "Invalid analysis data format")
                )
                return None

            # Ensure required fields are present with defaults
            analysis_data.setdefault("pros", [])
            analysis_data.setdefault("cons", [])
            analysis_data.setdefault("missing_keywords", [])

            # Ensure the score is within valid range
            if "suitability_score" in analysis_data:
                analysis_data["suitability_score"] = max(
                    0, min(100, int(analysis_data["suitability_score"]))
                )
            else:
                analysis_data["suitability_score"] = 0

            # Create the analysis result object
            analysis_result = JobAnalysisResult(**analysis_data)

            logger.info(
                f"JobAnalyzer: Suitability score for '{job_title}': {analysis_result.suitability_score}%"
            )
            current_span.set_status(
                trace.Status(trace.StatusCode.OK, "Analysis completed")
            )
            return analysis_result

        except Exception as e:
            log_exception(
                f"JobAnalyzer: Failed to process analysis result for '{job_title}': {e}",
                e,
            )
            logger.debug(
                f"Analysis data that caused error: {json.dumps(analysis_data, indent=2, default=str) if 'analysis_data' in locals() else 'N/A'}"
            )
            current_span.record_exception(e)
            current_span.set_status(
                trace.Status(
                    trace.StatusCode.ERROR, "Analysis result processing failed"
                )
            )
            return None


# --- Helper functions for async initialization ---
async def create_analyzer(analyzer_class, provider_name: Optional[str] = None):
    """Helper function to create and initialize an analyzer asynchronously.

    Args:
        analyzer_class: The analyzer class to instantiate (e.g., ResumeAnalyzer, JobAnalyzer)
        provider_name: Optional provider name. If not provided, uses the default from config.

    Returns:
        An initialized instance of the specified analyzer class.
    """
    if provider_name is None:
        provider_name = getattr(config, 'default_provider', 'openai')
    analyzer = analyzer_class(provider_name)
    await analyzer.initialize()
    return analyzer


# --- Standalone functions (current structure, to be refactored or removed if class methods are preferred everywhere) ---
# The following functions are from the current analyzer.py and might be deprecated or integrated into classes.
# For now, keeping them to see how they fit into the refactor.


async def original_extract_job_details_async(
    job_description_text: str, job_title: str = "N/A"
):
    """Extract job details asynchronously (legacy function).

    Note: This is a compatibility wrapper that will be phased out.
    Use JobAnalyzer class directly for new code.
    """
    analyzer = await create_analyzer(JobAnalyzer)
    return await analyzer.extract_job_details_async(job_description_text, job_title)


async def original_analyze_resume_suitability(
    resume_text: str, job_description_text: str, job_title: str = "N/A"
):
    """Analyze resume suitability asynchronously (legacy function).

    Note: This is a compatibility wrapper that will be phased out.
    Use ResumeAnalyzer class directly for new code.
    """
    analyzer = await create_analyzer(ResumeAnalyzer)
    job_analyzer = await create_analyzer(JobAnalyzer)

    # Extract resume data
    resume_data = await analyzer.extract_resume_data_async(resume_text)

    # Extract job details
    job_data = await job_analyzer.extract_job_details_async(
        job_description_text, job_title
    )

    # Analyze suitability
    return await job_analyzer.analyze_resume_suitability(resume_data, job_data)
