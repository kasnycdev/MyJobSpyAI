import openai
import ollama
# cspell:ignore generativeai, genai, ollama, Jinja2, autoescape, lstrip_blocks, api_core
import google.generativeai as genai # Corrected import alias
import google.api_core.exceptions # Added for DeadlineExceeded
import json
from contextlib import suppress # Added for Sourcery fix
import os
import asyncio
from typing import Dict, Optional, Any, Union # Removed List
# from rich.console import Console # No longer needed directly here
# import traceback # Unused
import time
from collections import defaultdict # Added for stats

# Import Jinja2 components
try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateNotFound, TemplateSyntaxError
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

from analysis.models import ResumeData, JobAnalysisResult, ParsedJobData
from filtering.filter_utils import DateEncoder
from config import settings
import logging
from logging_utils import MODEL_OUTPUT_LOGGER_NAME # Import the constant for model output logger

# Get a logger for this module
logger = logging.getLogger(__name__)
# Get the specific logger for model outputs
model_output_logger = logging.getLogger(MODEL_OUTPUT_LOGGER_NAME)

# Import OpenTelemetry trace and metrics modules
from opentelemetry import trace, metrics # Ensure trace and metrics modules are always imported

# Import tracer and meter instances from logging_utils
try:
    from logging_utils import tracer as global_tracer_instance, meter as global_meter_instance
    
    if global_tracer_instance is None: # Check if OTEL was disabled in logging_utils for tracer
        tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
        logger.warning("OpenTelemetry tracer not configured in logging_utils (global_tracer_instance is None), using NoOpTracer for analyzer.")
    else:
        tracer = global_tracer_instance # Use the instance from logging_utils
        logger.info("Using global_tracer_instance from logging_utils for analyzer.")

    if global_meter_instance is None: # Check if OTEL was disabled in logging_utils for meter
        meter = metrics.get_meter(__name__, meter_provider=metrics.NoOpMeterProvider())
        logger.warning("OpenTelemetry meter not configured in logging_utils (global_meter_instance is None), using NoOpMeter for analyzer.")
    else:
        meter = global_meter_instance # Use the instance from logging_utils
        logger.info("Using global_meter_instance from logging_utils for analyzer.")
        
except ImportError:
    # Fallback to NoOp versions if logging_utils or its tracer/meter cannot be imported
    tracer = trace.get_tracer(__name__, tracer_provider=trace.NoOpTracerProvider())
    meter = metrics.get_meter(__name__, meter_provider=metrics.NoOpMeterProvider())
    logger.error("Could not import global_tracer_instance/global_meter_instance from logging_utils. Using NoOp versions for analyzer.", exc_info=True)


# Helper function for logging exceptions (now uses standard logger)
def log_exception(message, exception): # This could also be traced if it becomes complex
    logger.error(message, exc_info=True)


# --- Jinja2 Environment Setup ---
PROMPT_TEMPLATE_LOADER = None
if JINJA2_AVAILABLE:
    try:
        prompts_dir = settings.get('analysis', {}).get('prompts_dir')
        if prompts_dir and os.path.isdir(prompts_dir):
            logger.info(f"Initializing Jinja2 environment for prompts in: {prompts_dir}")
            PROMPT_TEMPLATE_LOADER = Environment(
                loader=FileSystemLoader(prompts_dir),
                autoescape=select_autoescape(['html', 'xml']),
                trim_blocks=True,
                lstrip_blocks=True
            )
        else:
            logger.error(f"Jinja2 prompts directory not found or invalid in config: {prompts_dir}")
            JINJA2_AVAILABLE = False
    except Exception as jinja_err:
        log_exception(f"Failed to initialize Jinja2 environment: {jinja_err}", jinja_err)
        JINJA2_AVAILABLE = False
else:
    logger.error("Jinja2 library not installed. Prompts cannot be loaded.")

def load_template(template_name_key: str):
    if not JINJA2_AVAILABLE or not PROMPT_TEMPLATE_LOADER:
        raise RuntimeError("Jinja2 environment not available for loading prompt templates.")
    template_filename = settings.get('analysis', {}).get(template_name_key)
    if not template_filename:
        raise ValueError(f"Missing template filename configuration for key '{template_name_key}'")
    try:
        template = PROMPT_TEMPLATE_LOADER.get_template(template_filename)
        logger.info(f"Successfully loaded Jinja2 template: {template_filename}")
        return template
    except TemplateNotFound:
        logger.error(f"Jinja2 template file not found: {template_filename} in {settings.get('analysis', {}).get('prompts_dir')}")
        raise
    except TemplateSyntaxError as syn_err:
        logger.error(f"Syntax error in Jinja2 template {template_filename}: {syn_err}")
        raise
    except Exception as e:
        log_exception(f"Error loading Jinja2 template {template_filename}: {e}", e)
        raise

class BaseAnalyzer:
    def __init__(self, provider_config_key: str):
        self.provider = settings.get('llm_provider', 'openai').lower()
        logger.info(f"Initializing LLM client for provider: {self.provider} using config key '{provider_config_key}'")

        self.model_name: Optional[str] = None
        self.sync_client: Union[openai.OpenAI, ollama.Client, None] = None
        self.async_client: Union[openai.AsyncOpenAI, ollama.AsyncClient, genai.GenerativeModel, None] = None
        self.request_timeout: Optional[float] = None # Changed to float
        self.max_retries: int = 2
        self.retry_delay: int = 5
        self.ollama_base_url: Optional[str] = None

        with tracer.start_as_current_span("analyzer_init") as span:
            span.set_attribute("provider_config_key", provider_config_key)
            self._initialize_llm_client(provider_config_key)
            self._check_connection_and_model()

    # LLM Call Metrics (can be moved to a separate metrics module if it grows)
    llm_calls_counter = meter.create_counter(
        name="llm.calls.total",
        description="Total number of LLM calls.",
        unit="1"
    )
    llm_successful_calls_counter = meter.create_counter(
        name="llm.calls.successful",
        description="Number of successful LLM calls.",
        unit="1"
    )
    llm_failed_calls_counter = meter.create_counter(
        name="llm.calls.failed",
        description="Number of failed LLM calls (after retries).",
        unit="1"
    )
    llm_call_duration_histogram = meter.create_histogram(
        name="llm.call.duration",
        description="Duration of LLM calls.",
        unit="s"
    )
    llm_prompt_chars_histogram = meter.create_histogram(
        name="llm.prompt.chars",
        description="Number of characters in LLM prompts.",
        unit="char"
    )
    llm_response_chars_histogram = meter.create_histogram(
        name="llm.response.chars",
        description="Number of characters in LLM responses.",
        unit="char"
    )
    # Legacy stats, can be phased out or kept for internal logging
    _llm_call_stats: Dict[str, Any] = { # TODO: Review if this is still needed with OTEL metrics
        "total_calls": 0, # Covered by llm_calls_counter
        "successful_calls": 0,
        "failed_calls": 0,
        "total_prompt_chars": 0,
        "total_response_chars": 0,
        "total_duration_seconds": 0.0,
        "errors_by_type": defaultdict(int),
        "calls_by_task": defaultdict(lambda: defaultdict(int))
    }

    @classmethod
    def log_llm_call_summary(cls):
        if cls._llm_call_stats['total_calls'] == 0:
            logger.info("No LLM calls were made in this session.")
            return

        logger.info("--- LLM Call Statistics Summary ---")
        logger.info(f"Total LLM Calls Attempted: {cls._llm_call_stats['total_calls']}")
        logger.info(f"  Successful Calls: {cls._llm_call_stats['successful_calls']}")
        logger.info(f"  Failed Calls (after retries): {cls._llm_call_stats['failed_calls']}")
        
        successful_calls_count = cls._llm_call_stats['successful_calls']
        if successful_calls_count > 0:
            avg_duration = (cls._llm_call_stats['total_duration_seconds'] / successful_calls_count)
            logger.info(f"Average Call Duration (per successful call): {avg_duration:.2f}s")
        else:
            logger.info("Average Call Duration (per successful call): N/A (no successful calls)")

        logger.info(f"Total Prompt Characters Sent: {cls._llm_call_stats['total_prompt_chars']:,}")
        logger.info(f"Total Response Characters Received: {cls._llm_call_stats['total_response_chars']:,}")

        if cls._llm_call_stats['errors_by_type']:
            logger.info("\nErrors by Type (across all attempts):")
            for err_type, count in sorted(cls._llm_call_stats['errors_by_type'].items()):
                logger.info(f"  - {err_type}: {count}")
        
        if cls._llm_call_stats['calls_by_task']:
            logger.info("\nCalls by Task:")
            for task, stats in sorted(cls._llm_call_stats['calls_by_task'].items()):
                logger.info(f"  - Task: '{task}'")
                logger.info(f"    Successful: {stats.get('success', 0)}, Failed: {stats.get('fail', 0)}")
        logger.info("--- End LLM Call Statistics Summary ---")

    def _load_common_provider_config(self, provider_key: str) -> Dict[str, Any]:
        cfg = settings.get(provider_key, {})
        self.model_name = cfg.get('model')
        self.request_timeout = cfg.get('request_timeout')
        if self.request_timeout is None:
            self.request_timeout = 120.0 
            logger.warning(f"No request_timeout in config for '{provider_key}', defaulting to {self.request_timeout}s.")
        elif not isinstance(self.request_timeout, (int, float)):
            try:
                self.request_timeout = float(self.request_timeout)
                logger.warning(f"Converted request_timeout for '{provider_key}' to float: {self.request_timeout}s.")
            except ValueError:
                logger.error(f"Invalid request_timeout value '{self.request_timeout}' for '{provider_key}'. Using default 120.0s.")
                self.request_timeout = 120.0
        self.max_retries = cfg.get('max_retries', self.max_retries)
        self.retry_delay = cfg.get('retry_delay', self.retry_delay)
        return cfg

    def _initialize_openai_client(self, provider_config_key: str):
        cfg = self._load_common_provider_config(provider_config_key)
        base_url = cfg.get('base_url')
        api_key = cfg.get('api_key', 'lm-studio')
        if not base_url or not self.model_name:
            raise ValueError(f"OpenAI provider ('{provider_config_key}') requires 'base_url' and 'model' in config.")
        self.sync_client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=self.request_timeout)
        self.async_client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=self.request_timeout)
        logger.info(f"OpenAI client initialized for model '{self.model_name}' at {base_url}.")

    def _initialize_ollama_client(self, provider_config_key: str):
        cfg = self._load_common_provider_config(provider_config_key)
        base_url = cfg.get('base_url')
        self.ollama_base_url = base_url
        if not base_url or not self.model_name:
            raise ValueError(f"Ollama provider ('{provider_config_key}') requires 'base_url' and 'model' in config.")
        self.sync_client = ollama.Client(host=base_url, timeout=self.request_timeout) # type: ignore
        self.async_client = ollama.AsyncClient(host=base_url, timeout=self.request_timeout) # type: ignore
        logger.info(f"Ollama client initialized for model '{self.model_name}' at {base_url} with timeout {self.request_timeout}s.")

    def _initialize_gemini_client(self, provider_config_key: str):
        cfg = self._load_common_provider_config(provider_config_key)
        api_key = cfg.get('api_key') or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError(f"Gemini provider ('{provider_config_key}') requires 'api_key' in config or GOOGLE_API_KEY env var.")
        if not self.model_name:
            raise ValueError(f"Gemini provider ('{provider_config_key}') requires 'model' name in config.")
        genai.configure(api_key=api_key)
        safety_settings = [{"category": c, "threshold": "BLOCK_NONE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        self.async_client = genai.GenerativeModel(model_name=self.model_name, safety_settings=safety_settings)
        self.sync_client = None
        logger.info(f"Gemini client initialized for model '{self.model_name}'.")

    def _initialize_llm_client(self, provider_config_key: str):
        try:
            if self.provider == "openai": self._initialize_openai_client(provider_config_key)
            elif self.provider == "ollama": self._initialize_ollama_client(provider_config_key)
            elif self.provider == "gemini": self._initialize_gemini_client(provider_config_key)
            else: raise ValueError(f"Unsupported llm_provider: '{self.provider}'. Choose 'openai', 'ollama', or 'gemini'.")
        except Exception as e:
            log_exception(f"Failed to initialize LLM client for provider '{self.provider}': {e}", e)
            raise RuntimeError(f"LLM client initialization failed: {e}") from e

    def _check_openai_connection(self):
        if not self.sync_client: raise RuntimeError("OpenAI sync client not initialized.")
        base_url = str(getattr(self.sync_client, 'base_url', 'N/A'))
        logger.info(f"Attempting to list models from OpenAI-compatible server at {base_url}...")
        models_response = self.sync_client.models.list()
        available_model_ids = [model.id for model in models_response.data]
        logger.info(f"Available models reported by API: {available_model_ids}")
        logger.info(f"Successfully connected to OpenAI-compatible server at {base_url}.")
        logger.info(f"Configured to use model: '{self.model_name}'. Ensure this model is loaded/available.")

    def _check_ollama_connection(self):
        if not self.sync_client: raise RuntimeError("Ollama sync client not initialized.")
        if not self.ollama_base_url: raise RuntimeError("Ollama base URL not set.")
        logger.info(f"Attempting to list models from Ollama server at {self.ollama_base_url}...")
        response = self.sync_client.list() # type: ignore
        available_models = [m.get('name') for m in response.get('models', []) if m.get('name')]
        logger.info(f"Available Ollama models: {available_models}")
        if self.model_name not in available_models:
            logger.warning(f"Configured Ollama model '{self.model_name}' not found. Ensure it is pulled.")
        else: logger.info(f"Configured Ollama model '{self.model_name}' found.")
        logger.info(f"Successfully connected to Ollama server at {self.ollama_base_url}.")

    def _check_connection_and_model(self):
        logger.info(f"Checking connection for provider '{self.provider}' and model '{self.model_name}'...")
        try:
            if self.provider == "openai": self._check_openai_connection()
            elif self.provider == "ollama": self._check_ollama_connection()
            elif self.provider == "gemini":
                logger.info("Attempting to list models from Google AI...")
                found_model = any('generateContent' in m.supported_generation_methods and self.model_name in m.name for m in genai.list_models())
                if not found_model:
                    logger.error(f"Error: Configured Gemini model '{self.model_name}' not found or unsuitable.")
                    with suppress(Exception):
                        generative_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                        logger.info(f"Available generative models: {generative_models[:10]}")
                    raise ValueError(f"Gemini model '{self.model_name}' not suitable.")
                else: logger.info(f"Successfully verified Gemini model '{self.model_name}'.")
        except (openai.APIConnectionError, ollama.ResponseError, Exception) as e:
            log_exception(f"{self.provider.upper()} API Connection/Setup Error: {type(e).__name__} - {e}", e)
            raise ConnectionError(f"Failed to connect/setup {self.provider.upper()} provider: {e}") from e
        except Exception as e:
            log_exception(f"Unexpected error during LLM connection check: {e}", e)
            raise ConnectionError(f"Unexpected error during LLM connection check: {e}") from e

    async def _call_openai_llm_async(self, prompt: str) -> Optional[str]:
        if not isinstance(self.async_client, openai.AsyncOpenAI) or not self.model_name:
            raise TypeError("OpenAI client or model_name not configured correctly.")
        response = await self.async_client.chat.completions.create(model=self.model_name, messages=[{'role': 'user', 'content': prompt}], temperature=0.1)
        return response.choices[0].message.content

    async def _call_ollama_llm_async(self, prompt: str) -> Optional[str]:
        if not isinstance(self.async_client, ollama.AsyncClient) or not self.model_name:
            raise TypeError("Ollama client or model_name not configured correctly.")
        response = await self.async_client.chat(model=self.model_name, messages=[{'role': 'user', 'content': prompt}], format='json', options={'temperature': 0.1}) # type: ignore
        return response['message']['content'] # type: ignore

    async def _call_gemini_llm_async(self, prompt: str) -> Optional[str]:
        if not isinstance(self.async_client, genai.GenerativeModel) or not self.model_name:
            raise TypeError("Gemini client or model_name not configured correctly.")
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json", temperature=0.1)
        response = await self.async_client.generate_content_async(prompt, generation_config=generation_config, request_options={'timeout': self.request_timeout})
        if not response.candidates:
            raise genai.types.generation_types.BlockedPromptException(f"Gemini response blocked. Prompt feedback: {response.prompt_feedback}")
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            logger.debug(f"Gemini Usage - Prompt: {usage.prompt_token_count}, Candidates: {usage.candidates_token_count}, Total: {usage.total_token_count} tokens")
        return response.text

    @tracer.start_as_current_span("call_llm_async")
    async def _call_llm_async(self, prompt: str, task_name: str) -> Optional[Dict[str, Any]]:
        current_span = trace.get_current_span()
        current_span.set_attribute("llm.task_name", task_name)
        current_span.set_attribute("llm.provider", self.provider)
        current_span.set_attribute("llm.model_name", self.model_name or "N/A")
        current_span.set_attribute("llm.prompt_length_chars", len(prompt))

        if not self.async_client or not self.model_name:
            logger.error(f"LLM client or model name not initialized properly for _call_llm_async (Task: {task_name}).")
            current_span.set_status(trace.Status(trace.StatusCode.ERROR, "LLM client/model not initialized"))
            # Increment legacy stats for failure before error
            BaseAnalyzer._llm_call_stats["total_calls"] += 1
            BaseAnalyzer._llm_call_stats["calls_by_task"][task_name]["fail"] = BaseAnalyzer._llm_call_stats["calls_by_task"][task_name].get("fail", 0) + 1
            BaseAnalyzer._llm_call_stats["failed_calls"] +=1
            # OTEL Metrics
            self.llm_calls_counter.add(1, {"task": task_name, "provider": self.provider, "model": self.model_name or "N/A"})
            self.llm_failed_calls_counter.add(1, {"task": task_name, "provider": self.provider, "model": self.model_name or "N/A", "error_type": "InitializationError"})
            raise RuntimeError(f"LLM client or model name not initialized properly (Task: {task_name}).")

        # Legacy stats update (can be removed if fully relying on OTEL metrics)
        BaseAnalyzer._llm_call_stats["total_calls"] += 1
        BaseAnalyzer._llm_call_stats["total_prompt_chars"] += len(prompt)
        task_stats = BaseAnalyzer._llm_call_stats["calls_by_task"][task_name]
        
        # OTEL Metrics
        self.llm_calls_counter.add(1, {"task": task_name, "provider": self.provider, "model": self.model_name or "N/A"})
        self.llm_prompt_chars_histogram.record(len(prompt), {"task": task_name, "provider": self.provider, "model": self.model_name or "N/A"})
        
        provider_cfg = settings.get(self.provider, {})
        max_retries = provider_cfg.get('max_retries', self.max_retries)
        retry_delay = provider_cfg.get('retry_delay', self.retry_delay)
        analysis_cfg = settings.get('analysis', {})
        max_prompt_chars = analysis_cfg.get('max_prompt_chars', 24000)
        log_full_prompt = analysis_cfg.get('log_full_prompt', False)

        logger.info(f"ASYNC LLM Call ({task_name}): Provider='{self.provider}', Model='{self.model_name}', Prompt Chars={len(prompt)}")
        if log_full_prompt: logger.debug(f"Full prompt for '{task_name}':\n{prompt}")
        elif len(prompt) > 1000: logger.debug(f"Prompt snippet for '{task_name}':\n{prompt[:500]}...\n...{prompt[-500:]}")
        else: logger.debug(f"Prompt for '{task_name}':\n{prompt}")

        if len(prompt) > max_prompt_chars:
            logger.warning(f"Warning ({task_name}): Prompt length ({len(prompt)}) exceeds configured max ({max_prompt_chars}).")

        last_exception = None
        content_str: Optional[str] = None

        for attempt in range(max_retries + 1):
            logger.info(f"ASYNC LLM Call ({task_name}): Attempt {attempt + 1}/{max_retries + 1} starting...")
            attempt_start_time = time.monotonic()
            error_type_str = "UnknownError" 
            try:
                with tracer.start_as_current_span(f"llm_api_call_attempt_{attempt+1}") as attempt_span:
                    attempt_span.set_attribute("llm.attempt_number", attempt + 1)
                    if self.provider == "openai": content_str = await self._call_openai_llm_async(prompt)
                    elif self.provider == "ollama": content_str = await self._call_ollama_llm_async(prompt)
                    elif self.provider == "gemini": content_str = await self._call_gemini_llm_async(prompt)
                    else: raise ValueError(f"Unsupported provider '{self.provider}'.")
                
                attempt_duration = time.monotonic() - attempt_start_time
                self.llm_call_duration_histogram.record(attempt_duration, {"task": task_name, "provider": self.provider, "model": self.model_name, "status": "success" if content_str else "failure_content_none"})
                BaseAnalyzer._llm_call_stats["total_duration_seconds"] += attempt_duration # Legacy
                response_length = len(content_str) if content_str else 0
                self.llm_response_chars_histogram.record(response_length, {"task": task_name, "provider": self.provider, "model": self.model_name})
                BaseAnalyzer._llm_call_stats["total_response_chars"] += response_length # Legacy
                
                logger.info(f"ASYNC LLM Call ({task_name}): Attempt {attempt + 1} completed in {attempt_duration:.2f}s. Response Chars: {response_length}")
                current_span.set_attribute(f"llm.attempt_{attempt+1}.duration_s", attempt_duration)
                current_span.set_attribute(f"llm.attempt_{attempt+1}.response_length_chars", response_length)
                
                if content_str is not None:
                    model_output_logger.info(f"TASK: {task_name} | ATTEMPT: {attempt + 1}\nPROMPT:\n{prompt}\n---\nRESPONSE:\n{content_str}\n==========\n")
                else:
                    model_output_logger.info(f"TASK: {task_name} | ATTEMPT: {attempt + 1}\nPROMPT:\n{prompt}\n---\nRESPONSE: None\n==========\n")

                if content_str is None:
                    last_exception = ValueError("LLM response content was None.")
                    error_type_str = type(last_exception).__name__
                    logger.warning(f"ASYNC ({task_name}) LLM Response None (Attempt {attempt + 1})")
                else:
                    try:
                        content_strip = content_str.strip()
                        if content_strip.startswith("```json"): content_strip = content_strip[7:-3].strip() if content_strip.endswith("```") else content_strip[7:].strip()
                        elif content_strip.startswith("```"): content_strip = content_strip[3:-3].strip() if content_strip.endswith("```") else content_strip[3:].strip()
                        result = json.loads(content_strip)
                        
                        BaseAnalyzer._llm_call_stats["successful_calls"] += 1 # Legacy
                        task_stats["success"] += 1 # Legacy
                        self.llm_successful_calls_counter.add(1, {"task": task_name, "provider": self.provider, "model": self.model_name})
                        logger.info(f"ASYNC ({task_name}): Attempt {attempt + 1} successful. Parsed JSON response.")
                        current_span.set_status(trace.StatusCode.OK)
                        return result
                    except json.JSONDecodeError as json_err:
                        last_exception = json_err
                        error_type_str = type(json_err).__name__
                        logger.warning(f"ASYNC ({task_name}) JSON Decode Error (Attempt {attempt + 1}): {json_err}")
                        logger.debug(f"Problematic content from LLM for '{task_name}': {content_str}")
                        current_span.set_attribute(f"llm.attempt_{attempt+1}.error", error_type_str)
                        current_span.record_exception(json_err)
            
            except (openai.APIConnectionError, ollama.ResponseError, asyncio.TimeoutError, ConnectionError, TimeoutError) as conn_err: # type: ignore
                last_exception = conn_err; error_type_str = type(conn_err).__name__
                logger.warning(f"ASYNC ({task_name}) LLM Connection/Timeout (Attempt {attempt + 1}): {error_type_str} - {conn_err}")
                current_span.set_attribute(f"llm.attempt_{attempt+1}.error", error_type_str)
                current_span.record_exception(conn_err)
            except openai.RateLimitError as rate_err:
                last_exception = rate_err; error_type_str = type(rate_err).__name__
                logger.warning(f"ASYNC ({task_name}) OpenAI Rate Limit (Attempt {attempt + 1}): {rate_err}")
                current_span.set_attribute(f"llm.attempt_{attempt+1}.error", error_type_str)
                current_span.record_exception(rate_err)
            except openai.APIStatusError as status_err:
                last_exception = status_err; error_type_str = type(status_err).__name__
                logger.warning(f"ASYNC ({task_name}) OpenAI API Status {status_err.status_code} (Attempt {attempt + 1}): {status_err.response.text if status_err.response else 'N/A'}")
                current_span.set_attribute(f"llm.attempt_{attempt+1}.error", error_type_str)
                current_span.set_attribute(f"llm.attempt_{attempt+1}.status_code", status_err.status_code)
                current_span.record_exception(status_err)
            except genai.types.generation_types.BlockedPromptException as gemini_block_err: # type: ignore
                last_exception = gemini_block_err; error_type_str = type(gemini_block_err).__name__
                logger.error(f"ASYNC ({task_name}) Gemini Prompt Blocked (Attempt {attempt + 1}): {gemini_block_err}")
                current_span.set_attribute(f"llm.attempt_{attempt+1}.error", error_type_str)
                current_span.record_exception(gemini_block_err)
            except google.api_core.exceptions.DeadlineExceeded as deadline_err: # type: ignore
                last_exception = deadline_err; error_type_str = type(deadline_err).__name__
                logger.error(f"ASYNC ({task_name}) Google API Deadline Exceeded (Attempt {attempt + 1}): {deadline_err}")
                current_span.set_attribute(f"llm.attempt_{attempt+1}.error", error_type_str)
                current_span.record_exception(deadline_err)
            except TypeError as type_err: 
                last_exception = type_err; error_type_str = type(type_err).__name__
                logger.error(f"ASYNC ({task_name}) LLM Client Config Error (Attempt {attempt + 1}): {type_err}", exc_info=True);
                current_span.set_status(trace.Status(trace.StatusCode.ERROR, f"LLM Client Config Error: {type_err}"))
                current_span.record_exception(type_err)
                BaseAnalyzer._llm_call_stats["errors_by_type"][error_type_str] += 1 # Legacy
                self.llm_failed_calls_counter.add(1, {"task": task_name, "provider": self.provider, "model": self.model_name, "error_type": error_type_str})
                break 
            except Exception as e:
                last_exception = e; error_type_str = type(e).__name__
                logger.error(f"ASYNC ({task_name}) Unexpected LLM API Error (Attempt {attempt + 1}): {error_type_str} - {e}", exc_info=True)
                current_span.set_attribute(f"llm.attempt_{attempt+1}.error", error_type_str)
                current_span.record_exception(e)
            
            BaseAnalyzer._llm_call_stats["errors_by_type"][error_type_str] += 1 # Legacy
            self.llm_call_duration_histogram.record(time.monotonic() - attempt_start_time, {"task": task_name, "provider": self.provider, "model": self.model_name, "status": "failure", "error_type": error_type_str})


            if attempt < max_retries:
                current_delay = retry_delay * (2 ** attempt)
                logger.info(f"ASYNC ({task_name}): Retrying in {current_delay:.1f}s...")
                await asyncio.sleep(current_delay)
            else: 
                BaseAnalyzer._llm_call_stats["failed_calls"] += 1 # Legacy
                task_stats["fail"] += 1 # Legacy
                self.llm_failed_calls_counter.add(1, {"task": task_name, "provider": self.provider, "model": self.model_name, "error_type": error_type_str})
                logger.error(f"ASYNC ({task_name}): All {max_retries + 1} attempts failed for LLM call.")
                current_span.set_status(trace.Status(trace.StatusCode.ERROR, f"All LLM attempts failed. Last error: {error_type_str}"))
                if last_exception:
                    logger.error(f"ASYNC ({task_name}): Final error details: {type(last_exception).__name__} - {last_exception}")
        return None

class ResumeAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__(provider_config_key=settings.get('llm_provider', 'openai'))
        # Templates are now loaded dynamically in each method call
        # self._load_prompt_templates() # Removed

    # def _load_prompt_templates(self): # Removed
    #     try:
    #         self.resume_prompt_template = load_template("resume_prompt_file")
    #     except (RuntimeError, ValueError, TemplateNotFound, TemplateSyntaxError) as tmpl_err:
    #         log_exception(f"ResumeAnalyzer: Failed to load resume prompt template: {tmpl_err}", tmpl_err)
    #         raise RuntimeError(f"ResumeAnalyzer prompt template loading failed: {tmpl_err}") from tmpl_err

    @tracer.start_as_current_span("extract_resume_data_async")
    async def extract_resume_data_async(self, resume_text: str) -> Optional[ResumeData]:
        current_span = trace.get_current_span()
        current_span.set_attribute("resume_text_length", len(resume_text))
        MAX_RESUME_CHARS_FOR_LLM = settings.get('analysis', {}).get('max_prompt_chars', 15000) 
        if not resume_text or not resume_text.strip():
            logger.warning("Resume text empty for extraction.")
            return None
        
        resume_text_for_prompt = resume_text
        if len(resume_text) > MAX_RESUME_CHARS_FOR_LLM:
            logger.warning(f"Truncating resume text ({len(resume_text)} > {MAX_RESUME_CHARS_FOR_LLM}) for LLM.")
            resume_text_for_prompt = resume_text[:MAX_RESUME_CHARS_FOR_LLM]

        try:
            resume_prompt_template = load_template("resume_prompt_file") # Load dynamically
            prompt = resume_prompt_template.render(resume_text=resume_text_for_prompt)
        except Exception as render_err:
            log_exception(f"ResumeAnalyzer: Failed to load or render resume extraction prompt: {render_err}", render_err)
            return None

        extracted_json = await self._call_llm_async(prompt, task_name="Resume Extraction")
        if extracted_json:
            try:
                if isinstance(extracted_json, dict):
                    if 'contact_information' not in extracted_json or not isinstance(extracted_json['contact_information'], dict):
                        extracted_json['contact_information'] = {} 
                    
                    resume_data = ResumeData(**extracted_json)
                    logger.info("ResumeAnalyzer: Parsed extracted resume data.")
                    return resume_data
                else:
                    logger.error(f"ResumeAnalyzer: Resume extract response not dict: {type(extracted_json)}")
                    return None
            except Exception as e:
                log_exception(f"ResumeAnalyzer: Failed to validate extracted resume data: {e}", e)
                logger.debug(f"Invalid JSON from LLM for ResumeData: {extracted_json}")
                return None
        else:
            logger.error("ResumeAnalyzer: Failed to get response for resume extraction.")
            return None

class JobAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__(provider_config_key=settings.get('llm_provider', 'openai'))
        # Templates are now loaded dynamically in each method call
        # self._load_prompt_templates() # Removed

    # def _load_prompt_templates(self): # Removed
    #     try:
    #         self.suitability_prompt_template = load_template("suitability_prompt_file")
    #         self.job_extraction_prompt_template = load_template("job_extraction_prompt_file")
    #     except (RuntimeError, ValueError, TemplateNotFound, TemplateSyntaxError) as tmpl_err:
    #         log_exception(f"JobAnalyzer: Failed to load necessary prompt templates: {tmpl_err}", tmpl_err)
    #         raise RuntimeError(f"JobAnalyzer prompt template loading failed: {tmpl_err}") from tmpl_err

    @tracer.start_as_current_span("extract_job_details_async")
    async def extract_job_details_async(self, job_description_text: str, job_title: str = "N/A") -> Optional[ParsedJobData]:
        current_span = trace.get_current_span()
        current_span.set_attribute("job_title_param", job_title)
        current_span.set_attribute("job_description_length", len(job_description_text))

        if not job_description_text or not job_description_text.strip():
            logger.warning(f"Job description for '{job_title}' is empty. Skipping extraction.")
            return None
        
        try:
            job_extraction_prompt_template = load_template("job_extraction_prompt_file") # Load dynamically
            prompt = job_extraction_prompt_template.render(job_description=job_description_text)
        except Exception as render_err:
            log_exception(f"JobAnalyzer: Failed to load or render job extraction prompt for '{job_title}': {render_err}", render_err)
            current_span.record_exception(render_err)
            current_span.set_status(trace.Status(trace.StatusCode.ERROR, "Prompt rendering failed"))
            return None

        extracted_json = await self._call_llm_async(prompt, task_name=f"Job Details Extraction for '{job_title}'") # This is already traced
        if extracted_json:
            try:
                if isinstance(extracted_json, dict):
                    job_details = ParsedJobData(**extracted_json)
                    logger.info(f"JobAnalyzer: Parsed extracted job details for '{job_title}'.")
                    return job_details
                else:
                    logger.error(f"JobAnalyzer: Job details extraction response not a dict for '{job_title}': {type(extracted_json)}")
                    return None
            except Exception as e:
                log_exception(f"JobAnalyzer: Failed to validate extracted job details for '{job_title}': {e}", e)
                logger.debug(f"Invalid JSON for ParsedJobData ('{job_title}'): {extracted_json}")
                return None
        else:
            logger.error(f"JobAnalyzer: Failed to get response for job details extraction for '{job_title}'.")
            return None

    @tracer.start_as_current_span("analyze_resume_suitability")
    async def analyze_resume_suitability(self, resume_data: ResumeData, job_data: Dict[str, Any]) -> Optional[JobAnalysisResult]:
        current_span = trace.get_current_span()
        job_title = job_data.get('title', 'N/A')
        job_description = job_data.get('description')
        current_span.set_attribute("job_title_param", job_title)

        if not resume_data or not resume_data.summary: 
            logger.warning(f"Missing or incomplete structured resume data for '{job_title}'. Skipping suitability.")
            current_span.set_attribute("suitability_skipped_reason", "incomplete_resume_data")
            return None
        if not job_description:
            logger.warning(f"Missing job description for '{job_title}'. Skipping suitability analysis.")
            current_span.set_attribute("suitability_skipped_reason", "missing_job_description")
            return None

        try:
            with tracer.start_as_current_span("prepare_suitability_prompt"):
                resume_data_json_str = resume_data.model_dump_json(indent=2)
                job_data_serializable_str = json.dumps(job_data, cls=DateEncoder) 
                job_data_serializable = json.loads(job_data_serializable_str) 

                context = {
                    "resume_data_json": resume_data_json_str,
                    "job_data_json": job_data_serializable 
                }
                suitability_prompt_template = load_template("suitability_prompt_file") # Load dynamically
                prompt = suitability_prompt_template.render(context)
        except Exception as e:
            log_exception(f"JobAnalyzer: Error preparing/rendering suitability prompt for '{job_title}': {e}", e)
            current_span.record_exception(e)
            current_span.set_status(trace.Status(trace.StatusCode.ERROR, "Suitability prompt preparation failed"))
            return None

        combined_json_response = await self._call_llm_async(prompt, task_name=f"Suitability Analysis for '{job_title}'") # This is already traced
        if not combined_json_response or not isinstance(combined_json_response, dict):
            logger.error(f"JobAnalyzer: Failed to get valid JSON dict for suitability ('{job_title}').")
            logger.debug(f"Raw response for '{job_title}': {combined_json_response}")
            return None

        analysis_data = combined_json_response.get("analysis") 
        if not analysis_data or not isinstance(analysis_data, dict):
            logger.error(f"JobAnalyzer: Response JSON missing valid 'analysis' dict for '{job_title}'.")
            logger.debug(f"Full response for '{job_title}': {combined_json_response}")
            return None
        
        try:
            analysis_result = JobAnalysisResult(**analysis_data)
            logger.info(f"JobAnalyzer: Suitability score for '{job_title}': {analysis_result.suitability_score}%")
            return analysis_result
        except Exception as e:
            log_exception(f"JobAnalyzer: Failed to validate analysis result for '{job_title}': {e}", e)
            logger.debug(f"Invalid 'analysis' structure for '{job_title}': {analysis_data}")
            return None

# --- Standalone functions (current structure, to be refactored or removed if class methods are preferred everywhere) ---
# The following functions are from the current analyzer.py and might be deprecated or integrated into classes.
# For now, keeping them to see how they fit into the refactor.

async def original_extract_job_details_async(job_description_text: str, job_title: str = "N/A") -> Optional[ParsedJobData]: 
    logger.warning(f"Calling DEPRECATED original_extract_job_details_async for '{job_title}'")
    # The multi-line schema dictionary that was here has been removed as it was unused.
    return None 

async def original_analyze_resume_suitability(resume_text: str, job_description_text: str, job_title: str = "N/A") -> Optional[JobAnalysisResult]:
    logger.warning(f"Calling DEPRECATED original_analyze_resume_suitability for '{job_title}'")
    return None
