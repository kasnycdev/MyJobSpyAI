import openai
import ollama
import google.generativeai as genai # Corrected import alias
import google.api_core.exceptions # Added for DeadlineExceeded
import json
import os
import asyncio
from typing import Dict, Optional, Any, Union
from rich.console import Console
import traceback
import time
import datetime # For timing logs

# Import Jinja2 components
try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateNotFound, TemplateSyntaxError
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

from analysis.models import ResumeData, JobAnalysisResult, ParsedJobData # Added ParsedJobData
# Import the loaded settings dictionary from config.py
from config import settings

console = Console()

# Helper function for logging exceptions
def log_exception(message, exception):
    console.log(message)
    console.log(traceback.format_exc())

# --- Jinja2 Environment Setup ---
PROMPT_TEMPLATE_LOADER = None
if JINJA2_AVAILABLE:
    try:
        prompts_dir = settings.get('analysis', {}).get('prompts_dir')
        if prompts_dir and os.path.isdir(prompts_dir):
            console.log(f"Initializing Jinja2 environment for prompts in: {prompts_dir}")
            PROMPT_TEMPLATE_LOADER = Environment(
                loader=FileSystemLoader(prompts_dir),
                autoescape=select_autoescape(['html', 'xml']),  # Basic autoescape, adjust if needed
                trim_blocks=True,  # Helps clean up whitespace
                lstrip_blocks=True
            )
        else:
            console.log(f"[bold red]Jinja2 prompts directory not found or invalid in config:[/bold red] {prompts_dir}")
            JINJA2_AVAILABLE = False  # Disable Jinja2 usage
    except Exception as jinja_err:
        log_exception(f"[bold red]Failed to initialize Jinja2 environment:[/bold red] {jinja_err}", jinja_err)
        JINJA2_AVAILABLE = False
else:
    console.log("[bold red]Jinja2 library not installed. Prompts cannot be loaded.[/bold red]")
    # Consider exiting or falling back to basic .format() if needed, but requires different prompt files

# --- Load Templates (Now using Jinja2) ---
def load_template(template_name_key: str):
    """Loads a Jinja2 template object based on filename key in settings."""
    if not JINJA2_AVAILABLE or not PROMPT_TEMPLATE_LOADER:
        raise RuntimeError("Jinja2 environment not available for loading prompt templates.")

    template_filename = settings.get('analysis', {}).get(template_name_key)
    if not template_filename:
        raise ValueError(f"Missing template filename configuration for key '{template_name_key}'")

    try:
        template = PROMPT_TEMPLATE_LOADER.get_template(template_filename)
        console.log(f"Successfully loaded Jinja2 template: {template_filename}")
        return template
    except TemplateNotFound:
        console.log(f"[bold red]Jinja2 template file not found:[/bold red] {template_filename} in {settings.get('analysis', {}).get('prompts_dir')}")
        raise
    except TemplateSyntaxError as syn_err:
        console.log(f"[bold red]Syntax error in Jinja2 template {template_filename}:[/bold red] {syn_err}")
        raise
    except Exception as e:
        log_exception(f"[bold red]Error loading Jinja2 template {template_filename}:[/bold red] {e}", e)
        raise


class ResumeAnalyzer:
    def __init__(self):
        self.provider = settings.get('llm_provider', 'openai').lower()
        console.log(f"Initializing LLM client for provider: [bold cyan]{self.provider}[/bold cyan]")

        self.model_name: Optional[str] = None
        self.sync_client: Union[openai.OpenAI, ollama.Client, None] = None # Gemini doesn't have separate sync client in lib
        self.async_client: Union[openai.AsyncOpenAI, ollama.AsyncClient, genai.GenerativeModel, None] = None
        self.request_timeout: Optional[int] = None
        self.max_retries: int = 2
        self.retry_delay: int = 5

        try:
            if self.provider == "openai":
                cfg = settings.get('openai', {})
                self.model_name = cfg.get('model')
                base_url = cfg.get('base_url')
                api_key = cfg.get('api_key', 'lm-studio')
                self.request_timeout = cfg.get('request_timeout')
                self.max_retries = cfg.get('max_retries', 2)
                self.retry_delay = cfg.get('retry_delay', 5)

                if not base_url or not self.model_name:
                    raise ValueError("OpenAI provider requires 'base_url' and 'model' in config.")

                self.sync_client = openai.OpenAI(base_url=base_url, api_key=api_key, timeout=self.request_timeout)
                self.async_client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=self.request_timeout)
                console.log(f"[green]OpenAI client initialized for model '{self.model_name}' at {base_url}.[/green]")

            elif self.provider == "ollama":
                cfg = settings.get('ollama', {})
                self.model_name = cfg.get('model')
                base_url = cfg.get('base_url')
                self.request_timeout = cfg.get('request_timeout')
                self.max_retries = cfg.get('max_retries', 2)
                self.retry_delay = cfg.get('retry_delay', 5)

                if not base_url or not self.model_name:
                    raise ValueError("Ollama provider requires 'base_url' and 'model' in config.")

                # Ollama client timeout is handled differently (per-request option)
                self.sync_client = ollama.Client(host=base_url)
                self.async_client = ollama.AsyncClient(host=base_url)
                console.log(f"[green]Ollama client initialized for model '{self.model_name}' at {base_url}.[/green]")

            elif self.provider == "gemini":
                cfg = settings.get('gemini', {})
                self.model_name = cfg.get('model')
                api_key = cfg.get('api_key') or os.getenv('GOOGLE_API_KEY')
                # Timeout/retries often handled by google client library or need custom logic
                self.request_timeout = cfg.get('request_timeout') # Store for potential custom logic
                self.max_retries = cfg.get('max_retries', 2)
                self.retry_delay = cfg.get('retry_delay', 5)

                if not api_key:
                    raise ValueError("Gemini provider requires 'api_key' in config or GOOGLE_API_KEY env var.")
                if not self.model_name:
                    raise ValueError("Gemini provider requires 'model' name in config.")

                genai.configure(api_key=api_key)
                # Safety settings to allow potentially sensitive content from resumes/jobs
                safety_settings = [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                ]
                # For Gemini, the client is the model object itself
                self.async_client = genai.GenerativeModel(
                    model_name=self.model_name,
                    safety_settings=safety_settings
                    # generation_config can be set here if needed, e.g., temperature
                )
                self.sync_client = None # No separate sync client needed for basic checks
                console.log(f"[green]Gemini client initialized for model '{self.model_name}'.[/green]")

            else:
                raise ValueError(f"Unsupported llm_provider: '{self.provider}'. Choose 'openai', 'ollama', or 'gemini'.")

        except Exception as e:
            log_exception(f"[bold red]Failed to initialize LLM client for provider '{self.provider}':[/bold red] {e}", e)
            raise RuntimeError(f"LLM client initialization failed: {e}") from e

        # --- Load Jinja2 Templates ---
        try:
            self.resume_prompt_template = load_template("resume_prompt_file")
            self.suitability_prompt_template = load_template("suitability_prompt_file")
            self.job_extraction_prompt_template = load_template("job_extraction_prompt_file") # New
        except (RuntimeError, ValueError, TemplateNotFound, TemplateSyntaxError) as tmpl_err:
            log_exception(f"[bold red]Failed to load necessary prompt templates:[/bold red] {tmpl_err}", tmpl_err)
            raise RuntimeError(f"Prompt template loading failed: {tmpl_err}") from tmpl_err

        self._check_connection_and_model()

    def _check_connection_and_model(self):
        """Checks connection and model availability for the selected provider."""
        console.log(f"Checking connection for provider '{self.provider}' and model '{self.model_name}'...")

        try:
            if self.provider == "openai":
                # Check OpenAI/LM Studio connection
                if not self.sync_client: raise RuntimeError("OpenAI sync client not initialized.")
                base_url = getattr(self.sync_client, 'base_url', 'N/A') # Get base_url if available
                console.log(f"Attempting to list models from OpenAI-compatible server at {base_url}...")
                models_response = self.sync_client.models.list()
                available_model_ids = [model.id for model in models_response.data]
                console.log(f"Available models reported by API: {available_model_ids}")
                # Note: LM Studio might only list the loaded model. Check primarily for connection success.
                console.log(f"[green]Successfully connected to OpenAI-compatible server at {base_url}.[/green]")
                console.log(f"[cyan]Configured to use model: '{self.model_name}'. Ensure this model is loaded/available.[/cyan]")

            elif self.provider == "ollama":
                # Check Ollama connection
                if not self.sync_client: raise RuntimeError("Ollama sync client not initialized.")
                base_url = getattr(self.sync_client, '_client', {})._host # Access host from underlying client
                console.log(f"Attempting to list models from Ollama server at {base_url}...")
                response = self.sync_client.list()
                available_models = [m['name'] for m in response.get('models', [])]
                console.log(f"Available Ollama models: {available_models}")
                if self.model_name not in available_models:
                     console.log(f"[yellow]Warning: Configured Ollama model '{self.model_name}' not found in available models. Ensure it is pulled.[/yellow]")
                else:
                     console.log(f"[green]Configured Ollama model '{self.model_name}' found.[/green]")
                console.log(f"[green]Successfully connected to Ollama server at {base_url}.[/green]")

            elif self.provider == "gemini":
                # Check Gemini connection and model existence
                console.log(f"Attempting to list models from Google AI...")
                found_model = False
                for m in genai.list_models():
                    # Check if the model supports generateContent and matches the configured name
                    if 'generateContent' in m.supported_generation_methods and self.model_name in m.name:
                         console.log(f"Found compatible Gemini model: {m.name}")
                         found_model = True
                         break # Found a suitable match
                if not found_model:
                     console.log(f"[red]Error: Configured Gemini model '{self.model_name}' not found or does not support 'generateContent'.[/red]")
                     console.log("Please check available models in Google AI Studio and your config.yaml.")
                     # List some available generative models for user convenience
                     try:
                         generative_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                         console.log(f"Available generative models: {generative_models[:10]}") # Show first 10
                     except Exception:
                         pass # Ignore errors listing models if the initial check failed
                     raise ValueError(f"Gemini model '{self.model_name}' not suitable.")
                else:
                     console.log(f"[green]Successfully verified Gemini model '{self.model_name}' is available.[/green]")

        except (openai.APIConnectionError, ollama.ResponseError, Exception) as e:
            provider_name = self.provider.upper()
            error_type = type(e).__name__
            log_exception(f"[bold red]{provider_name} API Connection/Setup Error:[/bold red] {error_type} - {e}", e)
            raise ConnectionError(f"Failed to connect to/setup {provider_name} provider: {e}") from e
        except Exception as e: # Catch any other unexpected errors
             log_exception(f"[bold red]Unexpected error during LLM connection check:[/bold red] {e}", e)
             raise ConnectionError(f"Unexpected error during LLM connection check: {e}") from e

    # Ollama-specific model pulling methods are removed.

    async def _call_llm_async(self, prompt: str) -> Optional[Dict[str, Any]]: # Renamed
        """
        Calls the configured LLM provider (OpenAI, Ollama, Gemini) with the given prompt,
        handles retries, and attempts to parse a JSON response.
        """
        if not self.async_client or not self.model_name:
            raise RuntimeError("LLM client or model name not initialized properly.")

        # Use provider-specific config for retries/delay if available, else use defaults set in __init__
        provider_cfg = settings.get(self.provider, {})
        max_retries = provider_cfg.get('max_retries', self.max_retries)
        retry_delay = provider_cfg.get('retry_delay', self.retry_delay)
        request_timeout = provider_cfg.get('request_timeout', self.request_timeout) # Used for Ollama

        analysis_cfg = settings.get('analysis', {})
        max_prompt_chars = analysis_cfg.get('max_prompt_chars', 24000)
        log_full_prompt = analysis_cfg.get('log_full_prompt', False) # New setting

        console.log(f"ASYNC: Sending request via provider '{self.provider}' to model '{self.model_name}'. Prompt length: {len(prompt)} chars.")
        if log_full_prompt:
            console.log(f"ASYNC: Full prompt:\n{prompt}")
        elif len(prompt) > 1000: # Log a snippet if not logging full and prompt is long
            console.log(f"ASYNC: Prompt snippet:\n{prompt[:500]}...\n...{prompt[-500:]}")
        else:
            console.log(f"ASYNC: Prompt:\n{prompt}")

        if len(prompt) > max_prompt_chars:
            console.log(f"[yellow]Warning: Prompt length ({len(prompt)}) exceeds configured max_prompt_chars ({max_prompt_chars}). Model might truncate or error.[/yellow]")

        last_exception = None
        content = None
        for attempt in range(max_retries + 1):
            try:
                if self.provider == "openai":
                    if not isinstance(self.async_client, openai.AsyncOpenAI): raise TypeError("Invalid client type for OpenAI")
                    response = await self.async_client.chat.completions.create(
                        model=self.model_name,
                        messages=[{'role': 'user', 'content': prompt}],
                        # response_format={"type": "json_object"}, # Keep removed for LM Studio compatibility for now
                        temperature=0.1
                    )
                    content = response.choices[0].message.content

                elif self.provider == "ollama":
                    if not isinstance(self.async_client, ollama.AsyncClient): raise TypeError("Invalid client type for Ollama")
                    response = await self.async_client.chat(
                        model=self.model_name,
                        messages=[{'role': 'user', 'content': prompt}],
                        format='json', # Ollama's way to request JSON
                        options={'temperature': 0.1},
                        # Apply timeout per request if configured
                        # keep_alive=-1 # Consider keep_alive setting if needed
                    )
                    content = response['message']['content']

                elif self.provider == "gemini":
                    if not isinstance(self.async_client, genai.GenerativeModel): raise TypeError("Invalid client type for Gemini")
                    # Gemini requires specific generation config for JSON
                    generation_config = genai.types.GenerationConfig(
                        response_mime_type="application/json",
                        temperature=0.1
                    )
                    response = await self.async_client.generate_content_async(
                        prompt,
                        generation_config=generation_config,
                        request_options={'timeout': self.request_timeout} # Timeout for Gemini if needed
                    )
                    # Handle potential blocks by safety filters (though thresholds set to BLOCK_NONE)
                    if not response.candidates:
                         raise ValueError(f"Gemini response blocked. Prompt feedback: {response.prompt_feedback}")
                    content = response.text # Gemini returns JSON directly in .text when mime_type is set
                    # Log Gemini token usage if available
                    if hasattr(response, 'usage_metadata') and response.usage_metadata:
                        usage = response.usage_metadata
                        console.log(f"[dim]Gemini Usage - Prompt: {usage.prompt_token_count}, Candidates: {usage.candidates_token_count}, Total: {usage.total_token_count} tokens[/dim]")

                # --- Common Response Processing ---
                console.log(f"ASYNC: LLM raw response content (Attempt {attempt + 1}):\n{content}")

                if content is None:
                    console.log(f"[yellow]ASYNC LLM response content is None (Attempt {attempt + 1}). Provider: {self.provider}[/yellow]")
                    last_exception = ValueError("LLM response content was None.")
                else:
                    try:
                        # Strip markdown fences if they appear (might happen if JSON mode fails or isn't used)
                        content_strip = content.strip()
                        if content_strip.startswith("```json"):
                            content_strip = content_strip[7:-3].strip() if content_strip.endswith("```") else content_strip[7:].strip()
                        elif content_strip.startswith("```"):
                             content_strip = content_strip[3:-3].strip() if content_strip.endswith("```") else content_strip[3:].strip()

                        result = json.loads(content_strip)
                        console.log("[green]ASYNC: Parsed JSON response from LLM.[/green]")
                        return result # Success! Exit loop.
                    except json.JSONDecodeError as json_err:
                        console.log(f"[yellow]ASYNC JSON Decode Error (Attempt {attempt + 1}):[/yellow] {json_err}")
                        console.log(f"Problematic content from LLM: {content}")
                        last_exception = json_err
                        # Continue to retry logic

            # --- Exception Handling (Provider Agnostic where possible) ---
            except (openai.APIConnectionError, ollama.ResponseError, asyncio.TimeoutError, ConnectionError, TimeoutError) as conn_err:
                error_message = f"ASYNC LLM Connection/Timeout Error (Provider: {self.provider}, Attempt {attempt + 1}): {type(conn_err).__name__} - {conn_err}"
                console.log(f"[yellow]{error_message}[/yellow]")
                log_exception(error_message, conn_err) # Log full traceback
                last_exception = conn_err
            except openai.RateLimitError as rate_err: # Specific OpenAI error
                error_message = f"ASYNC LLM Rate Limit Error (Provider: openai, Attempt {attempt + 1}): {rate_err}"
                console.log(f"[yellow]{error_message}[/yellow]")
                log_exception(error_message, rate_err)
                last_exception = rate_err
            except openai.APIStatusError as status_err: # Specific OpenAI error
                error_message = f"ASYNC LLM API Status Error (Provider: openai, Attempt {attempt + 1}): Status {status_err.status_code}, Response: {status_err.response.text if status_err.response else 'N/A'}"
                console.log(f"[yellow]{error_message}[/yellow]")
                log_exception(error_message, status_err)
                last_exception = status_err
            except genai.types.generation_types.BlockedPromptException as gemini_block_err: # Specific Gemini error
                error_message = f"ASYNC Gemini Prompt Blocked (Attempt {attempt + 1}): {gemini_block_err}. Feedback: {gemini_block_err.response.prompt_feedback if hasattr(gemini_block_err, 'response') else 'N/A'}"
                console.log(f"[red]{error_message}[/red]")
                log_exception(error_message, gemini_block_err)
                last_exception = gemini_block_err
            except google.api_core.exceptions.DeadlineExceeded as deadline_err: # Specific Google API error
                error_message = f"ASYNC Google API Deadline Exceeded (Provider: gemini, Attempt {attempt + 1}): {deadline_err}"
                console.log(f"[red]{error_message}[/red]")
                log_exception(error_message, deadline_err)
                last_exception = deadline_err
            except Exception as e: # Catch any other unexpected errors
                error_message = f"ASYNC Unexpected LLM API Error (Provider: {self.provider}, Attempt {attempt + 1}): {type(e).__name__} - {e}"
                console.log(f"[red]{error_message}[/red]")
                log_exception(error_message, e)
                last_exception = e

            if attempt < max_retries:
                current_delay = retry_delay * (2 ** attempt)
                console.log(f"[dim]ASYNC: Retrying LLM call in {current_delay:.1f}s... (Attempt {attempt + 1} of {max_retries})[/dim]")
                await asyncio.sleep(current_delay)
            else:
                console.log(f"[bold red]ASYNC: LLM call failed after {max_retries + 1} attempts.[/bold red]")
                if last_exception:
                    console.log(f"ASYNC: Last error: {type(last_exception).__name__} - {last_exception}")
        return None

    async def extract_resume_data_async(self, resume_text: str) -> Optional[ResumeData]:
        MAX_RESUME_CHARS_FOR_LLM = settings.get('analysis', {}).get('max_prompt_chars', 15000)
        if not resume_text or not resume_text.strip():
            console.log("[yellow]Resume text empty.[/yellow]")
            return None
        resume_text_for_prompt = resume_text
        if len(resume_text) > MAX_RESUME_CHARS_FOR_LLM:
            console.log(f"[yellow]Truncating resume text ({len(resume_text)} > {MAX_RESUME_CHARS_FOR_LLM}).[/yellow]")
            resume_text_for_prompt = resume_text[:MAX_RESUME_CHARS_FOR_LLM]

        try:
            prompt = self.resume_prompt_template.render(resume_text=resume_text_for_prompt)
        except Exception as render_err:
            log_exception(f"[bold red]Failed to render resume extraction prompt:[/bold red] {render_err}", render_err)
            return None

        task_name = "Resume Extraction"
        console.log(f"ASYNC: Starting LLM call for: {task_name}")
        start_time = time.monotonic()
        extracted_json = await self._call_llm_async(prompt) # Use unified method
        duration = time.monotonic() - start_time
        console.log(f"ASYNC: Finished LLM call for: {task_name}. Duration: {duration:.2f}s")

        if extracted_json:
            try:
                if isinstance(extracted_json, dict):
                    resume_data = ResumeData(**extracted_json)
                    console.log("[green]ASYNC: Parsed extracted resume data.[/green]")
                    return resume_data
                else:
                    console.log(f"[red]ASYNC Resume extract response not dict:[/red] {type(extracted_json)}")
                    return None
            except Exception as e:
                log_exception(f"[bold red]ASYNC: Failed validate extracted resume:[/bold red] {e}", e)
                console.log(f"Invalid JSON: {extracted_json}")
                return None
        else:
            console.log("[bold red]ASYNC: Failed get response for resume extract.[/bold red]")
            return None

    async def analyze_suitability_async(self, resume_data: ResumeData, job_data: Dict[str, Any]) -> Optional[JobAnalysisResult]:
        job_title = job_data.get('title', 'N/A')
        if not resume_data:
            console.log(f"[yellow]Missing structured resume data for '{job_title}'.[/yellow]")
            return None
        if not job_data or not job_data.get("description"):
            console.log(f"[yellow]Missing job description for '{job_title}'. Skipping analysis.[/yellow]")
            return None

        try:
            resume_data_json_str = resume_data.model_dump_json(indent=2)
            context = {
                "resume_data_json": resume_data_json_str,
                "job_data_json": job_data
            }
            prompt = self.suitability_prompt_template.render(context)
        except Exception as e:
            log_exception(f"[red]Error preparing/rendering suitability prompt for '{job_title}':[/red] {e}", e)
            return None

        task_name = f"Suitability Analysis for '{job_title}'"
        console.log(f"ASYNC: Starting LLM call for: {task_name}")
        start_time = time.monotonic()
        combined_json_response = await self._call_llm_async(prompt) # Use unified method
        duration = time.monotonic() - start_time
        console.log(f"ASYNC: Finished LLM call for: {task_name}. Duration: {duration:.2f}s")

        if not combined_json_response or not isinstance(combined_json_response, dict):
            console.log(f"[red]ASYNC: Failed get valid JSON dict for suitability: {job_title}.[/red]")
            console.log(f"Raw response: {combined_json_response}")
            return None

        analysis_data = combined_json_response.get("analysis")
        if not analysis_data or not isinstance(analysis_data, dict):
            console.log(f"[red]ASYNC: Response JSON missing valid 'analysis' dict for: {job_title}.[/red]")
            console.log(f"Full response: {combined_json_response}")  # Log the full response for debugging
            return None

        try:
            analysis_result = JobAnalysisResult(**analysis_data)
            console.log(f"[cyan]ASYNC: Suitability score for '{job_title}': {analysis_result.suitability_score}%[/cyan]")
            return analysis_result
        except Exception as e:
            log_exception(f"[red]ASYNC: Failed validate analysis result for '{job_title}':[/red] {e}", e)
            console.log(f"Invalid 'analysis' structure: {analysis_data}")
            return None

    async def extract_job_details_async(self, job_description_text: str, job_title: str = "N/A") -> Optional[ParsedJobData]: # Added job_title for logging
        """
        Asynchronously extracts structured data from a job description string using an LLM.
        """
        if not job_description_text or not job_description_text.strip():
            console.log("[yellow]Job description text is empty. Skipping extraction.[/yellow]")
            return None

        # Truncate if necessary (using a general prompt char limit for now)
        # MAX_JOB_DESC_CHARS_FOR_LLM = settings.get('analysis', {}).get('max_prompt_chars', 15000) # Or a specific setting
        # if len(job_description_text) > MAX_JOB_DESC_CHARS_FOR_LLM:
        #     console.log(f"[yellow]Truncating job description text ({len(job_description_text)} > {MAX_JOB_DESC_CHARS_FOR_LLM}).[/yellow]")
        #     job_description_text_for_prompt = job_description_text[:MAX_JOB_DESC_CHARS_FOR_LLM]
        # else:
        #     job_description_text_for_prompt = job_description_text
        # For now, let _call_ollama_async handle truncation warning based on overall prompt length

        try:
            prompt = self.job_extraction_prompt_template.render(job_description=job_description_text)
        except Exception as render_err:
            log_exception(f"[bold red]Failed to render job extraction prompt:[/bold red] {render_err}", render_err)
            return None

        # For now, let _call_llm_async handle truncation warning based on overall prompt length

        try:
            prompt = self.job_extraction_prompt_template.render(job_description=job_description_text)
        except Exception as render_err:
            log_exception(f"[bold red]Failed to render job extraction prompt:[/bold red] {render_err}", render_err)
            return None

        task_name = f"Job Details Extraction for '{job_title}'"
        console.log(f"ASYNC: Starting LLM call for: {task_name}")
        start_time = time.monotonic()
        extracted_json = await self._call_llm_async(prompt) # Use unified method
        duration = time.monotonic() - start_time
        console.log(f"ASYNC: Finished LLM call for: {task_name}. Duration: {duration:.2f}s")

        if extracted_json:
            try:
                if isinstance(extracted_json, dict):
                    # Ensure all list fields are present even if empty, as per Pydantic model defaults
                    # Pydantic handles default empty lists, so direct instantiation is fine.
                    job_details = ParsedJobData(**extracted_json)
                    console.log("[green]ASYNC: Parsed extracted job details.[/green]")
                    return job_details
                else:
                    console.log(f"[red]ASYNC Job details extraction response not a dict:[/red] {type(extracted_json)}")
                    return None
            except Exception as e: # Catch Pydantic validation errors specifically if possible
                log_exception(f"[bold red]ASYNC: Failed to validate extracted job details:[/bold red] {e}", e)
                console.log(f"Invalid JSON for ParsedJobData: {extracted_json}")
                return None
        else:
            console.log("[bold red]ASYNC: Failed to get response for job details extraction.[/bold red]")
            return None
