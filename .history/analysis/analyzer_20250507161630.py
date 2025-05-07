import openai # Changed from ollama
import json
import os
import asyncio
from typing import Dict, Optional, Any
from rich.console import Console
import traceback

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
        console.log("Initializing OpenAI-compatible LLM client (for LM Studio)...")
        try:
            lm_config = settings.get('language_model', {})
            self.base_url = lm_config.get('base_url')
            self.api_key = lm_config.get('api_key', 'lm-studio') # Default to lm-studio if not set
            self.model_name = lm_config.get('model')
            request_timeout = lm_config.get('request_timeout')

            if not self.base_url:
                raise ValueError("LLM base_url is not configured.")
            if not self.model_name:
                raise ValueError("LLM model name is not configured.")

            self.sync_client = openai.OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=request_timeout
            )
            self.async_client = openai.AsyncOpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=request_timeout
            )
            console.log(f"[green]OpenAI-compatible LLM client initialized for model '{self.model_name}' at {self.base_url}.[/green]")
        except Exception as e:
            log_exception(f"[bold red]Failed to initialize OpenAI-compatible LLM client:[/bold red] {e}", e)
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
        """
        Checks the connection to the OpenAI-compatible LLM server (LM Studio)
        and verifies if the configured model seems available.
        Does not attempt to pull models as this is handled by LM Studio UI.
        """
        console.log(f"Checking LLM connection to '{self.base_url}' for model '{self.model_name}'...")
        try:
            # Attempt a lightweight API call, e.g., listing models.
            # For OpenAI-compatible APIs, this lists available model details.
            # LM Studio might only show the currently loaded model or a generic list.
            models_response = self.sync_client.models.list()
            
            available_model_ids = [model.id for model in models_response.data]
            console.log(f"Available models reported by API: {available_model_ids}")

            # With LM Studio, the 'model_name' in config should match what LM Studio expects
            # for the loaded model. A direct match in `available_model_ids` might not always occur
            # if LM Studio uses internal naming or if the /v1/models endpoint is generic.
            # The crucial part is that the server is responsive and the configured model name
            # is accepted by LM Studio when making chat completion requests.
            # A simple check here is if the list call succeeded.
            # A more robust check would be to see if self.model_name is in available_model_ids,
            # but this can be tricky with LM Studio's dynamic model loading.
            
            # For now, a successful list call implies the server is up.
            # The actual test of the model happens when a chat completion is attempted.
            console.log(f"[green]Successfully connected to LLM server at '{self.base_url}'.[/green]")
            console.log(f"[cyan]Configured to use model: '{self.model_name}'. Ensure this model is loaded and selected in LM Studio.[/cyan]")

        except openai.APIConnectionError as e:
            log_exception(f"[bold red]LLM API Connection Error to '{self.base_url}': {e}[/bold red]", e)
            raise ConnectionError(f"Failed to connect to LLM server at '{self.base_url}': {e}") from e
        except openai.APIStatusError as e:
            log_exception(f"[bold red]LLM API Status Error (e.g., 4xx, 5xx) from '{self.base_url}': {e}[/bold red]", e)
            raise ConnectionError(f"LLM server at '{self.base_url}' returned an error: {e.status_code} - {e.message}") from e
        except Exception as e:
            log_exception(f"[bold red]LLM connection/setup check failed: {e}[/bold red]", e)
            raise ConnectionError(f"LLM connection/setup check failed: {e}") from e

    # Ollama-specific model pulling methods are removed:
    # _extracted_from__check_connection_and_model_15
    # _pull_model_with_progress

    async def _call_llm_api_async(self, prompt: str) -> Optional[Dict[str, Any]]: # Renamed from _call_ollama_async
        lm_config = settings.get('language_model', {})
        analysis_cfg = settings.get('analysis', {})
        # self.model_name is set in __init__
        max_retries = lm_config.get('max_retries', 2)
        retry_delay = lm_config.get('retry_delay', 5)
        max_prompt_chars = analysis_cfg.get('max_prompt_chars', 24000) # This is a general setting for prompt length warning
        # Removed redundant/erroneous assignments from ollama_cfg
        console.log(f"ASYNC: Sending request to LLM model '{self.model_name}'. Prompt length: {len(prompt)} chars.")
        if len(prompt) > max_prompt_chars: # Check against analysis.max_prompt_chars
            console.log(f"[yellow]Warning: Prompt length ({len(prompt)}) exceeds configured max_prompt_chars ({max_prompt_chars}). Truncation might occur or be needed.[/yellow]")
            # Actual truncation should happen before rendering the prompt if it's a hard limit.
            # Or, rely on the model/API to handle very long prompts if possible.

        last_exception = None
        for attempt in range(max_retries + 1): # +1 to make max_retries the number of retries, so total attempts = max_retries + 1
            try:
                # For OpenAI compatible APIs, JSON mode can be requested if the model supports it.
                # This is generally more reliable than instructing "output JSON" in the prompt.
                response = await self.async_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{'role': 'user', 'content': prompt}],
                    # response_format={"type": "json_object"}, # Removed: LM Studio model reported incompatibility
                    temperature=0.1 # Low temperature for deterministic JSON structure
                )
                content = response.choices[0].message.content
                console.log(f"ASYNC: LLM raw response content: {content[:500]}...") # Log first 500 chars

                if content is None:
                    console.log(f"[yellow]ASYNC LLM response content is None (Attempt {attempt + 1})[/yellow]")
                    last_exception = ValueError("LLM response content was None.")
                    # Continue to retry logic
                else:
                    try:
                        # Since we requested json_object, the content should be a parsable JSON string.
                        # No need to strip ```json usually.
                        result = json.loads(content)
                        console.log("[green]ASYNC: Parsed JSON response from LLM.[/green]")
                        return result
                    except json.JSONDecodeError as json_err:
                        console.log(f"[yellow]ASYNC JSON Decode Error (Attempt {attempt + 1}):[/yellow] {json_err}")
                        console.log(f"Problematic content from LLM: {content}")
                        last_exception = json_err
                        # Continue to retry logic, as the model might produce valid JSON on a retry.
            
            except openai.APIConnectionError as e:
                console.log(f"[yellow]ASYNC LLM API Connection Error (Attempt {attempt + 1}): {e}[/yellow]")
                last_exception = e
            except openai.RateLimitError as e:
                console.log(f"[yellow]ASYNC LLM API Rate Limit Error (Attempt {attempt + 1}): {e}[/yellow]")
                last_exception = e # Could implement longer/specific backoff here
            except openai.APIStatusError as e: # Catches 4xx and 5xx errors
                console.log(f"[yellow]ASYNC LLM API Status Error (Attempt {attempt + 1}): Status {e.status_code}, Response: {e.response}[/yellow]")
                last_exception = e
            except asyncio.TimeoutError: # If the httpx client itself times out
                console.log(f"[yellow]ASYNC LLM Request Timeout (Attempt {attempt + 1})[/yellow]")
                last_exception = asyncio.TimeoutError("Request to LLM timed out.")
            except Exception as e: # Catch any other unexpected errors
                log_exception(f"[red]ASYNC Unexpected LLM API Error (Attempt {attempt + 1}):[/red] {e}", e)
                last_exception = e

            if attempt < max_retries:
                current_delay = retry_delay * (2 ** attempt)
                console.log(f"[dim]ASYNC: Retrying LLM call in {current_delay:.1f}s...[/dim]")
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

        console.log("ASYNC: Requesting resume data extraction from LLM...")
        extracted_json = await self._call_llm_api_async(prompt) # Changed call
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

        console.log(f"ASYNC: Requesting suitability analysis for: [cyan]{job_title}[/cyan] from LLM...")
        combined_json_response = await self._call_llm_api_async(prompt) # Changed call

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

    async def extract_job_details_async(self, job_description_text: str) -> Optional[ParsedJobData]:
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

        console.log("ASYNC: Requesting job details extraction from LLM...")
        extracted_json = await self._call_llm_api_async(prompt) # Changed call

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
