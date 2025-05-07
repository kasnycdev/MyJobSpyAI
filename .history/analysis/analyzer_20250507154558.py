import ollama
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
        console.log("Initializing Ollama clients...")
        try:
            ollama_cfg = settings.get('ollama', {})
            self.sync_client = ollama.Client(
                host=ollama_cfg.get('base_url'),
                timeout=ollama_cfg.get('request_timeout')
            )
            self.async_client = ollama.AsyncClient(
                host=ollama_cfg.get('base_url'),
                timeout=ollama_cfg.get('request_timeout')
            )
            console.log("[green]Ollama clients initialized.[/green]")
        except Exception as e:
            log_exception(f"[bold red]Failed init Ollama clients:[/bold red] {e}", e)
            raise RuntimeError(f"Ollama client init failed: {e}") from e

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
        try:
            ollama_cfg = settings.get('ollama', {})
            ollama_model = ollama_cfg.get('model')
            base_url = ollama_cfg.get('base_url')
            console.log(f"Checking Ollama connection at {base_url}...")
            self.sync_client.ps()
            console.log("[green]Ollama connection successful.[/green]")
            console.log("Fetching local Ollama models...")
            ollama_list_response = self.sync_client.list()
            console.log(f"Raw list response: {ollama_list_response}")
            models_data = ollama_list_response.get('models', [])
            if not isinstance(models_data, list):
                models_data = []
            local_models = [
                m.model for m in models_data
                if hasattr(m, 'model') and isinstance(m.model, str) and m.model
            ]
            console.log(f"Parsed local models: [cyan]{local_models}[/cyan]")
            if ollama_model not in local_models:
                console.log(f"[yellow]Model '{ollama_model}' not found locally. Attempting pull...[/yellow]")
                try:
                    self._extracted_from__check_connection_and_model_15(ollama_model)
                except Exception as pull_err:
                    log_exception(f"[bold red]Pull/verify failed for '{ollama_model}':[/bold red] {pull_err}", pull_err)
                    raise ConnectionError(f"Model '{ollama_model}' unavailable/pull failed.") from pull_err
            else:
                console.log(f"[cyan]Using configured Ollama model: {ollama_model}[/cyan]")
        except Exception as e:
            log_exception(f"[bold red]Ollama connection/setup failed:[/bold red] {e}", e)
            raise ConnectionError(f"Ollama connection/setup failed: {e}") from e

    def _extracted_from__check_connection_and_model_15(self, ollama_model):
        self._pull_model_with_progress(ollama_model)
        console.log("Re-fetching model list after pull...")
        updated_list_response = self.sync_client.list()
        updated_models_data = updated_list_response.get('models', [])
        updated_names = []
        for idx, m_upd in enumerate(updated_models_data):
            model_name = (
                m_upd.model if hasattr(m_upd, 'model') else
                (m_upd.get('name') if isinstance(m_upd, dict) else None)
            )
            if model_name:
                updated_names.append(model_name)
            else:
                console.log(f"[yellow]Could not extract name from updated item {idx}:[/yellow] {m_upd}")
        console.log(f"Model list after pull: {updated_names}")
        if ollama_model not in updated_names:
            console.log(f"[bold red]Model '{ollama_model}' still not found after pull.[/bold red]")
            raise ConnectionError(f"Model '{ollama_model}' unavailable.")
        else:
            console.log("[green]Model found after pull.[/green]")

    def _pull_model_with_progress(self, model_name: str):
        current_digest = ""
        status = ""
        try:
            for progress in ollama.pull(model_name, stream=True):
                digest = progress.get("digest", "")
                status = progress.get('status', '')
                if digest != current_digest != "":
                    print()
                if digest:
                    current_digest = digest
                console.log(f"Pulling {model_name}: {status}")
                if progress.get('error'):
                    raise Exception(f"Pull error: {progress['error']}")
                if 'status' in progress and 'success' in progress['status'].lower():
                    console.log(f"[green]Successfully pulled model {model_name}[/green]")
                    break
        except Exception as e:
            log_exception(f"[bold red]Error during model pull:[/bold red] {e}", e)
            raise
        finally:
            console.log("")

    async def _call_ollama_async(self, prompt: str) -> Optional[Dict[str, Any]]:
        ollama_cfg = settings.get('ollama', {})
        analysis_cfg = settings.get('analysis', {})
        ollama_model = ollama_cfg.get('model')
        max_retries = ollama_cfg.get('max_retries', 2)
        retry_delay = ollama_cfg.get('retry_delay', 5)
        max_prompt_chars = analysis_cfg.get('max_prompt_chars', 24000)
        console.log(f"ASYNC: Sending request to {ollama_model}. Prompt length: {len(prompt)} chars.")
        if len(prompt) > max_prompt_chars:
            console.log(f"[yellow]Prompt length exceeds {max_prompt_chars} chars.[/yellow]")
        last_exception = None
        for attempt in range(max_retries):
            try:
                response = await self.async_client.chat(
                    model=ollama_model,
                    messages=[{'role': 'user', 'content': prompt}],
                    format='json',
                    options={'temperature': 0.1}
                )
                content = response['message']['content']
                console.log(f"ASYNC: Ollama raw response: {content[:500]}...")
                try:
                    content_strip = content.strip()
                    if content_strip.startswith("```json"):
                        content_strip = content_strip[7:-3].strip() if content_strip.endswith("```") else content_strip[7:].strip()
                    elif content_strip.startswith("```"):
                        content_strip = content_strip[3:-3].strip() if content_strip.endswith("```") else content_strip[3:].strip()
                    result = json.loads(content_strip)
                    console.log("[green]ASYNC: Parsed JSON response.[/green]")
                    return result
                except json.JSONDecodeError as json_err:
                    console.log(f"[yellow]ASYNC JSON Decode Error (Attempt {attempt + 1}):[/yellow] {json_err}")
                    console.log(f"Problematic content: {content}")
                    last_exception = json_err
            except (ollama.ResponseError, asyncio.TimeoutError, ConnectionError, TimeoutError) as conn_err:
                console.log(f"[yellow]ASYNC Ollama API Error (Attempt {attempt + 1}):[/yellow] {conn_err}")
                last_exception = conn_err
            except Exception as e:
                log_exception(f"[red]ASYNC Unexpected Ollama Error (Attempt {attempt + 1}):[/red] {e}", e)
                last_exception = e
            if attempt < max_retries - 1:
                delay = retry_delay * (2 ** attempt)
                console.log(f"[dim]ASYNC: Retrying Ollama call in {delay:.1f}s...[/dim]")
                await asyncio.sleep(delay)
            else:
                console.log(f"[bold red]ASYNC: Ollama call failed after {max_retries} attempts.[/bold red]")
                console.log(f"ASYNC: Last error: {last_exception}")
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

        console.log("ASYNC: Requesting resume data extraction...")
        extracted_json = await self._call_ollama_async(prompt)
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

        console.log(f"ASYNC: Requesting suitability analysis for: [cyan]{job_title}[/cyan]")
        combined_json_response = await self._call_ollama_async(prompt)

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

        console.log("ASYNC: Requesting job details extraction...")
        extracted_json = await self._call_ollama_async(prompt)

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
