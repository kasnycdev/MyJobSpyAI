import ollama
import json
import logging
import time
import os
import asyncio
from typing import Dict, Optional, Any

# Import Jinja2 components
try:
    from jinja2 import Environment, FileSystemLoader, select_autoescape, TemplateNotFound, TemplateSyntaxError
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

from analysis.models import ResumeData, JobAnalysisResult, AnalyzedJob
# Import the loaded settings dictionary from config.py
from config import settings

log = logging.getLogger(__name__)

# --- Jinja2 Environment Setup ---
PROMPT_TEMPLATE_LOADER = None
if JINJA2_AVAILABLE:
    try:
        prompts_dir = settings.get('analysis', {}).get('prompts_dir')
        if prompts_dir and os.path.isdir(prompts_dir):
            log.info(f"Initializing Jinja2 environment for prompts in: {prompts_dir}")
            PROMPT_TEMPLATE_LOADER = Environment(
                loader=FileSystemLoader(prompts_dir),
                autoescape=select_autoescape(['html', 'xml']), # Basic autoescape, adjust if needed
                trim_blocks=True, # Helps clean up whitespace
                lstrip_blocks=True
            )
        else:
            log.error(f"[bold red]Jinja2 prompts directory not found or invalid in config:[/bold red] {prompts_dir}")
            JINJA2_AVAILABLE = False # Disable Jinja2 usage
    except Exception as jinja_err:
        log.error(f"[bold red]Failed to initialize Jinja2 environment:[/bold red] {jinja_err}", exc_info=True)
        JINJA2_AVAILABLE = False
else:
    log.error("[bold red]Jinja2 library not installed. Prompts cannot be loaded.[/bold red]")
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
        log.debug(f"Successfully loaded Jinja2 template: {template_filename}")
        return template
    except TemplateNotFound:
        log.error(f"[bold red]Jinja2 template file not found:[/bold red] {template_filename} in {settings.get('analysis',{}).get('prompts_dir')}")
        raise
    except TemplateSyntaxError as syn_err:
        log.error(f"[bold red]Syntax error in Jinja2 template {template_filename}:[/bold red] {syn_err}")
        raise
    except Exception as e:
        log.error(f"[bold red]Error loading Jinja2 template {template_filename}:[/bold red] {e}", exc_info=True)
        raise

class ResumeAnalyzer:
    def __init__(self):
        log.info("Initializing Ollama clients...")
        try:
            ollama_cfg = settings.get('ollama', {})
            self.sync_client = ollama.Client( host=ollama_cfg.get('base_url'), timeout=ollama_cfg.get('request_timeout') )
            self.async_client = ollama.AsyncClient( host=ollama_cfg.get('base_url'), timeout=ollama_cfg.get('request_timeout') )
            log.info("[green]Ollama clients initialized.[/green]")
        except Exception as e: log.critical(f"[bold red]Failed init Ollama clients:[/bold red] {e}", exc_info=True); raise RuntimeError(f"Ollama client init failed: {e}") from e

        # --- Load Jinja2 Templates ---
        try:
            self.resume_prompt_template = load_template("resume_prompt_file")
            self.suitability_prompt_template = load_template("suitability_prompt_file")
        except (RuntimeError, ValueError, TemplateNotFound, TemplateSyntaxError) as tmpl_err:
            log.critical(f"[bold red]Failed to load necessary prompt templates:[/bold red] {tmpl_err}")
            # Depending on desired behavior, you might exit or disable analysis
            raise RuntimeError(f"Prompt template loading failed: {tmpl_err}") from tmpl_err
        # --- End Template Loading ---

        self._check_connection_and_model() # Sync check uses sync_client

    # --- _check_connection_and_model remains the same ---
    def _check_connection_and_model(self):
        try:
            ollama_cfg = settings.get('ollama', {}); ollama_model = ollama_cfg.get('model'); base_url = ollama_cfg.get('base_url')
            log.info(f"Checking Ollama connection at {base_url}...")
            self.sync_client.ps(); log.info("[green]Ollama connection successful.[/green]")
            log.info("Fetching local Ollama models...")
            ollama_list_response = self.sync_client.list(); log.debug(f"Raw list response: {ollama_list_response}")
            models_data = ollama_list_response.get('models', []);
            if not isinstance(models_data, list): models_data = []
            local_models = [m.model for m in models_data if hasattr(m, 'model') and isinstance(m.model, str) and m.model]
            log.info(f"Parsed local models: [cyan]{local_models}[/cyan]")
            if ollama_model not in local_models:
                log.warning(f"[yellow]Model '{ollama_model}' not found locally. Attempting pull...[/yellow]");
                try:
                    self._pull_model_with_progress(ollama_model)
                    log.info("Re-fetching model list after pull...")
                    updated_list_response = self.sync_client.list(); updated_models_data = updated_list_response.get('models', [])
                    updated_names = []
                    for idx, m_upd in enumerate(updated_models_data):
                         model_name = m_upd.model if hasattr(m_upd, 'model') else (m_upd.get('name') if isinstance(m_upd, dict) else None)
                         if model_name: updated_names.append(model_name)
                         else: log.warning(f"[yellow]Could not extract name from updated item {idx}:[/yellow] {m_upd}")
                    log.debug(f"Model list after pull: {updated_names}")
                    if ollama_model not in updated_names: log.error(f"[bold red]Model '{ollama_model}' still not found after pull.[/bold red]"); raise ConnectionError(f"Model '{ollama_model}' unavailable.")
                    else: log.info("[green]Model found after pull.[/green]")
                except Exception as pull_err: log.error(f"[bold red]Pull/verify failed for '{ollama_model}':[/bold red] {pull_err}", exc_info=True); raise ConnectionError(f"Model '{ollama_model}' unavailable/pull failed.") from pull_err
            else: log.info(f"Using configured Ollama model: [cyan]{ollama_model}[/cyan]")
        except Exception as e: log.critical(f"[bold red]Ollama connection/setup failed:[/bold red] {e}", exc_info=True); raise ConnectionError(f"Ollama connection/setup failed: {e}") from e

    # --- _pull_model_with_progress remains the same ---
    def _pull_model_with_progress(self, model_name: str):
        # (Sync pulling logic remains the same)
        current_digest = ""; status = ""
        try:
            for progress in ollama.pull(model_name, stream=True):
                 digest = progress.get("digest", "");
                 if digest != current_digest and current_digest != "": print()
                 if digest: current_digest = digest; status = progress.get('status', ''); print(f"Pulling {model_name}: {status}", end='\r')
                 else: status = progress.get('status', ''); print(f"Pulling {model_name}: {status}")
                 if progress.get('error'): raise Exception(f"Pull error: {progress['error']}")
                 if 'status' in progress and 'success' in progress['status'].lower(): print(); log.info(f"[green]Successfully pulled model {model_name}[/green]"); break
        except Exception as e: print(); log.error(f"[bold red]Error during model pull:[/bold red] {e}"); raise
        finally: print()

    # --- _call_ollama_async remains the same ---
    async def _call_ollama_async(self, prompt: str) -> Optional[Dict[str, Any]]:
         ollama_cfg = settings.get('ollama', {}); analysis_cfg = settings.get('analysis', {})
         ollama_model = ollama_cfg.get('model'); max_retries = ollama_cfg.get('max_retries', 2)
         retry_delay = ollama_cfg.get('retry_delay', 5); max_prompt_chars = analysis_cfg.get('max_prompt_chars', 24000)
         log.debug(f"ASYNC: Sending request to {ollama_model}. Prompt length: {len(prompt)} chars.")
         if len(prompt) > max_prompt_chars: log.warning(f"[yellow]Prompt length exceeds {max_prompt_chars} chars.[/yellow]")
         last_exception = None
         for attempt in range(max_retries):
             try:
                 response = await self.async_client.chat( model=ollama_model, messages=[{'role': 'user', 'content': prompt}], format='json', options={'temperature': 0.1} )
                 content = response['message']['content']; log.debug(f"ASYNC: Ollama raw response: {content[:500]}...")
                 try:
                     content_strip = content.strip();
                     if content_strip.startswith("```json"): content_strip = content_strip[7:-3].strip() if content_strip.endswith("```") else content_strip[7:].strip()
                     elif content_strip.startswith("```"): content_strip = content_strip[3:-3].strip() if content_strip.endswith("```") else content_strip[3:].strip()
                     result = json.loads(content_strip); log.debug("ASYNC: Parsed JSON response."); return result
                 except json.JSONDecodeError as json_err: log.warning(f"[yellow]ASYNC JSON Decode Error (Attempt {attempt + 1}):[/yellow] {json_err}"); log.debug(f"Problematic content: {content}"); last_exception = json_err
             except (ollama.ResponseError, asyncio.TimeoutError, ConnectionError, TimeoutError) as conn_err: log.warning(f"[yellow]ASYNC Ollama API Error (Attempt {attempt + 1}):[/yellow] {conn_err}"); last_exception = conn_err
             except Exception as e: log.error(f"[red]ASYNC Unexpected Ollama Error (Attempt {attempt + 1}):[/red] {e}", exc_info=True); last_exception = e
             if attempt < max_retries - 1: delay = retry_delay * (2 ** attempt); log.info(f"[dim]ASYNC: Retrying Ollama call in {delay:.1f}s...[/dim]"); await asyncio.sleep(delay)
             else: log.error(f"[bold red]ASYNC: Ollama call failed after {max_retries} attempts.[/bold red]"); log.error(f"ASYNC: Last error: {last_exception}")
         return None

    # --- extract_resume_data_async MODIFIED to use Jinja2 ---
    async def extract_resume_data_async(self, resume_text: str) -> Optional[ResumeData]:
        MAX_RESUME_CHARS_FOR_LLM = settings.get('analysis', {}).get('max_prompt_chars', 15000) # Use config limit
        if not resume_text or not resume_text.strip(): log.warning("[yellow]Resume text empty.[/yellow]"); return None
        resume_text_for_prompt = resume_text
        if len(resume_text) > MAX_RESUME_CHARS_FOR_LLM:
            log.warning(f"[yellow]Truncating resume text ({len(resume_text)} > {MAX_RESUME_CHARS_FOR_LLM}).[/yellow]")
            resume_text_for_prompt = resume_text[:MAX_RESUME_CHARS_FOR_LLM]

        try:
            # Render the prompt using Jinja2
            prompt = self.resume_prompt_template.render(resume_text=resume_text_for_prompt)
        except Exception as render_err:
             log.error(f"[bold red]Failed to render resume extraction prompt:[/bold red] {render_err}", exc_info=True)
             return None

        log.info("ASYNC: Requesting resume data extraction..."); extracted_json = await self._call_ollama_async(prompt)
        if extracted_json:
            try:
                if isinstance(extracted_json, dict): resume_data = ResumeData(**extracted_json); log.info("[green]ASYNC: Parsed extracted resume data.[/green]"); return resume_data
                else: log.error(f"[red]ASYNC Resume extract response not dict:[/red] {type(extracted_json)}"); return None
            except Exception as e: log.error(f"[bold red]ASYNC: Failed validate extracted resume:[/bold red] {e}", exc_info=True); log.error(f"Invalid JSON: {extracted_json}"); return None
        else: log.error("[bold red]ASYNC: Failed get response for resume extract.[/bold red]"); return None

    # --- analyze_suitability_async MODIFIED to use Jinja2 ---
    async def analyze_suitability_async(self, resume_data: ResumeData, job_data: Dict[str, Any]) -> Optional[JobAnalysisResult]:
        job_title = job_data.get('title', 'N/A') # Get title early for logs
        if not resume_data: log.warning(f"[yellow]Missing structured resume data for '{job_title}'.[/yellow]"); return None
        if not job_data or not job_data.get("description"): log.warning(f"[yellow]Missing job description for '{job_title}'. Skipping analysis.[/yellow]"); return None

        try:
            # Prepare context for Jinja2 - pass Python objects directly
            resume_data_json_str = resume_data.model_dump_json(indent=2) # Still need JSON string for prompt text
            # Pass the full job_data dict to Jinja2
            context = {
                "resume_data_json": resume_data_json_str,
                "job_data_json": job_data # Pass the dictionary here
            }
            # Render the prompt using Jinja2
            prompt = self.suitability_prompt_template.render(context)

        except Exception as e:
            log.error(f"[red]Error preparing/rendering suitability prompt for '{job_title}':[/red] {e}", exc_info=True)
            return None

        log.info(f"ASYNC: Requesting suitability analysis for: [cyan]{job_title}[/cyan]")
        combined_json_response = await self._call_ollama_async(prompt) # Await async call

        if not combined_json_response or not isinstance(combined_json_response, dict):
            log.error(f"[red]ASYNC: Failed get valid JSON dict for suitability: {job_title}.[/red]");
            log.error(f"Raw response: {combined_json_response}"); return None

        analysis_data = combined_json_response.get("analysis")
        if not analysis_data or not isinstance(analysis_data, dict):
            log.error(f"[red]ASYNC: Response JSON missing valid 'analysis' dict for: {job_title}.[/red]");
            log.debug(f"Full response: {combined_json_response}"); return None

        try:
            analysis_result = JobAnalysisResult(**analysis_data)
            log.info(f"ASYNC: Suitability score for '[cyan]{job_title}[/cyan]': [bold cyan]{analysis_result.suitability_score}%[/bold cyan]")
            return analysis_result
        except Exception as e:
            log.error(f"[red]ASYNC: Failed validate analysis result for '{job_title}':[/red] {e}", exc_info=True)
            log.error(f"Invalid 'analysis' structure: {analysis_data}"); return None