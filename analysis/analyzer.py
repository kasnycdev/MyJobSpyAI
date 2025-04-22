import ollama
import json
import logging
import time
import os
import asyncio
from typing import Dict, Optional, Any

from analysis.models import ResumeData, JobAnalysisResult, AnalyzedJob
from config import settings


# OopCompanion:suppressRename

log = logging.getLogger(__name__)

def load_prompt(prompt_key: str) -> str:
    path = settings.get('analysis', {}).get(prompt_key)
    if not path: raise ValueError(f"Missing prompt configuration for {prompt_key}")
    try:
        with open(path, 'r', encoding='utf-8') as f: return f.read()
    except FileNotFoundError: log.error(f"Prompt file not found: {path}"); raise
    except Exception as e: log.error(f"Error reading prompt file {path}: {e}"); raise

class ResumeAnalyzer:
    def __init__(self):
        log.info("Initializing Ollama clients...")
        try:
            ollama_cfg = settings.get('ollama', {})
            self.sync_client = ollama.Client( host=ollama_cfg.get('base_url'), timeout=ollama_cfg.get('request_timeout') )
            self.async_client = ollama.AsyncClient( host=ollama_cfg.get('base_url'), timeout=ollama_cfg.get('request_timeout') )
            log.info("Ollama clients initialized.")
        except Exception as e: log.critical(f"Failed to initialize Ollama clients: {e}", exc_info=True); raise RuntimeError(f"Ollama client init failed: {e}") from e
        self.resume_prompt_template = load_prompt("resume_prompt_path")
        self.suitability_prompt_template = load_prompt("suitability_prompt_path")
        self._check_connection_and_model()

    def _check_connection_and_model(self):
        try:
            ollama_cfg = settings.get('ollama', {})
            ollama_model = ollama_cfg.get('model')
            base_url = ollama_cfg.get('base_url')

            # --- Ensure no stray backticks here ---
            log.info(f"Checking Ollama connection at {base_url}...")
            # --- End check ---

            self.sync_client.ps(); log.info("Ollama connection successful.")
            log.info("Fetching local Ollama models...")
            ollama_list_response = self.sync_client.list(); log.debug(f"Raw list response: {ollama_list_response}")
            models_data = ollama_list_response.get('models', []);
            if not isinstance(models_data, list): models_data = []
            local_models = []
            for m in models_data:
                if hasattr(m, 'model') and isinstance(m.model, str) and m.model: local_models.append(m.model)
                elif isinstance(m, dict) and m.get('name'): log.warning(f"Found dict item: {m}"); local_models.append(m.get('name'))
                else: log.warning(f"Could not extract model name from item: {m}")
            log.info(f"Parsed local models: {local_models}")

            if ollama_model not in local_models:
                log.warning(f"Model '{ollama_model}' not found locally. Attempting pull...");
                try:
                    self._pull_model_with_progress(ollama_model)
                    log.info("Re-fetching model list after pull...")
                    updated_list_response = self.sync_client.list(); updated_models_data = updated_list_response.get('models', [])
                    updated_names = []
                    # --- Corrected Indentation/Logic for Final Check ---
                    for idx, m_upd in enumerate(updated_models_data): # Iterate to potentially log index
                         model_name = None
                         if hasattr(m_upd, 'model') and isinstance(m_upd.model, str) and m_upd.model:
                              model_name = m_upd.model
                              updated_names.append(model_name)
                         elif isinstance(m_upd, dict) and m_upd.get('name'):
                              # Handle potential dictionary format as fallback
                              model_name = m_upd.get('name')
                              log.warning(f"Found dictionary item in updated model list: {m_upd}. Using 'name' key.")
                              updated_names.append(model_name)
                         # Log only if extraction failed for an item
                         if model_name is None:
                              log.warning(f"Could not extract name from updated model list item {idx}: {m_upd}")
                    # --- End Correction ---
                    log.debug(f"Model list after pull: {updated_names}")
                    if ollama_model not in updated_names:
                         log.error(f"Model '{ollama_model}' still not found after attempting pull and re-checking list.")
                         # Removed extra delay/check here for simplicity, assume list is final
                         raise ConnectionError(f"Ollama model pull seemed complete but model '{ollama_model}' not listed.")

                except Exception as pull_err:
                    log.error(f"Pull/verify failed for '{ollama_model}': {pull_err}", exc_info=True)
                    raise ConnectionError(f"Required Ollama model '{ollama_model}' unavailable/pull failed.") from pull_err
            else:
                log.info(f"Using configured Ollama model: {ollama_model}")

        except Exception as e:
            log.critical(f"Ollama connection/setup failed: {e}", exc_info=True)
            raise ConnectionError(f"Ollama connection/setup failed: {e}") from e

    # --- _pull_model_with_progress method remains unchanged ---
    def _pull_model_with_progress(self, model_name: str):
        current_digest = ""; status = ""
        try:
            for progress in ollama.pull(model_name, stream=True):
                 digest = progress.get("digest", "");
                 if digest != current_digest and current_digest != "": print()
                 if digest: current_digest = digest; status = progress.get('status', ''); print(f"Pulling {model_name}: {status}", end='\r')
                 else: status = progress.get('status', ''); print(f"Pulling {model_name}: {status}")
                 if progress.get('error'): raise Exception(f"Pull error: {progress['error']}")
                 if 'status' in progress and 'success' in progress['status'].lower(): print(); log.info(f"Successfully pulled model {model_name}"); break
        except Exception as e: print(); log.error(f"Error during model pull: {e}"); raise
        finally: print()


    # --- _call_ollama_async method remains unchanged ---
    async def _call_ollama_async(self, prompt: str) -> Optional[Dict[str, Any]]:
         ollama_cfg = settings.get('ollama', {}); analysis_cfg = settings.get('analysis', {})
         ollama_model = ollama_cfg.get('model'); max_retries = ollama_cfg.get('max_retries', 2)
         retry_delay = ollama_cfg.get('retry_delay', 5); max_prompt_chars = analysis_cfg.get('max_prompt_chars', 24000)
         log.debug(f"ASYNC: Sending request to {ollama_model}. Prompt length: {len(prompt)} chars.")
         if len(prompt) > max_prompt_chars: log.warning(f"Prompt length exceeds {max_prompt_chars} chars.")
         last_exception = None
         for attempt in range(max_retries):
             try:
                 response = await self.async_client.chat( model=ollama_model, messages=[{'role': 'user', 'content': prompt}], format='json', options={'temperature': 0.1} )
                 content = response['message']['content']; log.debug(f"ASYNC: Ollama raw response: {content[:500]}...")
                 try:
                     content_strip = content.strip()
                     if content_strip.startswith("```json"): content_strip = content_strip[7:]; content_strip = content_strip[:-3] if content_strip.endswith("```") else content_strip
                     elif content_strip.startswith("```"): content_strip = content_strip[3:]; content_strip = content_strip[:-3] if content_strip.endswith("```") else content_strip
                     result = json.loads(content_strip.strip()); log.debug("ASYNC: Parsed JSON response."); return result
                 except json.JSONDecodeError as json_err: log.warning(f"ASYNC JSON Decode Error (Attempt {attempt + 1}): {json_err}"); log.debug(f"Problematic content: {content}"); last_exception = json_err
             except (ollama.ResponseError, asyncio.TimeoutError, ConnectionError, TimeoutError) as conn_err: log.warning(f"ASYNC Ollama API Error (Attempt {attempt + 1}): {conn_err}"); last_exception = conn_err
             except Exception as e: log.error(f"ASYNC Unexpected Ollama Error (Attempt {attempt + 1}): {e}", exc_info=True); last_exception = e
             if attempt < max_retries - 1: delay = retry_delay * (2 ** attempt); log.info(f"ASYNC: Retrying in {delay:.1f}s..."); await asyncio.sleep(delay)
             else: log.error(f"ASYNC: Ollama call failed after {max_retries} attempts."); log.error(f"ASYNC: Last error: {last_exception}")
         return None

    # --- extract_resume_data_async method remains unchanged ---
    async def extract_resume_data_async(self, resume_text: str) -> Optional[ResumeData]:
        MAX_RESUME_CHARS_FOR_LLM = 15000
        if not resume_text or not resume_text.strip(): log.warning("Resume text empty."); return None
        if len(resume_text) > MAX_RESUME_CHARS_FOR_LLM: log.warning(f"Truncating resume text ({len(resume_text)} > {MAX_RESUME_CHARS_FOR_LLM})."); resume_text_for_prompt = resume_text[:MAX_RESUME_CHARS_FOR_LLM]
        else: resume_text_for_prompt = resume_text
        prompt = self.resume_prompt_template.format(resume_text=resume_text_for_prompt)
        log.info("ASYNC: Requesting resume data extraction..."); extracted_json = await self._call_ollama_async(prompt)
        if extracted_json:
            try:
                if isinstance(extracted_json, dict): resume_data = ResumeData(**extracted_json); log.info("ASYNC: Parsed extracted resume data."); return resume_data
                else: log.error(f"ASYNC Resume extract response not dict: {type(extracted_json)}"); return None
            except Exception as e: log.error(f"ASYNC: Failed validate extracted resume: {e}", exc_info=True); log.error(f"Invalid JSON: {extracted_json}"); return None
        else: log.error("ASYNC: Failed get response for resume extract."); return None

    # --- analyze_suitability_async method remains unchanged ---
    async def analyze_suitability_async(self, resume_data: ResumeData, job_data: Dict[str, Any]) -> Optional[JobAnalysisResult]:
        if not resume_data: log.warning("Missing structured resume data."); return None
        if not job_data or not job_data.get("description"): log.warning(f"Missing job description for '{job_data.get('title', 'N/A')}'. Skipping analysis."); return None
        try:
            resume_data_json = resume_data.model_dump_json(indent=2); job_data_json = json.dumps(job_data, indent=2, default=str)
            prompt = self.suitability_prompt_template.format( resume_data_json=resume_data_json, job_data_json=job_data_json )
        except Exception as e: log.error(f"Error preparing suitability prompt: {e}", exc_info=True); return None
        job_title = job_data.get('title', 'N/A'); log.info(f"ASYNC: Requesting suitability analysis for: {job_title}")
        combined_json_response = await self._call_ollama_async(prompt)
        if not combined_json_response or not isinstance(combined_json_response, dict): log.error(f"ASYNC: Failed get valid JSON dict for suitability: {job_title}."); log.error(f"Raw response: {combined_json_response}"); return None
        analysis_data = combined_json_response.get("analysis")
        if not analysis_data or not isinstance(analysis_data, dict): log.error(f"ASYNC: Response JSON missing valid 'analysis' dict for: {job_title}."); log.debug(f"Full response: {combined_json_response}"); return None
        try:
            analysis_result = JobAnalysisResult(**analysis_data)
            log.info(f"ASYNC: Suitability score for '{job_title}': {analysis_result.suitability_score}%")
            return analysis_result
        except Exception as e: log.error(f"ASYNC: Failed validate analysis result for '{job_title}': {e}", exc_info=True); log.error(f"Invalid 'analysis' structure: {analysis_data}"); return None