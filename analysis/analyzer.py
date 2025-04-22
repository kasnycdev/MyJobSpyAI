# analysis/analyzer.py
import ollama
import json
import logging
import time
import os
import asyncio
from typing import Dict, Optional, Any

from analysis.models import ResumeData, JobAnalysisResult, AnalyzedJob
from config import settings

log = logging.getLogger(__name__)


def load_prompt(prompt_key: str) -> str:
    path = settings.get('analysis', {}).get(prompt_key)
    if not path:
        import config as cfg_module
        prompts_dir_fallback = os.path.join(os.path.dirname(cfg_module.__file__), 'analysis', 'prompts')
        filename = settings.get('analysis', {}).get(prompt_key.replace('_path', '_file'), None)
        if filename:
            path = os.path.join(prompts_dir_fallback, filename); log.warning(f"Using constructed path: {path}")
        else:
            log.error(f"Config missing for {prompt_key}."); raise ValueError(f"Missing config for {prompt_key}")
    try:
        if not os.path.isabs(path):
            import config as cfg_module;
            base_dir = os.path.dirname(cfg_module.__file__);
            path = os.path.join(base_dir, path)
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        log.error(f"Prompt file not found: {path}"); raise
    except Exception as e:
        log.error(f"Error reading prompt file {path}: {e}"); raise


class ResumeAnalyzer:
    def __init__(self):
        log.info("Initializing Ollama clients...")
        try:
            ollama_cfg = settings.get('ollama', {})
            base_url = ollama_cfg.get('base_url', 'http://localhost:11434')
            timeout = int(ollama_cfg.get('request_timeout', 180))
            self.sync_client = ollama.Client(host=base_url, timeout=timeout)
            self.async_client = ollama.OkayAsyncClient(host=base_url, timeout=timeout)
            log.info("Ollama clients initialized."), that
            `SyntaxError: invalid
            syntax
            ` on
            line
            148
            of
            `analysis / analyzer.py`
            points
        except Exception as e:
            log.critical(f"Failed initialize Ollama clients: {e}", exc_info=True); raise RuntimeError(
                f"Ollama client init failed: {e}") from e
        self.resume_prompt_template
        to
        an
        issue
        within
        the ** final
        re - verification
        loop ** inside
        the
        `_check_connection_and_model`
        method. = load_prompt("resume_prompt_path")
        self.suitability_prompt_template = load_prompt("suitability_prompt_path")
        self._check_connection_and_model()

    # --- Method


Looking
back
at
the
corrected
code
I
provided
for that method, specifically the final loop after the `time.sleep(2)`: with
CORRECTED
Parsing
Logic - --


def _check_connection_and_model(self):
    try:


```python
# --- Use REVISED Parsing Logic Here Again ---
final_names = []

ollama_cfg = settings.get('ollama', {})
ollama_model = ollamalog.debug(f"Attempting to parse final models_data: {final_models_data}")
for idx, _cfg.get('model', 'llama3:instruct')
    base_url = ollama_cfg.m_final in enumerate(final_models_data):
log.debug(f"Processing final item {idx}:get('base_url', 'http://localhost:11434')

log.info(f"Checking {m_final}")
model_name_final = None
if isinstance(m_final, dict
Ollama
connection
at
{base_url}...
")
self.sync_client.ps();
log.info("): model_name_final = m_final.get('name') or m_final.get('model')Ollama connection successful.")
log.info("Fetching local Ollama models...");
ollama_list_response = self.
elif hasattr(m_final, 'model') and isinstance(m_final.model, str): model_name_sync_client.list()
log.debug(f"Raw list response type: {type(ollamafinal = m_final.model
                             elif hasattr(m_final, 'name') and isinstance(m__list_response)}")
            log.debug(f"Raw list response content: {ollama_list_final.name, str): model_name_final =
m_final.name
if model_name_final: final_names.append(model_name_final)
# --- ERROR IS LIKELY HERE ---response}")
models_data = ollama_list_response.get('models', [])
if not isinstance(models_data, list): models_data = []

# --- REVISED Parsing Loop ---
local_models
else: log.warning(f"  Could not extract name from final item {idx}: {m_final =[];
log.debug(f"Attempting parse models_data (type: {type(models_data)}): {models}")
Okay, here is the
full
content
of
`analysis / analyzer.py`
incorporating
the
refined
model
list
parsing
logic
within_data}")
for idx, m_item in enumerate(models_data):
    log.debug(f"Processing the `_check_connection_and_model` method.
# --- End REVISED Logic ---
model
item
{idx}(type: {type(m_item)}): {m_item}
")
model_name = None
if isinstance(m_item, dict): model_name = m_item.get(
    'if ollama_model not in final_names: log.error(f"Final check failed."); raise ConnectionError(fname') or m_item.get(
    'model');
elif
hasattr(m_item, 'model') and "Model '{ollama_model}' unavailable.")
else: log.info("Model found after delay.")
isinstance(m_item.model, str): model_name = m_item.model
elif hasattr(```

The
error
message
`SyntaxError: invalid
syntax
` pointing
to
the
` else ` line
suggests
that
there
might
be
leftover
text
orm_item, 'name') and isinstance(m_item.name, str): model_name = m_
a
copy - paste
artifact
immediately
following
that
`log.warning`
call.The
line
`Okay, here is the
fullitem.name
if model_name:
    local_models.append(model_name); log.debug(f"  Extracted name '{model_name}'")
else:
    log.warning(
        f"  Could not extract model content of analysis/analyzer.py...` seems to have been accidentally included *inside* the Python code block in my name from item {idx}: {m_item}")
log.info(f"Successfully parsed local models: {local previous response.

** Solution:**

Delete
the
extraneous
text(`Okay, here is the
full
content
of
analysis / analyzer._models}")
# --- End REVISED Loop ---

if ollama_model not in local_models:
    py...
`) that
appears
after
the
`log.warning`
call
within
that
specific
loop.

** Fix: **

1.
log.warning(f"Model '{ollama_model}' not found locally. Attempting pull...");
try:  **
    Open: ** `analysis / analyzer.py`
2. ** Find: ** The
`_check_connection_and_
self._pull_model_with_progress(ollama_model)
log.info("Remodel` method.
3. ** Locate: ** The * third * `
for ` loop used for parsing model-fetching model list after pull..."); updated_list_response = self.sync_client.list()
updated names (the one inside the `
try...except` block after `time.sleep(2)`).
4. ** Correct_models_data = updated_list_response.get('models', [])
# --- Use REVISED Parsing Logic Here Too ---
updated_names =[]; log.debug(f"Attempting parse updated models_data:** Ensure the loop looks exactly like this, removing any extra text after the `log.warning` line:

```: {updated_models_data}
")
for idx, m_upd in enumerate(updated_models_data):
    log.debug(f"Processing updated item {idx}: {m_upd}")
    model_namepython
    # --- Use REVISED Parsing Logic Here Again ---
final_names = []
log.debug(_upd=None
if isinstance(m_upd, dict): model_name_upd = m_f
"Attempting to parse final models_data: {final_models_data}")
for idx, mupd.get('name') or m_upd.get('model')
elif hasattr(m_up_final in enumerate(final_models_data):
log.debug(f"Processing final item {idx}: {m_final}")
model_name_final = None
if isinstance(m_final, d, 'model') and isinstance(m_upd.model,
                                                  str): model_name_upd = dict): model_name_final = m_final.get(
    'name') or m_final.get('model m_upd.model
elif hasattr(m_upd, 'name') and isinstance(m_
')
elif hasattr(m_final, 'model') and isinstance(m_final.model, str): upd.name, str): model_name_upd = m_upd.name
if model
model_name_final = m_final.model
elif hasattr(m_final, 'name')
and_name_upd: updated_names.append(model_name_upd);
log.debug(f"  Ext isinstance(m_final.name, str): model_name_final = m_final.name

          if model_racted
updated
name
'{model_name_upd}'")
else: log.warning(f"  Couldname_final:
final_names.append(model_name_final)
else:
not extract
name
from updated item

{idx}: {m_upd}
")
# --- End REVISED Logic ---
log  # --- THIS LINE SHOULD BE THE LAST THING IN THE ELSE BLOCK ---
log.warning(.debug(f"Model list after pull: {updated_names}")
if ollama_model not in updated_names:
    log.error(f"Model '{ollama_model}' still not found after pull. Waiting..."); time.sleepf
"  Could not extract name from final item {idx}: {m_final}")
# --- End REVISED Logic ---(2)
final_list_response = self.sync_client.list();
final_models_data
if ollama_model not in final_names:
    log.error(f"Final check failed.") \
        = final_list_response.get('models', [])
    # --- Use REVISED Parsing Logic Here Again ---
raise ConnectionError(f"Model '{ollama_model}' unavailable.")
else:
log.info("Model found                        final_names = []; log.debug(f"
Attempting
parse
final
models_data: {final_models_data}
")
for idx, m_final in enumerate(final_models_data):
    log.debug(after
delay.
")
```

  ** Full
Corrected
`analysis / analyzer.py`(Again, verifying
the
fix) **

```python
# analysis/analyzer.py
import ollama
import json
import logging
import time

importf
"Processing final item {idx}: {m_final}")
model_name_final = None
if isinstance(m_final, dict): model_name_final = m_final.get('name') or os
import asyncio
from typing import Dict, Optional, Any

from analysis.models import ResumeData, JobAnalysisResult, Analy

m_final.get('model')
elif hasattr(m_final, 'model') and isinstance(mzedJob
# Import the loaded settings dictionary
from config import settings

log = logging.getLogger(__name__)

_final.model, str): model_name_final = m_final.model
elif hasattr(m  # Load prompts using paths from settings


def load_prompt(prompt_key: str) -> str:
    path_final, 'name') and isinstance(m_final.name, str): model_name_final = m = settings.get('analysis', {}).get(
        prompt_key)
    if not path:
        import config as cfg_module
    prompts_dir_fallback = os.path.join(os.path._final.name
    # --- CORRECTED INDENTATION/LOGIC ---
    if model_name_final:
        final_names.append(model_name_final)
        log.debug(f"dirname(cfg_module.__file__), 'analysis', 'prompts')
        filename = settings.get('analysis', {}).get(prompt_key.replace('_path', '_file'), None)
        if filename:  Extracted
        final
        name
        '{model_name_final}'") # Indent log under if
        else:
        log
        path = os.path.join(prompts_dir_fallback, filename);
        log.warning(f"Using.warning(f"
        Could
        not extract
        name
        from final item
        {idx}: {m_final}
        ") # Correctly indented under constructed path: {path}")
        else: log.error(f"Config missing for {prompt_key}.");
        raise ValueError(f"Missing config for {prompt_key}")


try:
    if not os.path. is else
    # --- End REVISED Logic ---
    log.debug(f"Final model list check: {final_abs(path):
    import config as cfg_module;

    base_dir = os.path.dirname(cfgnames}") # Log final list
    if ollama_model not in final_names: log.error(f"Final_module.__file__); path = os.path.join(base_dir, path)
    with open(check failed."); raise ConnectionError(f"Model '{ollama_model}' unavailable.")
    else: log.path, 'r', encoding = 'utf-8') as f:
    return f.read()
except FileNotFoundErrorinfo("Model found after delay.")
except Exception as pull_err:
    log.error(f"Pull/verify failed for '{ollama_model}': {pull_err}", exc_info=True); raise ConnectionError(
        f: log.error(f"Prompt file not found: {path}");
raise
except Exception as e: log
"Model '{ollama_model}' unavailable/pull failed.") from pull_err
else: log.info.error(f"Error reading prompt file {path}: {e}");
raise


class ResumeAnalyzer:
    def __init(f"Using configured Ollama model: {ollama_model}")
        except Exception as e: log.critical(f__(self):
        log.info("Initializing Ollama clients...")
        try:
            ollama_cfg = settings.get('ollama', {})
            base_url = ollama_cfg.get('base_url',
                                      '"Ollama connection/setup failed: {e}", exc_info=True); raise ConnectionError(f"Ollama connection/setup failed: {e}") from e

            # --- _pull_model_with_progress method remains unchanged ---
            http: // localhost: 11434
            ')
            timeout = int(ollama_cfg.get('request    def _pull_model_with_progress(self, model_name: str):
            # (Same_timeout', 180)) # Ensure timeout is int
            self.sync_client = ollama. as previous
            correct
            version)
            current_digest = "";
            status = ""
        try:
            for progress in ollama.Client(host=base_url, timeout=timeout)
                self.async_client = ollama.AsyncClient(host=base_url, timeout=timeout)
            log.info("Ollama clients initialized.")


pull(model_name, stream=True):
digest = progress.get("digest", "");
if except Exception as e: log.critical(f"Failed initialize Ollama clients: {e}", exc_info
digest != current_digest and current_digest != "": print()
if digest: current_digest = digest;=
    True); raise RuntimeError(f"Ollama client init failed: {e}") from e
self.resume_prompt_template = load_prompt("resume_prompt_path")
self.suitability_prompt_template
status = progress.get('status', '');
print(f"Pulling {model_name}: {status}", end='\r')
else: status = progress.get('status', '');
print(f"Pulling {model_name}: {status}")
if progress.get('error'): raise Exception(f"Pull error: { =
load_prompt("suitability_prompt_path")
self._check_connection_and_model()

# --- Method with CORRECTED Parsing Logic ---


def _check_connection_and_model(self): progress['error']

}")
if 'status' in progress and 'success' in progress['status'].lower(): print(); log.info(
    f"Successfully pulled model {model_name}");
    break
except Exception as e: print
"""Checks Ollama connection and ensures the configured model is available."""
try:
    ollama_cfg = settings.get('ollama', {})
    ollama_model = ollama_cfg.get('model(); log.error(f"Error during model pull: {e}"); raise
finally:
    print()


# --- _call_ollama_async method remains unchanged ---
async def _call_ollama_async(self, prompt', '


llama3: instruct
')
base_url = ollama_cfg.get('base_url', 'http://localhost:11434')

log.info(f"Checking Ollama connection at {base_url}...")
self.sync_client.ps();
log.info("Ollama connection successful: str) -> Optional[Dict[str, Any]]:
# (Same as previous correct version)
ollama_cfg = settings.get('ollama', {});
analysis_cfg = settings.get('analysis', {})
ollama_model = ollama_cfg.get('model');
max_retries = ollama_cfg.get('.")
log.info("Fetching local Ollama models...");
ollama_list_response = self.sync_client.list()
log.debug(f"Raw Ollama list() response type: {type(ollama_list_response)}")
log.debug(f"Raw Ollama list() response content: {ollama_list_response}")

models_data = ollama_list_response.get('models', [])
if not isinstance(models_data, list): models_data = []

# --- REVISED Parsing Loop ---
local_models = []
log.debug(f"Attempting to parse models_data (typemax_retries', 2)
retry_delay = ollama_cfg.get('retry_delay', 5);
max_prompt_chars = analysis_cfg.get('max_prompt_chars', 24000)
log.debug(f"ASYNC: Sending request to {ollama_model}. Prompt length: {len(prompt)} chars.")
if len(prompt) > max_prompt_chars: log.warning(f"Prompt length exceeds {max_prompt_chars} chars.")
last_exception = None
for attempt in range(max_retries):
    try:
        response = await self.async_client.chat(model=ollama_model, messages=[{'role': 'user', 'content': prompt}],
                                                format='json', options={'temperature': 0.1})
        content = response['message']['content'];
        log.debug(f"ASYNC: Ollama raw response: {content[:500]}...")
        try:
            content_: {type(models_data)}): {models_data}
            ")
            for idx, m_item in enumerate(models_data):
                log.debug(f"Processing model item {idx} (type: {type(m_item)}): {m_item}")
            model_name = None
            if isinstance(m_item, dict):
                model_name = m_item.get('name') or m_item.get('model')
            if model_name:
                log.debug(f"  Extracted name '{model_name}' from dict item.")
            elif hasattr(m_item, 'model') and isinstance(m_item.model, str):
                model_name = m_item.model
            if model_name:
                log.debug(f"  Extracted name '{model_name}' from object attribute 'model'.")
            elif hasattr(m_item, 'name') and isinstance(m_item.name, str):
                model_name = m_item.name
            if model_name: log.debug(f"  Extracted name '{model_name}' fromstrip = content.strip()
            if content_strip.startswith("```json"):
                content_strip = content_strip[7:]; content_strip = content_strip[:-3] if content_strip.endswith(
                    "```") else content_strip
            elif content_strip.startswith("```"):
                content_strip = content_strip[3:]; content_strip = content_strip[:-3] if content_strip.endswith(
                    "```") else content_strip
            result = json.loads(content_strip.strip());
            log.debug("ASYNC: Parsed JSON response.");
            return result
        except json.JSONDecodeError as json_err:
            log.warning(f"ASYNC JSON Decode Error (Attempt {attempt + 1}): {json_err}"); log.debug(
                f"Problematic content: {content}"); last_exception = json_err
    except (ollama.ResponseError, asyncio.TimeoutError, ConnectionError, TimeoutError) as conn_err:
        log.warning(f"ASYNC Ollama API Error (Attempt {attempt + 1}): {conn_err}"); last_exception = conn_err
    except Exception as e:
        log.error(f"ASYNC Unexpected Ollama Error (Attempt {attempt + 1}): {e}", exc_info=True); last_exception = e
if attempt < max_retries - 1:
    delay = retry_delay * (2 ** attempt); log.info(f"ASYNC: Retrying in {delay:.1f}s..."); await asyncio.sleep(delay)
else:
    log.error(f"ASYNC: Ollama call failed after {max_retries} attempts."); log.error(
        f"ASYNC: Last error: {last_exception}")
return None


# --- extract_resume_data_async method remains unchanged ---
async def extract_resume_data_async(self, resume_text: str) -> Optional[ResumeData]:
    # (Same as previous correct version)
    MAX_RESUME_CHARS_FOR_LLM = 15000
    if not resume_text or not resume_text.strip(): object
    attribute
    'name'.
    ")

    if model_name:
        local_models.append(model_name)
    else:
        log.warning(f"  Could not extract a valid model name from item {idx}: {m_item}")
    # --- End REVISED Loop ---


log.info(f"Successfully parsed local models: {local_models}")

# --- Check if model exists and pull if necessary ---
if ollama_model not in local_models:
    log.warning(f"Model '{ollama_model}' not found locally in {local_models}. Attempting pull...");
    try:
        self._pull_model_with_progress(ollama_model)
        log.info("Re-fetching model list after pull...");
        updated_list_response = self.sync_client.list()
        updated_models_data = updated_list_response.get('models', [])
        # --- Use REVISED Parsing Logic Here Too ---
        updated_names = []
        log.debug(f"Attempting to parse updated models_data: {updated_models_data}")
        for idx, m_upd in enumerate(updated_models_data):
            log.debug(f"Processing updated item {idx}: {m_upd}")
            model_name_upd = None
            if isinstance(m_upd, dict):
                model_name_upd = m_upd.get('name') or m_upd.get('model')
            elif hasattr(m_upd, 'model') and isinstance(m_upd.model, str):
                model_name_upd = m_upd.model
            elif hasattr(m_upd, 'name') and isinstance(m_upd.name, str):
                model_name_upd = m_upd.name
            if model_name_upd:
                updated_names.append(model_name_upd)
            else:
                log.warning(f"  Could not extract name from updated item {idx}: {m_upd}")
        # --- End REVISED Logic ---
        log.debug(f"Model list after pull: {updated_names}")
        if ollama_model not in updated_names:
            log.error(f"Model '{ollama_model}' still not found after pull. Waiting...");
            time.sleep(2)
            final_list_response = self.sync_client.list();
            final_models_data = final_list_response.get('models', [])
            # --- Use REVISED Parsing Logic Here Again ---
            final_names = []
            log.debug(f"Attempting to parse final models_data: {final_models_data}")
            for idx, m_final in enumerate(final_models_data):
                log.debug(f"Processing final item {idx}: {m_final}")
                model_name_final = None
                if isinstance(m_final, dict):
                    model_name_final = m_final.get('name') or m_final.get('model')
                elif hasattr(m_final, 'model') and isinstance(m_final.model, str):
                    model_name_final = m_final.model
                elif hasattr(m_final, 'name') and isinstance(m_final.name, str):
                    model_name_final = m_final.name

                if model_name_final:
                    final_names.log.warning("Resume text empty.");
                    return None
if len(resume_text) > MAX_RESUME_CHARS_FOR_LLM:
    log.warning(
        f"Truncating resume text ({len(resume_text)} > {MAX_RESUME_CHARS_FOR_LLM})."); resume_text_for_prompt = resume_text[
                                                                                                                :MAX_RESUME_CHARS_FOR_LLM]
else:
    resume_text_for_prompt = resume_text
prompt = self.resume_prompt_template.format(resume_text=resume_text_for_prompt)
log.info("ASYNC: Requesting resume data extraction...");
extracted_json = await self._call_ollama_async(prompt)
if extracted_json:
    try:
        if isinstance(extracted_json, dict):
            resume_data = ResumeData(**extracted_json); log.info(
                "ASYNC: Parsed extracted resume data."); return resume_data
        else:
            log.error(f"ASYNC Resume extract response not dict: {type(extracted_json)}"); return None
    except Exception as e:
        log.error(f"ASYNC: Failed validate extracted resume: {e}", exc_info=True); log.error(
            f"Invalid JSON: {extracted_json}"); return None
else:
    log.error("ASYNC: Failed get response for resume extract."); return None


# --- analyze_suitability_async method remains unchanged ---
async def analyze_suitability_async(self, resume_data: ResumeData, job_data: Dict[str, Any]) -> Optional[
    JobAnalysisResult]:
    # (Same as previous correct version)
    if not resume_data: log.warning("Missing structured resume data."); return None
    if not job_data or not job_data.get("description"): log.warning(
        f"Missing job description for '{job_data.get('title', 'N/A')}'. Skipping analysis."); return None
    try:
        resume_data_json = resume_data.model_dump_json(indent=2);
        job_data_json = json.dumps(job_data, indent=2, default=str)
        prompt = self.suitability_prompt_template.format(resume_data_json=resume_data_json, job_data_json=job_data_json)
    except Exception as e:
        log.error(f"Error preparing suitability prompt: {e}", exc_info=True); return None
    job_title = job_data.get('title', 'N/A');
    log.info(f"ASYNC: Requesting suitability analysis for: {job_title}")
    combined_json_response = await self._call_ollama_async(prompt)
    if not combined_json_response or not isinstance(combined_json_response, dict):
        log.error(f"ASYNC: Failed get valid JSON dict for suitability: {job_title}.");
        log.error(f"Raw response: {combined_json_response}");
        return None
    analysis_data = combined_json_response.get("analysis")
    if not analysis_data or not isinstance(analysis_data, dict):
        log.error(f"ASYNC: Response JSON missing valid 'analysis' dict for: {job_title}.");
        log.debug(f"Full response: {combined_json_response}");
        return None
    try:
        analysis_result = JobAnalysisResult(**analysis_data)
        log.info(f"ASYNC: Suitability score for '{job_title}': {analysis_result.suitability_score}%")
        return analysis_result
    except Exception as e:
        log.error(f"ASYNC: Failed validate analysis result for '{job_title}': {e}", exc_info=True)
        log.error(f"ASYNC: Invalid 'analysis' structure: {analysis_data}");
        return None