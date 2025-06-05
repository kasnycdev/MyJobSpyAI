    # MyJobSpy Analyst: Scrape, Analyze, and Filter Jobs with GenAI

This project enhances job searching by combining the scraping power of **[JobSpy](https://github.com/speedyapply/JobSpy)** with Generative AI analysis via **multiple LLM backends (Ollama, OpenAI-compatible APIs like LM Studio, Google Gemini)** and advanced filtering, including location awareness.

**Core Workflow:**

1.  **Configure:** Select your desired LLM provider (`ollama`, `openai`, or `gemini`) and configure its specific settings (API endpoint, model, API key if needed) in `config.yaml`. Prepare LinkedIn cookies in `config.toml` (optional but recommended).
2.  **Scrape:** Uses `JobSpy` to find jobs based on your search criteria and location across multiple job boards (LinkedIn, Indeed, etc.).
3.  **Analyze (Async):** Parses your resume (DOCX/PDF) using `PyPDF2`/`python-docx`. It then uses the configured LLM backend concurrently via `asyncio` to:
    *   Extract structured data from your resume (results are cached for speed).
    *   Compare your structured resume profile against each scraped job description.
    *   Generate a suitability score (0-100%), detailed justification, and skill/experience match assessment based on enhanced comparison logic.
4.  **Filter & Rank:** Filters the analyzed jobs based on a wide range of criteria including salary, job type, work model, company names (include/exclude), title keywords, date posted, location (remote country or proximity), and minimum suitability score. Ranks the final list by suitability score.
5.  **Output:** Saves the detailed analysis results (including original job data) to a JSON file and prints a summary table of top matches to the console using `rich`.

## Features

*   **Multi-Site Scraping:** Leverages `JobSpy` to scrape from sites like LinkedIn, Indeed, ZipRecruiter, Glassdoor, etc. (Check `JobSpy` docs for current support).
*   **Asynchronous Analysis:** Significantly speeds up analysis by processing multiple jobs concurrently with the selected LLM via its API using `asyncio` and the appropriate client library (`ollama`, `openai`, `google-generativeai`).
*   **Resume Parsing:** Handles `.docx` and `.pdf` resume files.
*   **Flexible GenAI Analysis:** Supports multiple LLM backends:
    *   **Ollama:** Run models locally for privacy and control.
    *   **OpenAI-compatible APIs (e.g., LM Studio):** Use local servers like LM Studio or other compatible endpoints.
    *   **Google Gemini:** Leverage Google's cloud-based models via API.
    *   Configurable provider selection, model names, API endpoints/keys, timeouts, and retries via `config.yaml`.
    *   Structured resume data extraction with emphasis on quantifiable achievements.
    *   Detailed job suitability scoring based on recruiter-like evaluation criteria (essentials, relevance, impact).
    *   Evidence-based justification for scores.
*   **Advanced Filtering:**
    *   Salary range (min/max).
    *   Job Type(s) (Full-time, Contract, etc.).
    *   Work Model(s) (Remote, Hybrid, On-site).
    *   Company Name inclusion or exclusion lists.
    *   Job Title keyword matching (any keyword).
    *   Date Posted range (after/before YYYY-MM-DD).
    *   Minimum Suitability Score (0-100).
*   **JobSpy Native Filter Configuration:** Exposes and allows configuration of several native JobSpy scraping filters via `config.yaml` and command-line arguments, including:
    *   Google search term
    *   Distance from location
    *   Is remote flag
    *   Job type(s)
    *   Easy apply flag
    *   CA certificate path
    *   LinkedIn company IDs
    *   Enforce annual salary conversion
    Command-line arguments override `config.yaml` settings for these filters.
*   **Caching:**
    *   **Resume Analysis:** Caches structured resume data based on file hash to speed up subsequent runs with the same resume (`output/.resume_cache/`). Use `--force-resume-reparse` to override.
*   **Robustness:**
    *   More specific error handling for scraping, analysis, file I/O, and network issues.
    *   Retry logic for LLM API calls tailored to each provider.
    *   Graceful handling of `Ctrl+C` interruptions.
    *   Warning for long LLM prompts (truncation may be needed depending on model/context window).
    *   Handles jobs with missing descriptions scraped by `JobSpy` by skipping analysis for them.
*   **Configuration:** Centralized settings via `config.yaml` with environment variable overrides for key parameters.
*   **Rich Output:** Provides detailed JSON output and a configurable summary table in the console.

## Observability (OpenTelemetry)

This project integrates OpenTelemetry to provide insights into its execution through logs, traces, and metrics.

**Setup & Configuration:**

1.  **Install Dependencies:**
    Ensure you have installed the necessary OpenTelemetry packages by running:
    ```bash
    pip install -r requirements.txt
    ```
    This will install `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-grpc`, and `opentelemetry-instrumentation-logging`.

2.  **Local Collector:**
    You need a local OpenTelemetry collector (e.g., Jaeger, Grafana Agent, OpenTelemetry Collector) capable of receiving OTLP gRPC data.
    *   **Example (Jaeger with Docker):**
        ```bash
        docker run -d --name jaeger \
          -e COLLECTOR_OTLP_ENABLED=true \
          -p 16686:16686 \
          -p 4317:4317 \
          jaegertracing/all-in-one:latest
        ```
        The Jaeger UI will be accessible at `http://localhost:16686`.

3.  **Configuration (`config.yaml`):**
    OpenTelemetry settings can be configured in your `config.yaml` under the `opentelemetry` section:
    ```yaml
    opentelemetry:
      OTEL_ENABLED: true  # Set to false to disable OpenTelemetry
      OTEL_SERVICE_NAME: "MyJobSpyAI"  # Name of your service
      OTEL_EXPORTER_OTLP_ENDPOINT: "http://localhost:4317" # Collector endpoint
      OTEL_TRACES_SAMPLER: "always_on"  # Options: "always_on", "traceidratio"
      OTEL_TRACES_SAMPLER_CONFIG:
        ratio: 0.5  # Sampling ratio if "traceidratio" is used (0.0 to 1.0)
      OTEL_RESOURCE_ATTRIBUTES:
        environment: "development"
        version: "0.1.0" # Or your application version
        # Add any other custom resource attributes here
    ```
    *   **Environment Variable Overrides**: Standard OpenTelemetry environment variables (e.g., `OTEL_SERVICE_NAME`, `OTEL_EXPORTER_OTLP_ENDPOINT`, `OTEL_TRACES_SAMPLER`, `OTEL_TRACES_SAMPLER_ARG`, `OTEL_RESOURCE_ATTRIBUTES`) will override settings from `config.yaml` or internal defaults if set.
    *   **Disabling OpenTelemetry**:
        *   Set `OTEL_ENABLED: false` in the `opentelemetry` section of your `config.yaml`.
        *   Alternatively, set the environment variable `OTEL_SDK_DISABLED=true`. This will disable OpenTelemetry regardless of the `config.yaml` setting.
    *   The application code converts the `OTEL_TRACES_SAMPLER` string (from config or env var) to the appropriate OpenTelemetry Sampler instance.

**Viewing Telemetry:**
Once your application runs with a collector active, you can view:
*   **Traces:** In Jaeger (or your chosen backend) to see distributed traces of operations.
*   **Logs:** Standard Python logs are automatically enriched with trace context and exported via OTLP.
*   **Metrics:** Initial metrics for LLM calls (counts, duration, errors) are collected and can be visualized in backends like Prometheus/Grafana if your collector is configured to export them.


## Prerequisites

*   **Python 3.9+** (Recommended for enhanced `asyncio` support and type hinting)
*   **Git**
*   **LLM Backend (Choose one or more):**
    *   **Ollama:** Install and run Ollama locally. ([https://ollama.com/](https://ollama.com/)). Ensure the server is running and desired models are pulled (e.g., `ollama pull llama3:instruct`).
    *   **LM Studio (or other OpenAI-compatible server):** Install and run LM Studio locally ([https://lmstudio.ai/](https://lmstudio.ai/)). Download models via its interface and start the local API server (usually `http://localhost:1234/v1`).
    *   **Google Gemini API Key:** Obtain an API key from Google AI Studio ([https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)).
*   **Python LLM Libraries:** Ensure necessary libraries are installed (see Setup).
*   **Playwright Browsers:** `JobSpy` uses Playwright for some scrapers. Install required browsers (might take time/disk space):
    ```bash
    playwright install
    ```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kasnycdev/MyJobSpy.git
    cd MyJobSpy
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows: .\venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(This installs `openai`, `ollama`, `google-generativeai`, and other core libraries listed in `requirements.txt`)*

4.  **Install Playwright browsers:**
    ```bash
    playwright install
    ```

5.  **Configure `config.yaml`:**
    *   Review and edit `config.yaml` in the project root.
    *   **Set `llm_provider`:** Choose `"openai"`, `"ollama"`, or `"gemini"`.
    *   **Configure the corresponding section (`openai`, `ollama`, or `gemini`):**
        *   **`openai`:** Set `base_url` (e.g., for LM Studio), `model` identifier, and `api_key` (if needed).
        *   **`ollama`:** Set `base_url` and `model` name (must be pulled in Ollama).
        *   **`gemini`:** Set `model` name and provide your `api_key` (or set `GOOGLE_API_KEY` environment variable).
    *   Adjust other settings (scraping, analysis, logging) as needed.

6.  **Configure LinkedIn Cookies (Optional but Recommended):**
    *   Create a `config.toml` file in the project root (this file is ignored by git).
    *   Find your `li_at` and `JSESSIONID` cookies from your browser after logging into LinkedIn (use Developer Tools -> Application/Storage -> Cookies).
    *   Add them to `config.toml`:
        ```toml
        # config.toml
        [linkedin]
        li_at = "YOUR_LI_AT_COOKIE_VALUE"
        JSESSIONID = "YOUR_JSESSIONID_COOKIE_VALUE_WITH_QUOTES" # Include quotes if present
        ```
    *   `JobSpy` will automatically detect and use this file for authenticated LinkedIn scraping.

## Usage

Run the main pipeline script from the project's root directory:

```bash
python run_pipeline.py --search "Your Job Search" --resume /path/to/your/resume.pdf [LOCATION_FLAG] [OPTIONS]
```

### Command-line Options

The following command-line arguments can be used to customize the job search:

*   `--search`: The search term (e.g., "software engineer").
*   `--resume`: Path to your resume file (PDF or DOCX).
*   `--location`: Location to search for jobs (e.g., "New York, NY").
*   `--filter-remote-country`: Filter for remote jobs in a specific country (e.g., "United States").
*   `--min-salary`: Minimum salary (e.g., 100000).
*   `-v`: Verbosity level (0=errors, 1=warnings, 2=all logs).

These command-line arguments override the settings in `config.yaml`.

### config.yaml Options

The `config.yaml` file allows for detailed configuration of the job scraping and analysis process. Key options include:

*   `llm_provider`: Specifies the LLM provider ("openai", "ollama", or "gemini").
*   `openai`, `ollama`, or `gemini` sections: Configure the specific settings for the chosen LLM provider, including API keys, model names, and base URLs.
*   `linkedin`: Configure LinkedIn cookie values.
*   JobSpy Native Filter Configuration:
    *   `google_search_term`
    *   `distance`
    *   `is_remote`
    *   `job_type`
    *   `easy_apply`
    *   `ca_cert`
    *   `linkedin_company_ids`
    *   `enforce_annual_salary`
    # MyJobSpy Analyst: Scrape, Analyze, and Filter Jobs with GenAI

This project enhances job searching by combining the scraping power of **[JobSpy](https://github.com/speedyapply/JobSpy)** with Generative AI analysis via **multiple LLM backends (Ollama, OpenAI-compatible APIs like LM Studio, Google Gemini)** and advanced filtering, including location awareness.

**Core Workflow:**

1.  **Configure:** Select your desired LLM provider (`ollama`, `openai`, or `gemini`) and configure its specific settings (API endpoint, model, API key if needed) in `config.yaml`. Prepare LinkedIn cookies in `config.toml` (optional but recommended).
2.  **Scrape:** Uses `JobSpy` to find jobs based on your search criteria and location across multiple job boards (LinkedIn, Indeed, etc.).
3.  **Analyze (Async):** Parses your resume (DOCX/PDF) using `PyPDF2`/`python-docx`. It then uses the configured LLM backend concurrently via `asyncio` to:
    *   Extract structured data from your resume (results are cached for speed).
    *   Compare your structured resume profile against each scraped job description.
    *   Generate a suitability score (0-100%), detailed justification, and skill/experience match assessment based on enhanced comparison logic.
4.  **Filter & Rank:** Filters the analyzed jobs based on a wide range of criteria including salary, job type, work model, company names (include/exclude), title keywords, date posted, location (remote country or proximity), and minimum suitability score. Ranks the final list by suitability score.
5.  **Output:** Saves the detailed analysis results (including original job data) to a JSON file and prints a summary table of top matches to the console using `rich`.

## Features

*   **Multi-Site Scraping:** Leverages `JobSpy` to scrape from sites like LinkedIn, Indeed, ZipRecruiter, Glassdoor, etc. (Check `JobSpy` docs for current support).
*   **Asynchronous Analysis:** Significantly speeds up analysis by processing multiple jobs concurrently with the selected LLM via its API using `asyncio` and the appropriate client library (`ollama`, `openai`, `google-generativeai`).
*   **Resume Parsing:** Handles `.docx` and `.pdf` resume files.
*   **Flexible GenAI Analysis:** Supports multiple LLM backends:
    *   **Ollama:** Run models locally for privacy and control.
    *   **OpenAI-compatible APIs (e.g., LM Studio):** Use local servers like LM Studio or other compatible endpoints.
    *   **Google Gemini:** Leverage Google's cloud-based models via API.
    *   Configurable provider selection, model names, API endpoints/keys, timeouts, and retries via `config.yaml`.
    *   Structured resume data extraction with emphasis on quantifiable achievements.
    *   Detailed job suitability scoring based on recruiter-like evaluation criteria (essentials, relevance, impact).
    *   Evidence-based justification for scores.
*   **Advanced Filtering:**
    *   Salary range (min/max).
    *   Job Type(s) (Full-time, Contract, etc.).
    *   Work Model(s) (Remote, Hybrid, On-site).
    *   Company Name inclusion or exclusion lists.
    *   Job Title keyword matching (any keyword).
    *   Date Posted range (after/before YYYY-MM-DD).
    *   Minimum Suitability Score (0-100).
*   **JobSpy Native Filter Configuration:** Exposes and allows configuration of several native JobSpy scraping filters via `config.yaml` and command-line arguments, including:
    *   Google search term
    *   Distance from location
    *   Is remote flag
    *   Job type(s)
    *   Easy apply flag
    *   CA certificate path
    *   LinkedIn company IDs
    *   Enforce annual salary conversion
    Command-line arguments override `config.yaml` settings for these filters.
*   **Caching:**
    *   **Resume Analysis:** Caches structured resume data based on file hash to speed up subsequent runs with the same resume (`output/.resume_cache/`). Use `--force-resume-reparse` to override.
*   **Robustness:**
    *   More specific error handling for scraping, analysis, file I/O, and network issues.
    *   Retry logic for LLM API calls tailored to each provider.
    *   Graceful handling of `Ctrl+C` interruptions.
    *   Warning for long LLM prompts (truncation may be needed depending on model/context window).
    *   Handles jobs with missing descriptions scraped by `JobSpy` by skipping analysis for them.
*   **Configuration:** Centralized settings via `config.yaml` with environment variable overrides for key parameters.
*   **Rich Output:** Provides detailed JSON output and a configurable summary table in the console.


## Prerequisites

*   **Python 3.9+** (Recommended for enhanced `asyncio` support and type hinting)
*   **Git**
*   **LLM Backend (Choose one or more):**
    *   **Ollama:** Install and run Ollama locally. ([https://ollama.com/](https://ollama.com/)). Ensure the server is running and desired models are pulled (e.g., `ollama pull llama3:instruct`).
    *   **LM Studio (or other OpenAI-compatible server):** Install and run LM Studio locally ([https://lmstudio.ai/](https://lmstudio.ai/)). Download models via its interface and start the local API server (usually `http://localhost:1234/v1`).
    *   **Google Gemini API Key:** Obtain an API key from Google AI Studio ([https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)).
*   **Python LLM Libraries:** Ensure necessary libraries are installed (see Setup).
*   **Playwright Browsers:** `JobSpy` uses Playwright for some scrapers. Install required browsers (might take time/disk space):
    ```bash
    playwright install
    ```

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/kasnycdev/MyJobSpy.git
    cd MyJobSpy
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # Windows: .\venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(This installs `openai`, `ollama`, `google-generativeai`, and other core libraries listed in `requirements.txt`)*

4.  **Install Playwright browsers:**
    ```bash
    playwright install
    ```

5.  **Configure `config.yaml`:**
    *   Review and edit `config.yaml` in the project root.
    *   **Set `llm_provider`:** Choose `"openai"`, `"ollama"`, or `"gemini"`.
    *   **Configure the corresponding section (`openai`, `ollama`, or `gemini`):**
        *   **`openai`:** Set `base_url` (e.g., for LM Studio), `model` identifier, and `api_key` (if needed).
        *   **`ollama`:** Set `base_url` and `model` name (must be pulled in Ollama).
        *   **`gemini`:** Set `model` name and provide your `api_key` (or set `GOOGLE_API_KEY` environment variable).
    *   Adjust other settings (scraping, analysis, logging) as needed.

6.  **Configure LinkedIn Cookies (Optional but Recommended):**
    *   Create a `config.toml` file in the project root (this file is ignored by git).
    *   Find your `li_at` and `JSESSIONID` cookies from your browser after logging into LinkedIn (use Developer Tools -> Application/Storage -> Cookies).
    *   Add them to `config.toml`:
        ```toml
        # config.toml
        [linkedin]
        li_at = "YOUR_LI_AT_COOKIE_VALUE"
        JSESSIONID = "YOUR_JSESSIONID_COOKIE_VALUE_WITH_QUOTES" # Include quotes if present
        ```
    *   `JobSpy` will automatically detect and use this file for authenticated LinkedIn scraping.

## Usage

Run the main pipeline script from the project's root directory:

```bash
python run_pipeline.py --search "Your Job Search" --resume /path/to/your/resume.pdf [LOCATION_FLAG] [OPTIONS]
