CCCCCCCCCCC

# ===========

------------------------------------------------------------------------

------------------------------------------------------------------------

------------------------------------------------------------------------

------------------------------------------------------------------------

CCCCCCCCCCCCCCCCCCCCCCCCCCC

# ===========================

------------------------------------------------------------------------

------------------------------------------------------------------------

------------------------------------------------------------------------

------------------------------------------------------------------------

1.  `~/.config/myjobspyai/config.yaml` (recommended)
2.  `~/.myjobspyai/config.yaml`
3.  `./config.yaml` (in the current working directory)

## BBBBBBBBBBBBBBBBBBBBBBBBBBB

------------------------------------------------------------------------

------------------------------------------------------------------------

### LLM Providers

You can configure multiple LLM providers and switch between them. The
following providers are supported:

1.  **Ollama** (recommended for local development)
2.  **LangChain** (for production use with OpenAI, Anthropic, etc.)

Example configuration for multiple providers:

``` yaml
llm:
  default_provider: "ollama"
  providers:
    ollama:
      enabled: true
      model: "llama3:instruct"
      base_url: "http://localhost:11434"
    openai:
      enabled: false
      model: "gpt-4"
      api_key: ${OPENAI_API_KEY}  # Uses environment variable
      temperature: 0.7
      max_tokens: 1000
```

### Ollama

For local development, you can use Ollama with models like LLaMA 3.
Example configuration:

``` yaml
llm:
  providers:
    ollama:
      enabled: true
      model: "llama3:instruct"  # or any other model you have installed
      base_url: "http://localhost:11434"  # Default Ollama server URL
      temperature: 0.7
      num_predict: 1000  # Maximum number of tokens to generate
```

### OpenAI

``` yaml
openai:
  enabled: false  # Set to true to enable
  model: "gpt-4"  # or "gpt-3.5-turbo"
  api_key: ${OPENAI_API_KEY}  # Read from environment variable
  temperature: 0.7
  max_tokens: 1000
```

### Anthropic

``` yaml
anthropic:
  enabled: false
  model: "claude-3-opus-20240229"  # or "claude-3-sonnet-20240229"
  api_key: ${ANTHROPIC_API_KEY}  # Read from environment variable
  temperature: 0.7
  max_tokens: 1000
```

EEEEEEEEEEEEEEEEEEEE

# ====================

------------------------------------------------------------------------

------------------------------------------------------------------------

------------------------------------------------------------------------

------------------------------------------------------------------------

``` yaml
openai:
  api_key: ${OPENAI_API_KEY}  # Will be replaced with the value of the OPENAI_API_KEY environment variable
```

### Logging Configuration

You can configure logging in the configuration file:

``` yaml
# Logging configuration
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  file: "myjobspyai.log"  # Log file path (relative to config file or absolute)
  max_size: 10  # MB
  backup_count: 5  # Number of backup logs to keep
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

EEEEEEEEEEEEEEEEEEEE

# ====================

------------------------------------------------------------------------

------------------------------------------------------------------------

------------------------------------------------------------------------

------------------------------------------------------------------------

``` yaml
# Global settings
log_level: "INFO"
log_file: "myjobspyai.log"

# LLM Configuration
llm:
  default_provider: "ollama"
  providers:
    ollama:
      enabled: true
      model: "llama3:instruct"
      base_url: "http://localhost:11434"
      temperature: 0.7
      num_predict: 1000

    openai:
      enabled: false
      model: "gpt-4"
      api_key: ${OPENAI_API_KEY}
      temperature: 0.7
      max_tokens: 1000

# Logging configuration
logging:
  level: "INFO"
  file: "myjobspyai.log"
  max_size: 10
  backup_count: 5
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

EEEEEEEEEEEEEEEEEEEEEEEEEEEEEE

# ==============================

------------------------------------------------------------------------

------------------------------------------------------------------------

------------------------------------------------------------------------

------------------------------------------------------------------------

-   `MYJOBSPYAI_LOG_LEVEL`: Set the log level (DEBUG, INFO, WARNING,
    ERROR, CRITICAL)
-   `MYJOBSPYAI_LOG_FILE`: Set the log file path
-   `MYJOBSPYAI_LLM_DEFAULT_PROVIDER`: Set the default LLM provider
-   `MYJOBSPYAI_LLM_PROVIDERS_OLLAMA_ENABLED`: Enable/disable Ollama
    provider
-   `MYJOBSPYAI_LLM_PROVIDERS_OLLAMA_MODEL`: Set the Ollama model
-   `MYJOBSPYAI_LLM_PROVIDERS_OLLAMA_BASE_URL`: Set the Ollama server
    URL
-   `MYJOBSPYAI_LLM_PROVIDERS_OPENAI_ENABLED`: Enable/disable OpenAI
    provider
-   `MYJOBSPYAI_LLM_PROVIDERS_OPENAI_MODEL`: Set the OpenAI model
-   `OPENAI_API_KEY`: Set the OpenAI API key

### Job Search Configuration

You can configure job search settings:

``` yaml
# Job Search Configuration
jobspy:
  search_term: ""
  location: ""
  site_name:
    - linkedin
    - glassdoor
  is_remote: true
  results_wanted: 5
```

### Output Configuration

You can configure output settings:

``` yaml
# Output Configuration
output:
  output_dir: "output"
  scraped_jobs_filename: "scraped_jobs.json"
  analysis_filename: "analyzed_jobs.json"
```
