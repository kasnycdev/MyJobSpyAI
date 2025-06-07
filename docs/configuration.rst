Configuration
=============

MyJobSpyAI can be configured using a YAML configuration file. This file allows you to customize various aspects of the application, including LLM providers, logging, and API settings.

Configuration File Locations
---------------------------
MyJobSpyAI looks for configuration files in the following locations (in order of priority):

1. ``~/.config/myjobspyai/config.yaml`` (recommended)
2. ``~/.myjobspyai/config.yaml``
3. ``./config.yaml`` (in the current working directory)

Basic Configuration Structure
---------------------------
A basic configuration file looks like this:

.. code-block:: yaml

   # Global settings
   log_level: "INFO"
   log_file: "myjobspyai.log"

   # LLM Configuration
   llm:
     default_provider: "ollama"  # Default provider to use
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
         api_key: ${OPENAI_API_KEY}  # Read from environment variable
         temperature: 0.7
         max_tokens: 1000

LLM Providers
------------
MyJobSpyAI supports multiple LLM providers. Here's how to configure them:

### Ollama

.. code-block:: yaml

   ollama:
     enabled: true
     model: "llama3:instruct"  # Model to use
     base_url: "http://localhost:11434"  # Ollama server URL
     temperature: 0.7  # 0.0 to 1.0
     num_predict: 1000  # Maximum number of tokens to generate

### OpenAI

.. code-block:: yaml

   openai:
     enabled: false
     model: "gpt-4"  # or "gpt-3.5-turbo"
     api_key: ${OPENAI_API_KEY}  # Read from environment variable
     temperature: 0.7
     max_tokens: 1000

### Anthropic

.. code-block:: yaml

   anthropic:
     enabled: false
     model: "claude-3-opus-20240229"
     api_key: ${ANTHROPIC_API_KEY}  # Read from environment variable
     temperature: 0.7
     max_tokens: 1000

Environment Variables
-------------------
Sensitive information like API keys can be loaded from environment variables using the ``${VARIABLE_NAME}`` syntax in the config file.

Example:

.. code-block:: yaml

   openai:
     api_key: ${OPENAI_API_KEY}

Logging Configuration
-------------------
You can configure logging with these options:

.. code-block:: yaml

   log_level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
   log_file: "myjobspyai.log"  # Path to log file (optional)
   log_format: "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}"  # Custom log format

Example Configuration
-------------------

.. code-block:: yaml

   # LLM Configuration
   llm:
     default_provider: "ollama"
     providers:
       ollama:
         name: "ollama"
         enabled: true
         type: "ollama"
         model: "gemma3:latest"
         temperature: 0.7
         num_predict: 1000
         timeout: 120
         max_retries: 3
         base_url: "http://localhost:11434"

   # Job Search Configuration
   jobspy:
     search_term: ""
     location: ""
     site_name:
       - linkedin
       - glassdoor
     is_remote: true
     results_wanted: 5

   # Output Configuration
   output:
     output_dir: "output"
     scraped_jobs_filename: "scraped_jobs.json"
     analysis_filename: "analyzed_jobs.json"

Environment Variables
-------------------
You can also configure settings using environment variables:

- ``MYJOBSPYAI_LLM_DEFAULT_PROVIDER``
- ``MYJOBSPYAI_LLM_PROVIDERS_OLLAMA_MODEL``
- ``MYJOBSPYAI_JOBSPY_SEARCH_TERM``
- And more...
