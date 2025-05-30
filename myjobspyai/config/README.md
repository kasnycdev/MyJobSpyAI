# Configuration System

This module provides a centralized configuration system for MyJobSpyAI using Pydantic settings management.

## Features

- Type-safe configuration with validation
- Support for multiple configuration sources:
  - Environment variables
  - .env files
  - YAML/JSON configuration files
  - Direct Python configuration
- Nested configuration structure
- Automatic environment variable parsing
- Configuration reloading

## Usage

### Basic Usage

```python
from myjobspyai.config import settings

# Access configuration
print(f"Using {settings.llm.provider} with model {settings.llm.model}")
```

### Loading Configuration from File

```python
from myjobspyai.config import configure

# Load from YAML file
configure("config.yaml")

# Or with overrides
configure("config.yaml", debug=True, max_concurrent_tasks=20)
```

### Environment Variables

Configuration can be overridden using environment variables with the `MYJOBS_` prefix:

```bash
# Set LLM provider and model
export MYJOBS_LLM__PROVIDER=openai
export MYJOBS_LLM__MODEL=gpt-4

# Set cache settings
export MYJOBS_CACHE__ENABLED=true
export MYJOBS_CACHE__TTL=3600
```

### Configuration Precedence

1. Direct Python overrides (passed to `configure()`)
2. Environment variables
3. Configuration file settings
4. Default values

## Configuration Reference

### Core Settings

- `debug` (bool): Enable debug mode
- `environment` (str): Runtime environment (development, staging, production)
- `max_concurrent_tasks` (int): Maximum number of concurrent tasks
- `request_timeout` (float): Default request timeout in seconds

### LLM Configuration

- `llm.provider` (str): LLM provider (openai, ollama, gemini)
- `llm.model` (str): Model name to use
- `llm.api_key` (str, optional): API key for the provider
- `llm.base_url` (str, optional): Base URL for self-hosted instances
- `llm.timeout` (float): Request timeout in seconds
- `llm.max_retries` (int): Maximum number of retry attempts
- `llm.temperature` (float): Sampling temperature (0.0 to 2.0)

### Cache Configuration

- `cache.enabled` (bool): Whether caching is enabled
- `cache.directory` (str): Base directory for cache files
- `cache.ttl` (int): Default cache TTL in seconds

### Logging Configuration

- `logging.level` (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `logging.file` (str, optional): Path to log file (if None, logs to console only)
- `logging.format` (str): Log message format

## Best Practices

1. **Sensitive Information**:
   - Store API keys and other secrets in environment variables or a `.env` file
   - Never commit sensitive information to version control

2. **Environment-Specific Configuration**:
   - Use the `environment` setting to enable environment-specific behavior
   - Consider maintaining separate configuration files for different environments

3. **Validation**:
   - The configuration system validates all values at runtime
   - Invalid configuration will raise descriptive errors

## Example Configuration Files

See `config.example.yaml` for a complete example configuration file.
