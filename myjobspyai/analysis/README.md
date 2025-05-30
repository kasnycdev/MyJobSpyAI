# Analysis Module

This module provides LLM-based analysis capabilities for job search applications, including resume parsing, job description analysis, and suitability matching.

## Features

- Support for multiple LLM providers (OpenAI, Ollama, Gemini)
- Retry logic with exponential backoff
- Metrics collection and monitoring
- Async/await support
- Type hints throughout

## Installation

```bash
# Install with all providers
pip install -e .[all]

# Or install specific providers
pip install -e .[openai]
pip install -e .[ollama]
pip install -e .[gemini]
```

## Usage

### Basic Usage

```python
from analysis import BaseAnalyzer, get_factory

# Get the default factory
factory = get_factory()

# Create an analyzer with the default provider
analyzer = BaseAnalyzer()

# Or specify a provider
analyzer = BaseAnalyzer(provider="ollama")

# Use the analyzer
result = await analyzer.generate("Hello, world!")
print(result)
```

## 🔧 Configuration

The analysis module uses a hierarchical configuration system powered by Pydantic v2. Configuration can be provided through multiple sources with the following precedence:

1. Environment variables
2. `config.yaml` file
3. Default values in code

### Environment Variables

You can configure the analysis module using environment variables:

```bash
# Required API Keys
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_anthropic_key
export GOOGLE_API_KEY=your_google_key

# Optional Configuration
export DEBUG=true
export LOG_LEVEL=INFO

# Ollama Configuration
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama2
```

### Configuration File

Create a `config.yaml` file in your project root with the following structure:

```yaml
# Application settings
app:
  name: "MyJobSpyAI"
  debug: false
  log_level: "INFO"

# LLM Providers configuration
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    model: "gpt-4"
    temperature: 0.7
    max_tokens: 2048
    
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
    model: "claude-2"
    
  google:
    api_key: ${GOOGLE_API_KEY}
    model: "gemini-pro"

# Analysis settings
analysis:
  max_retries: 3
  timeout: 30
  cache_enabled: true
```

### Programmatic Configuration

You can also configure the analysis module programmatically:

```python
from myjobspyai.analysis import ProviderFactory
from myjobspyai.utils.config_utils import settings

# Access current configuration
print(f"Using model: {settings.providers.openai.model}")

# Create a provider with custom configuration
custom_config = {
    "providers": {
        "openai": {
            "api_key": "your-key",
            "model": "gpt-4-turbo",
            "temperature": 0.5
        }
    }
}

factory = ProviderFactory(custom_config)
analyzer = factory.create_analyzer("openai")

# Or update settings globally
settings.providers.openai.temperature = 0.8
```

### Available Providers

#### OpenAI

```python
from analysis.providers import OpenAIClient

client = OpenAIClient({
    "api_key": "your-key",
    "model": "gpt-4",
})

response = await client.generate("Hello, world!")
```

#### Ollama

```python
from analysis.providers import OllamaClient

client = OllamaClient({
    "base_url": "http://localhost:11434",
    "model": "llama2",
})

response = await client.generate("Hello, world!")
```

#### Gemini

```python
from analysis.providers import GeminiClient

client = GeminiClient({
    "api_key": "your-key",
    "model": "gemini-pro",
})

response = await client.generate("Hello, world!")
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
black .
ruff check .
mypy .
```

## License

MIT
