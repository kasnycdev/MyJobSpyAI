# LLM Providers with OTEL Integration

This directory contains implementations of various LLM providers with built-in OpenTelemetry (OTEL) support for observability and monitoring.

## Available Providers

1. **LangChain Provider** (`langchain_provider.py`)
   - Supports any LangChain-compatible LLM backend
   - Built-in OpenTelemetry tracing and metrics
   - Configurable via `config.yaml`

## Configuration

Add your provider configuration to `config.yaml` under the `llm.providers` section. Example:

```yaml
llm:
  default_provider: "langchain_default"
  providers:
    langchain_default:
      type: "langchain"
      class: "langchain_community.chat_models.ChatOpenAI"
      params:
        model_name: "gpt-3.5-turbo"
        temperature: 0.7
        max_tokens: 1000
        openai_api_key: "your-openai-api-key"
```

## Adding a New Provider

1. Create a new provider class in a new file (e.g., `my_provider.py`)
2. Inherit from `BaseProvider`
3. Implement the required methods
4. Register the provider in the `ProviderFactory`

## OTEL Integration

All providers automatically integrate with OpenTelemetry for:
- Distributed tracing
- Metrics collection
- Error tracking

Metrics are exposed via Prometheus and can be visualized in tools like Grafana.

## Usage

```python
from analysis.providers.factory import ProviderFactory

# Get a provider instance
provider = ProviderFactory.create_provider(
    provider_type="langchain",
    config=config["llm"]["providers"]["langchain_default"]
)

# Generate text
response = await provider.generate("Hello, world!")
```
