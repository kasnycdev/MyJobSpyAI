# Migration Guide: Upgrading to the New LLM Provider System

This guide will help you migrate your code to use the new LLM provider system with LangChain integration.

## Key Changes

1. **New Configuration Structure**:
   - LLM providers are now configured under the `llm` section in `config.yaml`
   - Each provider has a unique name and type
   - Support for multiple LangChain backends with custom configurations

2. **Updated Analyzer Class**:
   - New `BaseAnalyzer` class with support for multiple providers
   - Simplified provider initialization and management
   - Built-in OpenTelemetry instrumentation

3. **New LangChain Provider**:
   - Support for any LangChain-compatible LLM backend
   - Configurable via YAML with support for custom parameters

## Migration Steps

### 1. Update Your Configuration

Update your `config.yaml` to use the new provider structure:

```yaml
llm:
  # Default provider to use
  default_provider: "langchain_default"

  # Provider configurations
  providers:
    # OpenAI/LM Studio Configuration
    openai:
      type: "openai"
      base_url: "http://10.10.0.178:1337/v1"
      model: "hermes3:3b"
      api_key: "lm-studio"
      request_timeout: 600
      max_retries: 2
      retry_delay: 5

    # Ollama Configuration
    ollama:
      type: "ollama"
      base_url: "http://10.10.0.178:11434"
      model: "hermes3:3b"
      request_timeout: 600
      max_retries: 2
      retry_delay: 5

    # LangChain Provider Configuration
    langchain_default:
      type: "langchain"
      class: "langchain_community.chat_models.ChatOpenAI"
      params:
        model_name: "gpt-3.5-turbo"
        temperature: 0.7
        max_tokens: 1000
        openai_api_key: "your-openai-api-key"

    # Example of another LangChain provider with a different model
    langchain_ollama:
      type: "langchain"
      class: "langchain_community.chat_models.ChatOllama"
      params:
        base_url: "http://10.10.0.178:11434"
        model: "llama2"
        temperature: 0.7
        num_ctx: 4096
```

### 2. Update Your Code

#### Old Way:

```python
from analysis.analyzer import BaseAnalyzer, ResumeAnalyzer, JobAnalyzer

# Initialize with a provider config key
analyzer = ResumeAnalyzer()  # Uses default provider from settings
```

#### New Way:

```python
from analysis.analyzer_new import BaseAnalyzer, ResumeAnalyzer, JobAnalyzer

# Initialize with an optional provider name (uses default if not specified)
analyzer = ResumeAnalyzer(provider_name="langchain_default")

# Or use a different provider
analyzer = ResumeAnalyzer(provider_name="openai")
```

### 3. Update Dependencies

Make sure to install the required dependencies:

```bash
pip install langchain langchain-community langchain-openai
```

### 4. Testing

After migration, test your application with different providers to ensure everything works as expected:

```python
# Test with different providers
for provider in ["langchain_default", "openai", "ollama"]:
    try:
        print(f"\nTesting provider: {provider}")
        analyzer = ResumeAnalyzer(provider_name=provider)
        result = asyncio.run(analyzer.extract_resume_data_async("Test resume content"))
        print(f"Success with {provider}: {result}")
    except Exception as e:
        print(f"Error with {provider}: {str(e)}")
```

## New Features

### Multiple Provider Support

Easily switch between different LLM providers by changing the provider name:

```python
# Use different providers
openai_analyzer = ResumeAnalyzer(provider_name="openai")
ollama_analyzer = ResumeAnalyzer(provider_name="ollama")
langchain_analyzer = ResumeAnalyzer(provider_name="langchain_default")
```

### Custom LangChain Backends

Add custom LangChain backends in your configuration:

```yaml
llm:
  providers:
    my_custom_llm:
      type: "langchain"
      class: "my_module.CustomLLM"
      params:
        custom_param: "value"
        temperature: 0.8
```

### Advanced Configuration

Each provider can have its own configuration parameters:

```yaml
llm:
  providers:
    langchain_advanced:
      type: "langchain"
      class: "langchain_community.chat_models.ChatOpenAI"
      params:
        model_name: "gpt-4"
        temperature: 0.3
        max_tokens: 2000
        request_timeout: 120
        max_retries: 3
        streaming: true
```

## Troubleshooting

### Common Issues

1. **Provider Not Found**:
   - Make sure the provider name in your code matches exactly with the name in the config
   - Check that the provider type is one of the supported types (openai, ollama, langchain, etc.)

2. **Missing Dependencies**:
   - LangChain providers require additional packages
   - Install them with: `pip install langchain-community langchain-openai`

3. **Connection Issues**:
   - Verify the base URLs and API keys in your configuration
   - Check that the LLM services are running and accessible

## Next Steps

- Explore the new provider system with different LLM backends
- Add custom metrics and tracing for your specific use case
- Contribute new provider implementations to the codebase
