# myjobspyai.analysis.providers.langchain_provider

## Overview

The LangChain provider module implements integration with LangChain's LLM ecosystem, providing support for various LLM models and OpenTelemetry instrumentation for monitoring and tracing.

## Classes

### LangChainProvider

Provider for LangChain LLM integration with OTEL support.

#### Initialization

```python
provider = LangChainProvider(
    config={
        "class_name": "ChatOpenAI",
        "model_config": {
            "model_name": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 2000
        }
    }
)
```

#### Methods

##### \_initialize_langchain

```python
# Initialize the LangChain client based on configuration
provider._initialize_langchain()
```

Initializes the LangChain client with the specified model class and configuration. Handles automatic fallback to langchain_community if direct import fails.

##### \_initialize_provider

```python
# Initialize the LangChain provider
await provider._initialize_provider()
```

Initializes the provider with the given configuration. Sets up OpenTelemetry tracing and error handling.

##### generate_sync

```python
# Generate text synchronously
result = provider.generate_sync(
    prompt="Analyze this text",
    model="gpt-3.5-turbo",
    output_schema={
        "type": "object",
        "properties": {
            "analysis": {"type": "string"}
        }
    }
)
```

Generates text synchronously using the configured LLM. Supports structured output via JSON schema.

##### close_sync

```python
# Clean up resources
provider.close_sync()
```

Closes the provider's resources, including any active connections or sessions.

### SyncLangChainProvider

Synchronous wrapper for the LangChain provider.

#### Methods

##### generate_sync

```python
# Generate text synchronously using the wrapper
result = provider.generate_sync(
    prompt="Analyze this text",
    model="gpt-3.5-turbo",
    output_schema={
        "type": "object",
        "properties": {
            "analysis": {"type": "string"}
        }
    }
)
```

Same functionality as LangChainProvider.generate_sync but with additional error handling and resource management.

##### close_sync

```python
# Clean up resources using the wrapper
provider.close_sync()
```

Closes the provider's resources with additional cleanup steps.

## Configuration

The LangChain provider supports the following configuration options:

```python
{
    "class_name": "ChatOpenAI",  # LLM class name (e.g., "ChatOpenAI", "ChatAnthropic")
    "model_config": {
        "model_name": "gpt-3.5-turbo",  # Model name
        "temperature": 0.7,            # Temperature for randomness
        "max_tokens": 2000,            # Maximum tokens in response
        "api_key": "your-api-key"      # API key (optional)
    },
    "provider_type": "langchain",     # Provider type identifier
    "name": "langchain_provider"      # Provider instance name
}
```

## Error Handling

The provider implements robust error handling:

- Invalid configuration
- Model initialization failures
- Network errors
- Rate limiting
- Model response validation

All errors are wrapped in `ProviderError` with detailed context and status codes.

## OpenTelemetry Integration

The provider includes OpenTelemetry instrumentation for:

- Request tracing
- Error tracking
- Performance metrics
- Resource monitoring

Traces are tagged with:
- Provider name
- Model name
- Request type
- Response status
- Duration metrics

## Usage Example

```python
from myjobspyai.analysis.providers.langchain_provider import LangChainProvider

# Initialize provider
provider = LangChainProvider(
    config={
        "class_name": "ChatOpenAI",
        "model_config": {
            "model_name": "gpt-3.5-turbo",
            "api_key": "your-api-key"
        }
    }
)

try:
    # Generate text
    result = await provider.generate_sync(
        prompt="Analyze this job description",
        output_schema={
            "type": "object",
            "properties": {
                "skills": {"type": "array", "items": {"type": "string"}},
                "experience": {"type": "integer"},
                "responsibilities": {"type": "array", "items": {"type": "string"}}
            }
        }
    )
    print(f"Analysis result: {result}")
finally:
    # Clean up
    await provider.close_sync()
```

## Best Practices

1. Always use configuration validation
2. Implement proper error handling
3. Use structured output schemas
4. Monitor performance metrics
5. Clean up resources properly
6. Use appropriate model configuration
7. Handle rate limiting gracefully
8. Validate model responses
9. Use OpenTelemetry for monitoring
10. Follow security best practices
