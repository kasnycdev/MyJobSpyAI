# myjobspyai.llm.base

## Overview

The base module provides the foundational classes for LLM providers, defining common interfaces and functionality for all LLM integrations.

## Classes

### BaseProvider

Base class for LLM providers.

#### Initialization

```python
provider = BaseProvider(
    config={
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 2000,
        "api_key": "your-api-key"
    },
    name="openai_provider"
)
```

#### Methods

##### \_validate_model

```python
# Validate the LLM model configuration
is_valid = provider._validate_model()
```

Validates the current model configuration against provider requirements.

##### generate

```python
# Generate text using the LLM
result = await provider.generate(
    prompt="Analyze this text",
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=2000
)
```

Generates text using the configured LLM. Supports model-specific parameters.

##### close

```python
# Clean up resources
await provider.close()
```

Closes the provider's resources, including any active connections or sessions.

### SyncProvider

Synchronous wrapper for the provider.

#### Methods

##### generate_sync

```python
# Generate text synchronously
result = provider.generate_sync(
    prompt="Analyze this text",
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=2000
)
```

Synchronous version of the generate method for non-async environments.

##### close_sync

```python
# Clean up resources synchronously
provider.close_sync()
```

Synchronously closes the provider's resources with additional cleanup steps.

## Configuration

The base provider supports the following configuration options:

```python
{
    "model": "gpt-3.5-turbo",  # Model name
    "temperature": 0.7,        # Temperature for randomness
    "max_tokens": 2000,        # Maximum tokens in response
    "api_key": "your-api-key", # API key (optional)
    "provider_type": "base",    # Provider type identifier
    "name": "base_provider"    # Provider instance name
}
```

## Error Handling

The base provider implements robust error handling:

- Invalid configuration
- Model initialization failures
- Network errors
- Rate limiting
- Model response validation

All errors are wrapped in `ProviderError` with detailed context and status codes.

## OpenTelemetry Integration

The base provider includes OpenTelemetry instrumentation for:

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
from myjobspyai.llm.base import BaseProvider

# Initialize provider
provider = BaseProvider(
    config={
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 2000
    },
    name="openai_provider"
)

try:
    # Generate text
    result = await provider.generate(
        prompt="Analyze this text",
        temperature=0.7,
        max_tokens=2000
    )
    print(f"Generation result: {result}")
finally:
    # Clean up
    await provider.close()
```

## Best Practices

1. Always validate configuration
2. Use appropriate model parameters
3. Handle errors gracefully
4. Clean up resources properly
5. Monitor performance
6. Use OpenTelemetry for monitoring
7. Follow security best practices
8. Implement proper error handling
9. Use configuration validation
10. Monitor resource usage
