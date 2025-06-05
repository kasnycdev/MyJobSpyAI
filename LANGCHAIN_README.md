# LangChain Provider Integration

## Overview

The LangChain provider offers a unified interface to interact with multiple LLM providers through the LangChain library. This integration simplifies the process of switching between different LLM backends while maintaining a consistent API.

## Features

- **Unified Interface**: Single interface for multiple LLM providers
- **Chat Model Support**: Native support for chat-based models with system messages and conversation history
- **Streaming**: Real-time token streaming for better user experience
- **Flexible Configuration**: Easy configuration through YAML or code
- **Standardized Responses**: Consistent response format across all providers
- **Error Handling**: Comprehensive error handling and retry logic

## Supported Providers

The following LLM providers are supported through the LangChain integration:

| Provider | Model Example | Required Environment Variable |
|----------|---------------|-------------------------------|
| OpenAI | gpt-4-turbo-preview | `OPENAI_API_KEY` |
| Anthropic | claude-3-opus-20240229 | `ANTHROPIC_API_KEY` |
| Google | gemini-pro | `GOOGLE_API_KEY` |
| Ollama | llama2 | `OLLAMA_BASE_URL` (optional) |
| OpenRouter | openai/gpt-4-turbo | `OPENROUTER_API_KEY` |
| Together AI | togethercomputer/llama-2-70b-chat | `TOGETHER_API_KEY` |
| Fireworks | fireworks-llama-v2-70b-chat | `FIREWORKS_API_KEY` |
| DeepInfra | meta-llama/Llama-2-70b-chat-hf | `DEEPINFRA_API_TOKEN` |
| Perplexity | pplx-70b-chat | `PERPLEXITY_API_KEY` |
| Vertex AI | gemini-pro | `GOOGLE_APPLICATION_CREDENTIALS` |
| Cohere | command-nightly | `COHERE_API_KEY` |
| HuggingFace | meta-llama/Llama-2-70b-chat-hf | `HUGGINGFACEHUB_API_TOKEN` |
| Jina | jina-embeddings-v2-base-en | `JINACHAT_API_KEY` |
| MLX | mlx-community/Qwen1.5-0.5B-Chat | - |
| OpenRouter | openai/gpt-4-turbo | `OPENROUTER_API_KEY` |
| Tongyi | qwen-turbo | `DASHSCOPE_API_KEY` |
| YandexGPT | yandexgpt | `YANDEX_API_KEY` |
| ZhipuAI | chatglm_pro | `ZHIPUAI_API_KEY` |

## Configuration

### YAML Configuration

Add the following to your `config.yaml`:

```yaml
# LLM Provider Configuration
llm_provider:
  type: "langchain"  # Using LangChain as the unified provider
  config:
    # General configuration
    provider: "openai"  # The underlying provider to use
    model: "gpt-4-turbo-preview"  # Model to use
    api_key: ${OPENAI_API_KEY}  # Read from environment variable
    
    # Generation parameters
    temperature: 0.7
    max_tokens: 1000
    top_p: 1.0
    frequency_penalty: 0.0
    presence_penalty: 0.0
    stop: null
    
    # Connection settings
    streaming: true
    timeout: 60
    max_retries: 3
    
    # Provider-specific configuration
    provider_config:
      # OpenAI-specific settings
      openai:
        organization: ${OPENAI_ORG_ID}  # Optional
      
      # Anthropic-specific settings
      anthropic:
        max_tokens_to_sample: 1000
        
      # Google-specific settings
      google:
        model_name: "gemini-pro"
        
      # Ollama settings
      ollama:
        base_url: "http://localhost:11434"
```

### Environment Variables

Set the appropriate environment variables for your chosen provider:

```bash
# For OpenAI
export OPENAI_API_KEY='your-api-key-here'

# For Anthropic
export ANTHROPIC_API_KEY='your-api-key-here'

# For Google
export GOOGLE_API_KEY='your-api-key-here'
```

## Usage

### Basic Usage

```python
from myjobspyai.llm.providers import LangChainProvider

# Initialize the provider
provider = LangChainProvider({
    "provider": "openai",
    "model": "gpt-4-turbo-preview",
    "api_key": "your-api-key-here",
    "temperature": 0.7,
    "max_tokens": 1000
})

# Generate a response
response = await provider.generate(
    "Hello, how are you?",
    system_message="You are a helpful assistant."
)
print(response.text)

# Don't forget to close the provider
await provider.close()
```

### With Conversation History

```python
response = await provider.generate(
    "What's the weather like today?",
    system_message="You are a helpful assistant that knows about weather.",
    messages=[
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you today?"},
        {"role": "user", "content": "I'd like to know about the weather."}
    ]
)
print(response.text)
```

### Streaming Response

```python
async for chunk in provider.generate_stream(
    "Tell me a short story about AI in the future.",
    system_message="You are a creative storyteller.",
    max_tokens=200
):
    print(chunk.text, end="", flush=True)
print()  # New line after streaming
```

## Testing

Run the test script to verify the integration:

```bash
# Set your API key
export OPENAI_API_KEY='your-api-key-here'

# Run the test script
python test_langchain_chat.py
```

The test script will run through a series of tests with the configured providers and display the results.

## Error Handling

The provider includes comprehensive error handling for common issues:

- Missing API keys
- Invalid provider configurations
- Network timeouts
- Rate limiting
- Invalid responses

Errors are raised as `LLMRequestError` with descriptive messages to help with debugging.

## Extending with New Providers

To add support for additional LangChain chat models:

1. Ensure the required LangChain integration package is installed (e.g., `langchain-openai`, `langchain-anthropic`)
2. Add the provider initialization logic in the `_initialize_model()` method of `LangChainProvider`
3. Update the provider configuration schema in the README

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
