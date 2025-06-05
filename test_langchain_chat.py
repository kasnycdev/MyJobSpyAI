"""Test script for the LangChain chat provider.

This script demonstrates how to use the LangChain chat provider with different configurations.

Note: To run this test, you'll need to set up the required API keys:
1. For OpenAI: Set OPENAI_API_KEY environment variable
2. For other providers, set the appropriate API keys

Example:
    export OPENAI_API_KEY='your-api-key-here'
"""

import asyncio
import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the provider after setting up logging
from myjobspyai.llm.providers import LangChainProvider

# Configuration for different providers
PROVIDER_CONFIGS = {
    "openai": {
        "provider": "openai",
        "model": "gpt-4-turbo-preview",
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": 0.7,
        "max_tokens": 100,
        "provider_config": {
            "openai": {
                "organization": os.getenv("OPENAI_ORG_ID"),
            }
        }
    },
    "anthropic": {
        "provider": "anthropic",
        "model": "claude-3-opus-20240229",
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "temperature": 0.7,
        "max_tokens": 100,
        "provider_config": {
            "max_tokens_to_sample": 100
        }
    },
    "google": {
        "provider": "google",
        "model": "gemini-pro",
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "temperature": 0.7,
        "max_tokens": 100,
        "provider_config": {
            "model_name": "gemini-pro"
        }
    },
    "ollama": {
        "provider": "ollama",
        "model": "llama2",
        "base_url": "http://localhost:11434",
        "temperature": 0.7,
        "max_tokens": 100,
        "provider_config": {
            "base_url": "http://localhost:11434"
        }
    }
}

async def test_provider(config: Dict[str, Any], provider_name: str):
    """Test a provider with the given configuration.
    
    Args:
        config: Provider configuration
        provider_name: Name of the provider for logging
    """
    logger.info(f"\n{'='*50}")
    logger.info(f"Testing {provider_name.upper()} provider")
    logger.info(f"{'='*50}")
    
    try:
        # Initialize the provider
        logger.info(f"Initializing {provider_name} provider...")
        provider = LangChainProvider(config)
        
        try:
            # Test basic generation
            logger.info("\nTesting basic text generation...")
            response = await provider.generate("Hello, how are you?")
            print(f"\n{provider_name.upper()} Response:")
            print(response.text)
            
            # Test with system message and conversation history
            logger.info("\nTesting with system message and conversation history...")
            response = await provider.generate(
                "What's the weather like today?",
                system_message="You are a helpful assistant that knows about weather.",
                messages=[
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi there! How can I help you today?"},
                    {"role": "user", "content": "I'd like to know about the weather."}
                ]
            )
            print(f"\n{provider_name.upper()} Response with context:")
            print(response.text)
            
            # Test streaming
            logger.info("\nTesting streaming...")
            print(f"\n{provider_name.upper()} Streaming Response:")
            async for chunk in provider.generate_stream(
                "Tell me a short story about AI in the future.",
                system_message="You are a creative storyteller.",
                max_tokens=200
            ):
                print(chunk.text, end="", flush=True)
            print("\n")
            
        finally:
            await provider.close()
            logger.info(f"{provider_name} provider closed successfully.")
            
    except Exception as e:
        logger.error(f"Error testing {provider_name} provider: {str(e)}")
        if "API key" in str(e):
            logger.error(f"Please make sure you have set the required {provider_name.upper()}_API_KEY environment variable.")
        raise

async def main():
    """Run tests for all configured providers."""
    # Only test providers that have their API keys set
    providers_to_test = []
    
    for provider_name, config in PROVIDER_CONFIGS.items():
        # Skip providers that don't have their API key set (except for Ollama which might be local)
        if provider_name == "ollama" or config.get("api_key") or config.get("base_url"):
            providers_to_test.append((provider_name, config))
    
    if not providers_to_test:
        logger.warning("No providers configured with valid API keys. Please set the appropriate environment variables.")
        return
    
    # Test each provider
    for provider_name, config in providers_to_test:
        try:
            await test_provider(config, provider_name)
        except Exception as e:
            logger.error(f"Error testing {provider_name}: {str(e)}")
            continue

if __name__ == "__main__":
    asyncio.run(main())
