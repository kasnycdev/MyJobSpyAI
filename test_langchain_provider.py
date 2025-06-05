"""Test script for LangChain provider.

This script demonstrates how to use the LangChain provider with different configurations.

Note: To run this test, you'll need to set up the required API keys:
1. For OpenAI: Set OPENAI_API_KEY environment variable

Example:
    export OPENAI_API_KEY='your-api-key-here'
"""

import asyncio
import os
import logging
from myjobspyai.analysis.providers import LangChainProvider, SyncLangChainProvider, ProviderError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration for the LangChain provider
CONFIG = {
    "class_name": "ChatOpenAI",
    "model_config": {
        "model_name": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 100
    },
    "system_message": "You are a helpful assistant."
}

async def test_async_provider():
    """Test the async LangChain provider."""
    try:
        # Check for required API keys
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY environment variable not set. Some tests may fail.")
            logger.info("To set the API key, run: export OPENAI_API_KEY='your-api-key-here'")
        
        logger.info("Initializing LangChain provider...")
        provider = LangChainProvider(
            config=CONFIG,
            name="test-langchain"
        )
        
        try:
            # Test basic generation
            logger.info("Testing basic text generation...")
            response = await provider.generate("Hello, how are you?")
            print(f"\nAsync Response: {response}")
            
            # Test with JSON output
            logger.info("\nTesting JSON output generation...")
            json_schema = {
                "type": "object",
                "properties": {
                    "greeting": {"type": "string"},
                    "feeling": {"type": "string"}
                },
                "required": ["greeting", "feeling"]
            }
            
            json_response = await provider.generate(
                "Tell me how you're feeling today in JSON format.",
                output_schema=json_schema
            )
            print(f"\nJSON Response: {json_response}")
            
        finally:
            await provider.close()
            logger.info("Provider closed successfully.")
            
    except ProviderError as e:
        logger.error(f"Provider error: {str(e)}")
        if "API key" in str(e):
            logger.error("Please make sure you have set the required API key.")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

def test_sync_provider():
    """Test the sync LangChain provider."""
    try:
        # Check for required API keys
        if not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY environment variable not set. Some tests may fail.")
            logger.info("To set the API key, run: export OPENAI_API_KEY='your-api-key-here'")
        
        logger.info("Initializing SyncLangChain provider...")
        provider = SyncLangChainProvider(
            config=CONFIG,
            name="test-sync-langchain"
        )
        
        try:
            # Test basic generation
            logger.info("Testing basic text generation (sync)...")
            response = provider.generate_sync("Hello, how are you?")
            print(f"\nSync Response: {response}")
            
        finally:
            provider.close_sync()
            logger.info("Sync provider closed successfully.")
            
    except ProviderError as e:
        logger.error(f"Provider error: {str(e)}")
        if "API key" in str(e):
            logger.error("Please make sure you have set the required API key.")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        # Run async tests
        print("\n" + "="*50)
        print("Running async tests...")
        print("="*50)
        asyncio.run(test_async_provider())
        
        # Run sync tests
        print("\n" + "="*50)
        print("Running sync tests...")
        print("="*50)
        test_sync_provider()
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        import sys
        sys.exit(1)
