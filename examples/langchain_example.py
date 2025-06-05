"""
Example script demonstrating how to use the LangChain provider.

This script shows how to initialize the provider, generate text, and handle streaming responses.
"""

import asyncio
import os
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Import the provider
from myjobspyai.llm.providers import LangChainProvider

async def main():
    """Main async function to demonstrate provider usage."""
    # Configuration for the provider
    config = {
        "provider": "openai",  # or 'anthropic', 'google', 'ollama', etc.
        "model": "gpt-4-turbo-preview",  # or any other supported model
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": 0.7,
        "max_tokens": 500,
        "streaming": True,
        "provider_config": {
            "openai": {
                "organization": os.getenv("OPENAI_ORG_ID", ""),
            }
        }
    }

    # Initialize the provider
    print("Initializing LangChain provider...")
    provider = LangChainProvider(config)

    try:
        # Example 1: Simple text generation
        print("\n--- Example 1: Simple Text Generation ---")
        response = await provider.generate(
            "Tell me a short joke about artificial intelligence.",
            system_message="You are a helpful AI assistant with a great sense of humor."
        )
        print(f"Response: {response.text}")

        # Example 2: With conversation history
        print("\n--- Example 2: With Conversation History ---")
        response = await provider.generate(
            "What's the weather like today?",
            system_message="You are a helpful assistant that knows about weather.",
            messages=[
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there! How can I help you today?"},
                {"role": "user", "content": "I'd like to know about the weather."}
            ]
        )
        print(f"Response: {response.text}")

        # Example 3: Streaming response
        print("\n--- Example 3: Streaming Response ---")
        print("Response (streaming): ")
        full_response = ""
        async for chunk in provider.generate_stream(
            "Write a haiku about programming:",
            system_message="You are a creative poet who writes in haiku format (5-7-5 syllables).",
            max_tokens=100
        ):
            print(chunk.text, end="", flush=True)
            full_response += chunk.text
        print("\n")

        # Example 4: Using different provider (Anthropic)
        if os.getenv("ANTHROPIC_API_KEY"):
            print("\n--- Example 4: Using Anthropic ---")
            anthropic_config = {
                "provider": "anthropic",
                "model": "claude-3-opus-20240229",
                "api_key": os.getenv("ANTHROPIC_API_KEY"),
                "temperature": 0.7,
                "max_tokens": 500,
                "provider_config": {
                    "anthropic": {
                        "max_tokens_to_sample": 500
                    }
                }
            }
            
            anthropic_provider = LangChainProvider(anthropic_config)
            try:
                response = await anthropic_provider.generate(
                    "What are the key principles of effective prompt engineering?",
                    system_message="You are an AI expert specializing in prompt engineering."
                )
                print(f"Anthropic Response: {response.text[:200]}...")  # Print first 200 chars
            finally:
                await anthropic_provider.close()

    finally:
        # Always close the provider when done
        await provider.close()
        print("\nProvider closed successfully.")

if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it before running this example:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
    else:
        asyncio.run(main())
