import sys
import os
import logging
import asyncio
import json

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

async def main():
    logger.info("=== Starting Simple Test ===")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Python executable: {sys.executable}")
    
    # Test basic imports
    try:
        import requests
        logger.info("Successfully imported requests")
    except ImportError as e:
        logger.error(f"Failed to import requests: {e}")
    
    # Test HTTP request
    try:
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        response.raise_for_status()
        logger.info(f"Ollama version: {response.text}")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {e}")
    
    logger.info("=== Test Completed ===")

if __name__ == "__main__":
    asyncio.run(main())
