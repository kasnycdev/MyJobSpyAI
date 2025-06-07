"""Ollama provider implementation for LLM integration."""

import json
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast

from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import OllamaLLM
from opentelemetry.trace import Status, StatusCode

from .base import BaseProvider, ProviderError, SyncProvider

# Type variable for the response type
T = TypeVar('T')

logger = logging.getLogger(__name__)


class OllamaProvider(BaseProvider[str]):
    """Provider for Ollama LLM integration with OTEL support."""

    def __init__(self, config: Dict[str, Any], **kwargs):
        super().__init__(config, **kwargs)

        # Convert config to dict if it's a Pydantic model
        if hasattr(config, 'model_dump'):
            config = config.model_dump()
        elif hasattr(config, 'dict'):
            config = config.dict()

        # Extract configuration with defaults
        self.model = config.get(
            "model", "llama3:instruct"
        )  # Default to llama3 if not specified
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.temperature = min(
            max(float(config.get("temperature", 0.7)), 0.1), 1.0
        )  # Clamp between 0.1 and 1.0

        # Handle both num_predict and max_tokens for backward compatibility
        self.num_predict = config.get("num_predict", config.get("max_tokens", 1000))

        # Configure HTTPX client with detailed logging
        import http.client as http_client
        import logging

        # Enable HTTP client debugging
        http_client.HTTPConnection.debuglevel = 1

        # Configure logging for HTTP requests
        logging.basicConfig()
        logging.getLogger().setLevel(logging.DEBUG)
        requests_log = logging.getLogger("urllib3")
        requests_log.setLevel(logging.DEBUG)
        requests_log.propagate = True

        logger.debug(f"Initializing Ollama provider with model: {self.model}")
        logger.debug(f"Ollama base URL: {self.base_url}")
        logger.debug(f"Temperature: {self.temperature}")
        logger.debug(f"Num predict: {self.num_predict}")

        # Initialize the Ollama LLM with basic parameters
        # Note: We'll pass additional parameters like num_predict in the generate method
        self.llm = OllamaLLM(
            model=self.model,
            base_url=self.base_url,
            temperature=self.temperature,
            num_predict=self.num_predict,
            verbose=True,  # Enable verbose logging from LangChain
        )

        logger.debug("Ollama provider initialized successfully")

        # Initialize JSON parser
        self.json_parser = JsonOutputParser()

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Generate text from a prompt asynchronously.

        Args:
            prompt: The prompt to generate text from
            model: Optional model name override
            output_schema: Optional JSON schema for structured output
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            Generated text or JSON string if output_schema is provided

        Raises:
            ProviderError: If there's an error generating the response
        """
        current_span = self.tracer.start_span("ollama_generate")
        current_span.set_attribute("model", model or self.model)
        current_span.set_attribute("prompt", prompt)

        try:
            # Use the specified model or fall back to the default
            model_to_use = model or self.model

            # Prepare parameters for the LLM call
            llm_params = {
                "stop": ["\n"],
                "num_predict": self.num_predict,  # Use num_predict instead of max_tokens
                **{
                    k: v for k, v in kwargs.items() if k != "max_tokens"
                },  # Exclude max_tokens from kwargs
            }

            logger.debug(f"Calling Ollama with params: {llm_params}")
            logger.debug(f"Prompt length: {len(prompt)} characters")

            try:
                # Log the request details
                logger.debug(f"Sending request to Ollama API at: {self.base_url}")
                logger.debug(f"Model: {self.model}")

                # Call the LLM
                response = await self.llm.agenerate(
                    [prompt],
                    **llm_params,
                )

                logger.debug("Received response from Ollama API")
                if hasattr(response, 'generations') and response.generations:
                    logger.debug(
                        f"Response contains {len(response.generations)} generations"
                    )
                    if response.generations[0]:
                        logger.debug(
                            f"First generation length: {len(response.generations[0][0].text) if hasattr(response.generations[0][0], 'text') else 'N/A'}"
                        )
            except Exception as e:
                logger.error(f"Error calling Ollama API: {str(e)}", exc_info=True)
                raise

            # Extract the generated text
            if not response.generations or not response.generations[0]:
                raise ProviderError("No generations returned from Ollama")

            generated_text = response.generations[0][0].text.strip()

            # If an output schema is provided, try to parse the response as JSON
            if output_schema:
                try:
                    # Clean the JSON string
                    cleaned_text = generated_text.strip()
                    if cleaned_text.startswith('```json'):
                        cleaned_text = cleaned_text[7:].strip()
                    if cleaned_text.endswith('```'):
                        cleaned_text = cleaned_text[:-3].strip()

                    # Parse the JSON
                    parsed = json.loads(cleaned_text)

                    # Validate against the schema if provided
                    if output_schema:
                        # Simple validation - just check that required fields exist
                        required_fields = output_schema.get("required", [])
                        for field in required_fields:
                            if field not in parsed:
                                raise ValueError(f"Missing required field: {field}")

                    generated_text = json.dumps(parsed)
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse JSON from LLM response: {e}")
                    # Return the raw text if JSON parsing fails
                except Exception as e:
                    logger.warning(f"Error processing JSON response: {e}")

            current_span.set_status(Status(StatusCode.OK))
            return generated_text

        except Exception as e:
            error_msg = f"Error generating text with Ollama: {str(e)}"
            logger.error(error_msg, exc_info=True)
            current_span.record_exception(e)
            current_span.set_status(Status(StatusCode.ERROR, error_msg))
            raise ProviderError(error_msg) from e

        finally:
            current_span.end()

    async def close(self) -> None:
        """Close the provider's resources asynchronously."""
        # Ollama doesn't require explicit cleanup, but we'll implement it for completeness
        pass


class SyncOllamaProvider(SyncProvider):
    """Synchronous wrapper for the Ollama provider."""

    def __init__(self, config: Dict[str, Any], **kwargs):
        self._provider = OllamaProvider(config, **kwargs)
        super().__init__(self._provider)

    def generate_sync(
        self,
        prompt: str,
        model: Optional[str] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Synchronous version of generate.

        Args:
            prompt: The prompt to generate text from
            model: Optional model name override
            output_schema: Optional JSON schema for structured output
            **kwargs: Additional arguments to pass to the LLM

        Returns:
            Generated text or JSON string if output_schema is provided

        Raises:
            ProviderError: If there's an error generating the response
        """
        import asyncio

        try:
            return asyncio.run(
                self._provider.generate(
                    prompt, model=model, output_schema=output_schema, **kwargs
                )
            )
        except Exception as e:
            raise ProviderError(
                f"Error in synchronous Ollama generation: {str(e)}"
            ) from e

    def close_sync(self) -> None:
        """Synchronously close the provider's resources."""
        # Ollama doesn't require explicit cleanup
        pass
