"""LangChain provider implementation for LLM integration."""

import json
import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, cast

from langchain_core.output_parsers import JsonOutputParser
from opentelemetry.trace import Status, StatusCode

from .base import BaseProvider, ProviderError, SyncProvider

# Type variable for the response type

logger = logging.getLogger(__name__)


def clean_json_string(json_str: str) -> str:
    """Clean JSON string by handling escape sequences and invalid characters."""
    if not json_str:
        return json_str

    # Common problematic unicode characters
    replacements = {
        "\x2013": "-",  # en-dash
        "\x2014": "--",  # em-dash
        "\x2018": "'",  # left single quote
        "\x2019": "'",  # right single quote
        "\x201c": '"',  # left double quote
        "\x201d": '"',  # right double quote
        "\x2026": "...",  # ellipsis
        "\u2013": "-",  # en-dash (unicode)
        "\u2014": "--",  # em-dash (unicode)
        "\u2018": "'",  # left single quote (unicode)
        "\u2019": "'",  # right single quote (unicode)
    }

    for old, new in replacements.items():
        json_str = json_str.replace(old, new)

    return json_str


class LangChainProvider(BaseProvider[str]):
    """Provider for LangChain LLM integration with OTEL support."""

    def __init__(self, config: Dict[str, Any], **kwargs):
        """Initialize the LangChain provider.

        Args:
            config: Configuration dictionary
            **kwargs: Additional arguments
                - name: Optional name for the provider instance
                - provider_type: Optional provider type (defaults to 'langchain')
        """
        super().__init__(config, **kwargs)
        self.llm = None
        self.parser = JsonOutputParser()

    def _initialize_provider(self) -> None:
        """Initialize the LangChain provider with the given configuration."""
        try:
            self._initialize_langchain()
            logger.info("Initialized LangChain provider: %s", self.provider_name)
        except Exception as e:
            error_msg = f"Failed to initialize LangChain provider: {str(e)}"
            logger.exception(error_msg)
            raise ProviderError(
                message=error_msg,
                provider=self.provider_name,
                error_type="initialization_error",
                details={"config_keys": list(self.config.keys())},
            ) from e

    def _initialize_langchain(self) -> None:
        """Initialize the LangChain client based on configuration."""
        try:
            # Get the LLM class from configuration or use default
            llm_class_name = self.get_config_value("class_name", "ChatOpenAI")

            # Import the module and get the class
            module_name = f"langchain_community.chat_models.{llm_class_name.lower()}"
            try:
                module = __import__(module_name, fromlist=[llm_class_name])
                llm_class = getattr(module, llm_class_name)
            except (ImportError, AttributeError):
                # Fall back to langchain_community if direct import fails
                try:
                    module = __import__(
                        "langchain_community.chat_models", fromlist=[llm_class_name]
                    )
                    llm_class = getattr(module, llm_class_name)
                except (ImportError, AttributeError) as e:
                    error_msg = f"Failed to import LLM class: {llm_class_name}"
                    logger.exception(error_msg)
                    raise ImportError(error_msg) from e

            # Get model configuration
            model_config = self.get_config_value("model_config", {})

            # Create LLM instance
            self.llm = llm_class(**model_config)

            logger.info("Initialized LangChain LLM: %s", llm_class_name)

        except Exception as e:
            error_msg = f"Failed to initialize LangChain: {str(e)}"
            logger.exception(error_msg)
            raise ProviderError(
                message=error_msg,
                provider=self.provider_name,
                error_type="initialization_error",
            ) from e

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> str:
        """Generate text using the configured LangChain LLM with optional JSON output parsing.

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
        if not self.llm:
            raise ProviderError(
                message="LLM not initialized. Call _initialize_langchain() first.",
                provider=self.provider_name,
                error_type="initialization_error",
            )

        # Start a new span for the generation
        with self.tracer.start_as_current_span("langchain.generate") as span:
            try:
                # Set model if provided
                current_model = model or self.get_config_value("model_name")
                if current_model:
                    span.set_attribute("model", current_model)
                    kwargs["model"] = current_model

                # Update metrics
                self.request_counter.add(1, {"provider": self.provider_name})

                # Prepare the prompt with system message if available
                system_message = self.get_config_value("system_message")
                messages = []

                if system_message:
                    messages.append(SystemMessage(content=system_message))

                messages.append(HumanMessage(content=prompt))

                # Log the request
                logger.debug(
                    "LangChain request - model: %s, messages: %s",
                    current_model,
                    [str(m) for m in messages],
                )

                # Invoke the LLM
                response = await self.llm.ainvoke(messages, **kwargs)

                # Log the response
                logger.debug("LangChain response: %s", response.content)

                # Process the response
                if output_schema:
                    try:
                        # Try to parse as JSON if schema is provided
                        parsed = json.loads(response.content)
                        # Validate against schema if needed
                        return json.dumps(parsed, ensure_ascii=False)
                    except json.JSONDecodeError:
                        # If parsing fails, try to clean and parse again
                        cleaned = clean_json_string(response.content)
                        return json.dumps(json.loads(cleaned), ensure_ascii=False)

                return response.content

            except Exception as e:
                error_msg = f"Error in LangChain provider: {str(e)}"
                logger.exception(error_msg)

                # Update error metrics
                self.error_counter.add(
                    1, {"provider": self.provider_name, "error_type": type(e).__name__}
                )

                # Record the error in the span
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))

                # Raise a ProviderError with details
                raise ProviderError(
                    message=error_msg,
                    provider=self.provider_name,
                    error_type="generation_error",
                    details={
                        "model": current_model,
                        "prompt_length": len(prompt),
                        "error": str(e),
                    },
                ) from e

    async def close(self) -> None:
        """Clean up resources used by the provider."""
        if hasattr(self.llm, 'close'):
            await self.llm.aclose()
        self.llm = None
        logger.info("Closed LangChain provider: %s", self.provider_name)


class SyncLangChainProvider(SyncProvider[str], LangChainProvider):
    """Synchronous wrapper for the LangChain provider."""

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

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            super().generate(prompt, model, output_schema, **kwargs)
        )

    def close_sync(self) -> None:
        """Synchronously close the provider's resources."""
        import asyncio

        if hasattr(self, 'llm') and self.llm is not None:
            if hasattr(self.llm, 'aclose'):
                loop = asyncio.get_event_loop()
                loop.run_until_complete(self.llm.aclose())
            self.llm = None
        logger.info("Synchronously closed LangChain provider: %s", self.provider_name)
