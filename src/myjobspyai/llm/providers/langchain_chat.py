"""LangChain chat model provider implementation."""

import importlib
import logging
from typing import Any, Dict, List, Optional, Union, AsyncGenerator, Type, TYPE_CHECKING

from myjobspyai.llm.base import BaseLLMProvider, LLMResponse, LLMError, LLMRequestError

# Lazy imports to avoid loading all providers at once
if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.runnables import RunnableConfig
    from langchain_core.runnables.config import run_in_executor
else:
    # Define these as Any when not type checking to avoid import issues
    BaseChatModel = Any
    HumanMessage = Any
    SystemMessage = Any
    AIMessage = Any
    BaseMessage = Any
    ChatGeneration = Any
    ChatResult = Any
    RunnableConfig = Any
    run_in_executor = Any

# Mapping of provider names to their module and class names
PROVIDER_MAPPING = {
    # Core providers
    "openai": ("langchain_openai", "ChatOpenAI"),
    "anthropic": ("langchain_anthropic", "ChatAnthropic"),
    "google": ("langchain_google_genai", "ChatGoogleGenerativeAI"),
    
    # Community providers
    "ollama": ("langchain_community.chat_models", "ChatOllama"),
    "anyscale": ("langchain_community.chat_models", "ChatAnyscale"),
    "fireworks": ("langchain_community.chat_models", "ChatFireworks"),
    "cohere": ("langchain_community.chat_models", "ChatCohere"),
    "deepinfra": ("langchain_community.chat_models", "ChatDeepInfra"),
    "deepseek": ("langchain_community.chat_models", "ChatDeepSeek"),
    "google-palm": ("langchain_community.chat_models", "ChatGooglePalm"),
    "huggingface": ("langchain_community.chat_models", "ChatHuggingFace"),
    "jina": ("langchain_community.chat_models", "ChatJinaChat"),
    "mlflow": ("langchain_community.chat_models", "ChatMlflow"),
    "mlx": ("langchain_community.chat_models", "ChatMLX"),
    "openrouter": ("langchain_community.chat_models", "ChatOpenRouter"),
    "perplexity": ("langchain_community.chat_models", "ChatPerplexity"),
    "tongyi": ("langchain_community.chat_models", "ChatTongyi"),
    "vertexai": ("langchain_community.chat_models", "ChatVertexAI"),
    "yandex": ("langchain_community.chat_models", "ChatYandexGPT"),
    "yuan2": ("langchain_community.chat_models", "ChatYuan2"),
    "zhipuai": ("langchain_community.chat_models", "ChatZhipuAI"),
}

# Check if langchain is available
try:
    import langchain_core
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)

class LangChainChatProvider(BaseLLMProvider):
    """LangChain chat model provider implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the LangChain chat model provider.
        
        Args:
            config: Configuration dictionary with the following keys:
                - provider: The provider name (e.g., 'openai', 'anthropic', 'google')
                - model: The model name to use
                - api_key: API key for the provider (can be set via environment variable)
                - base_url: Custom API base URL (optional)
                - temperature: Generation temperature (0.0 to 1.0)
                - max_tokens: Maximum number of tokens to generate
                - top_p: Nucleus sampling parameter
                - frequency_penalty: Frequency penalty
                - presence_penalty: Presence penalty
                - stop: List of stop sequences
                - streaming: Whether to enable streaming
                - timeout: Request timeout in seconds
                - max_retries: Number of retries for failed requests
                - provider_config: Additional provider-specific configuration
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Please install it with: "
                "pip install langchain-core langchain-openai langchain-community"
            )
        
        # Initialize the base class with the provider name
        provider_type = config.get("provider", "openai").lower()
        super().__init__(provider_name=f"langchain_{provider_type}")
        
        self.config = config
        self.provider = provider_type
        self.model = config.get("model", "gpt-4")
        self.temperature = float(config.get("temperature", 0.7))
        self.max_tokens = int(config.get("max_tokens", 1000))
        self.top_p = float(config.get("top_p", 1.0))
        self.frequency_penalty = float(config.get("frequency_penalty", 0.0))
        self.presence_penalty = float(config.get("presence_penalty", 0.0))
        self.stop = config.get("stop")
        self.streaming = bool(config.get("streaming", False))
        self.timeout = int(config.get("timeout", 60))
        self.max_retries = int(config.get("max_retries", 3))
        self.provider_config = config.get("provider_config", {})
        
        # Set API key from config or environment variable
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")
        
        # Initialize the chat model
        self._model = self._initialize_model()
    
    def _initialize_model(self) -> BaseChatModel:
        """Initialize the LangChain chat model based on provider."""
        common_kwargs = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "streaming": self.streaming,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            **self.provider_config
        }
        
        # Add API key and base URL if provided
        if self.api_key and self.api_key != "ollama":  # Skip API key for Ollama if it's just 'ollama'
            common_kwargs["api_key"] = self.api_key
        if self.base_url:
            common_kwargs["base_url"] = self.base_url
        
        # Special handling for Ollama
        if self.provider == "ollama":
            try:
                from langchain_community.chat_models import ChatOllama
                # Ensure base_url is set for Ollama
                if "base_url" not in common_kwargs:
                    common_kwargs["base_url"] = "http://10.10.0.178:11434"
                # Remove any None values that might cause issues
                common_kwargs = {k: v for k, v in common_kwargs.items() if v is not None}
                return ChatOllama(**common_kwargs)
            except ImportError as e:
                logger.error("Failed to import Ollama. Install with: pip install langchain-community")
                raise
        
        # Initialize other providers
        try:
            module_name, class_name = PROVIDER_MAPPING.get(
                self.provider, 
                ("langchain_openai", "ChatOpenAI")
            )
            module = importlib.import_module(module_name)
            model_class = getattr(module, class_name)
            # Remove any None values that might cause issues
            common_kwargs = {k: v for k, v in common_kwargs.items() if v is not None}
            return model_class(**common_kwargs)
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to initialize provider {self.provider}: {str(e)}")
            logger.warning(f"Provider '{self.provider}' not found or not properly configured. Defaulting to OpenAI.")
            
            # Fall back to OpenAI if the configured provider fails
            try:
                from langchain_openai import ChatOpenAI
                common_kwargs = {k: v for k, v in common_kwargs.items() if v is not None}
                return ChatOpenAI(**common_kwargs)
            except ImportError:
                raise ImportError(
                    "Failed to initialize default OpenAI provider. "
                    "Please install langchain-openai: pip install langchain-openai"
                ) from e
    
    async def generate(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate text from a prompt.
        
        Args:
            prompt: The prompt to generate text from.
            **kwargs: Additional generation parameters.
                - system_message: Optional system message to set the behavior of the assistant
                - messages: Optional list of message dictionaries with 'role' and 'content' keys
                - temperature: Override the default temperature
                - max_tokens: Override the default max tokens
                - stop: Override the default stop sequences
                - Any other model-specific parameters
                
        Returns:
            LLMResponse containing the generated text and metadata
            
        Raises:
            LLMError: If the request fails
        """
        try:
            # Extract common parameters
            system_message = kwargs.pop("system_message", None)
            messages = kwargs.pop("messages", None)
            
            # Override instance settings with any provided in kwargs
            temperature = kwargs.pop("temperature", self.temperature)
            max_tokens = kwargs.pop("max_tokens", self.max_tokens)
            stop = kwargs.pop("stop", self.stop)
            
            # Prepare messages
            chat_messages = []
            
            # Add system message if provided
            if system_message:
                chat_messages.append(SystemMessage(content=system_message))
            
            # Add conversation history if provided
            if messages:
                for msg in messages:
                    role = msg.get("role", "user").lower()
                    content = msg.get("content", "")
                    
                    if role == "system":
                        chat_messages.append(SystemMessage(content=content))
                    elif role == "assistant":
                        chat_messages.append(AIMessage(content=content))
                    else:  # default to user
                        chat_messages.append(HumanMessage(content=content))
            
            # Add the current user prompt
            chat_messages.append(HumanMessage(content=prompt))
            
            # Prepare generation parameters
            generation_kwargs = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": stop,
                **kwargs  # Allow any other model-specific parameters
            }
            
            # Generate response
            response = await self._model.agenerate(
                messages=[chat_messages],
                **generation_kwargs
            )
            
            # Extract the generated text
            if isinstance(response, ChatResult) and response.generations:
                generation = response.generations[0]
                if isinstance(generation, list):
                    generation = generation[0]
                text = generation.text
            else:
                text = str(response)
            
            # Calculate token usage if available
            usage = {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None
            }
            
            if hasattr(response, "usage"):
                usage.update({
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(response.usage, "completion_tokens", None),
                    "total_tokens": getattr(response.usage, "total_tokens", None),
                })
            
            return LLMResponse(
                text=text,
                model=self.model,
                usage=usage,
                metadata={
                    "model": self.model,
                    "provider": self.provider,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stop": stop,
                }
            )
            
        except Exception as e:
            logger.error(f"Error in LangChain chat model generation: {str(e)}", exc_info=True)
            raise LLMRequestError(f"Error generating response: {str(e)}")
    
    async def get_embeddings(
        self,
        texts: Union[str, List[str]],
        **kwargs: Any,
    ) -> List[List[float]]:
        """Get embeddings for the given texts.
        
        Args:
            texts: A single text or a list of texts to get embeddings for.
            **kwargs: Additional parameters for the embedding model.
            
        Returns:
            A list of embeddings, one for each input text.
            
        Raises:
            NotImplementedError: If the provider doesn't support embeddings
        """
        # Most LangChain chat models don't support embeddings directly
        # This would require a separate embedding model
        raise NotImplementedError("Embeddings not supported by this provider. Use a dedicated embedding model.")
    
    async def generate_stream(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncGenerator[LLMResponse, None]:
        """Stream a response from the chat model.
        
        Args:
            prompt: The prompt to generate text from.
            **kwargs: Additional generation parameters.
                - system_message: Optional system message to set the behavior of the assistant
                - messages: Optional list of message dictionaries with 'role' and 'content' keys
                - temperature: Override the default temperature
                - max_tokens: Override the default max tokens
                - stop: Override the default stop sequences
                - Any other model-specific parameters
                
        Yields:
            LLMResponse chunks containing the generated text and metadata
            
        Raises:
            LLMError: If the request fails
        """
        # Enable streaming for this request
        streaming = kwargs.pop("streaming", True)
        if not streaming:
            response = await self.generate(prompt, **kwargs)
            yield response
            return
        
        try:
            # Extract common parameters
            system_message = kwargs.pop("system_message", None)
            messages = kwargs.pop("messages", None)
            
            # Override instance settings with any provided in kwargs
            temperature = kwargs.pop("temperature", self.temperature)
            max_tokens = kwargs.pop("max_tokens", self.max_tokens)
            stop = kwargs.pop("stop", self.stop)
            
            # Prepare messages
            chat_messages = []
            
            if system_message:
                chat_messages.append(SystemMessage(content=system_message))
            
            if messages:
                for msg in messages:
                    role = msg.get("role", "user").lower()
                    content = msg.get("content", "")
                    
                    if role == "system":
                        chat_messages.append(SystemMessage(content=content))
                    elif role == "assistant":
                        chat_messages.append(AIMessage(content=content))
                    else:
                        chat_messages.append(HumanMessage(content=content))
            
            chat_messages.append(HumanMessage(content=prompt))
            
            # Prepare generation parameters
            generation_kwargs = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stop": stop,
                **kwargs  # Allow any other model-specific parameters
            }
            
            # Stream the response
            full_response = ""
            async for chunk in self._model.astream(chat_messages, **generation_kwargs):
                if isinstance(chunk, str):
                    chunk_text = chunk
                elif hasattr(chunk, 'content'):
                    chunk_text = chunk.content
                else:
                    chunk_text = str(chunk)
                
                full_response += chunk_text
                
                yield LLMResponse(
                    text=chunk_text,
                    model=self.model,
                    usage={
                        "prompt_tokens": None,
                        "completion_tokens": len(chunk_text.split()),  # Approximate
                        "total_tokens": None
                    },
                    metadata={
                        "model": self.model,
                        "provider": self.provider,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "chunk": True
                    }
                )
            
            # Final response to mark completion
            yield LLMResponse(
                text=full_response,
                model=self.model,
                usage={
                    "prompt_tokens": None,
                    "completion_tokens": len(full_response.split()),  # Approximate
                    "total_tokens": None
                },
                metadata={
                    "model": self.model,
                    "provider": self.provider,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "complete": True
                }
            )
            
        except Exception as e:
            logger.error(f"Error in LangChain chat model streaming: {str(e)}", exc_info=True)
            raise LLMRequestError(f"Error streaming response: {str(e)}")
    
    async def close(self):
        """Close the provider and release any resources."""
        # Most LangChain models don't need explicit cleanup, but some might
        if hasattr(self._model, 'close'):
            if callable(self._model.close):
                if hasattr(self._model.close, '__aexit__'):
                    await self._model.close()
                else:
                    self._model.close()
        
        # Also check for async close methods
        if hasattr(self._model, 'aclose'):
            await self._model.aclose()


# Alias for backward compatibility
LangChainProvider = LangChainChatProvider
