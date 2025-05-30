"""
JSON utilities for handling LLM responses and data parsing.
"""
import re
import json
import logging
import asyncio
from typing import Any, Dict, Optional, TypeVar, Callable, cast, Type, Union
from functools import wraps

T = TypeVar('T', bound=Callable[..., Any])
logger = logging.getLogger(__name__)

def retry_on_json_error(max_retries: int = 3, delay: float = 1.0) -> Callable[[T], T]:
    """
    Decorator to retry a function when JSON parsing fails.
    
    Args:
        max_retries: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
    """
    def decorator(func: T) -> T:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            last_error = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except (json.JSONDecodeError, ValueError) as e:
                    last_error = e
                    if attempt < max_retries:
                        logger.warning(
                            f"JSON parse attempt {attempt + 1} failed, retrying in {current_delay:.1f}s... Error: {str(e)}"
                        )
                        await asyncio.sleep(current_delay)
                        current_delay *= 2  # Exponential backoff
            
            logger.error(f"Failed to parse JSON after {max_retries} attempts")
            raise ValueError(f"Failed to parse JSON after {max_retries} attempts") from last_error
        
        return cast(T, async_wrapper)
    return decorator

def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract JSON from text that might contain markdown code blocks or other formatting.
    
    Args:
        text: Text that might contain JSON
        
    Returns:
        Parsed JSON as dict or None if no valid JSON found
    """
    if not text or not isinstance(text, str):
        return None
        
    text = text.strip()
    
    # Try direct JSON parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
        
    # Try to extract JSON from markdown code blocks
    try:
        # Handle ```json ... ``` or ``` ... ```
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
            
        # Try to find any JSON object in the text
        json_match = re.search(r'(\{.*\})', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
    except (json.JSONDecodeError, AttributeError) as e:
        logger.debug(f"Failed to extract JSON: {str(e)}")
        pass
        
    return None

def validate_json_structure(data: Dict[str, Any], required_fields: list) -> bool:
    """
    Validate that the JSON contains all required fields.
    
    Args:
        data: Parsed JSON data
        required_fields: List of required field names
        
    Returns:
        bool: True if all required fields are present
    """
    if not isinstance(data, dict):
        return False
    return all(field in data for field in required_fields)

def format_llm_prompt(prompt_template: str, **kwargs: Any) -> str:
    """
    Format a prompt template with the given variables.
    Ensures consistent formatting across all prompts.
    
    Args:
        prompt_template: Template string with {placeholders}
        **kwargs: Variables to format into the template
        
    Returns:
        Formatted prompt string
    """
    try:
        return prompt_template.format(**kwargs)
    except KeyError as e:
        logger.error(f"Missing required prompt variable: {e}")
        raise ValueError(f"Missing required prompt variable: {e}") from e
