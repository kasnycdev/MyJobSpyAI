"""Prompt utility functions for MyJobSpy AI.

This module provides utility functions for working with prompt templates,
including loading, formatting, and managing prompts.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from myjobspyai import prompts

logger = logging.getLogger(__name__)


def get_prompt(
    prompt_name: str,
    params: Optional[Dict[str, Any]] = None,
    prompt_dir: Optional[Union[str, Path]] = None,
) -> str:
    """
    Get a formatted prompt with the given parameters.

    Args:
        prompt_name: Name of the prompt file (with or without .prompt extension)
        params: Dictionary of parameters to format the prompt with
        prompt_dir: Optional directory containing prompt files. If not provided,
                   uses the built-in prompts.

    Returns:
        Formatted prompt string

    Raises:
        FileNotFoundError: If the prompt file does not exist
        KeyError: If a required parameter is missing
    """
    try:
        if prompt_dir is not None:
            # Load prompt from the specified directory
            prompt_path = Path(prompt_dir) / (
                prompt_name
                if prompt_name.endswith(".prompt")
                else f"{prompt_name}.prompt"
            )
            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

            with open(prompt_path, "r", encoding="utf-8") as f:
                template = f.read()
        else:
            # Use built-in prompts
            template = prompts.load_prompt(prompt_name)

        # Format the template with the provided parameters
        if params:
            try:
                return template.format(**params)
            except KeyError as e:
                logger.error(f"Missing required parameter for prompt: {e}")
                raise

        return template

    except Exception as e:
        logger.error(f"Error loading prompt '{prompt_name}': {e}")
        raise


def validate_prompt_parameters(template: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate that all required parameters are present in the params dictionary.

    Args:
        template: The prompt template string
        params: Dictionary of parameters to validate

    Returns:
        Dictionary of missing required parameters

    Example:
        >>> template = "Hello {name}, welcome to {company}!"
        >>> params = {"name": "Alice"}
        >>> missing = validate_prompt_parameters(template, params)
        >>> print(missing)
        {'company'}
    """
    import string
    from collections import defaultdict

    # Parse the template to find all format placeholders
    formatter = string.Formatter()
    parsed = formatter.parse(template)

    # Extract field names (excluding None and empty strings)
    field_names = [fname for _, fname, _, _ in parsed if fname is not None and fname]

    # Find missing required parameters
    missing = {}
    for field in field_names:
        if field not in params or params[field] is None:
            missing[field] = "Required parameter missing"

    return missing
