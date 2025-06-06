"""
Prompt templates for MyJobSpy AI.

This package contains prompt templates used for various natural language
processing tasks such as job information extraction, resume parsing, and job
suitability analysis.
"""

from pathlib import Path
from typing import Dict, Optional

# Get the directory containing this file
PACKAGE_DIR = Path(__file__).parent

# Dictionary to store loaded prompt templates
_loaded_prompts: Dict[str, str] = {}


def load_prompt(prompt_name: str) -> str:
    """
    Load a prompt template by name.

    Args:
        prompt_name: Name of the prompt file (with or without .prompt extension)

    Returns:
        The content of the prompt template as a string

    Raises:
        FileNotFoundError: If the prompt file does not exist
    """
    # Ensure the prompt name has the .prompt extension
    if not prompt_name.endswith(".prompt"):
        prompt_name = f"{prompt_name}.prompt"

    # Check if the prompt is already loaded
    if prompt_name in _loaded_prompts:
        return _loaded_prompts[prompt_name]

    # Build the path to the prompt file
    prompt_path = PACKAGE_DIR / prompt_name

    # Check if the file exists
    if not prompt_path.is_file():
        available = [f.name for f in PACKAGE_DIR.glob("*.prompt")]
        raise FileNotFoundError(
            f"Prompt '{prompt_name}' not found. "
            f"Available prompts: {', '.join(available)}"
        )

    # Read and cache the prompt content
    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()

    _loaded_prompts[prompt_name] = content
    return content


def clear_prompt_cache() -> None:
    """Clear the in-memory prompt cache."""
    _loaded_prompts.clear()


# List of all available prompt files
AVAILABLE_PROMPTS = [
    "job_extraction.prompt",
    "resume_extraction.prompt",
    "suitability_analysis.prompt",
]
