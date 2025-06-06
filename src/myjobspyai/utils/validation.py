"""Configuration validation utilities."""

from pathlib import Path
from typing import Any, Dict, Optional, Union


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate the application configuration.

    Args:
        config: The configuration dictionary to validate.

    Returns:
        The validated configuration.

    Raises:
        ValueError: If the configuration is invalid.
    """
    try:
        # Validate required sections
        required_sections = ["app", "logging", "provider"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate provider configuration
        provider_type = config["provider"].get("type")
        if not provider_type:
            raise ValueError("Provider type must be specified in configuration")

        # Validate provider-specific settings
        provider_settings = config["provider"].get("settings", {})
        if provider_type == "openai" and not provider_settings.get("api_key"):
            raise ValueError("OpenAI API key is required in provider settings")
        elif provider_type == "gemini" and not provider_settings.get("api_key"):
            raise ValueError("Gemini API key is required in provider settings")

        # Validate file paths
        if "data_dir" in config["app"]:
            data_dir = Path(config["app"]["data_dir"]).expanduser().resolve()
            if not data_dir.exists():
                data_dir.mkdir(parents=True, exist_ok=True)
            config["app"]["data_dir"] = str(data_dir)

        # Validate logging configuration
        if "log_file" in config["logging"]:
            log_file = Path(config["logging"]["log_file"]).expanduser().resolve()
            log_file.parent.mkdir(parents=True, exist_ok=True)
            config["logging"]["log_file"] = str(log_file)

        return config

    except Exception as e:
        raise ValueError(f"Configuration validation failed: {str(e)}")


def ensure_absolute_path(
    path: Union[str, Path], base_dir: Optional[Union[str, Path]] = None
) -> Path:
    """Ensure a path is absolute, optionally relative to a base directory.

    Args:
        path: The path to make absolute.
        base_dir: Optional base directory for relative paths.

    Returns:
        The absolute Path object.
    """
    path = Path(path).expanduser()

    if not path.is_absolute() and base_dir is not None:
        base_dir = Path(base_dir).expanduser().resolve()
        path = (base_dir / path).resolve()

    return path.resolve()


def check_file_exists(file_path: Union[str, Path], description: str = "File") -> Path:
    """Check if a file exists and return its Path.

    Args:
        file_path: Path to the file.
        description: Description of the file for error messages.

    Returns:
        The resolved Path object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(file_path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"{description} not found: {file_path}")
    return file_path
