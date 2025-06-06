"""Environment variable and configuration utilities for MyJobSpy AI."""

import os
import re
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar, Union

import dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

T = TypeVar("T")


class EnvConfig(BaseSettings):
    """Configuration loaded from environment variables."""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra environment variables
        env_prefix = ""
        case_sensitive = False
        use_enum_values = True

    # Debug and logging
    DEBUG: bool = Field(False, description="Enable debug mode")
    LOG_LEVEL: str = Field("INFO", description="Logging level")

    # Database
    DATABASE_URL: str = Field(
        "sqlite:///./myjobspyai.db", description="Database connection URL"
    )

    # API Keys
    OPENAI_API_KEY: Optional[str] = Field(None, description="OpenAI API key")
    GEMINI_API_KEY: Optional[str] = Field(None, description="Google Gemini API key")

    # LinkedIn credentials
    LINKEDIN_USERNAME: Optional[str] = Field(
        None, description="LinkedIn username/email"
    )
    LINKEDIN_PASSWORD: Optional[str] = Field(None, description="LinkedIn password")

    # Rate limiting
    RATE_LIMIT_REQUESTS: int = Field(10, description="Requests per minute")

    # Proxies
    HTTP_PROXY: Optional[str] = Field(None, description="HTTP proxy URL")
    HTTPS_PROXY: Optional[str] = Field(None, description="HTTPS proxy URL")
    NO_PROXY: Optional[str] = Field(
        "localhost,127.0.0.1",
        description="Comma-separated list of hosts to bypass proxy",
    )

    # File system paths
    DATA_DIR: str = Field(
        str(Path("~/.myjobspyai").expanduser()),
        description="Directory for application data",
    )
    CACHE_DIR: str = Field(
        "{{DATA_DIR}}/cache", description="Directory for cached data"
    )
    LOG_DIR: str = Field("{{DATA_DIR}}/logs", description="Directory for log files")

    # Feature flags
    ENABLE_WEB_UI: bool = Field(True, description="Enable the web interface")
    ENABLE_API: bool = Field(True, description="Enable the REST API")

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate the log level."""
        v = v.upper()
        if v not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(
                "LOG_LEVEL must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL"
            )
        return v

    @field_validator("DATA_DIR", "CACHE_DIR", "LOG_DIR", mode='before')
    @classmethod
    def resolve_paths(cls, v: str, info: Any) -> str:
        """Resolve paths and expand variables."""
        # Skip if the value is already processed
        if not isinstance(v, str):
            return v

        # Replace placeholders
        data_dir = str(Path("~/.myjobspyai").expanduser())  # Default DATA_DIR
        if info.data and "DATA_DIR" in info.data:
            data_dir = str(Path(info.data["DATA_DIR"]).expanduser())

        v = v.replace("{{DATA_DIR}}", data_dir)

        # Expand user and resolve path
        path = Path(v).expanduser().resolve()
        # Create the directory if it doesn't exist
        if "NO_CREATE_DIRS" not in os.environ:  # For testing
            path.mkdir(parents=True, exist_ok=True)

        return str(path)


def load_dotenv(
    env_file: Optional[Union[str, Path]] = None,
    override: bool = False,
    **kwargs: Any,
) -> bool:
    """Load environment variables from a .env file.

    Args:
        env_file: Path to the .env file. If None, searches in standard locations.
        override: Whether to override existing environment variables.
        **kwargs: Additional arguments to pass to python-dotenv.

    Returns:
        True if a .env file was found and loaded, False otherwise.
    """
    if env_file is None:
        # Look for .env files in standard locations
        search_dirs = [
            Path.cwd(),
            Path.home(),
            Path("/etc/myjobspyai"),
        ]

        for dir_path in search_dirs:
            env_path = dir_path / ".env"
            if env_path.exists():
                env_file = env_path
                break
        else:
            return False

    env_file = Path(env_file).expanduser().resolve()

    if not env_file.exists():
        return False

    # Load the .env file
    dotenv.load_dotenv(env_file, override=override, **kwargs)
    return True


def get_env_config() -> EnvConfig:
    """Load and validate environment configuration.

    Returns:
        Validated environment configuration.
    """
    # Load .env file if it exists
    load_dotenv()

    # Create and validate the config
    return EnvConfig()


def get_env_var(
    name: str,
    default: Optional[T] = None,
    required: bool = False,
    var_type: Type[T] = str,  # type: ignore
) -> Optional[T]:
    """Get an environment variable with type conversion and validation.

    Args:
        name: Name of the environment variable.
        default: Default value if the variable is not set.
        required: Whether the variable is required.
        var_type: Type to convert the variable to (e.g., int, bool, str).

    Returns:
        The parsed environment variable value, or the default if not set.

    Raises:
        ValueError: If the variable is required but not set, or if type conversion fails.
    """
    value = os.environ.get(name)

    if value is None:
        if required and default is None:
            raise ValueError(f"Environment variable {name} is required but not set")
        return default

    # Convert the value to the specified type
    try:
        if var_type is bool:
            # Handle boolean values
            value = value.lower() in ("true", "1", "t", "y", "yes")
        elif var_type is list or var_type is list[str]:
            # Handle comma-separated lists
            value = [item.strip() for item in value.split(",") if item.strip()]
        elif var_type is dict or var_type is dict[str, str]:
            # Handle key=value pairs separated by commas
            value = dict(item.split("=", 1) for item in value.split(",") if "=" in item)
        else:
            # Use the type's constructor for other types
            value = var_type(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Failed to parse {name}: {e}")

    return cast(T, value)


def set_env_var(name: str, value: Any, overwrite: bool = True) -> None:
    """Set an environment variable.

    Args:
        name: Name of the environment variable.
        value: Value to set.
        overwrite: Whether to overwrite an existing value.
    """
    if name not in os.environ or overwrite:
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)


def unset_env_var(name: str) -> None:
    """Unset an environment variable.

    Args:
        name: Name of the environment variable to unset.
    """
    os.environ.pop(name, None)


def get_bool_env(name: str, default: bool = False) -> bool:
    """Get a boolean environment variable.

    Args:
        name: Name of the environment variable.
        default: Default value if the variable is not set.

    Returns:
        The boolean value of the environment variable.
    """
    return get_env_var(name, default, var_type=bool)  # type: ignore


def get_int_env(name: str, default: int = 0) -> int:
    """Get an integer environment variable.

    Args:
        name: Name of the environment variable.
        default: Default value if the variable is not set.

    Returns:
        The integer value of the environment variable.
    """
    return get_env_var(name, default, var_type=int)  # type: ignore


def get_list_env(name: str, default: Optional[List[str]] = None) -> List[str]:
    """Get a list from a comma-separated environment variable.

    Args:
        name: Name of the environment variable.
        default: Default value if the variable is not set.

    Returns:
        The list of values from the environment variable.
    """
    if default is None:
        default = []
    return get_env_var(name, default, var_type=list)  # type: ignore


def get_dict_env(name: str, default: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Get a dictionary from a key=value,key2=value2 environment variable.

    Args:
        name: Name of the environment variable.
        default: Default value if the variable is not set.

    Returns:
        The dictionary of key-value pairs from the environment variable.
    """
    if default is None:
        default = {}
    return get_env_var(name, default, var_type=dict)  # type: ignore


def interpolate_env_vars(value: str) -> str:
    """Interpolate environment variables in a string.

    Supports both ${VAR} and $VAR syntax.

    Args:
        value: The string to interpolate.

    Returns:
        The interpolated string.
    """
    if not value:
        return value

    def replace_var(match: re.Match) -> str:
        var_name = match.group(1) or match.group(2)
        return os.environ.get(var_name, match.group(0))

    # Match ${VAR} or $VAR
    pattern = r"\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)"
    return re.sub(pattern, replace_var, value)
