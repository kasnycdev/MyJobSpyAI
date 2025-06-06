"""File and directory utilities for MyJobSpy AI."""

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Path to the directory.

    Returns:
        The resolved Path object.
    """
    path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(file_path: Union[str, Path]) -> Any:
    """Read JSON data from a file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        The parsed JSON data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    file_path = Path(file_path).expanduser().resolve()
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(
    data: Any, file_path: Union[str, Path], indent: int = 2, ensure_ascii: bool = False
) -> None:
    """Write data to a JSON file.

    Args:
        data: Data to serialize to JSON.
        file_path: Path to the output file.
        indent: Number of spaces for indentation.
        ensure_ascii: Whether to escape non-ASCII characters.
    """
    file_path = Path(file_path).expanduser().resolve()
    ensure_dir(file_path.parent)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def read_yaml(file_path: Union[str, Path]) -> Any:
    """Read YAML data from a file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        The parsed YAML data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        yaml.YAMLError: If the file contains invalid YAML.
    """
    import yaml

    file_path = Path(file_path).expanduser().resolve()
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(
    data: Any,
    file_path: Union[str, Path],
    default_flow_style: bool = False,
    sort_keys: bool = False,
) -> None:
    """Write data to a YAML file.

    Args:
        data: Data to serialize to YAML.
        file_path: Path to the output file.
        default_flow_style: Whether to use flow style for collections.
        sort_keys: Whether to sort dictionary keys.
    """
    import yaml

    file_path = Path(file_path).expanduser().resolve()
    ensure_dir(file_path.parent)

    with open(file_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            default_flow_style=default_flow_style,
            sort_keys=sort_keys,
            allow_unicode=True,
            encoding="utf-8",
        )


def file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """Calculate the hash of a file.

    Args:
        file_path: Path to the file.
        algorithm: Hash algorithm to use (e.g., 'md5', 'sha1', 'sha256').

    Returns:
        The hexadecimal digest of the file's hash.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the algorithm is not supported.
    """
    file_path = Path(file_path).expanduser().resolve()

    try:
        hash_func = getattr(hashlib, algorithm)()
    except AttributeError as e:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from e

    with open(file_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            hash_func.update(byte_block)

    return hash_func.hexdigest()


def find_files(
    directory: Union[str, Path],
    patterns: Union[str, List[str]] = "*",
    recursive: bool = True,
    case_sensitive: bool = False,
) -> List[Path]:
    """Find files matching the given patterns.

    Args:
        directory: Directory to search in.
        patterns: Glob patterns to match (str or list of str).
        recursive: Whether to search recursively.
        case_sensitive: Whether the pattern matching is case-sensitive.

    Returns:
        List of matching file paths.
    """
    directory = Path(directory).expanduser().resolve()

    if isinstance(patterns, str):
        patterns = [patterns]

    matches = set()

    for pattern in patterns:
        if recursive:
            glob_method = directory.rglob if case_sensitive else directory.glob
        else:
            glob_method = directory.glob

        try:
            for path in glob_method(pattern):
                if path.is_file():
                    matches.add(path.resolve())
        except Exception as e:
            logger.warning(f"Error searching for pattern '{pattern}': {e}")

    return sorted(matches)


def copy_file(
    src: Union[str, Path],
    dst: Union[str, Path],
    overwrite: bool = False,
    create_dirs: bool = True,
) -> Path:
    """Copy a file to a new location.

    Args:
        src: Source file path.
        dst: Destination path (file or directory).
        overwrite: Whether to overwrite existing files.
        create_dirs: Whether to create parent directories if they don't exist.

    Returns:
        The path to the copied file.

    Raises:
        FileNotFoundError: If the source file doesn't exist.
        FileExistsError: If the destination exists and overwrite is False.
    """
    src = Path(src).expanduser().resolve()
    dst = Path(dst).expanduser()

    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    if dst.is_dir():
        dst = dst / src.name

    dst = dst.resolve()

    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination exists and overwrite is False: {dst}")

    if create_dirs:
        ensure_dir(dst.parent)

    shutil.copy2(src, dst)
    return dst


def move_file(
    src: Union[str, Path],
    dst: Union[str, Path],
    overwrite: bool = False,
    create_dirs: bool = True,
) -> Path:
    """Move a file to a new location.

    Args:
        src: Source file path.
        dst: Destination path (file or directory).
        overwrite: Whether to overwrite existing files.
        create_dirs: Whether to create parent directories if they don't exist.

    Returns:
        The path to the moved file.

    Raises:
        FileNotFoundError: If the source file doesn't exist.
        FileExistsError: If the destination exists and overwrite is False.
    """
    src = Path(src).expanduser().resolve()
    dst = Path(dst).expanduser()

    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")

    if dst.is_dir():
        dst = dst / src.name

    dst = dst.resolve()

    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Destination exists and overwrite is False: {dst}")
        dst.unlink()

    if create_dirs:
        ensure_dir(dst.parent)

    shutil.move(str(src), str(dst))
    return dst


def delete_file(file_path: Union[str, Path], missing_ok: bool = True) -> None:
    """Delete a file.

    Args:
        file_path: Path to the file to delete.
        missing_ok: If False, raise FileNotFoundError if the file doesn't exist.

    Raises:
        FileNotFoundError: If the file doesn't exist and missing_ok is False.
    """
    file_path = Path(file_path).expanduser().resolve()

    try:
        file_path.unlink()
    except FileNotFoundError:
        if not missing_ok:
            raise


def clear_directory(
    directory: Union[str, Path], pattern: str = "*", exclude: Optional[List[str]] = None
) -> None:
    """Remove all files in a directory matching the pattern.

    Args:
        directory: Directory to clear.
        pattern: Glob pattern to match files.
        exclude: List of patterns to exclude from deletion.
    """
    directory = Path(directory).expanduser().resolve()

    if not directory.exists():
        return

    exclude = exclude or []

    for item in directory.glob(pattern):
        if any(item.match(e) for e in exclude):
            continue

        try:
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
        except Exception as e:
            logger.warning(f"Failed to delete {item}: {e}")
