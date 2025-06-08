#!/usr/bin/env python3
"""
Script to generate API documentation using sphinx-apidoc.
"""
import os
import sys
from pathlib import Path


def generate_api_docs() -> None:
    """Generate API documentation using sphinx-apidoc."""
    # Project root directory
    project_root: Path = Path(__file__).parent
    src_dir: Path = project_root.parent / "src"
    docs_dir: Path = project_root
    api_dir: Path = docs_dir / "api"

    # Create api directory if it doesn't exist
    api_dir.mkdir(exist_ok=True)

    # Run sphinx-apidoc
    cmd: str = f"sphinx-apidoc -f -o {api_dir} {src_dir} --separate --module-first --tocfile index"

    print(f"Running: {cmd}")
    os.system(cmd)

    # Create or update the main API index
    with open(api_dir / "index.rst", "w", encoding="utf-8") as f:
        f.write(
            """API Reference
============

.. toctree::
   :maxdepth: 4
   :caption: Modules

   modules
"""
        )

    print("API documentation generated successfully!")


if __name__ == "__main__":
    generate_api_docs()
