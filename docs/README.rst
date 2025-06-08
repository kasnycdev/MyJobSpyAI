.. _readme:

Documentation
============

This directory contains the source files for the MyJobSpyAI documentation.

Prerequisites
------------

- Python 3.8+
- pip
- Sphinx and other documentation dependencies (installed via ``requirements-docs.txt``)

Building Locally
---------------

1. Install the documentation dependencies:

   .. code-block:: bash

      pip install -r requirements-docs.txt

2. Generate the API documentation:

   .. code-block:: bash

      python generate_api_docs.py

3. Build the HTML documentation:

   .. code-block:: bash

      # On Unix/macOS
      ./build_docs.sh

      # Or on Windows
      .\build_docs.bat

4. The built documentation will be available in the ``_build/html`` directory.

Documentation Structure
----------------------

- :doc:`index` - Main documentation entry point
- :doc:`getting_started` - Getting started guide
- :doc:`installation` - Installation instructions
- :doc:`configuration` - Configuration reference
- :doc:`usage` - Usage guide
- :doc:`examples` - Code examples
- :doc:`api/modules` - API Reference
- :doc:`contributing` - Contribution guidelines
- :doc:`changelog` - Release history

## Writing Documentation

- Use reStructuredText (`.rst`) or Markdown (`.md`) for documentation files
- Follow the existing style and formatting
- Use the `.. note::`, `.. warning::`, and `.. tip::` directives for callouts
- Use code blocks with syntax highlighting when showing code examples

## Viewing the Documentation

After building, open `_build/html/index.html` in your web browser to view the documentation locally.

## Publishing

The documentation is automatically published to Read the Docs when changes are pushed to the `main` branch.
