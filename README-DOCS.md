# MyJobSpyAI Documentation

This directory contains the source files for the MyJobSpyAI documentation.

## Prerequisites

- Python 3.8 or higher
- pip
- Git (for version control)

## Setting Up the Documentation Environment

1. Clone the repository if you haven't already:
   ```bash
   git clone https://github.com/yourusername/MyJobSpyAI.git
   cd MyJobSpyAI
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # On Unix/macOS
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. Install the documentation dependencies:
   ```bash
   pip install -r docs/requirements-docs.txt
   ```

## Building the Documentation

### On Unix/macOS:
```bash
cd docs
./build_docs.sh
```

### On Windows:
```cmd
cd docs
build_docs.bat
```

## Viewing the Documentation

After a successful build, open `docs/_build/html/index.html` in your web browser to view the documentation locally.

## Documentation Structure

- `docs/` - Documentation source files
  - `api/` - Auto-generated API documentation
  - `_static/` - Static files (CSS, images, etc.)
  - `_templates/` - Custom Sphinx templates
  - `*.rst` - Documentation source files in reStructuredText format
  - `*.md` - Documentation source files in Markdown format

## Writing Documentation

- Use reStructuredText (`.rst`) for most documentation
- Markdown (`.md`) is also supported but may have limited features
- Follow the existing style and formatting
- Use the following Sphinx directives for special formatting:
  - `.. note::` - For important notes
  - `.. warning::` - For warnings
  - `.. tip::` - For tips and tricks
  - `.. code-block:: python` - For code examples

## Updating API Documentation

When you add or modify Python code, update the API documentation by running:

```bash
python docs/generate_api_docs.py
```

## Publishing the Documentation

The documentation is automatically published to Read the Docs when changes are pushed to the `main` branch.

## Troubleshooting

If you encounter any issues:
1. Make sure all dependencies are installed
2. Check for any error messages in the build output
3. Verify that all required files exist
4. Check the Sphinx documentation for help with specific errors

## License

The documentation is licensed under the same license as the MyJobSpyAI project.
