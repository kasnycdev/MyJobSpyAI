with open('docs/README.rst', 'w', newline='\n', encoding='utf-8') as f:
    f.write('''.. _readme:

Documentation
============

This directory contains the source files for the MyJobSpyAI documentation.

Prerequisites
============

- Python 3.8+
- pip
- Sphinx and other documentation dependencies (installed via ``requirements-docs.txt``)

Building Locally
===============

1. Install the documentation dependencies::

      pip install -r requirements-docs.txt

2. Generate the API documentation::

      python generate_api_docs.py

3. Build the HTML documentation::

      make html

   or::

      sphinx-build -b html docs/ docs/_build/html

4. Open the generated documentation in your browser::

      open docs/_build/html/index.html  # On macOS
      xdg-open docs/_build/html/index.html  # On Linux
      start docs/_build/html/index.html  # On Windows

Documentation Structure
=====================

- ``source/``: Contains the source files for the documentation
  - ``_static/``: Static files (CSS, images, etc.)
  - ``_templates/``: Custom Sphinx templates
  - ``api/``: Auto-generated API documentation
  - ``guides/``: User guides and tutorials
  - ``examples/``: Example code and usage
  - ``conf.py``: Sphinx configuration file
  - ``index.rst``: The main documentation page

Writing Documentation
====================

- Use reStructuredText (RST) syntax for all documentation
- Follow the `Python Documentation Style Guide <https://devguide.python.org/documentation/style-guide/>`_
- Use Sphinx directives and roles where appropriate
- Add cross-references using the ``:ref:`` role
- Include code examples with proper syntax highlighting

Updating API Documentation
========================

To update the API documentation after making changes to the code::

   python generate_api_docs.py
   make html

Troubleshooting
==============

- If you get build errors, check the error messages in the console output
- Make sure all dependencies are installed
- Check for syntax errors in RST files
- Ensure that all referenced files and modules exist

For more information, see the `Sphinx documentation <https://www.sphinx-doc.org/>`_.
''')
