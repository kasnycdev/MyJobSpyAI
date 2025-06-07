#!/bin/bash

# Install documentation dependencies
echo "Installing documentation dependencies..."
pip install -r requirements-docs.txt

# Initialize Sphinx
echo "Initializing Sphinx documentation..."
sphinx-quickstart --quiet --sep --project="MyJobSpyAI" --author="Your Name" -v 1.0.0 --ext-autodoc --ext-viewcode --ext-mathjax --extensions="sphinx.ext.napoleon" --makefile --no-batchfile -r 1.0.0 -l "en" .

# Create necessary directories
mkdir -p _static _templates

# Copy configuration file with our custom settings
cat > conf.py << 'EOL'
# Configuration file for Sphinx documentation builder.
import os
import sys
from datetime import datetime

# Add project to path
sys.path.insert(0, os.path.abspath('../../src'))

# Project information
project = 'MyJobSpyAI'
copyright = f"{datetime.now().year}, Your Name"
author = 'Your Name'
release = '1.0.0'

# General configuration
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosummary',
    'sphinx_rtd_theme',
    'sphinx_copybutton',
    'sphinx_tabs.tabs',
    'myst_parser',
]

# Templates
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# HTML theme options
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    'navigation_depth': 3,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False
}

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# Enable Markdown support
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
EOL

# Create main index.rst
cat > index.rst << 'EOL'
.. MyJobSpyAI documentation master file, created by
   sphinx-quickstart on Sat Jun  7 2025.

Welcome to MyJobSpyAI's documentation!
====================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   installation
   configuration
   usage
   api/modules
   examples
   contributing
   changelog

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
EOL

# Create basic documentation pages
mkdir -p api

echo "Creating basic documentation pages..."

# Getting Started
cat > getting_started.rst << 'EOL'
Getting Started
==============

.. toctree::
   :maxdepth: 2

   installation
   configuration
   usage
EOL

# Installation
cat > installation.rst << 'EOL'
Installation
===========


From PyPI
--------
.. code-block:: bash

   pip install myjobspyai


From Source
-----------
.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/myjobspyai.git
   cd myjobspyai

   # Install in development mode
   pip install -e .


Dependencies
------------
All dependencies will be installed automatically when installing the package.

EOL

# Configuration
cat > configuration.rst << 'EOL'
Configuration
============

MyJobSpyAI can be configured using a YAML configuration file. By default, it looks for configuration in the following locations:

1. ``~/.config/myjobspyai/config.yaml``
2. ``~/.myjobspyai/config.yaml``
3. ``./config.yaml``

Example Configuration
-------------------

.. code-block:: yaml

   # LLM Configuration
   llm:
     default_provider: "ollama"
     providers:
       ollama:
         name: "ollama"
         enabled: true
         type: "ollama"
         model: "gemma3:latest"
         temperature: 0.7
         num_predict: 1000
         timeout: 120
         max_retries: 3
         base_url: "http://localhost:11434"

   # Job Search Configuration
   jobspy:
     search_term: ""
     location: ""
     site_name:
       - linkedin
       - glassdoor
     is_remote: true
     results_wanted: 5

   # Output Configuration
   output:
     output_dir: "output"
     scraped_jobs_filename: "scraped_jobs.json"
     analysis_filename: "analyzed_jobs.json"

Environment Variables
-------------------
You can also configure settings using environment variables:

- ``MYJOBSPYAI_LLM_DEFAULT_PROVIDER``
- ``MYJOBSPYAI_LLM_PROVIDERS_OLLAMA_MODEL``
- ``MYJOBSPYAI_JOBSPY_SEARCH_TERM``
- And more...
EOL

# Usage
cat > usage.rst << 'EOL'
Usage
=====

Basic Usage
-----------

.. code-block:: python

   from myjobspyai import MyJobSpyAI

   # Initialize with default settings
   client = MyJobSpyAI()

   # Search for jobs
   jobs = client.search_jobs(
       search_term="Software Engineer",
       location="Remote",
       is_remote=True
   )

   # Analyze jobs with a resume
   analysis = client.analyze_jobs_with_resume(
       resume_path="path/to/your/resume.pdf",
       jobs=jobs
   )

   # Save results
   client.save_results(analysis, "job_analysis_results.json")


Command Line Interface
---------------------

.. code-block:: bash

   # Search for jobs
   myjobspyai search --search-term "Software Engineer" --location "Remote" --is-remote

   # Analyze jobs with a resume
   myjobspyai analyze --resume path/to/your/resume.pdf --jobs jobs.json

   # Get help
   myjobspyai --help
EOL

# API Documentation
cat > api/modules.rst << 'EOL'
API Reference
============

.. toctree::
   :maxdepth: 4

   myjobspyai
EOL

# Create module documentation
sphinx-apidoc -o api/ ../src/myjobspyai --separate --module-first -f

# Examples
cat > examples.rst << 'EOL'
Examples
=======

Basic Job Search
----------------

.. code-block:: python

   from myjobspyai import MyJobSpyAI

   client = MyJobSpyAI()
   jobs = client.search_jobs(
       search_term="Data Scientist",
       location="New York, NY",
       is_remote=True,
       results_wanted=10
   )
   print(f"Found {len(jobs)} jobs")

Resume Analysis
--------------

.. code-block:: python

   from myjobspyai import MyJobSpyAI

   client = MyJobSpyAI()
   jobs = client.search_jobs("Machine Learning Engineer", "Remote")

   # Analyze jobs with your resume
   analysis = client.analyze_jobs_with_resume(
       resume_path="path/to/your/resume.pdf",
       jobs=jobs
   )

   # Print analysis results
   for job_analysis in analysis:
       print(f"Job: {job_analysis['job_title']}")
       print(f"Match Score: {job_analysis['match_score']}%")
       print("-" * 50)

Custom Configuration
-------------------

.. code-block:: python

   from myjobspyai import MyJobSpyAI
   from pathlib import Path

   # Load custom config
   config_path = Path.home() / ".config" / "myjobspyai" / "config.yaml"

   client = MyJobSpyAI(config_path=config_path)
   # Use the client as usual...
EOL

# Contributing
cat > contributing.rst << 'EOL'
Contributing
===========

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch: ``git checkout -b feature/your-feature``
3. Commit your changes: ``git commit -m 'Add some feature'``
4. Push to the branch: ``git push origin feature/your-feature``
5. Open a pull request

Development Setup
---------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/yourusername/myjobspyai.git
   cd myjobspyai

   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install development dependencies
   pip install -e ".[dev]"

   # Run tests
   pytest

Code Style
----------
- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Write docstrings for all public functions and classes
- Keep lines under 100 characters

Pull Request Guidelines
----------------------
- Make sure all tests pass
- Update documentation as needed
- Add tests for new features
- Ensure code is properly formatted with black
EOL

# Changelog
cat > changelog.rst << 'EOL'
Changelog
========

1.0.0 (2025-06-07)
------------------
- Initial release
EOL

echo "Documentation setup complete!"
echo "To build the documentation, run:"
echo "cd docs && make html"
echo "Then open _build/html/index.html in your browser"
