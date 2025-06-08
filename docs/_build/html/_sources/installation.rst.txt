Installation
============

This guide will help you install MyJobSpyAI and its dependencies.

Prerequisites
------------
- Python 3.8 or higher
- pip (Python package manager)
- Git (for development installation)

Installation Methods
-------------------

### Using pip (Recommended)

The easiest way to install MyJobSpyAI is using pip:

.. code-block:: bash

   pip install myjobspyai

### From Source

If you want to contribute to the project or need the latest development version:

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/kasnycdev/MyJobSpyAI.git
      cd MyJobSpyAI

2. Install in development mode:

   .. code-block:: bash

      pip install -e .


### Verifying the Installation

After installation, you can verify that MyJobSpyAI is installed correctly by running:

.. code-block:: bash

   python -c "import myjobspyai; print('MyJobSpyAI version:', myjobspyai.__version__)"

Dependencies
------------
MyJobSpyAI automatically installs the following dependencies:

- requests
- pydantic
- pyyaml
- python-dotenv
- tqdm
- loguru
- langchain
- langchain-community
- langchain-openai
- langchain-anthropic
- langchain-ollama
- beautifulsoup4
- lxml
- python-multipart
- fastapi
- uvicorn
- pydantic-settings

Next Steps
----------
- :doc:`configuration`: Learn how to configure MyJobSpyAI
- :doc:`getting_started`: Get started with basic usage
