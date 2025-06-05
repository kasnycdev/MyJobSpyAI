# Development Guide

[![Tests](https://github.com/yourusername/myjobspyai/actions/workflows/tests.yml/badge.svg)](https://github.com/yourusername/myjobspyai/actions/workflows/tests.yml)
[![Code Coverage](https://codecov.io/gh/yourusername/myjobspyai/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/myjobspyai)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running Tests](#running-tests)
  - [Unit Tests](#unit-tests)
  - [Integration Tests](#integration-tests)
  - [Test Coverage](#test-coverage)
- [Code Style](#code-style)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- Python 3.8+
- [Poetry](https://python-poetry.org/) (recommended) or pip
- [Docker](https://www.docker.com/) (for running tests with services like Ollama)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/myjobspyai.git
   cd myjobspyai
   ```

2. **Set up a virtual environment and install dependencies:**
   ```bash
   # Using Poetry (recommended)
   poetry install

   # Or using pip
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .[dev]  # Install package in development mode with dev dependencies
   ```

3. **Install pre-commit hooks (optional but recommended):**
   ```bash
   pre-commit install
   ```

## Running Tests

### Unit Tests

Run all unit tests:

```bash
# Using make
make test

# Or directly with pytest
pytest -v tests/unit/
```

Run a specific test file or test case:

```bash
# Run a specific test file
pytest tests/unit/test_config.py

# Run a specific test case
pytest tests/unit/test_config.py::TestAppConfig::test_environment_variables
```

### Integration Tests

Integration tests require additional services (like Ollama) to be running. You can use Docker Compose to set these up:

```bash
# Start required services
docker-compose up -d

# Run integration tests
pytest -v tests/integration/

# Stop services when done
docker-compose down
```

### Test Coverage

Generate a coverage report:

```bash
# Using make
make test-cov

# Or directly with pytest
pytest --cov=myjobspyai --cov-report=term-missing
```

Generate an HTML coverage report:

```bash
pytest --cov=myjobspyai --cov-report=html
open htmlcov/index.html  # Open the report in your browser
```

## Code Style

This project uses:
- **Black** for code formatting
- **isort** for import sorting
- **Flake8** for linting
- **Mypy** for static type checking

To automatically format your code:

```bash
make format
```

To check code style and type hints:

```bash
# Run all linters and type checking
make lint check-types

# Or run them individually
flake8 myjobspyai tests
black --check myjobspyai tests
mypy myjobspyai tests
```

## Project Structure

```
myjobspyai/
├── src/
│   └── myjobspyai/
│       ├── __init__.py
│       ├── config.py       # Configuration management
│       ├── logging.py      # Logging configuration
│       ├── llm/           # LLM provider implementations
│       │   ├── __init__.py
│       │   ├── base.py    # Base classes and interfaces
│       │   └── providers/ # Provider implementations
│       │       └── ollama.py
│       └── utils/         # Utility modules
│           ├── __init__.py
│           ├── files.py
│           ├── async_utils.py
│           ├── http_client.py
│           └── env.py
├── tests/                  # Test files
│   ├── __init__.py
│   ├── conftest.py        # Test fixtures
│   ├── unit/              # Unit tests
│   │   ├── __init__.py
│   │   ├── test_config.py
│   │   ├── test_llm_provider.py
│   │   └── test_http_client.py
│   └── integration/       # Integration tests
│       ├── __init__.py
│       └── test_ollama_integration.py
├── .github/workflows/     # GitHub Actions workflows
│   └── tests.yml
├── .coveragerc            # Coverage configuration
├── .flake8                # Flake8 configuration
├── .pre-commit-config.yaml # Pre-commit hooks
├── Makefile               # Common development commands
├── pyproject.toml         # Project metadata and dependencies
├── README.md              # User documentation
└── DEVELOPMENT.md         # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please make sure to:
- Update tests as appropriate
- Ensure all tests pass before submitting a PR
- Update documentation if needed
- Follow the code style guidelines

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
