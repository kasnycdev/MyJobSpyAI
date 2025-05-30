# Contributing to MyJobSpyAI

Thank you for your interest in contributing to MyJobSpyAI! We welcome contributions from everyone, whether it's bug reports, feature requests, documentation improvements, or code contributions.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
  - [Your First Code Contribution](#your-first-code-contribution)
  - [Pull Requests](#pull-requests)
- [Development Environment](#development-environment)
- [Code Style](#code-style)
- [Testing](#testing)
- [Documentation](#documentation)
- [Commit Messages](#commit-messages)
- [License](#license)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report any unacceptable behavior to the project maintainers.

## How Can I Contribute?

### Reporting Bugs

Bugs are tracked as [GitHub issues](https://github.com/yourusername/myjobspyai/issues). When creating a bug report, please include the following information:

1. A clear and descriptive title
2. A description of the problem
3. Steps to reproduce the issue
4. Expected behavior
5. Actual behavior
6. Screenshots if applicable
7. Your environment (OS, Python version, installed packages)

### Suggesting Enhancements

Feature requests are also tracked as [GitHub issues](https://github.com/yourusername/myjobspyai/issues). When suggesting an enhancement, please:

1. Use a clear and descriptive title
2. Describe the current behavior
3. Describe the desired behavior
4. Explain why this enhancement would be useful
5. List any alternative solutions you've considered

### Your First Code Contribution

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your changes
4. Make your changes
5. Run the test suite
6. Commit your changes with a descriptive commit message
7. Push your changes to your fork
8. Open a pull request

### Pull Requests

1. Keep pull requests focused on a single feature or bug fix
2. Ensure your code follows the project's code style
3. Update the documentation as needed
4. Add tests for new features or bug fixes
5. Make sure all tests pass
6. Reference any related issues in your pull request description

## Development Environment

### Prerequisites

- Python 3.8 or higher
- [Poetry](https://python-poetry.org/) for dependency management
- [pre-commit](https://pre-commit.com/) for git hooks

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/myjobspyai.git
   cd myjobspyai
   ```

2. Install dependencies:
   ```bash
   poetry install --with dev
   ```

3. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

4. Create a `.env` file with your environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

## Code Style

This project uses the following tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **mypy** for static type checking
- **flake8** for linting

Run the following commands to ensure your code meets our standards:

```bash
# Format code with Black
poetry run black .

# Sort imports with isort
poetry run isort .

# Run type checking with mypy
poetry run mypy .
# Run linter with flake8
poetry run flake8 .
```

## Testing

We use `pytest` for testing. To run the test suite:

```bash
# Run all tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=myjobspyai --cov-report=term-missing

# Run a specific test file
poetry run pytest tests/test_module.py

# Run a specific test
poetry run pytest tests/test_module.py::test_function
```

## Documentation

Documentation is written in Markdown and built with [MkDocs](https://www.mkdocs.org/). To build the documentation locally:

```bash
# Install documentation dependencies
poetry install --with docs

# Build the documentation
poetry run mkdocs build

# Serve the documentation locally
poetry run mkdocs serve
```

## Commit Messages

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for our commit messages. This allows us to automatically generate changelogs and determine semantic versioning.

Format: `type(scope): description`

Example:
```
feat(api): add support for new API endpoint

Add a new endpoint to handle user authentication. This endpoint will
allow users to authenticate using their email and password.

Closes #123
```

### Types:

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing or correcting existing tests
- **chore**: Changes to the build process or auxiliary tools and libraries such as documentation generation

### Scopes:

Scopes are optional but encouraged. They should be a noun describing a section of the codebase.

Examples:
- api
- cli
- docs
- ci
- deps

## License

By contributing to this project, you agree that your contributions will be licensed under the [MIT License](LICENSE).
