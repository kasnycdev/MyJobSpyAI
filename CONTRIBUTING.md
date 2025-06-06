# Contributing to MyJobSpy AI

[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)
[![Open Source Love](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

Thank you for your interest in contributing to MyJobSpy AI! We welcome contributions from the community to help improve this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Environment](#development-environment)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Style](#code-style)
- [Documentation](#documentation)
- [Submitting a Pull Request](#submitting-a-pull-request)
- [Reporting Issues](#reporting-issues)
- [Feature Requests](#feature-requests)
- [License](#license)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [your-email@example.com](mailto:your-email@example.com).

## Getting Started

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/myjobspyai.git
   cd myjobspyai
   ```
3. **Set up the development environment** (see below).
4. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-number-description
   ```

## Development Environment

### Prerequisites

- Python 3.9 or higher
- [Poetry](https://python-poetry.org/) (recommended) or pip
- [Git](https://git-scm.com/)
- [pre-commit](https://pre-commit.com/) (recommended)
- [Docker](https://www.docker.com/) (for local development with dependencies)

### Setup

1. **Install dependencies** using Poetry:
   ```bash
   poetry install
   ```
   Or using pip:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .[dev]
   ```

2. **Install pre-commit hooks** (optional but recommended):
   ```bash
   pre-commit install
   ```

## Making Changes

1. **Sync your fork** with the main repository:
   ```bash
   git remote add upstream https://github.com/original-owner/myjobspyai.git
   git fetch upstream
   git checkout main  # or develop, depending on your workflow
   git merge upstream/main
   ```

2. **Make your changes** in your feature branch.

3. **Run tests** to ensure everything works (see [Testing](#testing)).

4. **Commit your changes** with a descriptive commit message:
   ```bash
   git add .
   git commit -m "Add feature/fix: brief description of changes"
   ```

5. **Push your changes** to your fork:
   ```bash
   git push origin your-branch-name
   ```

## Testing

### Running Tests

Run the test suite to ensure your changes don't break existing functionality:

```bash
# Run all tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_module.py

# Run tests with verbose output
pytest -v

# Run tests and stop after first failure
pytest -x
```

### Test Structure

- Unit tests are located in the `tests/unit` directory
- Integration tests are in `tests/integration`
- End-to-end tests are in `tests/e2e`

### Writing Tests

- Follow the Arrange-Act-Assert pattern
- Use descriptive test names
- Keep tests focused and independent
- Use fixtures for common setup/teardown
- Mock external dependencies

### Test Coverage

We aim to maintain high test coverage. Before submitting a PR, ensure:

1. New code has appropriate test coverage
2. Existing tests still pass
3. Coverage doesn't decrease significantly

To generate a coverage report:

```bash
pytest --cov=src --cov-report=html
open htmlcov/index.html  # View coverage report in browser
```

# Run a specific test file
pytest tests/unit/test_feature.py

# Run a specific test case
pytest tests/unit/test_feature.py::TestFeature::test_specific_case
```

## Submitting a Pull Request

1. **Ensure all tests pass** and the code follows the project's style guidelines.

2. **Update documentation** if your changes affect the API or add new features.

3. **Push your changes** to your fork on GitHub.

4. **Create a Pull Request** (PR) from your fork to the main repository's `main` or `develop` branch.

5. **Fill out the PR template** with a clear description of your changes, any related issues, and any additional context.

6. **Request a review** from one of the project maintainers.

## Reporting Issues

If you find a bug or have a suggestion, please open an issue on GitHub with the following information:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected vs. actual behavior
- Environment information (Python version, OS, etc.)
- Any relevant error messages or logs
- Screenshots or examples if applicable

## Feature Requests

We welcome feature requests! Please open an issue and use the "Feature Request" template to describe:

- The problem you're trying to solve
- A clear description of the feature
- Any alternative solutions you've considered
- Additional context or examples

# Format code with Black
black .

# Sort imports with isort
isort .

# Check for style issues with Flake8
flake8

# Run type checking with Mypy
mypy .
```

## Documentation

- Keep docstrings up to date following the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
- Update the README.md and other documentation files if your changes affect them.
- Add comments to explain complex or non-obvious code.

## License

By contributing to this project, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

Thank you for your contribution! Your work helps make MyJobSpy AI better for everyone.
