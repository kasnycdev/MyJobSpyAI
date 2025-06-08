Contributing to MyJobSpyAI
=========================

Thank you for your interest in contributing to MyJobSpyAI! We welcome contributions from the community to help improve this project.

Table of Contents
----------------

.. contents::
   :local:
   :depth: 2

Ways to Contribute
-----------------

Report Bugs
~~~~~~~~~~
- Search existing issues to avoid duplicates
- Provide detailed information about your environment
- Include steps to reproduce the issue
- Share any relevant error messages or logs

Suggest Enhancements
~~~~~~~~~~~~~~~~~~
- Describe the feature or improvement
- Explain why it would be valuable
- Provide examples of how it would work

Submit Code Changes
~~~~~~~~~~~~~~~~~~
- Follow the development setup instructions
- Create a feature branch for your changes
- Submit a pull request with a clear description

Improve Documentation
~~~~~~~~~~~~~~~~~~~~
- Fix typos and improve clarity
- Add missing documentation
- Update outdated information

Development Setup
----------------

Prerequisites
~~~~~~~~~~~~
- Python 3.8+
- Git
- pip
- (Optional) virtualenv or conda

Installation
~~~~~~~~~~~

1. **Fork the repository** on GitHub

2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/MyJobSpyAI.git
   cd MyJobSpyAI
   ```

3. **Set up a virtual environment**:
   ```bash
   # Using venv
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Or using conda
   conda create -n myjobspyai python=3.10
   conda activate myjobspyai
   ```

4. **Install the package in development mode**:
   ```bash
   pip install -e .[dev]
   ```

5. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

Running Tests
~~~~~~~~~~~~

Run the full test suite:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=myjobspyai --cov-report=term-missing
```

Run specific test files or test cases:
```bash
pytest tests/test_analyzer.py
pytest tests/test_analyzer.py::TestAnalyzer::test_specific_case
```

Code Quality
~~~~~~~~~~~

Before submitting code, run these checks:

1. **Linting**:
   ```bash
   flake8 myjobspyai tests
   black --check myjobspyai tests
   ```

2. **Type checking**:
   ```bash
   mypy myjobspyai
   ```

3. **Formatting** (if needed):
   ```bash
   black myjobspyai tests
   ```

Documentation
~~~~~~~~~~~~

Build the documentation locally:
```bash
cd docs
make html
```

View the documentation by opening `_build/html/index.html` in your browser.

Pull Request Guidelines
----------------------

1. **Branch Naming**:
   - Use descriptive branch names (e.g., `feature/add-llm-caching`, `bugfix/fix-login-issue`)
   - Prefix with `feature/`, `bugfix/`, `docs/`, `test/`, etc.

2. **Commit Messages**:
   - Use the imperative mood ("Add feature" not "Added feature" or "Adds feature")
   - Keep the first line under 50 characters
   - Include a blank line between the subject and body
   - Reference issues and pull requests liberally

   Example:
   .. code-block:: text

      feat(ollama): Add support for Ollama LLM provider

      - Implement OllamaProvider class
      - Add configuration options
      - Update documentation

      Fixes #123

3. **Pull Request Process**:
   - Ensure tests pass and coverage remains high
   - Update documentation as needed
   - Request reviews from maintainers
   - Address all review comments

4. **Code Review**:
   - Be open to feedback
   - Keep discussions focused on the code
   - Be respectful and constructive

Project Structure
-----------------

myjobspyai/
├── src/                    # Source code
│   ├── myjobspyai/         # Main package
│   │   ├── analysis/       # Analysis modules
│   │   ├── models/         # Data models
│   │   ├── providers/      # LLM providers
│   │   ├── utils/          # Utility functions
│   │   └── __init__.py
│   └── tests/              # Test files
├── docs/                   # Documentation
├── examples/               # Example scripts
├── .github/               # GitHub workflows and templates
├── .pre-commit-config.yaml # Pre-commit hooks
├── pyproject.toml         # Project configuration
└── README.md              # Project README

Code Style
---------

- Follow `PEP 8 <https://www.python.org/dev/peps/pep-0008/>`_ style guide
- Use type hints for all functions and methods
- Write docstrings following `Google style <https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings>`_
- Keep functions small and focused (preferably < 50 lines)
- Use meaningful variable and function names
- Add comments to explain "why" not just "what"

Testing Guidelines
-----------------

- Write tests for all new features and bug fixes
- Follow the Arrange-Act-Assert pattern
- Use descriptive test names
- Test edge cases and error conditions
- Keep tests independent and isolated
- Mock external dependencies

Documentation Standards
----------------------

- Keep documentation up to date with code changes
- Use reStructuredText for all documentation
- Add docstrings to all public modules, classes, and functions
- Include examples in docstrings
- Document all public API endpoints
- Keep README.md up to date
- Update CHANGELOG.md for all user-facing changes

Release Process
--------------

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create a release tag
4. Push the tag to trigger the release workflow

Code of Conduct
---------------

We follow the `Contributor Covenant <https://www.contributor-covenant.org/>`_ code of conduct. By participating, you are expected to uphold this code.

Getting Help
------------

- For questions, open a discussion on GitHub
- For bugs, open an issue with the bug template
- For security issues, please email security@example.com

Thank you for contributing to MyJobSpyAI! Your help is greatly appreciated.
