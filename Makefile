.PHONY: test test-cov lint format check-types install install-dev clean docs docs-serve

# Variables
PYTHON = python3
PIP = pip3
PYTEST = pytest
COVERAGE = coverage
FLAKE8 = flake8
BLACK = black
MYPY = mypy
ISORT = isort

# Directories
SRC_DIR = src
TESTS_DIR = tests
DOCS_DIR = docs

# Default target
all: test lint check-types

# Documentation targets
docs:
	$(PYTHON) -m pip install -r requirements-docs.txt
	mkdocs build

docs-serve:
	$(PYTHON) -m pip install -r requirements-docs.txt
	mkdocs serve

# Install the package in development mode with docs
install-dev:
	$(PIP) install -e .
	$(PIP) install -r requirements-test.txt
	$(PIP) install -r requirements-docs.txt

# Install the package in development mode
install-dev:
	$(PIP) install -e .
	$(PIP) install -r requirements-test.txt

# Run tests
test:
	$(PYTHON) -m pytest $(TESTS_DIR) -v

# Run tests with coverage
test-cov:
	$(PYTHON) -m pytest $(TESTS_DIR) -v --cov=$(SRC_DIR) --cov-report=term-missing

# Run linter
lint:
	$(FLAKE8) $(SRC_DIR) $(TESTS_DIR)
	$(BLACK) --check $(SRC_DIR) $(TESTS_DIR)

# Format code
format:
	$(BLACK) $(SRC_DIR) $(TESTS_DIR)
	$(ISORT) $(SRC_DIR) $(TESTS_DIR)

# Run type checking
check-types:
	$(MYPY) $(SRC_DIR) $(TESTS_DIR)

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type d -name ".mypy_cache" -exec rm -r {} +
	find . -type f -name "*.py[co]" -delete
	find . -type f -name "*~" -delete
	find . -type f -name ".coverage" -delete
	find . -type f -name "coverage.xml" -delete
	find . -type f -name "htmlcov" -exec rm -r {} +

# Help target
help:
	@echo "Available targets:"
	@echo "  install-dev     Install the package in development mode"
	@echo "  test            Run tests"
	@echo "  test-cov        Run tests with coverage report"
	@echo "  lint            Run linter"
	@echo "  format          Format code"
	@echo "  check-types     Run type checking"
	@echo "  clean           Clean up temporary files"
	@echo "  help            Show this help message"
