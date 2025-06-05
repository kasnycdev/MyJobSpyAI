#!/bin/bash

# Exit on error
set -e

# Print commands as they are executed
set -x

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -e .[dev]

# Run linters
echo "Running linters..."
flake8 myjobspyai tests
black --check myjobspyai tests
isort --check-only myjobspyai tests

# Run type checking
echo "Running type checking..."
mypy myjobspyai tests

# Run tests with coverage
echo "Running tests..."
pytest --cov=myjobspyai --cov-report=term-missing -v tests/

echo "All tests passed!"
