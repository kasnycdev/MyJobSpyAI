#!/bin/bash
# Script to build the documentation

# Exit on error
set -e

# Change to the docs directory
cd "$(dirname "$0")"

# Generate API documentation
echo "Generating API documentation..."
python generate_api_docs.py

# Create _static directory if it doesn't exist
mkdir -p _static

# Build the documentation
echo "Building documentation..."
sphinx-build -b html . _build/html

echo "Documentation built successfully!"
echo "Open _build/html/index.html in your browser to view the documentation."
