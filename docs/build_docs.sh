#!/bin/bash
# Script to build the documentation

# Exit on error
set -e

# Change to the docs directory
cd "$(dirname "$0")"

# Generate API documentation
echo "Generating API documentation..."
python generate_api_docs.py

# Create assets directory if it doesn't exist
mkdir -p assets

# Build the documentation
echo "Building documentation..."
mkdocs build --clean

# Serve documentation
echo "Starting documentation server..."
mkdocs serve
echo "Open _build/html/index.html in your browser to view the documentation."
