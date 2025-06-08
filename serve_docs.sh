#!/bin/bash

# Install dependencies
pip install -r requirements-docs.txt

# Build and serve documentation
mkdocs serve --dev-addr 0.0.0.0:8000
