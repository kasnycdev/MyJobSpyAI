#!/bin/bash

# Install dependencies
pip install -r requirements-docs.txt

# Build documentation
mkdocs build

# Serve documentation locally
mkdocs serve
