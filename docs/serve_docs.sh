#!/bin/bash

# Start a simple HTTP server to serve the documentation
PORT=8000

# Check if Python 3 is available
if command -v python3 &> /dev/null; then
    echo "Starting Python 3 HTTP server on port $PORT..."
    python3 -m http.server $PORT --directory _build/
elif command -v python &> /dev/null; then
    echo "Starting Python HTTP server on port $PORT..."
    python -m http.server $PORT --directory _build/
else
    echo "Python not found. Please install Python to serve the documentation."
    exit 1
fi

echo "Documentation is available at: http://localhost:$PORT"
echo "Press Ctrl+C to stop the server."
