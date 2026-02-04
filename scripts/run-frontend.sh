#!/bin/bash
# Run the Streamlit frontend
# Usage: ./scripts/run-frontend.sh [python_path]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FRONTEND_DIR="$PROJECT_ROOT/frontend"

# Python interpreter (override via argument)
PYTHON="${1:-python3}"

echo "================================================"
echo "RAG Frontend Startup"
echo "================================================"
echo "Python: $PYTHON ($($PYTHON --version 2>&1))"
echo "Frontend: $FRONTEND_DIR"
echo ""

cd "$FRONTEND_DIR"

# Create venv if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
fi

# Activate and install deps
echo "Installing dependencies..."
./venv/bin/pip install --upgrade pip -q
./venv/bin/pip install -r requirements.txt -q

echo ""
echo "Starting Streamlit on http://0.0.0.0:8501"
echo "Press Ctrl+C to stop"
echo ""

# Run streamlit
exec ./venv/bin/streamlit run app.py --server.address 0.0.0.0 --server.port 8501
