#!/bin/bash
# Run the FastAPI backend
# Usage: ./scripts/run-backend.sh [python_path]
#
# Examples:
#   ./scripts/run-backend.sh                    # Uses python3
#   ./scripts/run-backend.sh python3.11         # Uses python3.11
#   ./scripts/run-backend.sh /opt/python311/bin/python3

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_ROOT/backend"

# Python interpreter (override via argument)
PYTHON="${1:-python3}"

echo "================================================"
echo "RAG Backend Startup"
echo "================================================"
echo "Python: $PYTHON ($($PYTHON --version 2>&1))"
echo "Backend: $BACKEND_DIR"
echo ""

# Check Python version (3.11+ required for type syntax)
$PYTHON -c "import sys; v=sys.version_info; exit(0 if v >= (3,11) else 1)" 2>/dev/null || {
    echo "ERROR: Python 3.11+ required. Found: $($PYTHON --version 2>&1)"
    echo ""
    echo "The codebase uses Python 3.10+ type syntax (e.g., 'str | None')."
    echo ""
    echo "Options:"
    echo "  1. Install Python 3.11+:"
    echo "     brew install python@3.11"
    echo ""
    echo "  2. Run with specific Python:"
    echo "     ./scripts/run-backend.sh /opt/homebrew/bin/python3.11"
    echo "     ./scripts/run-backend.sh python3.11"
    echo ""
    exit 1
}

cd "$BACKEND_DIR"

# Create venv if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
fi

# Activate and install deps
echo "Installing dependencies..."
./venv/bin/pip install --upgrade pip -q
./venv/bin/pip install -r requirements.txt -q

# Verify imports work
echo "Verifying imports..."
./venv/bin/python -c "from app.main import app; print('[OK] Imports OK')" || {
    echo "ERROR: Import failed. Check error above."
    exit 1
}

echo ""
echo "Starting server on http://0.0.0.0:8000"
echo "Swagger docs: http://localhost:8000/docs"
echo "Press Ctrl+C to stop"
echo ""

# Run uvicorn
exec ./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
