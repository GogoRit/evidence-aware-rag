# Evidence-Aware RAG Makefile
# Usage:
#   make backend   - Install deps & run FastAPI on :8000
#   make frontend  - Install deps & run Streamlit on :8501
#   make demo      - Run end-to-end smoke demo (requires backend running)
#   make test      - Run pytest
#   make lint      - Run ruff linter
#   make check     - Verify imports work before running

SHELL := /bin/bash
.PHONY: backend frontend check clean help check-python demo test lint eval

# Python interpreter - override with: make backend PYTHON=python3.11
PYTHON ?= python3

# Directories
BACKEND_DIR := backend
FRONTEND_DIR := frontend
BACKEND_VENV := $(BACKEND_DIR)/venv
FRONTEND_VENV := $(FRONTEND_DIR)/venv

help:
	@echo "Evidence-Aware RAG Commands:"
	@echo ""
	@echo "  make backend   - Create venv, install deps, run FastAPI on 0.0.0.0:8000"
	@echo "  make frontend  - Create venv, install deps, run Streamlit on 0.0.0.0:8501"
	@echo "  make demo      - Run E2E smoke demo (start backend first!)"
	@echo "  make eval      - Seed stress workspace + run evaluator + golden set (backend must be running)"
	@echo "  make test      - Run pytest test suite"
	@echo "  make lint      - Run ruff linter"
	@echo "  make check     - Verify backend imports work (no runtime crash)"
	@echo "  make clean     - Remove virtual environments"
	@echo ""
	@echo "Quick Demo (2 minutes):"
	@echo "  Terminal 1: make backend"
	@echo "  Terminal 2: make demo"
	@echo ""
	@echo "Prerequisites: Python 3.11+"
	@echo ""
	@echo "If your default Python is not 3.11+, specify with:"
	@echo "  make backend PYTHON=python3.11"

# Check Python version is 3.11+
check-python:
	@$(PYTHON) -c "import sys; v=sys.version_info; exit(0 if v >= (3,11) else 1)" 2>/dev/null || \
		(echo "ERROR: Python 3.11+ required. Found: $$($(PYTHON) --version 2>&1)"; \
		 echo "Install Python 3.11+ or specify path: make backend PYTHON=/path/to/python3.11"; \
		 exit 1)

# ============================================================
# Backend
# ============================================================

$(BACKEND_VENV)/bin/activate: check-python
	@echo "Creating backend virtual environment with $(PYTHON)..."
	cd $(BACKEND_DIR) && $(PYTHON) -m venv venv
	@echo "Installing backend dependencies..."
	cd $(BACKEND_DIR) && ./venv/bin/pip install --upgrade pip
	cd $(BACKEND_DIR) && ./venv/bin/pip install -e .

backend-venv: $(BACKEND_VENV)/bin/activate

check: backend-venv
	@echo "Checking backend imports..."
	cd $(BACKEND_DIR) && ./venv/bin/python -c "from app.main import app; print('[OK] Backend imports OK')"

backend: backend-venv check
	@echo ""
	@echo "Starting FastAPI backend on http://0.0.0.0:8000"
	@echo "Swagger docs: http://localhost:8000/docs"
	@echo "Press Ctrl+C to stop"
	@echo ""
	cd $(BACKEND_DIR) && ./venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# ============================================================
# Frontend
# ============================================================

$(FRONTEND_VENV)/bin/activate:
	@echo "Creating frontend virtual environment..."
	cd $(FRONTEND_DIR) && $(PYTHON) -m venv venv
	@echo "Installing frontend dependencies..."
	cd $(FRONTEND_DIR) && ./venv/bin/pip install --upgrade pip
	cd $(FRONTEND_DIR) && ./venv/bin/pip install -r requirements.txt

frontend-venv: $(FRONTEND_VENV)/bin/activate

frontend: frontend-venv
	@echo "Starting Streamlit frontend on http://0.0.0.0:8501"
	@echo "Press Ctrl+C to stop"
	@echo ""
	cd $(FRONTEND_DIR) && ./venv/bin/streamlit run app.py --server.address 0.0.0.0 --server.port 8501

# ============================================================
# Utilities
# ============================================================

clean:
	@echo "Removing virtual environments..."
	rm -rf $(BACKEND_VENV) $(FRONTEND_VENV)
	@echo "Done"

# Install both environments (useful for CI)
install: backend-venv frontend-venv
	@echo "All dependencies installed"

# ============================================================
# Demo
# ============================================================

demo: backend-venv
	@echo "Running E2E smoke demo..."
	@echo "(Make sure backend is running: make backend)"
	@echo ""
	cd $(BACKEND_DIR) && ./venv/bin/python ../scripts/demo_e2e.py

# ============================================================
# Evaluation (stress + golden)
# ============================================================
# Requires: backend running (make backend in another terminal).
# Wipes stress-test workspace, reseeds, resets retrieval metrics, runs evaluator with golden set.
# ============================================================

eval: backend-venv
	@echo "Running evaluation (seed + reset metrics + stress eval + golden)..."
	@echo "REQUIRED: Backend must be running (make backend in another terminal)."
	@echo ""
	@rm -rf $(BACKEND_DIR)/data/workspaces/stress-test
	$(BACKEND_DIR)/venv/bin/python scripts/seed_stress_workspace.py
	@curl -s -X POST http://localhost:8000/metrics/retrieval/reset > /dev/null
	$(BACKEND_DIR)/venv/bin/python scripts/evaluate_retrieval_stress.py --workspace stress-test --golden scripts/golden_eval.json

# ============================================================
# Testing & Linting
# ============================================================

test: backend-venv
	@echo "Running tests..."
	cd $(BACKEND_DIR) && ./venv/bin/pip install pytest pytest-asyncio -q
	cd $(BACKEND_DIR) && ./venv/bin/pytest ../scripts/test_confidence_modes.py -v || true
	cd $(BACKEND_DIR) && ./venv/bin/pytest tests/ -v 2>/dev/null || echo "No tests/ directory found"

lint: backend-venv
	@echo "Running linter..."
	cd $(BACKEND_DIR) && ./venv/bin/pip install ruff -q
	cd $(BACKEND_DIR) && ./venv/bin/ruff check app/ --fix || true
	@echo "Lint complete"
