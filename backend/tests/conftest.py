"""Pytest configuration and fixtures."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add backend directory to path so tests can import app modules
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

# Mock external dependencies that may not be installed in test environment
# This allows running basic unit tests without the full dependency stack
_mock_modules = [
    'structlog',
    'aiofiles', 
    'pydantic_settings',
    'fastapi',
    'faiss',
    'numpy',
]

for mod in _mock_modules:
    if mod not in sys.modules:
        sys.modules[mod] = MagicMock()
