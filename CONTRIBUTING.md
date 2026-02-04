# Contributing

Thanks for your interest in contributing to Evidence-Aware RAG.

## Development Setup

1. Clone the repository
2. Ensure Python 3.11+ is installed
3. Run the backend:

```bash
make backend
```

4. In another terminal, verify with the demo:

```bash
make demo
```

## Makefile Targets

| Command | Description |
|---------|-------------|
| `make backend` | Create venv, install deps, run FastAPI on port 8000 |
| `make frontend` | Create venv, install deps, run Streamlit on port 8501 |
| `make demo` | Run end-to-end smoke test (requires backend running) |
| `make test` | Run pytest test suite |
| `make lint` | Run ruff linter |
| `make clean` | Remove virtual environments |

## Branch Naming

Use prefixes to categorize your work:

- `feat/<topic>` - New features
- `fix/<topic>` - Bug fixes
- `chore/<topic>` - Maintenance, docs, refactoring

Examples:
- `feat/streaming-responses`
- `fix/retrieval-confidence-calibration`
- `chore/update-dependencies`

## Commit Messages

Follow conventional commit style:

```
<type>: <short description>

[optional body]
```

Types:
- `feat:` - New feature
- `fix:` - Bug fix
- `chore:` - Maintenance, docs, config
- `refactor:` - Code restructuring
- `test:` - Adding tests

Examples:
- `feat: add streaming response support`
- `fix: correct confidence threshold for retrieval-only mode`
- `chore: update README with workflow docs`

## Pull Request Process

1. **Never commit directly to main**
2. Create a feature branch from latest main
3. Make focused changes (one topic per PR)
4. Run local checks:
   ```bash
   make test
   make demo
   ```
5. Push and open a PR with a clear title and description
6. Address review feedback
7. Squash and merge when approved

## Workflow Commands

Standard sequence for every change:

```bash
# Start from latest main
git checkout main
git pull origin main

# Create feature branch
git checkout -b feat/my-feature

# Make changes, then verify
make test
make demo

# Commit
git add -A
git status
git commit -m "feat: description of change"

# Push and open PR
git push -u origin feat/my-feature
# Then open PR in GitHub
```

## Questions?

Open an issue for discussion.
