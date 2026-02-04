# Evidence-Aware RAG

A production-grade Retrieval-Augmented Generation system that prioritizes **evidence transparency**. Features confidence-gated responses, intelligent refusal when evidence is insufficient, mode-aware thresholds (retrieval-only vs generation), and comprehensive observability. Works offline with local embeddings - no API keys required for the demo.

## Demo (2 Minutes)

Run the complete system locally without any API keys:

```bash
# Terminal 1 - Start backend
make backend

# Terminal 2 - Run demo
make demo
```

The demo ingests a sample policy document and runs test queries, showing confidence scores, sources, and refusal behavior.

## Quick Start

**Prerequisites:** Python 3.11+

```bash
# Start backend (port 8000)
make backend

# Start frontend (port 8501) - in another terminal
make frontend
```

Then open http://localhost:8501 to use the web interface.

## Configuration

Copy `.env.example` to `.env` to customize settings. Key options:

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDINGS_BACKEND` | `local` (sentence-transformers) or `openai` | `local` |
| `GENERATION_ENABLED` | Enable LLM generation | `false` |
| `CONFIDENCE_THRESHOLD` | Minimum confidence for answers | `0.4` |
| `OPENAI_API_KEY` | Required only if using OpenAI embeddings/generation | - |

**Note:** With `EMBEDDINGS_BACKEND=local` (default), the system works completely offline using sentence-transformers. No API keys needed.

## Project Structure

```
evidence-aware-rag/
├── backend/
│   ├── app/
│   │   ├── main.py           # FastAPI application
│   │   ├── config.py         # Settings (pydantic-settings)
│   │   ├── routes/           # API endpoints
│   │   │   ├── health.py     # Health checks
│   │   │   ├── ingest.py     # Document ingestion
│   │   │   ├── chat.py       # Query endpoint
│   │   │   └── metrics.py    # Observability
│   │   └── modules/
│   │       ├── ingestion/    # Loaders, chunking, storage
│   │       ├── retrieval/    # FAISS index, confidence scoring
│   │       ├── generation/   # LLM routing (cheap/expensive)
│   │       └── observability/# Latency, cost tracking
│   ├── data/workspaces/      # Document storage (gitignored)
│   └── demo_data/            # Sample documents for demo
├── frontend/
│   └── app.py                # Streamlit UI
├── scripts/
│   ├── demo_e2e.py           # End-to-end demo script
│   └── test_confidence_modes.py
├── Makefile                  # Build/run commands
└── .env.example              # Configuration template
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with component status |
| `/workspaces` | GET | List available workspaces |
| `/ingest` | POST | Upload and process documents |
| `/chat` | POST | Query with confidence-gated response |
| `/metrics` | GET | System metrics (latency, costs, counts) |

## Development

### Makefile Targets

| Command | Description |
|---------|-------------|
| `make backend` | Start FastAPI server on port 8000 |
| `make frontend` | Start Streamlit UI on port 8501 |
| `make demo` | Run E2E smoke test |
| `make test` | Run pytest suite |
| `make lint` | Run ruff linter |
| `make clean` | Remove virtual environments |

### Developer Workflow

We follow a strict branch-based workflow:

1. **Create branch** from latest main:
   ```bash
   git checkout main && git pull
   git checkout -b feat/my-feature
   ```

2. **Run checks** before committing:
   ```bash
   make test
   make demo
   ```

3. **Commit** with conventional style:
   ```bash
   git commit -m "feat: add new feature"
   ```

4. **Push and open PR**:
   ```bash
   git push -u origin feat/my-feature
   ```

Branch prefixes: `feat/`, `fix/`, `chore/`
Commit prefixes: `feat:`, `fix:`, `chore:`, `refactor:`, `test:`

See [CONTRIBUTING.md](CONTRIBUTING.md) for full details.

## License

MIT License - see [LICENSE](LICENSE) for details.
