"""FastAPI application entrypoint."""

import structlog
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routes import health, ingest, chat, metrics, workspaces
from app.modules.observability import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    settings = get_settings()
    logger = structlog.get_logger()
    
    # Startup
    logger.info("Starting RAG Backend", env=settings.app_env)
    
    # Ensure workspace directory exists
    settings.workspaces_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Workspaces directory ready", path=str(settings.workspaces_dir))
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG Backend")


def create_app() -> FastAPI:
    """Application factory."""
    settings = get_settings()
    setup_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        description="Production-grade Retrieval-Augmented Generation System",
        version="0.1.0",
        docs_url="/docs" if not settings.is_production else None,
        redoc_url="/redoc" if not settings.is_production else None,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.debug else ["http://localhost:8501"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(workspaces.router, prefix="/workspaces", tags=["Workspaces"])
    app.include_router(ingest.router, prefix="/ingest", tags=["Ingestion"])
    app.include_router(chat.router, prefix="/chat", tags=["Chat"])
    app.include_router(metrics.router, prefix="/metrics", tags=["Metrics"])

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
