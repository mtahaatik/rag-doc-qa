"""
FastAPI application entrypoint. Wires together routes, lifespan, and config.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import ingest, query
from app.core.config import settings
from app.core.db import close_pool, init_pool
from app.core.logging import setup_logging

setup_logging(settings.log_level)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: open the connection pool, register the vector type
    log.info("starting up, opening db pool")
    await init_pool()
    yield
    log.info("shutting down, closing db pool")
    await close_pool()


app = FastAPI(
    title="RAG Document QA",
    description="Production-style retrieval-augmented generation service",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(ingest.router)
app.include_router(query.router)


@app.get("/health", tags=["health"])
async def health():
    # Keep this cheap. No DB call, no model load. Used by docker healthcheck
    # and orchestrators. If you need a deep health check, make a separate
    # /ready endpoint.
    return {"status": "ok"}
