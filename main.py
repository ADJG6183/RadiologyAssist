"""
Application entry point.

Run with:
    TEST_MODE=1 uvicorn main:app --reload

The `lifespan` context manager replaces the older @app.on_event("startup")
pattern. It runs setup code before the server starts accepting requests,
and teardown code after it shuts down.  Using it for table creation means
SQLite (TEST_MODE=1) always has its schema ready without a separate migration step.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.studies import router as studies_router
from app.core.logging import get_logger
from app.db.connection import engine
from app.db.models import Base

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    # Create all tables if they don't exist yet.
    # In production (MS SQL) the schema is managed by Alembic migrations
    # and the tables already exist, so create_all is a safe no-op.
    # In TEST_MODE=1 (SQLite in-memory) this creates the tables fresh.
    Base.metadata.create_all(bind=engine)
    log.info("app.startup", db_url=engine.url.render_as_string(hide_password=True))
    yield
    # --- shutdown (nothing to clean up) ---


app = FastAPI(
    title="Radiology AI Assistant",
    description="Dictation → draft radiology report pipeline backed by Claude.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(studies_router, prefix="/api/v1")

# Serve the single-page UI from /
# The API routes registered above always win because FastAPI evaluates routes
# in order of registration — the catch-all below only fires for non-API paths.
@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))
