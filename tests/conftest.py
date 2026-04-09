"""
Shared test fixtures.

How the test database works
---------------------------
Each test gets its own fresh SQLite in-memory database.
We use StaticPool so that all SQLAlchemy sessions share the exact same
underlying connection — otherwise each new session would get a blank DB
and the tables created a moment ago would be invisible to it.

How the test HTTP client works
------------------------------
FastAPI lets you swap out any dependency via app.dependency_overrides.
We use this to replace the real get_db() with a version that points at
the test database instead of the production one.
TestClient (from Starlette, bundled with FastAPI) lets us make real HTTP
requests against the app without a running server.
"""

import os
import pytest

# Force test settings BEFORE any app module is imported.
# TEST_MODE=1              → SQLite in-memory instead of MS SQL
# LLM_PROVIDER=mock        → never call the real Anthropic API in tests
# TRANSCRIPTION_PROVIDER=mock → never call the real Whisper API in tests
# AUDIO_UPLOAD_DIR=/tmp/... → don't try to write to /data which is read-only
os.environ["TEST_MODE"] = "1"
os.environ["LLM_PROVIDER"] = "mock"
os.environ["TRANSCRIPTION_PROVIDER"] = "mock"
os.environ["AUDIO_UPLOAD_DIR"] = "/tmp/radiology_test_audio"

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from starlette.testclient import TestClient

from app.db.models import Base
from app.db.connection import get_db
from main import app


def _make_test_engine():
    """Create a brand new in-memory SQLite engine with tables created."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return engine


@pytest.fixture
def db():
    """
    Yields a SQLAlchemy session connected to a fresh in-memory SQLite DB.
    Tables are created before the test and dropped after — complete isolation.
    """
    engine = _make_test_engine()
    TestingSession = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)


@pytest.fixture
def client(db):
    """
    Yields a TestClient wired to the test database.

    The key line is app.dependency_overrides[get_db] — this tells FastAPI
    'whenever a route asks for get_db, give it this test version instead.'
    We clear the override after the test so other tests aren't affected.
    """
    def override_get_db():
        yield db

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c
    app.dependency_overrides.clear()
