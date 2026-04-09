from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from app.core.config import settings

if settings.test_mode:
    # SQLite in-memory: use StaticPool so every session shares the same
    # single connection — otherwise each new connection gets a blank DB
    # and the tables created at startup disappear.
    engine = create_engine(
        settings.db_url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _):
        dbapi_conn.execute("PRAGMA foreign_keys=ON")
else:
    engine = create_engine(
        settings.db_url,
        pool_pre_ping=True,
    )

# SessionLocal is a factory: call SessionLocal() to get a new DB session.
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def get_db() -> Session:
    """
    FastAPI dependency — yields a DB session per request, always closes it.

    On exception: rolls back any uncommitted changes before closing.
    This guarantees the DB is never left in a partial state.

    Usage in a route:
        def my_route(db: Session = Depends(get_db)): ...
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
