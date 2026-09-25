"""
Database Connection and Session Management.

Supports:
- PostgreSQL + PostGIS via psycopg/SQLAlchemy
- Resilient fallback to SQLite if PostgreSQL is offline or unconfigured
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

# Normalize Render/Heroku PostgreSQL URLs:
# Render provides "postgres://..." or "postgresql://..." which requires psycopg driver in SQLAlchemy 2
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://") and not DATABASE_URL.startswith("postgresql+psycopg://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

# Test primary connection and fallback gracefully if needed
engine = None
if DATABASE_URL and ("postgresql" in DATABASE_URL or "postgres" in DATABASE_URL):
    try:
        # Quick connect check with 3s timeout
        test_engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True,
            connect_args={"connect_timeout": 3}
        )
        with test_engine.connect() as conn:
            pass
        engine = test_engine
        print(f"[Database] Successfully connected to PostGIS PostgreSQL database at {DATABASE_URL.split('@')[-1]}.")
    except Exception as e:
        print(f"[Database] Primary PostGIS connection unavailable ({e}). Activating SQLite local fallback mode.")
        engine = None

if engine is None:
    # Use SQLite fallback database
    fallback_path = os.environ.get(
        "SQLITE_PATH",
        os.path.join(os.path.dirname(__file__), "..", "..", "sonar_intel_fallback.db")
    )
    fallback_dir = os.path.dirname(os.path.abspath(fallback_path))
    if fallback_dir:
        os.makedirs(fallback_dir, exist_ok=True)
    engine = create_engine(
        f"sqlite:///{os.path.abspath(fallback_path)}",
        connect_args={"check_same_thread": False}
    )
    print(f"[Database] Initialized local fallback SQLite at {fallback_path}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initializes tables in database."""
    try:
        from backend.app.database import models
        Base.metadata.create_all(bind=engine)
        print("[Database] Schema synchronized successfully.")
    except Exception as ex:
        print(f"[Database] Warning during schema creation: {ex}")

# Ensure tables exist immediately upon import
init_db()

def get_db():
    """FastAPI Dependency for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

