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

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://sonar_user:sonar_password@localhost:5433/sonar_intel"
)

# Test primary connection and fallback gracefully if needed
engine = None
try:
    if "postgresql" in DATABASE_URL:
        # Quick connect check with 2s timeout
        test_engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=False,
            connect_args={"connect_timeout": 2}
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
    fallback_path = os.path.join(os.path.dirname(__file__), "..", "..", "sonar_intel_fallback.db")
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

