from __future__ import annotations

from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session


# backend/app/database.py
# Go up to backend/ folder
BASE_DIR = Path(__file__).resolve().parent.parent

# SQLite database file path
DB_PATH = BASE_DIR / "predictions.db"

# SQLAlchemy connection string for SQLite
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Create SQLAlchemy engine
# check_same_thread=False is needed for SQLite when used with FastAPI
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
)

# Create a session factory for database access
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

# Base class for all ORM models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    # Create a new database session for each request
    db = SessionLocal()
    try:
        # Give the session to the API route
        yield db
    finally:
        # Always close the session after request is finished
        db.close()