"""Database configuration for MyJobSpyAI."""

import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from . import settings

logger = logging.getLogger('database')

# Create SQLAlchemy engine
engine = create_engine(
    f"postgresql://{settings.settings.database.user}:{settings.settings.database.password}@"
    f"{settings.settings.database.host}:{settings.settings.database.port}/"
    f"{settings.settings.database.name}",
    pool_size=settings.settings.database.pool_size,
    max_overflow=settings.settings.database.max_overflow,
    echo=settings.settings.database.echo,
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Dependency for getting database session."""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """Initialize database tables."""
    logger.info("Initializing database...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def get_db_url() -> str:
    """Get the database URL from settings."""
    return (
        f"postgresql://{settings.settings.database.user}:"
        f"{settings.settings.database.password}@"
        f"{settings.settings.database.host}:"
        f"{settings.settings.database.port}/"
        f"{settings.settings.database.name}"
    )
