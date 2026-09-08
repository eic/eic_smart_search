from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


settings = get_settings()

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    # Sized against the anyio threadpool (capped in main.py): enough
    # connections that request handlers never block on pool checkout, with
    # a short timeout so overload fails fast instead of hanging 30 s.
    pool_size=20,
    max_overflow=10,
    pool_timeout=5,
    pool_recycle=1800,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

